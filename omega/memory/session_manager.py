import json
import logging
from omega.storage.memory_queries import (
    create_session,
    get_session,
    find_resumable_session,
    reopen_session,
    close_session,
    get_session_messages,
    append_message,
    create_session_summary,
    create_compression_summary_and_mark,
    get_final_session_summary,
    get_session_summary_spans,
    find_orphaned_sessions
)
from omega.memory.context_build import estimate_tokens, build_system_context
from omega.memory.core import ensure_memory_dir
from omega.llm.client import get_llm_provider
from omega.environment.conf_loader import omega_settings

logger = logging.getLogger("SessionManager")

COMPRESSION_PROMPT = """Summarize the following conversation segment concisely. Preserve:
- Key topics discussed
- Any decisions, conclusions, or facts established
- Any user preferences or personal information revealed
- Any tool calls made and their outcomes

Be factual and compact. Do not add information that wasn't in the coversation."""

SESSION_CLOSE_PROMPT = """Provide a comprehensive summary of this entire conversation session.
Include:
- All major topics discussed
- Key facts, decisions, and outcomes
- Any preferences or personal details the user shared
- Tools used and their results
- The overall purpose/goal of the session if one was apparent

Be thorough but concise. This summary will be used to provide continuity in future sessions"""


class SessionManager:
    def __init__(self):
        self.active_session_id: str | None = None
        self.system_context: str | None = None
        self.llm_client = get_llm_provider()
        ensure_memory_dir()

    def _require_active_session_id(self) -> str:
        if self.active_session_id is None:
            raise RuntimeError("No active session is available")
        return self.active_session_id

    async def ensure_session(self, resume: bool = False, session_id: str | None = None) -> str:
        if session_id is not None:
            return await self.attach_session(session_id)
        if self.active_session_id:
            return self.active_session_id
        if resume:
            resume_session_id = await self.resume_latest_session()
            if resume_session_id:
                return resume_session_id

        await self._recover_orphaned_sessions()
        self.active_session_id = await create_session()
        self.system_context = await build_system_context()
        logger.info(f"Session started: {self.active_session_id}")
        return self.active_session_id

    async def attach_session(self, session_id: str) -> str:
        if self.active_session_id and self.active_session_id != session_id:
            raise RuntimeError("This agent instance is already bound to another session")

        session = await get_session(session_id)
        if session is None:
            raise ValueError("Unknown session_id")
        if session["status"] == "closed":
            if not await reopen_session(session_id):
                raise RuntimeError(f"Could not reopen session {session_id}")

        self.active_session_id = str(session["id"])
        self.system_context = await build_system_context()
        logger.info(f"Attached to session {session_id}")
        return self.active_session_id

    async def resume_latest_session(self) -> str | None:
        if self.active_session_id:
            return self.active_session_id

        candidate = await find_resumable_session()
        if not candidate:
            logger.info("No prior session is available to resume")
            return None

        session_id = str(candidate["id"])
        reopened = await reopen_session(session_id)
        if not reopened:
            logger.info("Resuming already-active session %s", session_id)

        self.active_session_id = session_id
        self.system_context = await build_system_context()
        logger.info("Resumed session %s", session_id)
        return self.active_session_id

    async def start_new_session(self) -> str:
        if self.active_session_id:
            await self.close_current_session()

        self.active_session_id = await create_session()
        self.system_context = await build_system_context()
        logger.info(f"New session started: {self.active_session_id}")
        return self.active_session_id

    async def close_current_session(self):
        if not self.active_session_id:
            return

        messages = await get_session_messages(self.active_session_id, include_compressed=True)
        if len(messages) >= 2:
            try:
                await self._create_session_summary(messages)
            except Exception as e:
                logger.error(f"Failed to create session summary during session close: {e}")
                return

        await close_session(self.active_session_id)
        logger.info(f"Session {self.active_session_id} closed")
        self.active_session_id = None
        self.system_context = None

    async def _create_session_summary_for(self, session_id: str, messages: list[dict], trigger: str = "session_close"):
        existing = await get_final_session_summary(session_id)
        if existing:
            logger.info(f"Final session summary already exists for {session_id}")
            return

        conversation_text = self._format_message_for_summary(messages)
        summary = await self.llm_client.generate_answer(
            SESSION_CLOSE_PROMPT, conversation_text
        )
        await create_session_summary(
            session_id=session_id,
            summary_kind=trigger,
            content=summary,
            first_message_id=str(messages[0]["id"]) if messages else None,
            last_message_id=str(messages[-1]["id"]) if messages else None,
            message_count=len(messages),
            metadata={"trigger": trigger},
        )
        logger.info(
            f"Created session summary for {session_id} ({len(messages)} messages)"
        )

    async def _create_session_summary(self, messages: list[dict]):
        await self._create_session_summary_for(self._require_active_session_id(), messages, trigger="session_close")

    async def add_message(self, role: str, content: str, tool_name: str | None = None, metadata: dict | None = None) -> str:
       return await append_message(self._require_active_session_id(), role, content, tool_name, metadata)

    @staticmethod
    def _metadata_dict(message: dict) -> dict:
        metadata = message.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _is_valid_tool_arguments(arguments) -> bool:
        if not isinstance(arguments, str):
            return False
        try:
            return isinstance(json.loads(arguments), dict)
        except json.JSONDecodeError:
            return False

    def _provider_transaction_units(self, messages: list[dict]) -> list[list[dict]]:
        units: list[list[dict]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            metadata = self._metadata_dict(message)
            tool_calls = metadata.get("tool_calls") if message["role"] == "assistant" else None
            if tool_calls is not None:
                if not isinstance(tool_calls, list) or not tool_calls:
                    index += 1
                    continue
                call_ids = [call.get("id") for call in tool_calls if isinstance(call, dict)]
                valid_calls = all(
                    isinstance(call, dict)
                    and call.get("type") == "function"
                    and isinstance(call.get("function"), dict)
                    and isinstance(call["function"].get("name"), str)
                    and call["function"]["name"]
                    and self._is_valid_tool_arguments(call["function"].get("arguments"))
                    for call in tool_calls
                )
                if not valid_calls or len(call_ids) != len(tool_calls) or not all(call_ids) or len(set(call_ids)) != len(call_ids):
                    index += 1
                    continue
                results = messages[index + 1:index + 1 + len(call_ids)]
                result_ids = [
                    self._metadata_dict(result).get("tool_call_id")
                    for result in results
                    if result["role"] == "tool"
                ]
                if len(results) == len(call_ids) and set(result_ids) == set(call_ids) and len(set(result_ids)) == len(result_ids):
                    units.append([message, *results])
                    index += len(results) + 1
                    continue
                index += 1
                while index < len(messages) and messages[index]["role"] == "tool":
                    index += 1
                continue
            if message["role"] not in {"user", "assistant"}:
                index += 1
                continue
            units.append([message])
            index += 1
        return units


    async def get_context_messages(self) -> list[dict]:
        session_id = self._require_active_session_id()
        summaries = await get_session_summary_spans(session_id)
        units = self._provider_transaction_units(await get_session_messages(session_id))
        messages = await get_session_messages(session_id)
        context = []
        if summaries:
            summary_text = "\n\n".join(summary["content"] for summary in summaries)
            context.append({"role": "system", "content": f"Earlier in this session:\n{summary_text}"})
        for unit in units:
            for message in unit:
                role = message["role"]
                metadata = self._metadata_dict(message)
                if role == "assistant":
                    context_message = {"role": "assistant", "content": message["content"]}
                    if metadata.get("tool_calls"):
                        context_message["tool_calls"] = metadata["tool_calls"]
                    context.append(context_message)
                elif role == "user":
                    context.append({"role": "user", "content": message["content"]})
                else:
                    context.append({
                        "role": "tool",
                        "tool_call_id": metadata["tool_call_id"],
                        "content": message["content"],
                    })
        return context

    async def check_and_compress(self, tool_schemas_text: str = ""):
        messages = await get_session_messages(self._require_active_session_id())
        if len(messages) < 4:
            return

        all_content = " ".join(m["content"] for m in messages) + tool_schemas_text
        total_tokens = estimate_tokens(all_content)
        budget = omega_settings.session_token_budget
        ratio = total_tokens / budget

        try:
            if ratio >= omega_settings.session_emergency_ratio:
                logger.warning(f"Emergency compression triggered ({ratio:.0%} of budget)")
                await self._compress(messages, aggressive=True)
            elif ratio >= omega_settings.session_compression_ratio:
                logger.info(f"standard compression triggered ({ratio:.0%} of budget)")
                await self._compress(messages, aggressive=False)
        except Exception as e:
            logger.error(f"Compression failed: {e}")
    
    async def compress_now(self, aggressive: bool = False) -> bool:
        messages = await get_session_messages(self._require_active_session_id())
        if len(messages) < 2:
            return False
        return await self._compress(messages, aggressive=aggressive)

    async def _compress(self, messages: list[dict], aggressive: bool = False) -> bool:
        units = self._provider_transaction_units(messages)
        if len(units) < 2:
            return False

        tail_budget = omega_settings.tail_preserve_tokens
        if aggressive:
            tail_budget = tail_budget // 2

        tail_start = len(units) - 1
        tail_tokens = sum(estimate_tokens(message["content"]) for message in units[tail_start])
        for index in range(len(units) -2, -1, -1):
            unit_tokens = sum(estimate_tokens(messages["content"]) for message in units[index])
            if tail_tokens + unit_tokens > tail_budget:
                break
            tail_tokens += unit_tokens
            tail_start = index

        if tail_start == 0:
            logger.warning(
                "nothing to compress - tail covers the complete provider-valid conversation"
            )
            return False

        compress_span = [message for unit in units[:tail_start] for message in unit]
        pruned_count = sum(
            1 for message in compress_span
            if message.get("tool_name") and len(message["content"]) > 200
        )
        summary = await self.llm_client.generate_answer(
            COMPRESSION_PROMPT, self._format_message_for_summary(compress_span)
        )
        message_ids = [str(message["id"]) for message in compress_span]
        await create_compression_summary_and_mark(
            session_id=self._require_active_session_id(),
            summary_kind="emergency_compression" if aggressive else "standard_compression",
            content=summary,
            first_message_id=message_ids[0],
            last_message_id=message_ids[-1],
            message_ids=message_ids,
            metadata={
                "trigger": "emergency_compression" if aggressive else "standard_compression",
                "pruned_tool_outputs": pruned_count,
            },
        )
        logger.info(
            f"Compressed {len(compress_span)} messages into summary "
            f"(pruned {pruned_count} tool outputs, aggressive={aggressive})"
        )
        return True

    def _format_message_for_summary(self, messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = m["role"].upper()
            content = m["content"]
            if m.get("tool_name"):
                role = f"TOOL({m['tool_name']})"
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    async def _recover_orphaned_sessions(self, exclude_session_id: str | None = None):
        orphans = await find_orphaned_sessions(exclude_session_id=exclude_session_id)
        if not orphans:
            return

        for orphan in orphans:
            orphan_id = orphan["id"]
            logger.warning(f"found orphaned session {orphan_id} - attempting crash recovery summary")
            should_close = True
            try:
                messages = await get_session_messages(orphan_id, include_compressed=True)
                if len(messages) >= 2:
                    await self._create_session_summary_for(orphan_id, messages, trigger="crash_recovery")
                    logger.info(f"Crash recovery: created summary for orphaned session {orphan_id}")
                else:
                    logger.info(f"Crash recovery: orphaned session {orphan_id} had <2 messages, skipping summary")
            except Exception as e:
                should_close = False
                logger.error(f"Crash recovery: failed to summarize orphaned session {orphan_id}: {e}")
            if should_close:
                await close_session(orphan_id)
                logger.info(f"Crash recovery closed orphaned session {orphan_id}")