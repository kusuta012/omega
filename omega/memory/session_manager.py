import logging
from omega.storage.memory_queries import (
    create_session,
    get_active_session,
    close_session,
    get_session_messages,
    append_message,
    store_memory_entry,
    mark_messages_compressed,
    get_message_span,
    find_orphaned_sessions
)
from omega.memory.context_build import estimate_tokens, build_system_context
from omega.memory.core import ensure_memory_dir
from omega.llm.client import get_llm_provider
from omega.embeddings.embedding_service import EmbeddingService
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
        self.active_session_id = None
        self.system_context = None
        self.llm_client = get_llm_provider()
        self.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        ensure_memory_dir()

    async def ensure_session(self) -> str:
        if self.active_session_id:
            return self.active_session_id
        await self._recover_orphaned_sessions()
        self.active_session_id = await create_session()
        self.system_context = await build_system_context()
        logger.info(f"Session started: {self.active_session_id}")
        return self.active_session_id

    async def start_new_session(self) -> str:
        if self.active_session_id:
            await self._close_current_session()

        self.active_session_id = await create_session()
        self.system_context = await build_system_context()
        logger.info(f"New session started: {self.active_session_id}")
        return self.active_session_id

    async def _close_current_session(self):
        if not self.active_session_id:
            return

        messages = await get_session_messages(self.active_session_id)
        if len(messages) >= 2:
            try:
                await self._create_session_summary(messages)
            except Exception as e:
                logger.error(f"Failed to create session summary during session close: {e}")

        await close_session(self.active_session_id)
        logger.info(f"Session {self.active_session_id} closed")
        self.active_session_id = None
        self.system_context = None

    async def _create_session_summary_for(self, session_id: str, messages: list[dict], trigger: str = "session_close"):
        conversation_text = self._format_message_for_summary(messages)
        summary = await self.llm_client.generate_answer(
            SESSION_CLOSE_PROMPT, conversation_text
        )
        embedding = self.embedding_service.generate_single_embedding(summary)

        await store_memory_entry(
            memory_type="session_summary",
            content=summary,
            embedding=embedding,
            source_session_id=session_id,
            metadata={"message_count": len(messages), "trigger": trigger},
        )
        logger.info(
            f"Created session summary for {session_id} ({len(messages)} messages)"
        )

    async def _create_session_summary(self, messages: list[dict]):
        await self._create_session_summary_for(self.active_session_id, messages, trigger="session_close")

    async def add_message(self, role: str, content: str, tool_name: str = None):
        await append_message(self.active_session_id, role, content, tool_name)

    async def get_context_messages(self) -> list[dict]:
        messages = await get_session_messages(self.active_session_id)
        return [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")]

    async def check_and_compress(self, tool_schemas_text: str = ""):
        messages = await get_session_messages(self.active_session_id)
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

    async def _compress(self, messages: list[dict], aggressive: bool = False):
        tail_budget = omega_settings.tail_preserve_tokens
        if aggressive:
            tail_budget = tail_budget // 2

        tail_tokens = 0
        tail_start_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            msg_tokens = estimate_tokens(messages[i]["content"])
            if tail_tokens + msg_tokens > tail_budget:
                tail_start_idx = i + 1
                break
            tail_tokens += msg_tokens

        if tail_start_idx <= 1:
            logger.warning(
                "nothing to compress - tail covers almost entire conversation"
            )
            return

        compress_span = messages[:tail_start_idx]
        if not compress_span:
            return

        pruned_count = 0
        for msg in compress_span:
            if msg.get("tool_name") and len(msg["content"]) > 200:
                pruned_count += 1

        conversation_text = self._format_message_for_summary(compress_span)
        summary = await self.llm_client.generate_answer(
            COMPRESSION_PROMPT, conversation_text
        )
        embedding = self.embedding_service.generate_single_embedding(summary)

        await store_memory_entry(
            memory_type="session_summary",
            content=summary,
            embedding=embedding,
            source_session_id=self.active_session_id,
            metadata={
                "message_count": len(compress_span),
                "trigger": "emergency_compression"
                if aggressive
                else "standard_compression",
                "pruned_tool_outputs": pruned_count,
            },
        )

        msg_ids = [m["id"] for m in compress_span if "id" in m]
        if msg_ids:
            await mark_messages_compressed(msg_ids)

        logger.info(
            f"Compressed {len(compress_span)} messages into summary"
            f"(pruned {pruned_count} tool outputs, aggressive={aggressive})"
        )

    def _format_message_for_summary(self, messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = m["role"].upper()
            content = m["content"]
            if m.get("tool_name"):
                role = f"TOOL({m['tool_name']})"
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    async def _recover_orphaned_sessions(self):
        orphans = await find_orphaned_sessions()
        if not orphans:
            return

        for orphan in orphans:
            orphan_id = orphan["id"]
            logger.warning(f"found orphaned session {orphan_id} - attempting crash recovery summary")
            try:
                messages = await get_session_messages(orphan_id)
                if len(messages) >= 2:
                    await self._create_session_summary_for(orphan_id, messages, trigger="crash_recovery")
                    logger.info(f"Crash recovery: created summary for orphaned session {orphan_id}")
                else:
                    logger.info(f"Crash recovery: orphaned session {orphan_id} had <2 messages, skipping summary")
            except Exception as e:
                logger.error(f"Crash recovery: failed to summarize orphaned session {orphan_id}: {e}")

            await close_session(orphan_id)
            logger.info(f"Crash recovery closed orphaned session {orphan_id}")