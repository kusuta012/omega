import json
import logging
from collections.abc import AsyncIterator
from omega.agent.events import AgentEvent
from omega.llm.client import get_llm_provider
from omega.agent.tool_registry import ToolExecutor, TOOLS_OPENAI_FORMAT
from omega.memory.session_manager import SessionManager
from omega.memory.loop_detector import LoopDetector
from omega.memory.context_build import EXEC_HARNESS
from omega.memory.consolidation import get_consolidation_job
from omega.environment.conf_loader import omega_settings
from omega.memory.turn_context import TurnContextManager
from omega.memory.provenance import DURABLE_MEMORY_TOOLS, DirectUserProvenance
from omega.memory.session_locks import session_turn_locks
import asyncio

logger = logging.getLogger("AgentLoop")

class AgentLoop:
    def __init__(self):
        self.llm_client = get_llm_provider()
        self.tool_executor = ToolExecutor()
        self.session_manager = SessionManager()
        self.tools_schema_text = json.dumps(TOOLS_OPENAI_FORMAT)

    async def process(self, user_message: str, *, session_id: str | None = None) -> dict:
        active_session_id = await self.session_manager.ensure_session(session_id=session_id)
        async with session_turn_locks.acquire(active_session_id):
            return await self._process_in_session(user_message)
    
    async def _process_in_session(self, user_message: str) -> dict:
        user_message_id = await self.session_manager.add_message("user", user_message)
        memory_provenance = DirectUserProvenance(
            session_id=str(self.session_manager.active_session_id),
            message_id=user_message_id,
            user_message=user_message,
        )

        system_context = self.session_manager.system_context
        conversation = await self.session_manager.get_context_messages()
        messages = [{"role": "system", "content": system_context}] + conversation

        tool_calls_log = []
        all_sources = []
        loop_detector = LoopDetector()
        tool_round = 0
        tool_results_seen_this_turn = False
        turn_scratchpad = []
        turn_context = TurnContextManager()

        while True:
            current_system_content = system_context + "\n\n" + EXEC_HARNESS
            if turn_scratchpad:
                scratchpad_text = "\n\n[CURRENT SCRATCHPAD NOTES]\n" + "\n".join(f"- {note}" for note in turn_scratchpad)
                current_system_content += scratchpad_text

            messages[0]["content"] = current_system_content

            response = await self.llm_client.chat_with_tools(
                messages=messages, tools=TOOLS_OPENAI_FORMAT
            )

            if not response.tool_calls:
                answer = response.content or "I'm not sure how to respond to that."
                await self.session_manager.add_message("assistant", answer)
                await self.session_manager.check_and_compress(self.tools_schema_text)
                await self._maybe_consolidate()
                return {
                    "session_id": str(self.session_manager.active_session_id),
                    "question": user_message,
                    "answer": answer,
                    "sources": all_sources,
                    "tool_calls": tool_calls_log
                }

            tool_round += 1
            if tool_round > omega_settings.max_tool_rounds_per_turn:
                bail = f"I've reached my limit of {omega_settings.max_tool_rounds_per_turn} tool-calling rounds. Let me answer with what I have"
                logger.warning(f"Hard tool-round cap hit {omega_settings.max_tool_rounds_per_turn} for message: {user_message[:80]}")
                await self.session_manager.add_message("assistant", bail)
                await self.session_manager.check_and_compress(self.tools_schema_text)
                await self._maybe_consolidate()
                return {
                    "session_id": str(self.session_manager.active_session_id),
                    "question": user_message, "answer": bail,
                    "sources": all_sources, "tool_calls": tool_calls_log
                }

            assistant_msg = {"role": "assistant", "content": response.content or ""}
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in response.tool_calls 
            ]
            messages.append(assistant_msg)

            tool_decision_text = response.content or ""


            if response.tool_calls and not tool_decision_text.strip():
                logger.warning("Structured Reasoning Enforcer: LLM emitted tool call without CoT monologue. Intercepting.")
                err_msg = "[SYSTEM REJECTION: You attempted to call a tool without explaining your reasoning first. State your thought process and plan before calling any tools.]"
                await self.session_manager.add_message(
                    "assistant", tool_decision_text, metadata={"tool_calls": assistant_msg["tool_calls"]}
                )
                for tc in response.tool_calls:
                    await self.session_manager.add_message("tool", err_msg, tool_name=tc.name, metadata={"tool_call_id": tc.id})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": err_msg
                    })
                continue

            await self.session_manager.add_message("assistant", tool_decision_text, metadata={"tool_calls": assistant_msg["tool_calls"]})

            safe_tool_calls = []
            for tc in response.tool_calls:
                if tool_results_seen_this_turn and tc.name in DURABLE_MEMORY_TOOLS:
                    err = f"{tc.name} blocked: tool results exist since the current user message"
                    logger.warning(err)
                    await self.session_manager.add_message("tool", err, tool_name=tc.name)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": err})
                    continue

                if loop_detector.record_call(tc.name, tc.arguments):
                    bail = "I noticed I was repeating the same operations. Let me answer with what I have so far"
                    await self.session_manager.add_message("assistant", bail)
                    await self.session_manager.check_and_compress(self.tools_schema_text)
                    await self._maybe_consolidate()
                    return {
                        "session_id": str(self.session_manager.active_session_id),
                        "question": user_message, "answer": bail,
                        "sources": all_sources, "tool_calls": tool_calls_log
                    }

                safe_tool_calls.append(tc)

            async def run_tool(tc):
                logger.info(f"Executing tool: {tc.name}({tc.arguments})")
                try:
                    result = await self.tool_executor.execute(tc.name, tc.arguments, turn_context=turn_context, memory_provenance=memory_provenance)
                    return tc, result
                except Exception as e:
                    logger.error(f"Tool {tc.name} failed with unhandled exception: {e}")
                    return tc, {"success": False, "error": str(e), "result_summary": f"Unhandled error: {e}"}
                
            execution_tasks = [run_tool(tc) for tc in safe_tool_calls]
            executed_results = await asyncio.gather(*execution_tasks)
            
            for i, (tc, result) in enumerate(executed_results):
                tool_calls_log.append({
                    "tool": tc.name, "input": tc.arguments,
                    "result_summary": result.get("result_summary", "no summary")
                })
                tool_results_seen_this_turn = True

                if result.get("sources"):
                    all_sources.extend(result["sources"])
                if result.get("scratchpad_note"):
                    turn_scratchpad.append(result["scratchpad_note"])
                
                raw_tool_content = result.get("answer", result.get("error", "No result"))
                tool_content = turn_context.truncate_and_cache(tc.name, raw_tool_content)

                await self.session_manager.add_message("tool", tool_content, tool_name=tc.name, metadata={"tool_call_id": tc.id})

                ephemeral_content = tool_content
                if i == len(executed_results) - 1:
                    reflection = f"\n\n[SYSTEM INSTRUCTION: You have completed {tool_round} of {omega_settings.max_tool_rounds_per_turn} tool rounds. Evaluate the results above. If you have enough information to fully answer the user, do so. If not, state what is missing and use another tool.]"
                    ephemeral_content += reflection

                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": ephemeral_content 
                })

    async def process_stream(self, user_message: str, *, resume: bool = False) -> AsyncIterator[AgentEvent]:
        active_session_id = await self.session_manager.ensure_session(resume=resume)
        async with session_turn_locks.acquire(active_session_id):
            async for event in self._process_stream_in_session(user_message):
                yield event
    async def _process_stream_in_session(self, user_message: str) -> AsyncIterator[AgentEvent]:
        user_message_id = await self.session_manager.add_message("user", user_message)
        memory_provenance = DirectUserProvenance(
            session_id=str(self.session_manager.active_session_id),
            message_id=user_message_id,
            user_message=user_message,
        )
        system_context = self.session_manager.system_context
        if system_context is None:
            raise RuntimeError("session started without a system context")
        conversation = await self.session_manager.get_context_messages()
        messages = [{"role": "system", "content": system_context}] + conversation
        tool_calls_log: list[dict] = []
        all_sources: list[dict] = []
        loop_detector = LoopDetector()
        tool_round = 0
        tool_results_seen_this_turn = False
        turn_scratchpad: list[str] = []
        turn_context = TurnContextManager()
        total_usage: dict[str, int] = {}

        while True:
            current_system_content = system_context + "\n\n" + EXEC_HARNESS
            if turn_scratchpad:
                current_system_content += "\n\n[CURRENT SCRATCHPAD NOTES]\n" + "\n".join(
                    f"- {note}" for note in turn_scratchpad
                )
            messages[0]["content"] = current_system_content
            response_parts: list[str] = []
            streamed_tool_calls = []
            stream_finished = False
            stream_usage: dict[str, int] = {}

            async for event in self.llm_client.chat_with_tools_stream(
                messages=messages, tools=TOOLS_OPENAI_FORMAT
            ):
                if event.type == "text_delta":
                    if event.text:
                        response_parts.append(event.text)
                        yield AgentEvent.text_delta(event.text)
                elif event.type == "tool_call":
                    if event.tool_call is None:
                        raise RuntimeError("Provider emitted a tool event without a completed tool call")
                    streamed_tool_calls.append(event.tool_call)
                elif event.type == "message_end":
                    stream_finished = True
                    stream_usage.update(event.usage)

            if not stream_finished:
                raise RuntimeError("LLM stream ended without a completion event; assistant output was not saved")
            for key, value in stream_usage.items():
                total_usage[key] = total_usage.get(key, 0) + value

            response_text = "".join(response_parts)
            if not streamed_tool_calls:
                answer = response_text or "I'm not sure how to respond to that"
                if not response_text:
                    yield AgentEvent.text_delta(answer)
                await self.session_manager.add_message("assistant", answer)
                await self.session_manager.check_and_compress(self.tools_schema_text)
                await self._maybe_consolidate()
                result = {
                    "session_id": str(self.session_manager.active_session_id),
                    "question": user_message,
                    "answer": answer,
                    "sources": all_sources,
                    "tool_calls": tool_calls_log,
                }
                yield AgentEvent.turn_complete(result, total_usage)
                return

            tool_round += 1
            if tool_round > omega_settings.max_tool_rounds_per_turn:
                bail = (
                    f"I've reached my limit of {omega_settings.max_tool_rounds_per_turn}"
                    "tool-calling rounds, let me answer with what I have"
                )
                logger.warning(f"Hard tool-round cap hit {omega_settings.max_tool_rounds_per_turn, user_message[:80]}")
                yield AgentEvent.text_delta(bail)
                await self.session_manager.add_message("assistant", bail)
                await self.session_manager.check_and_compress(self.tools_schema_text)
                await self._maybe_consolidate()
                result = {
                    "session_id": str(self.session_manager.active_session_id),
                    "question": user_message,
                    "answer": bail,
                    "sources": all_sources,
                    "tool_calls": tool_calls_log,
                }
                yield AgentEvent.turn_complete(result, total_usage)
                return

            assistant_msg = {
                "role": "assistant",
                "content": response_text,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in streamed_tool_calls
                ],
            }
            messages.append(assistant_msg)
            
            if not response_text.strip():
                logger.warning("structured reasoning enforcer: LLM emitted tool calls without visible reasoning")
                error_message = (
                    "[SYSTEM: You attempted to call a tool without explaining your reasoning first"
                    "State your thought process and plan before calling any tools"
                )
                await self.session_manager.add_message(
                    "assistant", response_text, metadata={"tool_calls": assistant_msg["tool_calls"]}
                )
                for tool_call in streamed_tool_calls:
                    await self.session_manager.add_message("tool", error_message, tool_name=tool_call.name, metadata={"tool_call_id": tool_call.id})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_message,
                    })
                continue

            await self.session_manager.add_message("assistant", response_text, metadata={"tool_calls": assistant_msg["tool_calls"]})
            safe_tool_calls = []
            for tool_call in streamed_tool_calls:
                if tool_results_seen_this_turn and tool_call.name in DURABLE_MEMORY_TOOLS:
                    error_message = (
                        f"{tool_call.name} blocked: tool results exist since the current user message"
                    )
                    logger.warning(error_message)
                    await self.session_manager.add_message("tool", error_message, tool_name=tool_call.name, metadata={"tool_call_id": tool_call.id})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_message,
                    })
                    yield AgentEvent.tool_completed(tool_call.name, "blocked by memory safety rule")
                    continue

                if loop_detector.record_call(tool_call.name, tool_call.arguments):
                    bail = "I noticed I was repeating the same operations, let me answer with what I have so far"
                    yield AgentEvent.text_delta(bail)
                    await self.session_manager.add_message("assistant", bail)
                    await self.session_manager.check_and_compress(self.tools_schema_text)
                    await self._maybe_consolidate()
                    result = {
                        "session_id": str(self.session_manager.active_session_id),
                        "question": user_message,
                        "answer": bail,
                        "sources": all_sources,
                        "tool_calls": tool_calls_log,
                    }
                    yield AgentEvent.turn_complete(result, total_usage)
                    return
                safe_tool_calls.append(tool_call)

            async def run_tool(tool_call):
                logger.info(f"Executing tool: {tool_call.name}({tool_call.arguments})")
                try:
                    result = await self.tool_executor.execute(
                        tool_call.name, tool_call.arguments, turn_context=turn_context, memory_provenance=memory_provenance,
                    )
                    return tool_call, result
                except Exception as e:
                    logger.error(f"Tool {tool_call.name} failed with an unhandled err: {e}")
                    return tool_call, {
                        "success": False,
                        "error": str(e),
                        "result_summary": f"Unhandled error: {e}"
                    }
            
            for tool_call in safe_tool_calls:
                yield AgentEvent.tool_started(tool_call.name)
            executed_results = await asyncio.gather(*(run_tool(tool_call) for tool_call in safe_tool_calls))

            for index, (tool_call, result) in enumerate(executed_results):
                summary = result.get("result_summary", "no summary")
                tool_calls_log.append({
                    "tool": tool_call.name,
                    "input": tool_call.arguments,
                    "result_summary": summary,
                })
                tool_results_seen_this_turn = True
                if result.get("sources"):
                    all_sources.extend(result["sources"])
                if result.get("scratchpad_note"):
                    turn_scratchpad.append(result["scratchpad_note"])

                raw_tool_content = result.get("answer", result.get("error", "No result"))
                tool_content = turn_context.truncate_and_cache(tool_call.name, raw_tool_content)
                await self.session_manager.add_message("tool", tool_content, tool_name=tool_call.name, metadata={"tool_call_id"})
                
                ephemeral_content = tool_content
                if index == len(executed_results) - 1:
                    ephemeral_content += (
                        f"\n\n[SYSTEM: You have completed {tool_round} of"
                        f"{omega_settings.max_tool_rounds_per_turn} tool rounds. Evaluate the results above"
                        "If you have enough information to fully answer the user, do so. If not state what is missing and use another tool"
                    )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": ephemeral_content,
                })
                yield AgentEvent.tool_completed(tool_call.name, summary)

    async def close_session(self):
        await self.session_manager.close_current_session()

    async def new_session(self) -> str:
        return await self.session_manager.start_new_session()

    async def _maybe_consolidate(self):
        try:
            job = get_consolidation_job()
            await job.trigger_if_needed()
        except Exception as e:
            logger.error(f"consolidation trigger failed: {e}")