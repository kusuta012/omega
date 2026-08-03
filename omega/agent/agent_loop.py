import json
import logging
import asyncio
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
from omega.agent.turn_exec import TurnExecutionState


logger = logging.getLogger("AgentLoop")

class AgentLoop:
    def __init__(self):
        self.llm_client = get_llm_provider()
        self.tool_executor = ToolExecutor()
        self.session_manager = SessionManager()
        self.tools_schema_text = json.dumps(TOOLS_OPENAI_FORMAT)

    async def _record_tool_result(self, tool_call, result: dict, *, turn_context: TurnContextManager, execution_state: TurnExecutionState, messages: list[dict], tool_calls_log: list[dict], all_sources: list[dict]) -> str:
        summary = result.get("result_summary", "no summary")
        tool_calls_log.append({
            "tool": tool_call.name,
            "input": tool_call.arguments,
            "result_summary": summary,
        })
        execution_state.record_tool_result(tool_call.name, result)
        if result.get("sources"):
            all_sources.extend(result["sources"])

        raw_tool_content = result.get("answer", result.get("error", "No result"))
        tool_content = turn_context.truncate_and_cache(tool_call.name, raw_tool_content)
        await self.session_manager.add_message(
            "tool",
            tool_content,
            tool_name=tool_call.name,
            metadata={"tool_call_id": tool_call.id},
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_content,
        })
        return summary

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
        loop_detector = LoopDetector(omega_settings.max_tool_rounds_per_turn)   
        tool_round = 0
        tool_results_seen_this_turn = False
        turn_scratchpad = []
        turn_context = TurnContextManager()
        execution_state = TurnExecutionState(user_message)

        while True:
            current_system_content = system_context + "\n\n" + EXEC_HARNESS
            current_system_content += "\n\n[PRIVATE TURN EXECUTION STATE]\n" + execution_state.decision_context()
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

            await self.session_manager.add_message("assistant", tool_decision_text, metadata={"tool_calls": assistant_msg["tool_calls"]})

            rejected_results: dict[str, dict] = {}
            safe_tool_calls = []
            for tc in response.tool_calls:
                if tool_results_seen_this_turn and tc.name in DURABLE_MEMORY_TOOLS:
                    err = f"{tc.name} blocked: tool results exist since the current user message"
                    logger.warning(err)
                    rejected_results[tc.id] = {
                        "success": False,
                        "error": err,
                        "result_summary": f"{tc.name}: blocked by memory safety rule",
                    }
                    continue

                decision = loop_detector.record_call(tc.name, tc.arguments)
                if not decision.allowed:
                    rejected_results[tc.id] = {
                        "success": False,
                        "error": decision.message,
                        "result_summary": f"{tc.name}: {decision.code}",
                    }
                    continue
                safe_tool_calls.append(tc)

            async def run_tool(tc):
                logger.info(f"Executing tool: {tc.name}({tc.arguments})")
                try:
                    result = await self.tool_executor.execute(tc.name, tc.arguments, turn_context=turn_context, memory_provenance=memory_provenance)
                    return tc.id, result
                except Exception as e:
                    logger.error(f"Tool {tc.name} failed with unhandled exception: {e}")
                    return tc.id, {"success": False, "error": str(e), "result_summary": f"Unhandled error: {e}"}
                
            executed_results = await asyncio.gather(*(run_tool(tc) for tc in safe_tool_calls))
            results_by_call_id = {**rejected_results, **dict(executed_results)}
            
            for tc in response.tool_calls:
                result = results_by_call_id[tc.id]
                await self._record_tool_result(
                    tc, result, turn_context=turn_context, execution_state=execution_state,
                    messages=messages, tool_calls_log=tool_calls_log, all_sources=all_sources,
                )
                tool_results_seen_this_turn = True
                if result.get("scratchpad_note"):
                    turn_scratchpad.append(result["scratchpad_note"])

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
        loop_detector = LoopDetector(omega_settings.max_tool_rounds_per_turn)
        tool_round = 0
        tool_results_seen_this_turn = False
        turn_scratchpad: list[str] = []
        turn_context = TurnContextManager()
        execution_state = TurnExecutionState(user_message)
        total_usage: dict[str, int] = {}

        while True:
            current_system_content = system_context + "\n\n" + EXEC_HARNESS
            current_system_content += "\n\n[PRIVATE TURN EXECUTION STATE]\n" + execution_state.decision_context()
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

            await self.session_manager.add_message("assistant", response_text, metadata={"tool_calls": assistant_msg["tool_calls"]})
            rejected_results: dict[str, dict] = {}
            safe_tool_calls = []
            for tool_call in streamed_tool_calls:
                if tool_results_seen_this_turn and tool_call.name in DURABLE_MEMORY_TOOLS:
                    err = (f"{tool_call.name} blocked: tool results exist since the current user message")
                    logger.warning(err)
                    rejected_results[tool_call.id] = {
                        "success": False,
                        "error": err,
                        "result_summary": f"{tool_call.name}: blocked by memory safety rule",
                    }
                    continue

                decision = loop_detector.record_call(tool_call.name, tool_call.arguments)
                if not decision.allowed:
                    rejected_results[tool_call.id] = {
                        "success": False,
                        "error": decision.message,
                        "result_summary": f"{tool_call.name}: {decision.code}",
                    }
                    continue
                safe_tool_calls.append(tool_call)

            async def run_tool(tool_call):
                logger.info(f"Executing tool: {tool_call.name}({tool_call.arguments})")
                try:
                    result = await self.tool_executor.execute(
                        tool_call.name, tool_call.arguments, turn_context=turn_context, memory_provenance=memory_provenance,
                    )
                    return tool_call.id, result
                except Exception as e:
                    logger.error(f"Tool {tool_call.name} failed with an unhandled err: {e}")
                    return tool_call.id, {
                        "success": False,
                        "error": str(e),
                        "result_summary": f"Unhandled error: {e}"
                    }
            
            for tool_call in safe_tool_calls:
                yield AgentEvent.tool_started(tool_call.name)
            executed_results = await asyncio.gather(*(run_tool(tool_call) for tool_call in safe_tool_calls))
            results_by_call_id = {**rejected_results, **dict(executed_results)}

            for tool_call in streamed_tool_calls:
                result = results_by_call_id[tool_call.id]
                summary = await self._record_tool_result(
                    tool_call, result, turn_context=turn_context,
                    execution_state=execution_state, messages=messages,
                    tool_calls_log=tool_calls_log, all_sources=all_sources
                )
                tool_results_seen_this_turn = True
                if result.get("scratchpad_note"):
                    turn_scratchpad.append(result["scratchpad_note"])
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