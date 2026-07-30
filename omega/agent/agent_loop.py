import json
import logging
from omega.llm.client import get_llm_provider
from omega.agent.tool_registry import ToolExecutor, TOOLS_OPENAI_FORMAT
from omega.memory.session_manager import SessionManager
from omega.memory.loop_detector import LoopDetector
from omega.memory.context_build import EXEC_HARNESS
from omega.memory.consolidation import get_consolidation_job
from omega.environment.conf_loader import omega_settings
from omega.memory.turn_context import TurnContextManager
import asyncio

logger = logging.getLogger("AgentLoop")

class AgentLoop:
    def __init__(self):
        self.llm_client = get_llm_provider()
        self.tool_executor = ToolExecutor()
        self.session_manager = SessionManager()
        self.tools_schema_text = json.dumps(TOOLS_OPENAI_FORMAT)

    async def process(self, user_message: str) -> dict:
        await self.session_manager.ensure_session()
        await self.session_manager.add_message("user", user_message)

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
                for tc in response.tool_calls:
                    await self.session_manager.add_message("tool", err_msg, tool_name=tc.name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": err_msg
                    })
                continue

            await self.session_manager.add_message("assistant", tool_decision_text)

            safe_tool_calls = []
            for tc in response.tool_calls:
                if tool_results_seen_this_turn and tc.name == "remember":
                    err = "remember blocked: tool results exist since last user turn - safety rule prevents existing facts from search/KB output"
                    logger.warning(err)
                    await self.session_manager.add_message("tool", err, tool_name="remember")
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
                    result = await self.tool_executor.execute(tc.name, tc.arguments, turn_context=turn_context)
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

                await self.session_manager.add_message("tool", tool_content, tool_name=tc.name)

                ephemeral_content = tool_content
                if i == len(executed_results) - 1:
                    reflection = f"\n\n[SYSTEM INSTRUCTION: You have completed {tool_round} of {omega_settings.max_tool_rounds_per_turn} tool rounds. Evaluate the results above. If you have enough information to fully answer the user, do so. If not, state what is missing and use another tool.]"
                    ephemeral_content += reflection

                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": ephemeral_content 
                })

    async def new_session(self) -> str:
        return await self.session_manager.start_new_session()

    async def _maybe_consolidate(self):
        try:
            job = get_consolidation_job()
            await job.trigger_if_needed()
        except Exception as e:
            logger.error(f"consolidation trigger failed: {e}")