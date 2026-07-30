import json
import logging
from omega.llm.client import get_llm_provider
from omega.agent.tool_registry import ToolExecutor, TOOLS_OPENAI_FORMAT
from omega.memory.session_manager import SessionManager
from omega.memory.loop_detector import LoopDetector
from omega.memory.context_build import EXEC_HARNESS
from omega.memory.consolidation import get_consolidation_job
from omega.environment.conf_loader import omega_settings

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

        while True:
            current_system_content = system_context + "\n\n" + EXEC_HARNESS
            if turn_scratchpad:
                scratchpad_text = "\n\n[CURRENT SCRATCHPAD NOTES\n" + "\n".join(f"- {note}" for note in turn_scratchpad)
                current_system_context += scratchpad_text

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
            tool_names = [tc.name for tc in response.tool_calls]
            if not tool_decision_text:
                tool_decision_text = f"[Calling tools: {', '.join(tool_names)}]"
            await self.session_manager.add_message("assistant", tool_decision_text)

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

                logger.info(f"Executing tool: {tc.name}({tc.arguments})")
                result = await self.tool_executor.execute(tc.name, tc.arguments)
                tool_calls_log.append({
                    "tool": tc.name, "input": tc.arguments,
                    "result_summary": result.get("result_summary", "no summary")
                })
                tool_results_seen_this_turn = True
                if result.get("sources"):
                    all_sources.extend(result["sources"])
                if result.get("scratchpad_note"):
                    turn_scratchpad.append(result["scratchpad_note"])
                
                tool_content = result.get("answer", result.get("error", "No result"))
                await self.session_manager.add_message("tool", tool_content, tool_name=tc.name)

                ephemeral_content = tool_content
                if tc == response.tool_calls[-1]:
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