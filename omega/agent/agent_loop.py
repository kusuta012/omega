import json
import logging
from omega.llm.client import get_llm_provider
from omega.agent.tool_registry import ToolExecutor, TOOLS_OPENAI_FORMAT
from omega.memory.session_manager import SessionManager
from omega.memory.loop_detector import LoopDetector
from omega.memory.context_build import estimate_tokens

logger = logging.getLogger("AgentLoop")

# CLASSIFIER_SYS_PROMPT = """You are Omega's intent classifier. Given a user's request, determine which tool to call

# Availabile tools:

# 1. search_knowledge_base(query)
#     search saved content to answer a factual question.
#     This is the DEFAULT - use it for any request seeking an answer, and for ANY ambigious request.

# 2. summarize_item(title_query)
#     Summarize a specific saved item. ONLY use when user explicitly names an item AND asks for a summary.

# 3. list_recent_items(count, source_type)
#     List saved items. Use when the user wants to see what they have, not get an answer.
#     count: number of items (default 10). source_type: optional filter (url/pdf/text/code)

# 4. get_item_status(title_query)
#     Check processing status of a named item. Use when the user asks about an item's state.

# CRITICAL RULE: If a request could plausibly map to more than one tool, you MUST choose search_knowledge_base. Never guess an item identity for summarize_item - only use it when the user clearly names something specific.

# Respond with ONLY a JSON object:
# {"tool": "tool_name", "arguments": {"key": "value"}}"""

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

        while True:
            response = await self.llm_client.chat_with_tools(
                messages=messages, tools=TOOLS_OPENAI_FORMAT
            )

            if not response.tool_calls:
                answer = response.content or "I'm not sure how to respond to that."
                await self.session_manager.add_message("assistant", answer)
                await self.session_manager.check_and_compress(self.tools_schema_text)
                return {
                    "session_id": str(self.session_manager.active_session_id),
                    "question": user_message,
                    "answer": answer,
                    "sources": all_sources,
                    "tool_calls": tool_calls_log
                }

            assistant_msg = {"role": "assistant", "content": response.content or ""}
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in response.tool_calls 
            ]
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                if loop_detector.record_call(tc.name, tc.arguments):
                    bail = "I noticed I was repeating the same operations. Let me answer with what I have so far"
                    await self.session_manager.add_message("assistant", bail)
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
                if result.get("sources"):
                    all_sources.extend(result["sources"])

                tool_content = result.get("answer", result.get("error", "No result"))
                await self.session_manager.add_message("tool", tool_content, tool_name=tc.name)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": tool_content 
                })

    async def new_session(self) -> str:
        return await self.session_manager.start_new_session()