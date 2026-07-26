import json
import logging
from omega.llm.client import get_llm_provider
from omega.agent.tool_registry import ToolExecutor, TOOL_DESC

logger = logging.getLogger("AgentLoop")

MAX_TOOL_CALLS = 3

CLASSIFIER_SYS_PROMPT = """You are Omega's intent classifier. Given a user's request, determine which tool to call

Availabile tools:

1. search_knowledge_base(query)
    search saved content to answer a factual question.
    This is the DEFAULT - use it for any request seeking an answer, and for ANY ambigious request.

2. summarize_item(title_query)
    Summarize a specific saved item. ONLY use when user explicitly names an item AND asks for a summary.

3. list_recent_items(count, source_type)
    List saved items. Use when the user wants to see what they have, not get an answer.
    count: number of items (default 10). source_type: optional filter (url/pdf/text/code)

4. get_item_status(title_query)
    Check processing status of a named item. Use when the user asks about an item's state.

CRITICAL RULE: If a request could plausibly map to more than one tool, you MUST choose search_knowledge_base. Never guess an item identity for summarize_item - only use it when the user clearly names something specific.

Respond with ONLY a JSON object:
{"tool": "tool_name", "arguments": {"key": "value"}}"""

class AgentLoop:
    def __init__(self):
        self.classifier = get_llm_provider()
        self.tool_executor = ToolExecutor()

    async def process(self, user_message: str) -> dict:
        tool_calls_log = []
        last_error = None

        for attempt in range(MAX_TOOL_CALLS):
            classifier_input = user_message
            if last_error:
                classifier_input += f"\n\n[Previous attempt failed: {last_error}. Try a different approach.]"
            
            intent = await self._classify(classifier_input)
            logger.info(f"Agent attempt {attempt + 1}: tool={intent['tool']}, args={intent['arguments']}")

            result = await self.tool_executor.execute(intent["tool"], intent["arguments"])
            tool_calls_log.append({
                "tool": intent["tool"],
                "input": intent["arguments"],
                "result_summary": result.get("result_summary", "no summary")
            })

            if result["success"]:
                return {
                    "question": user_message,
                    "answer": result["answer"],
                    "sources": result.get("sources", []),
                    "tool_calls": tool_calls_log
                }

            last_error = result.get("error", "Unknown error")
            logger.warning(f"Tool '{intent['tool']}' failed: {last_error}")

        return {
            "question": user_message,
            "answer": f"I wasn't able to resolve your request after {MAX_TOOL_CALLS} attempts. Last issue: {last_error}. Try being more specific",
            "sources": [],
            "tool_calls": tool_calls_log
        }
    
    async def _classify(self, user_message: str) -> dict:
        try:
            raw = await self.classifier.generate_json(CLASSIFIER_SYS_PROMPT, user_message)
            parsed = json.loads(raw)

            tool = parsed.get("tool", "search_knowledge_base")
            arguments = parsed.get("arguments", {})

            if tool not in [t["name"] for t in TOOL_DESC]:
                logger.warning(f"LLM returned unknown tool '{tool}', falling back to search")
                tool = "search_knowledge_base"
                arguments = {"query": user_message}

            return {"tool": tool, "arguments": arguments}

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Intent classification failed ({e}), defaulting to search")
            return {
                "tool": "search_knowledge_base",
                "arguments": {"query": user_message}
            }