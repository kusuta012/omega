import logging
from omega.rag.synthesis import Synthesis
from omega.storage.management_queries import (
    list_items_paginated, get_item_detail
)
from omega.storage.item_queries import fetch_item_by_id
from omega.storage.postgres_session import db_pool
from omega.llm.client import get_llm_provider

logger = logging.getLogger("ToolRegistry")

TOOL_DESC = [
    {
        "name": "search_knowledge_base",
        "description": "Search the user's saved knowledge base to answer a factual question. This is the DEFAULT tool , use it for any request that seek an answer from saved content, and for any ambiguous request.",
        "parameters": {
            "query": "The natural language search query"
        }
    },
    {
        "name": "summarize_item",
        "description": "Summarize a specific saved item. ONLY use when the user explicitly names a particular item and asks for a summary or overview of it.",
        "parameters": {
            "title_query": "The title or identifying name of the item to summarize"
        }
    },
    {
        "name": "list_recent_items",
        "description": "List items saved in the knowledge base. Use when the user wants to browse or see what they have saved, not when they want an answer to a question",
        "parameters": {
            "count": "Number of items to return (default 10)",
            "source_type": "Optional filters: url, pdf, text, code"
        }
    },
    {
        "name": "get_item_status",
        "description": "Check the processing status of a specific item. Use when the user asks whether an item has finished processing or wants to know its current state",
        "parameters": {
            "title_query": "The title or identifying name of the item to check"
        }
    }
]

TOOLS_OPENAI_FORMAT = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the user's saved knowledge base to answer a factual question. This is the DEFAULT tool , use it for any request that seek an answer from saved content, and for any ambiguous request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The natural language search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search BOTH the knowledge base AND the agent's own memory of past conversations. Use this when you need to find something that might be in saved documents OR something the user you in a previous conversation. Always labels results by source (document vs memory)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The natural language search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_item",
            "description": "Summarize a specific saved item. ONLY use when the user explicitly names a particular item and asks for a summary or overview of it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title_query": {"type": "string", "description": "The title or identifying name of the item to summarize"}
                },
                "required": ["title_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_recent_items",
            "description": "List items saved in the knowledge base. Use when the user wants to browse or see what they have saved, not when they want an answer to a question",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of items to return (default 10)"},
                    "source_type": {"type": "string", "description": "Optional filters: url, pdf, text, code", "enum": ["url", "pdf", "text", "code"]}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_status",
            "description": "Check the processing status of a specific item. Use when the user asks whether an item has finished processing or wants to know its current state",
            "parameters": {
                "type": "object",
                "properties": {
                    "title_query": {"type": "string", "description": "The title or identifying name of the item to check"}
                },
                "required": ["title_query"]
            }
        }
    }
]

SUMMARIZE_SYSTEM_PROMPT = """You are Omega, a factual summarization assistant.
Summarize the following document concisely. Capture the key points, main arguments and important details.
Keep your summary strictly factual - do not add information not present in the text.
If the document is very short, say so and present its contents directly."""

async def resolve_item_by_title(title_query: str):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, title, source_type, status
            FROM items
            WHERE title ILIKE '%' || $1 || '%'
            ORDER BY created_at DESC
            LIMIT 5
        """, title_query)
    return [dict(r) for r in rows]

class ToolExecutor:
    def __init__(self):
        self.synthesis = Synthesis()
        self.llm_client = get_llm_provider()

    async def execute(self, tool_name: str, arguments: dict) -> dict:
        executors = {
            "search_knowledge_base": self._search,
            "search_memory": self._search_memory_tool,
            "summarize_item": self._summarize,
            "list_recent_items": self._list_items,
            "get_item_status": self._get_status
        }

        executor = executors.get(tool_name)
        if not executor:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "result_summary": f"Tool '{tool_name}' does not exist"
            }

        try:
            return await executor(arguments)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "result_summary": f"Tool '{tool_name}' encountered an error: {e}"
            }

    async def _search(self, args: dict) -> dict:
        query = args.get("query", "")
        if not query:
            return {"success": False, "error": "No query provided", "result_summary": "Empty query"}

        result = await self.synthesis.answer_question(query, top_k=5)

        if not result.get("sources"):
            return {
                "success": True,
                "answer": "I searched your knowledge base but found nothing relevant to that query",
                "sources": [],
                "result_summary": "Search returned 0 relevant chunks"
            }

        return {
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "result_summary": f"Search returned {len(result['sources'])} sources"
        }

    async def _summarize(self, args: dict) -> dict:
        title_query = args.get("title_query", "")
        if not title_query:
            return {"success": False, "error": "No item specified", "result_summary": "Empty title query"}

        matches = await resolve_item_by_title(title_query)

        if not matches:
            logger.info(f"no ILIKE match for '{title_query}', falling back to semantic search")
            semantic_results = await self.synthesis.search_knowledge(title_query, top_k=3)
            if semantic_results:
                top_result = semantic_results[0]
                item_id = top_result["item_id"]
                item_title = top_result["source_title"] or "Untitled"
                logger.info(f"semantic fallback: resolved {title_query} to '{item_title}' ({item_id})")
                item = await fetch_item_by_id(item_id)
                if item and item.get("raw_content"):
                    content = item["raw_content"]
                    if len(content) > 12000:
                        content = content[:12000] + "\n\n[content truncated for summarization]"
                    user_prompt = f"DOCUMENT TITLE: {item_title}\n\nDOCUMENT CONTENT:\n{content}\n\nSUMMARY"
                    summary = await self.llm_client.generate_answer(SUMMARIZE_SYSTEM_PROMPT, user_prompt)
                    note = f"\n\n[Note: no item matched '{title_query}' exactly, But I found this similar item '{item_title}']"
                    return{
                        "success": True,
                        "answer": summary + note,
                        "sources": [{"title": item_title, "item_id": str(item_id), "match_type": "semantic_fallback"}],
                        "result_summary": f"semantic fallback: summarized '{item_title}' matched from '{title_query}'"
                    }
            return {
                "success": False,
                "error": f"No item found matching '{title_query}', tried exact match and semantic search",
                "result_summary": f"Title and semantic lookup for '{title_query}' returned 0 results"
            }

        ambiguity_note = ""
        if len(matches) > 1:
            match_titles = [f"'{m['title']}'" for m in matches]
            ambiguity_note = f"\n\n[Note: Found {len(matches)} matching items ({', '.join(match_titles)}). Summarizing the most recent one '{matches[0]['title']}']"

        item_id = matches[0]["id"]
        item_title = matches[0]["title"] or "Untitled"
        logger.info(f"Resolved '{title_query}' to item '{item_title}' ({item_id})")

        item = await fetch_item_by_id(item_id)
        if not item or not item.get("raw_content"):
            return {
                "success": False,
                "error": f"Item '{item_title}' has no content to summarize",
                "result_summary": f"Item '{item_title}' found but contains no text"
            }

        content = item["raw_content"]
        if len(content) > 12000:
            content = content[:12000] + "\n\n[Content truncated for summarization]"

        user_prompt = f"DOCUMENT TITLE: {item_title}\n\nDOCUMENT CONTENT:\n{content}\n\nSUMMARY:"
        summary = await self.llm_client.generate_answer(SUMMARIZE_SYSTEM_PROMPT, user_prompt)

        final_ans = summary + ambiguity_note

        result_summary = f"Summarized '{item_title}' ({len(content)} chars)"
        if len(matches) > 1:
            result_summary += f" (Note: {len(matches)} matches found, picked most recent)"

        return {
            "success": True,
            "answer": final_ans,
            "sources": [{"title": item_title, "item_id": str(item_id)}],
            "result_summary": result_summary
        }

    async def _list_items(self, args: dict) -> dict:
        count = min(int(args.get("count", 10)), 20)
        source_type = args.get("source_type")

        items, total_count = await list_items_paginated(
            page=1, page_size=count,
            status=None, source_type=source_type
        )

        if not items:
            return {
                "success": True,
                "answer": "Your knowledge base is empty, No items have been saved yet",
                "sources": [],
                "result_summary": "0 items in knowledge base"
            }

        lines = [f"You have {total_count} item(s) in your knowledge base. Here are the most recent:\n"]
        for i, item in enumerate(items, 1):
            status = "Done" if item["status"] == "done" else "Pending" if item["status"] == "pending" else "Failed"
            chunks_info = f"{item['chunk_count']} chunks" if item["chunk_count"] > 0 else "no chunks"
            lines.append(
                f"{i}. [{status}] {item['title'] or 'Untitled'} "
                f"({item['source_type']}, {chunks_info})"
            )

        return {
            "success": True,
            "answer": "\n".join(lines),
            "sources": [],
            "result_summary": f"Listed {len(items)} of {total_count} items"
        }
    
    async def _get_status(self, args: dict) -> dict:
        title_query = args.get("title_query", "")
        if not title_query:
            return {"success": False, "error": "No item specified", "result_summary": "Empty title query"}

        matches = await resolve_item_by_title(title_query)
        if not matches:
            logger.info(f"No exact match for '{title_query}' in get_item_status, falling back to semantic search")
            semantic_results = await self.synthesis.search_knowledge(title_query, top_k=3)
            if semantic_results:
                top_result = semantic_results[0]
                item_id = top_result["item_id"]
                item_title = top_result["source_title"] or "Untitled"
                detail = await get_item_detail(item_id)
                if detail:
                    item = detail["item"]
                    jobs = detail["jobs"]
                    status_line = f"'{item['title']}' (semantically matched from '{title_query}'), status: {item['status'].upper()}"
                    if jobs:
                        latest_job = jobs[0]
                        status_line += f"\nLatest job: {latest_job['status']} attempt {latest_job['attempts']}"
                        if latest_job["last_error"]:
                            status_line += f"\nLast error: {latest_job["last_error"]}"
                    status_line += f"\nChunks: {len(detail['chunks'])}"
                    status_line += f"\nNote: no item name '{title_query}' exactly, but found this similar item"
                    return {
                        "success": True,
                        "answer": status_line,
                        "sources": [{"title": item_title, "item_id": str(item_id), "match_type": "semantic_fallback"}],
                        "result_summary": f"semantic fallback: status check for '{item_title}' matched from '{title_query}'"
                    }
            return {
                "success": False,
                "error": f"No item found matching '{title_query}' tried exact match and semantic search",
                "result_summary": f"Title and semantic lookup for '{title_query}' returned 0 results"
            }

        ambiguity_note = ""
        if len(matches) > 1:
            match_titles = [f"'{m['title']}'" for m in matches]
            ambiguity_note = f"\n[Note: found {len(matches)} matching items ({', '.join(match_titles)}). showing status for the most recent one.]"

        item_id = matches[0]["id"]
        detail = await get_item_detail(item_id)
        if not detail:
            return {"success": False, "error": "Item not found", "result_summary": "Item detail fetch failed"}

        item = detail["item"]
        jobs = detail["jobs"]

        status_line = f"'{item['title']}', status: {item['status'].upper()}"
        if jobs:
            latest_job = jobs[0]
            status_line += f"\nLatest job: {latest_job['status']} (attempt {latest_job['attempts']})"
            if latest_job["last_error"]:
                status_line += f"\nLast error: {latest_job['last_error']}"

        chunk_count = len(detail["chunks"])
        status_line += f"\nChunks: {chunk_count}"
        status_line += ambiguity_note

        result_summary = f"Status check for '{item['title']}': {item['status']}"
        if len(matches) > 1:
            result_summary += f" (Note: {len(matches)} matches found, picked most recent)"

        return {
            "success": True,
            "answer": status_line,
            "sources": [{"title": item["title"], "item_id": str(item_id)}],
            "result_summary": result_summary
        }
    
    async def _search_memory(self, args: dict) -> dict:
        query = args.get("query", "")
        if not query:
            return {"success": False, "error": "no query provided", "results_summary": "Empty query"}

        result = await self.synthesis.answer_with_memory(query, top_k=5)

        if not result.get("sources"):
            return {
                "success": True,
                "answer": "I searched both your knowledge base and memory but found nothing relevant",
                "sources": [],
                "result_summary": "unified search returned 0 results across KB + memory"
            }

        kb_count = sum(1 for s in result["sources"] if s.get("source_type") == "knowledge_base")
        mem_count = sum(1 for s in result["sources"] if s.get("source_type") == "memory")

        return {
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "result_summary": f"unified search: {kb_count} KB sources + {mem_count} memory entries"
        }