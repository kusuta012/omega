import logging
from omega.memory.core import read_memory_md, read_user_md
from omega.storage.memory_queries import get_recent_session_summaries
from omega.environment.conf_loader import omega_settings

logger = logging.getLogger("ContextBuilder")

BASE_SYS_PROMPT = """You are Omega, a personal knowledge assistant. You have access to a knowledge base of saved documents and artifacts and a growing memory of past conversations.

You speak naturally and conversationally. You are direct, honest and concise - you don't pad answers with fillers. When you don't know something, you say so clearly. When you use tools, you do so beacause you genuinely need the information, not on every turn.

You have tools available but you are not required to use one on every message. If the user is just chatting chat back. If they ask a question that requires searching their knowledge base, use the search tool, If they ask to see their items, use the list tool. Use your judgement."""

def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4

async def build_system_context() -> str:
    sections = [BASE_SYS_PROMPT]
    memory_md = read_memory_md()
    if memory_md.strip() and "No memories yet" not in memory_md:
        sections.append(f"## Your Core Memory\n{memory_md}")

    user_md = read_user_md()
    if user_md.strip() and "No profile information yet" not in user_md:
        sections.append(f"## User Profile\n{user_md}")

    recent_summaries = await get_recent_session_summaries(
        limit=omega_settings.memory_inject_recent_summaries
    )
    if recent_summaries:
        summary_lines = []
        for s in reversed(recent_summaries):
            summary_lines.append(f"- {s['content']}")
        sections.append(
            "## Recent conversations\n" + "\n".join(summary_lines)
        )

    full_context = "\n\n".join(sections)
    token_est = estimate_tokens(full_context)
    logger.info(f"system context built: ~{token_est} tokens")
    return full_context