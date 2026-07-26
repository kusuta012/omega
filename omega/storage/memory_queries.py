import json
import logging
from omega import memory
from omega.storage.postgres_session import db_pool

logger = logging.getLogger("MemoryQueries")

async def create_session() -> str:
    async with db_pool.acquire() as conn:
        async with conn.transactions():
            await conn.execute(
                "UPDATE sessions SET status = 'closed', ended_at = now() WHERE status = 'active'"
            )
            session_id = await conn.fetchval(
                "INSERT INTO sessions (status) VALUES ('active') RETURNING id"
            )
    logger.info(f"Created new session {session_id}")
    return session_id

async def get_active_session():
    async with db_pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT id, started_at FROM sessions WHERE status = 'active' ORDER BY started_at DESC LIMIT 1"
        )

async def close_session(session_id: str):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE session SET status = 'closed', ended_at = now() WHERE id = $1",
            session_id
        )
    logger.info(f"closed session {session_id}")

async def append_message(session_id: str, role: str, content: str, tool_name: str = None):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_name) VALUES ($1, $2, $3, $4)",
            session_id, role, content, tool_name
        )

# Hello unc reviewer, don't be too lazy to review projects

async def get_session_messages(session_id: str) -> list[dict]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content, tool_name, created_at FROM messages"
            "WHERE session_id = $1 ORDER BY created_at ASC",
            session_id
        )
    return [dict(r) for r in rows]

async def get_message_span(session_id: str, before_timestamp) -> list[dict]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, role, content, tool_name, created_at, FROM messages "
            "WHERE session_id = $1 AND created_at < $2 ORDER BY created_at ASC",
            session_id, before_timestamp
        )
    return [dict(r)for r in rows]

async def mark_messages_compressed(message_ids: list[str]):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM messages WHERE id = ANY($1::uuid[])",
            message_ids
        )

async def store_memory_entry(
    memory_type: str,
    content: str,
    embedding: list[float],
    source_session_id: str = None,
    metadata: dict = None
):
    embedding_json = json.dumps(embedding)
    metadata_json= json.dumps(metadata) if metadata else None

    async with db_pool.acquire() as conn:
        entry_id = await conn.fetchval("""
            INSERT INTO memory_entries
                (memory_type, content, embedding, source_session_id, occured_at, metadata)
            VALUES ($1, $2, $3::vector, $4, now(), $5::jsonb)
            RETURNING id
        """, memory_type, content, embedding_json, source_session_id, metadata_json)

    logger.info(f"Stored memory entry {entry_id} (type={memory_type})")
    return entry_id

async def get_recent_session_summaries(limit: int = 2) -> list[dict]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT me.content, me.created_at, s.started_at AS session_started
            FROM memory_entries me
            LEFT JOIN sessions s ON me.source_session_id = s.id
            WHERE me.memory_type = 'session_summary'
            ORDER BY me.created_at DESC
            LIMIT $1
        """, limit)
    return [dict(r) for r in rows]