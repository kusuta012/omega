import asyncio
import asyncpg
import json
import logging
from omega.storage.postgres_session import db_pool

logger = logging.getLogger("MemoryQueries")

def _message_from_row(row) -> dict:
    message = dict(row)
    metadata = message.get("metadata")
    if isinstance(metadata, str):
        message["metadata"] = json.loads(metadata)
    return message

def _summary_from_row(row) -> dict:
    summary = dict(row)
    metadata = summary.get("metadata")
    if isinstance(metadata, str):
        summary["metadata"] = json.loads(metadata)
    return summary

async def create_session() -> str:
    async with db_pool.acquire() as conn:
        session_id = await conn.fetchval(
            "INSERT INTO sessions (status) VALUES ('active') RETURNING id"
        )
    logger.info(f"Created new session {session_id}")
    return str(session_id)

async def get_active_session() -> dict | None:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, started_at FROM sessions WHERE status = 'active' ORDER BY started_at DESC LIMIT 1"
        )
    return dict(row) if row else None

async def find_resumable_session() -> dict | None:
    active_session = await get_active_session()
    if active_session:
        return active_session

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, started_at, ended_at
            FROM sessions
            WHERE status = 'closed'
            ORDER BY ended_at DESC NULLS LAST, started_at DESC
            LIMIT 1
        """)
    return dict(row) if row else None

async def get_session(session_id: str) -> dict | None:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, status, started_at, ended_at FROM sessions WHERE id = $1",
            session_id,
        )
    return dict(row) if row else None

async def reopen_session(session_id: str) -> bool:
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            result = await conn.execute("""
                UPDATE sessions
                SET status = 'active', ended_at = NULL
                WHERE id = $1 AND status = 'closed'
            """, session_id)
            reopened = result == "UPDATE 1"
            if reopened:
                await conn.execute("""
                    DELETE FROM session_summaries
                    WHERE session_id = $1
                      AND summary_kind IN ('session_close', 'crash_recovery')
                """, session_id)
    if reopened:
        logger.info(f"Reopened session {session_id}")
    return reopened

async def find_orphaned_sessions(exclude_session_id: str | None = None) -> list[dict]:
    async with db_pool.acquire() as conn:
        if exclude_session_id is None:
            rows = await conn.fetch(
                "SELECT id, started_at FROM sessions WHERE status = 'active' ORDER BY started_at ASC"
            )
        else:
            rows = await conn.fetch("""
                SELECT id, started_at
                FROM sessions
                WHERE status = 'active' AND id != $1
                ORDER BY started_at ASC
            """, exclude_session_id)
    return [dict(r) for r in rows]

async def close_session(session_id: str):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE sessions SET status = 'closed', ended_at = now() WHERE id = $1",
            session_id
        )
    logger.info(f"closed session {session_id}")

async def append_message(session_id: str, role: str, content: str, tool_name: str | None = None, metadata: dict | None = None) -> str:
    metadata_json = json.dumps(metadata) if metadata is not None else "null"
    async with db_pool.acquire() as conn:
        message_id = await conn.fetchval(
            "INSERT INTO messages (session_id, role, content, tool_name, metadata) VALUES ($1, $2, $3, $4, $5::jsonb) RETURNING id",
            session_id,
            role,
            content,
            tool_name,
            metadata_json,
        )
    return str(message_id)
        

# Hello unc reviewer, don't be too lazy to review projects

async def get_session_messages(session_id: str, include_compressed: bool = False) -> list[dict]:
    async with db_pool.acquire() as conn:
        if include_compressed:
            rows = await conn.fetch(
                "SELECT id, role, content, tool_name, metadata, compressed, created_at FROM messages "
                "WHERE session_id = $1 ORDER BY created_at ASC",
                session_id
            )
        else:
            rows = await conn.fetch(
                "SELECT id, role, content, tool_name, metadata, compressed, created_at FROM messages "
                "WHERE session_id = $1 AND compressed = FALSE ORDER BY created_at ASC",
                session_id
            )
    return [_message_from_row(row) for row in rows]

async def get_message_span(session_id: str, before_timestamp) -> list[dict]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, role, content, tool_name, created_at FROM messages "
            "WHERE session_id = $1 AND created_at < $2 ORDER BY created_at ASC",
            session_id, before_timestamp
        )
    return [_message_from_row(row) for row in rows]

async def mark_messages_compressed(message_ids: list[str]):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE messages SET compressed = TRUE WHERE id = ANY($1::uuid[])",
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
                (memory_type, content, embedding, source_session_id, occurred_at, metadata)
            VALUES ($1, $2, $3::vector, $4, now(), $5::jsonb)
            RETURNING id
        """, memory_type, content, embedding_json, source_session_id, metadata_json)

    logger.info(f"Stored memory entry {entry_id} (type={memory_type})")
    return entry_id

async def _create_session_summary(
    conn,
    session_id: str,
    summary_kind: str,
    content: str,
    first_message_id: str,
    last_message_id: str,
    message_count: int,
    metadata: dict | None,
) -> dict:
    allowed_kinds = {
        "standard_compression",
        "emergency_compression",
        "session_close",
        "crash_recovery",
    }
    if summary_kind not in allowed_kinds:
        raise ValueError(f"Unknown session summary kind: {summary_kind}")
    if message_count < 0:
        raise ValueError("Session summary message_count cannot be negative")
    if not first_message_id or not last_message_id:
        raise ValueError("Session summaries require complete source-message boundaries")

    row = await conn.fetchrow("""
        INSERT INTO session_summaries
            (session_id, summary_kind, content, first_message_id, last_message_id, message_count, metadata)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
        ON CONFLICT DO NOTHING
        RETURNING id, session_id, summary_kind, content, first_message_id, last_message_id,
                  message_count, created_at, metadata
    """, session_id, summary_kind, content, first_message_id, last_message_id, message_count, json.dumps(metadata or {}))
    if row:
        return _summary_from_row(row)

    if summary_kind in {"session_close", "crash_recovery"}:
        row = await conn.fetchrow("""
            SELECT id, session_id, summary_kind, content, first_message_id, last_message_id,
                   message_count, created_at, metadata
            FROM session_summaries
            WHERE session_id = $1
              AND summary_kind IN ('session_close', 'crash_recovery')
            ORDER BY created_at ASC
            LIMIT 1
        """, session_id)
    else:
        row = await conn.fetchrow("""
            SELECT id, session_id, summary_kind, content, first_message_id, last_message_id,
                    message_count, created_at, metadata
            FROM session_summaries
            WHERE session_id = $1
              AND summary_kind = $2
              AND first_message_id = $3
              AND last_message_id = $4
            LIMIT 1
        """, session_id, summary_kind, first_message_id, last_message_id)
    if row:
        return _summary_from_row(row)
    raise RuntimeError("Session summary insert did not create or locate a summary")
    
async def create_session_summary(
    session_id: str,
    summary_kind: str,
    content: str,
    first_message_id: str,
    last_message_id: str,
    message_count: int,
    metadata: dict | None = None,
) -> dict:
    async with db_pool.acquire() as conn:
        return await _create_session_summary(
            conn, session_id, summary_kind, content, first_message_id,
            last_message_id, message_count, metadata,
        )

async def create_compression_summary_and_mark(
    session_id: str,
    summary_kind: str,
    content: str,
    first_message_id: str,
    last_message_id: str,
    message_ids: list[str],
    metadata: dict | None = None,
) -> dict:
    if not message_ids:
        raise ValueError("Compression summaries require source messages")
    if len(set(message_ids)) != len(message_ids):
        raise ValueError("Compression summary source messages must be unique")

    for attempt in range(3):
        try:
            async with db_pool.acquire() as conn:
                async with conn.transaction():
                    summary = await _create_session_summary(
                        conn, session_id, summary_kind, content, first_message_id,
                        last_message_id, len(message_ids), metadata,
                    )
                    result = await conn.execute("""
                        UPDATE messages
                        SET compressed = TRUE
                        WHERE session_id = $1
                          AND compressed = FALSE
                          AND id = ANY($2::uuid[])
                    """, session_id, message_ids)
                    updated_count = int(result.rsplit(" ", 1)[-1])
                    if updated_count != len(message_ids):
                        raise RuntimeError(
                            f"Compression source mismatch: expected {len(message_ids)} messages, updated {updated_count}"
                        )
            return summary
        except asyncpg.PostgresError as exc:
            if getattr(exc, "sqlstate", None) not in {"40001", "40P01"} or attempt == 2:
                raise
            await asyncio.sleep(0.05 * (2 ** attempt))
    raise RuntimeError("Compression transaction retry loop exhausted")


async def get_final_session_summary(session_id: str) -> dict | None:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, session_id, summary_kind, content, first_message_id, last_message_id,
                   message_count, created_at, metadata
            FROM session_summaries
            WHERE session_id = $1
              AND summary_kind IN ('session_close', 'crash_recovery')
            ORDER BY created_at ASC
            LIMIT 1
        """, session_id)
    return _summary_from_row(row) if row else None

async def get_session_summary_spans(session_id: str) -> list[dict]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, session_id, summary_kind, content, first_message_id, last_message_id,
                   message_count, created_at, metadata
            FROM session_summaries
            WHERE session_id = $1
              AND summary_kind IN ('standard_compression', 'emergency_compression')
            ORDER BY created_at ASC
        """, session_id)
    return [_summary_from_row(row) for row in rows]

async def get_recent_closed_session_summaries(limit: int = 2) -> list[dict]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT ss.id, ss.session_id, ss.summary_kind, ss.content, ss.first_message_id,
                   ss.last_message_id, ss.message_count, ss.created_at, ss.metadata
            FROM session_summaries ss
            JOIN sessions s ON s.id = ss.session_id
            WHERE s.status = 'closed'
              AND ss.summary_kind IN ('session_close', 'crash_recovery')
            ORDER BY ss.created_at DESC
            LIMIT $1
        """, limit)
    return [_summary_from_row(row) for row in rows]

async def record_memory_access(entry_id: str):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE memory_entries
            SET access_count = access_count + 1,
                last_accessed_at = now()
            WHERE id = $1
        """, entry_id)

async def store_extracted_fact(
    content: str,
    embedding: list[float],
    source_session_id: str = None,
    occurred_at=None,
    importance: float = 0.5,
    supersedes_id: str = None,
    metadata: dict = None
) -> str:
    embedding_json = json.dumps(embedding)
    metadata_json = json.dumps(metadata) if metadata else None

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            entry_id = await conn.fetchval("""
                INSERT INTO memory_entries
                    (memory_type, content, embedding, source_session_id, occurred_at, importance, metadata, superseded_by)
                VALUES ('extracted_fact', $1, $2::vector, $3, $4, $5, $6::jsonb, $7)
                RETURNING id
            """, content, embedding_json, source_session_id, occurred_at, importance, metadata_json, supersedes_id)

            if supersedes_id:
                await conn.execute("""
                    UPDATE memory_entries SET superseded_by = $1 WHERE id = $2
                """, entry_id, supersedes_id)
    logger.info(f"Stored extracted fact {entry_id} (supersedes={supersedes_id})")
    return entry_id

async def find_similar_facts(embedding: list[float], limit: int = 3) -> list[dict]:
    embedding_json = json.dumps(embedding)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance,
                   (1 - (embedding <=> $1::vector)) as similarity
            FROM memory_entries
            WHERE memory_type = 'extracted_fact' AND superseded_by IS NULL
              AND embedding <=> $1::vector < 0.5
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """, embedding_json, limit)
    return [dict(r) for r in rows]

async def get_unconsolidated_count() -> int:
    async with db_pool.acquire() as conn:
        count = await conn.fetchval("""
            SELECT COUNT(*) FROM memory_entries
            WHERE (metadata->>'consolidated' IS NULL
                   OR metadata->>'consolidated' = 'false')
        """)
    return count or 0

async def mark_entries_consolidated(entry_ids: list[str]):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE memory_entries
            SET metadata = COALESCE(metadata, '{}'::jsonb) || '{"consolidated": true}'::jsonb
            WHERE id = ANY($1::uuid[])
        """, entry_ids)

async def apply_importance_decay(entry_ids: list[str], decay_factor: float = 0.95):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE memory_entries
            SET importance = GREATEST(0.05, importance * $2)
            WHERE id = ANY($1::uuid[])
        """, entry_ids, decay_factor)

async def count_total_msgs() -> int:
    async with db_pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM messages")
    return count or 0