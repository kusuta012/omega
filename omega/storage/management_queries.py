from omega.storage.postgres_session import db_pool
import logging

logger = logging.getLogger("ManagementQueries")

async def list_items_paginated(page: int = 1, page_size: int = 20, status: str | None = None, source_type: str | None = None):
    offset = (page - 1) * page_size

    conditions = []
    params = []
    param_index = 1

    if status:
        conditions.append(f"i.status = ${param_index}")
        params.append(status)
        param_index += 1

    if source_type:
        conditions.append(f"i.source_type = ${param_index}")
        params.append(source_type)
        param_index += 1

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        SELECT
            i.id, i.source_type, i.title, i.source_ref, i.status, i.created_at,
            COUNT(c.id) AS chunk_count,
            COUNT(*) OVER() AS total_count
        FROM items i
        LEFT JOIN chunks c ON c.item_id = i.id
        {where_clause}
        GROUP BY i.id
        ORDER BY i.created_at DESC
        LIMIT ${param_index} OFFSET ${param_index + 1}
    """

    params.extend([page_size, offset])

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    if not rows:
        return [], 0

    total_count = rows[0]['total_count']
    items = [
        {
            "id": str(row['id']),
            "source_type": row['source_type'],
            "title": row['title'],
            "source_ref": row['source_ref'],
            "status": row['status'],
            "created_at": row['created_at'],
            "chunk_count": row['chunk_count']
        }
        for row in rows
    ]
    return items, total_count

async def get_item_detail(item_id: str):
    async with db_pool.acquire() as conn:
        item = await conn.fetchrow("""
            SELECT id, source_type, title, source_ref, content_hash, status, created_at
            FROM items WHERE id = $1
        """, item_id)

        if not item:
            return None

        chunks = await conn.fetch("""
            SELECT id, chunk_index, content
            FROM chunks WHERE item_id = $1
            ORDER BY chunk_index ASC
        """, item_id)

        jobs = await conn.fetch("""
            SELECT id, job_type, status, attempts, last_error, created_at, updated_at
            FROM jobs WHERE item_id = $1
            ORDER BY created_at DESC
        """, item_id)

    return {
        "item": dict(item),
        "chunks": [dict(c) for c in chunks],
        "jobs": [dict(j)for j in jobs]
    }

async def delete_item(item_id: str):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            DELETE FROM items WHERE id = $1
            RETURNING id, title, source_type
        """, item_id)
        return dict(row) if row else None

async def list_jobs(page: int = 1, page_size: int = 20, status: str | None = None):
    offset = (page - 1) * page_size
    params = []
    param_index = 1

    if status:
        where_clause = f"WHERE j.status = ${param_index}"
        params.append(status)
        param_index += 1
    else:
        where_clause = ""

    query = f"""
        SELECT
            j.id, j.item_id, j.job_type, j.status, j.attempts,
            j.last_error, j.created_at, j.updated_at,
            i.title AS item_title,
            COUNT(*) OVER() AS total_count
        FROM jobs j
        JOIN items i ON j.item_id = i.id
        {where_clause}
        ORDER BY j.updated_at DESC
        LIMIT ${param_index} OFFSET ${param_index + 1}
    """

    params.extend([page_size, offset])

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    if not rows:
        return [], 0

    total_count = rows[0]['total_count']
    jobs = [
        {
            "id": str(row['id']),
            "item_id": str(row['item_id']),
            "item_title": row['item_title'],
            "job_type": row['job_type'],
            "status": row['status'],
            "attempts": row['attempts'],
            "last_error": row['last_error'],
            "created_at": row['created_at'],
            "updated_at": row['updated_at']
        }
        for row in rows
    ]
    return jobs, total_count

async def retry_failed_item(item_id: str) -> dict:
    async with db_pool.acquire() as conn:
        item = await conn.fetchrow(
            "SELECT id, status, title FROM items WHERE id = $1", item_id
        )

        if not item:
            return {"error": "not_found", "message": "item not found"}

        if item['status'] != 'failed':
            return {
                "error": "invalid_status",
                "message": f"Item is currently in '{item['status']}' state, only failed items can be retired"
            }

        job = await conn.fetchrow("""
            SELECT id, status FROM jobs
            WHERE item_id = $1
            ORDER BY created_at DESC
            LIMIT 1
        """, item_id)

        async with conn.transaction():
            if job and job['status'] == 'failed':
                await conn.execute("""
                    UPDATE jobs
                    SET status = 'pending', attempts = 0, last_error = NULL, updated_at = now()
                    WHERE id = $1
                """, job['id'])
                await conn.execute(
                    "UPDATE items SET status = 'pending' WHERE id = $1", item_id
                )
                return {
                    "success": True,
                    "item_id": str(item_id),
                    "job_id": str(job['id']),
                    "message": f"Re-queued '{item['title']}' for processing"
                }
            
            elif job and job['status'] in ('pending', 'running'):
                return {
                    "error": "conflict",
                    "message": f"A job for this item is already {job['status']}, cannot retry"
                }

            else:
                job_id = await conn.fetchval("""
                    INSERT INTO jobs (item_id, job_type, status)
                    VALUES ($1, 'ingest', 'pending')
                    RETURNING id
                """, item_id)
                await conn.execute(
                    "UPDATE items SET status = 'pending' WHERE id = $1", item_id
                )
                return {
                    "success": True,
                    "item_id": str(item_id),
                    "job_id": str(job_id),
                    "message": f"Created new job for '{item['title']}' (original job was missing)"
                }