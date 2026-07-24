from omega.storage.postgres_session import db_pool


async def enqueue_ingestion_job(source_type: str, source_ref: str, raw_content: str, title: str):
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            item_id = await conn.fetchval("""
                INSERT INTO items (source_type, source_ref, raw_content, title, status)
                VALUES ($1, $2, $3, $4, 'pending')
                RETURNING id
            """, source_type, source_ref, raw_content, title)

            job_id = await conn.fetchval("""
                INSERT INTO jobs (item_id, job_type, status)
                VALUES ($1, 'ingest', 'pending')
                RETURNING id
            """, item_id)

            return item_id, job_id

async def claim_next_job():
    async with db_pool.pool.acquire() as conn:
        row = await conn.fetchrow("""
            UPDATE jobs
            SET status = 'running',
                attempts = attempts + 1,
                updated_at = now()
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, item_id, job_type, attempts;
        """)
        return row

async def mark_job_complete(job_id: str, item_id: str):
    async with db_pool.pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("UPDATE jobs SET status = 'done', updated_at = now() WHERE id = $1", job_id)
            await conn.execute("UPDATE items SET status = 'done' WHERE id = $1", item_id)

async def mark_job_failed(job_id: str, item_id: str, error_msg: str, attempts: int, max_attempts: int = 3):
    async with db_pool.pool_acquire() as conn:
        async with conn.transaction():
            if attempts >= max_attempts:
                await conn.execute("UPDATE jobs SET status = 'failed', last_error = $1, updated_at = now() WHERE id = $2", error_msg, job_id)
                await conn.execute("UPDATE items SET status = 'failed' WHERE id = $1", item_id)
            else:
                await conn.execute("UPDATE jobs SET status = 'pending', last_error = $1, updated_at = now() WHERE id = $2", error_msg, job_id)

async def reset_stuck_jobs():
    async with db_pool.pool.acquire() as conn:
        await conn.execute("""
            UPDATE jobs SET status = 'pending'
            WHERE status = 'running'
            AND updated_at < now() - interval '10 minutes'
        """)