from omega.storage.postgres_session import db_pool
import json

async def replace_item_content_and_chunks(
    item_id: str,
    title: str,
    parsed_content: str,
    chunks: list[str],
    embeddings: list[list[float]],
    job_id: str | None = None,
    claim_token: str | None = None,
):
    if (job_id is None) != (claim_token is None):
        raise ValueError("job_id and claim_token must be provided together")
    if len(chunks) != len(embeddings):
        raise ValueError("chunk text list and embeddings list must be the same length")
    if not chunks:
        raise ValueError("parsed content produced no chunks")

    records = [
        (item_id, index, chunk_text, json.dumps(embedding))
        for index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings))
    ]
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            updated = await conn.execute("""
                UPDATE items
                SET title = $1, raw_content = $2
                WHERE id = $3
            """, title, parsed_content, item_id)
            if updated != "UPDATE 1":
                raise ValueError(f"Item record {item_id} not found")
            await conn.execute("DELETE FROM chunks WHERE item_id = $1", item_id)
            await conn.executemany("""
                INSERT INTO chunks (item_id, chunk_index, content, embedding)
                VALUES ($1, $2, $3, $4::vector)
            """, records)
            if job_id is not None:
                completed = await conn.execute("""
                    UPDATE jobs
                    SET status = 'done', updated_at = now()
                    WHERE id = $1 AND item_id = $2 AND status = 'running' AND claim_token = $3
                """, job_id, item_id, claim_token)
                if completed != "UPDATE 1":
                    raise RuntimeError(f"Ingestion job {job_id} was not running for item {item_id}")
                await conn.execute("UPDATE items SET status = 'done' WHERE id = $1", item_id)

async def save_document_chunks(item_id: str, chunks: list[str], embeddings: list[list[float]]):
    if len(chunks) != len(embeddings):
        raise ValueError("chunk text list and embeddings list must be the same length")

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM chunks WHERE item_id = $1", item_id)

            records = [
                (item_id, index, chunk_text, json.dumps(embedding))
                for index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings))
            ]
            
            await conn.executemany("""
                INSERT INTO chunks (item_id, chunk_index, content, embedding)
                VALUES ($1, $2, $3, $4::vector)
            """, records)