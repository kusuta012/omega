from omega.storage.postgres_session import db_pool
import json

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