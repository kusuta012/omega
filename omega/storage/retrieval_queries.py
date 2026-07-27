from omega.storage.postgres_session import db_pool
import json
import logging

logger = logging.getLogger("RetrievalQueries")

async def search_memory_entries(query_text: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
    embedding_json = json.dumps(query_embedding)
    query = """
        SELECT
            id, memory_type, content, importance, access_count,
            created_at, source_session_id, metadata, superseded_by,
            (1 - (embedding <=> $1::vector)) as semantic_score,
            GREATEST(0.0, EXP(-0.02 * EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0)) as recency_score,
            (
                (1 - (embedding <=> $1::vector)) * 1.0 +
                GREATEST(0.0, EXP(-0.02 * EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0)) * 1.0 +
                COALESCE(importance, 0.5) * 1.0 +
                LN(COALESCE(access_count, 0) + 1) * 1.0
            ) as composite_score
        FROM memory_entries
        WHERE superseded_by IS NULL
          AND embedding <=> $1::vector < 1.5
        ORDER BY composite_score DESC
        LIMIT $2
    """

    async with db_pool.acquire() as conn:
        records = await conn.fetch(query, embedding_json, top_k)
    return [dict(r) for r in records]

async def search_hybrid_chunks(query_text: str, query_embedding: list[float], top_k: int = 5, distance_threshold: float = 0.6, min_rrf_score: float = 0.01, min_keyword_rank: float = 0.05):
    embedding_json = json.dumps(query_embedding)
    hybrid_query = """
        WITH vector_search AS (
            SELECT id, item_id, chunk_index, content,
                ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) as rank
            FROM chunks
            WHERE embedding <=> $1::vector < $4
            ORDER BY embedding <=> $1::vector
            LIMIT 20
        ),
        keyword_search AS(
            SELECT id, item_id, chunk_index, content,
                    ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english',$2)) DESC) as rank
            FROM chunks
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $2)
            AND ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $2)) >= $5
            ORDER BY ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', $2)) DESC
            LIMIT 20
        ),
        rrf_fusion AS (
            SELECT
                COALESCE(v.id, k.id) as chunk_id,
                COALESCE(v.item_id, k.item_id) as item_id,
                COALESCE(v.content, k.content) as chunk_text,
                (COALESCE(1.0 / (60 + v.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0)) as rrf_score
            FROM vector_search v
            FULL OUTER JOIN keyword_search k ON v.id = k.id
        )
        SELECT
            r.chunk_id,
            r.item_id,
            r.chunk_text,
            r.rrf_score,
            i.title AS source_title,
            i.source_ref
        FROM rrf_fusion r
        JOIN items i ON r.item_id = i.id
        WHERE r.rrf_score >= $6
        ORDER BY r.rrf_score DESC
        LIMIT $3;
    """

    async with db_pool.acquire() as conn:
        records = await conn.fetch(hybrid_query, embedding_json, query_text, top_k, distance_threshold, min_keyword_rank, min_rrf_score)
        return [dict (record) for record in records]