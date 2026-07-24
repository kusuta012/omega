from omega.storage.postgres_session import db_pool

async def fetch_item_by_id(item_id: str):
    async with db_pool.acquire() as conn:
        return await conn.fetchrow("""
            SELECT id, source_type, source_ref, raw_content, title, status
            FROM items WHERE id = $1
        """, item_id)

async def update_item_parsed_content(item_id: str, title: str, parsed_content: str):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE items
            SET title = $1, raw_content = $2
            WHERE id = $3
        """, title, parsed_content, item_id)