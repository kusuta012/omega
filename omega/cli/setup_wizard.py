from __future__ import annotations
import asyncio
from pathlib import Path
import asyncpg
from omega.cli.configuration import prompt_llm_settings, read_env, validate_llm_settings, write_env
from omega.embeddings.embedding_service import EMBEDDING_DIM, get_embedding_service


def _apply_schema(database_url: str) -> None:
    async def apply() -> None:
        connection = await asyncpg.connect(database_url, timeout=10)
        try:
            await connection.execute(Path("schema.sql").read_text(encoding="utf-8"))
        finally:
            await connection.close()
    asyncio.run(apply())

def run_setup() -> int:
    existing = read_env()
    try:
        settings = prompt_llm_settings(existing)
        database_url = input(f"Database URL [{existing.get('DATABASE_URL')}]: ").strip() or existing.get("DATABASE_URL")
        print("Validating LLM settings...")
        asyncio.run(validate_llm_settings(settings["LLM_PROVIDER"], settings["LLM_BASE_URL"], settings["LLM_API_KEY"], settings["LLM_MODEL"]))
        print("Applying database schema...")
        _apply_schema(database_url)
        print("Verifying embedding model")
        service = get_embedding_service()
        if len(service.generate_single_embedding("Omega setup check")) != EMBEDDING_DIM:
            raise RuntimeError("Embedding model produced an unexpected vector dimension")
        write_env({**settings, "DATABASE_URL": database_url})
    except (ValueError, OSError, RuntimeError, asyncpg.PostgresError) as ex:
        print(f"Setup was not completed: {ex}")
        return 1
    print("Setup complete. Run 'omega' to start.")
    return 0