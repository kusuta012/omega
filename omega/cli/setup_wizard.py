from __future__ import annotations
import asyncio
import shutil
import subprocess
import time
from pathlib import Path
import asyncpg
from omega.cli.configuration import prompt_llm_settings, read_env, validate_llm_settings, write_env
from omega.embeddings.embedding_service import EMBEDDING_DIM, get_embedding_service
from omega.environment.conf_loader import omega_settings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATABASE_SERVICE = "omega_db"
_DATABASE_START_TIMEOUT = 45


def _start_bundled_database() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is required for Omega. Install Docker and Docker compose, then rerun omega setup"
        )
    compose_file = _PROJECT_ROOT / "docker-compose.yml"
    if not compose_file.is_file():
        raise RuntimeError(f"Bundled Docker Compose file is missing: {compose_file}")

    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", _DATABASE_SERVICE],
            cwd=_PROJECT_ROOT,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as ex:
        raise RuntimeError(
            "could not start omega's database with Docker Compose. "
            "Ensure Docker is running, then rerun omega setup."
        ) from ex

async def _connect_database() -> asyncpg.Connection:
    deadline = time.monotonic() + _DATABASE_START_TIMEOUT
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            return await asyncpg.connect(omega_settings.database_url, timeout=3)
        except Exception as ex:
            last_error = ex
            await asyncio.sleep(1)

    raise RuntimeError(
        "Omega's database did not become ready within"
        f"{_DATABASE_START_TIMEOUT} seconds: {last_error}"
    )

def _apply_schema() -> None:
    async def apply() -> None:
        connection = await _connect_database()
        try:
            await connection.execute(
                (_PROJECT_ROOT / "schema.sql").read_text(encoding="utf-8")
            )
        finally:
            await connection.close()
    asyncio.run(apply())

def run_setup() -> int:
    existing = read_env()
    try:
        print("Starting Omega's database")
        _start_bundled_database()
        print("Applying database schema")
        _apply_schema()
        settings = prompt_llm_settings(existing)
        print("Validating LLM settings...")
        asyncio.run(validate_llm_settings(settings["LLM_PROVIDER"], settings["LLM_BASE_URL"], settings["LLM_API_KEY"], settings["LLM_MODEL"]))
        print("Verifying embedding model")
        service = get_embedding_service()
        if len(service.generate_single_embedding("Omega setup check")) != EMBEDDING_DIM:
            raise RuntimeError("Embedding model produced an unexpected vector dimension")
        write_env(settings)
    except (ValueError, OSError, RuntimeError, asyncpg.PostgresError) as ex:
        print(f"Setup was not completed: {ex}")
        return 1
    print("Setup complete. Run 'omega' to start.")
    return 0