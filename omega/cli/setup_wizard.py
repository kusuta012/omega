from __future__ import annotations
import asyncio
import shutil
import subprocess
import time
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
import asyncpg
from omega.cli.configuration import prompt_llm_settings, read_env, validate_llm_settings, write_env
from omega.llm.provider_specs import sanitize_provider_error
from omega.embeddings.embedding_service import EMBEDDING_DIM, get_embedding_service
from omega.environment.conf_loader import omega_settings

_DATABASE_SERVICE = "omega_db"
_DATABASE_START_TIMEOUT = 45

@contextmanager
def _bundled_files():
    assets = resources.files("omega").joinpath("resources")
    with resources.as_file(assets.joinpath("docker-compose.yml")) as compose_file, resources.as_file(assets.joinpath("schema.sql")) as schema_file:
        yield compose_file, schema_file
        
def _start_bundled_database(compose_file: Path) -> None:
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is required for Omega. Install Docker and Docker compose, then rerun omega setup"
        )

    if not compose_file.is_file():
        raise RuntimeError(f"Bundled Docker Compose file is missing: {compose_file}")

    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", _DATABASE_SERVICE],
            cwd=compose_file.parent,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as ex:
        raise RuntimeError(
            "Could not start omega's database with Docker Compose. "
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

def _apply_schema(schema_file: Path) -> None:
    async def apply() -> None:
        connection = await _connect_database()
        try:
            await connection.execute(
                schema_file.read_text(encoding="utf-8")
            )
        finally:
            await connection.close()
    asyncio.run(apply())

def run_setup() -> int:
    existing = read_env()
    try:
        with _bundled_files() as (compose_file, schema_file):
            print("Starting Omega's database")
            _start_bundled_database(compose_file)
            print("Applying database schema")
            _apply_schema(schema_file)
        settings = prompt_llm_settings(existing)
        print("Validating LLM settings...")
        try:
            asyncio.run(validate_llm_settings(settings["LLM_PROVIDER"], settings["LLM_BASE_URL"], settings["LLM_API_KEY"], settings["LLM_MODEL"]))
        except Exception as e:
            raise ValueError(sanitize_provider_error(e)) from e
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