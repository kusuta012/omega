from __future__ import annotations
import asyncio
import uuid
import logging
from dataclasses import dataclass
from pathlib import Path
import asyncpg
from omega.cli.logging_config import get_log_file
from omega.embeddings.embedding_service import EMBEDDING_DIM, get_embedding_service
from omega.environment.conf_loader import omega_settings
from omega.llm.client import get_llm_provider
from omega.llm.provider_specs import sanitize_provider_error

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DoctorCheck:
    name: str
    passed: bool
    detail: str

async def _connect_database() -> asyncpg.Connection:
    return await asyncpg.connect(omega_settings.database_url, timeout=5)

async def _check_database() -> DoctorCheck:
    connection = None
    try:
        connection = await _connect_database()
        await connection.fetchval("SELECT 1")
        return DoctorCheck("database", True, "Connected to PostgreSQL")
    except Exception:
        return DoctorCheck("database", False, "Database is unavailable; run omega setup")
    finally:
        if connection is not None:
            await connection.close()

async def _check_pgvector() -> DoctorCheck:
    connection = None
    try:
        connection = await _connect_database()
        installed = await connection.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        if installed:
            return DoctorCheck("pgvector", True, "The vector extension is installed")
        return DoctorCheck("pgvector", False, "The vector extension is not installed")
    except Exception as e:
        return DoctorCheck("pgvector", False, f"Could not check the database extension")
    finally:
        if connection is not None:
            await connection.close()

async def _check_embedding_model() -> DoctorCheck:
    try:
        service = await asyncio.to_thread(get_embedding_service)
        vector = await asyncio.to_thread(service.generate_single_embedding, "omega doctor check")
        if len(vector) != EMBEDDING_DIM:
            return DoctorCheck(
                "embedding model",
                False,
                f"Expected {EMBEDDING_DIM} dimensions, received {len(vector)}.",
            )
        return DoctorCheck(
            "embedding model",
            True,
            f"Loaded {omega_settings.embedding_model} ({len(vector)} dimensions)."
        )
    except Exception as e:
        return DoctorCheck("embedding model", False, f"Could not load the configured embedding model")


async def _check_llm() -> DoctorCheck:
    if not omega_settings.llm_api_key.strip():
        return DoctorCheck("LLM provider", False, "LLM_API_KEY is not configured")

    try:
        provider = get_llm_provider()
        response = await provider.generate_answer(
            "You are a connectivity check, reply with exactly OK",
            "Respond now",
        )
        if not response or not response.strip():
            return DoctorCheck("LLM provider", False, "Provider returned an empty response")
        return DoctorCheck(
            "LLM provider",
            True,
            f"Validated {omega_settings.llm_provider}/{omega_settings.llm_model}",
        )
    except Exception as e:
        return DoctorCheck("LLM provider", False, sanitize_provider_error(e))

async def _check_memory_dir() -> DoctorCheck:
    path = Path(omega_settings.memory_dir).expanduser()
    probe = path / f".omega-doctor-{uuid.uuid4().hex}.tmp"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        if probe.read_text(encoding="utf-8") != "ok":
            return DoctorCheck("memory directory", False, "Write verification returned unexpected content")
        return DoctorCheck("memory directory", True, f"Writable: {path.resolve()}")
    except Exception as e:
        return DoctorCheck("memory directory", False, f"Not writable {e}")
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove doctor probe file %s", probe)

async def run_doctor() -> list[DoctorCheck]:
    checks = await asyncio.gather(
        _check_database(),
        _check_pgvector(),
        _check_embedding_model(),
        _check_llm(),
        _check_memory_dir()
    )
    return list(checks)

def print_doctor_report(checks: list[DoctorCheck]) -> bool:
    print("Omega doctor:")
    print(f"Log file: {get_log_file()}")
    print()
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    healthy = all(check.passed for check in checks)
    print()
    print("Omega is ready" if healthy else "Omega needs attention; see failed checks above")
    return healthy