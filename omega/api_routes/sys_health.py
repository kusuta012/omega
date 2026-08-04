from fastapi import APIRouter
from fastapi.responses import JSONResponse
from omega.storage.postgres_session import db_pool
import logging


logger = logging.getLogger(__name__)
health_router = APIRouter()

@health_router.get("/health")
async def check_system_health():
    return {"status": "ok", "service": "omega"}

@health_router.get("/ready")
async def check_sys_readiness():
    if db_pool.pool is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "service": "omega"},
        )

    try:
        async with db_pool.pool.acquire() as connection:
            await connection.fetchval("SELECT 1")
    except Exception as error:
        logger.exception("Omega readiness check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "service": "omega"}
        )

    return {"status": "ready", "service": "omega"}