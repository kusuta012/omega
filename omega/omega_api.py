import logging
from sys import exception
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import asyncpg
from omega.storage.postgres_session import db_pool
from omega.api_routes.sys_health import health_router
from omega.api_routes.ingestion_endpoints import ingestion_router
from omega.api_routes.retrieval_endpoints import retrieval_router
from omega.api_routes.management_endpoints import management_router
from omega.api_routes.agent_endpoints import router as agent_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def sys_lifespan(app: FastAPI):
    await db_pool.connect()
    yield
    await db_pool.disconnect()

omega_app = FastAPI(
    title="Omega",
    description="Omega local API",
    lifespan=sys_lifespan
)

@omega_app.exception_handler(asyncpg.PostgresError)
@omega_app.exception_handler(OSError)
@omega_app.exception_handler(TimeoutError)
async def deps_err_handler(request: Request, err: Exception):
    logger.exception(f"API dependency failure for {request.method} {request.url.path}")
    return JSONResponse(status_code=503, content={"detail": "service temporary unavailable"})

@omega_app.exception_handler(Exception)
async def internal_error_handler(request: Request, error: Exception):
    logger.exception(f"Unhandled API faliure for {request.method} {request.url.path}")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})

omega_app.include_router(health_router, prefix="/api/v1")
omega_app.include_router(ingestion_router, prefix="/api/v1")
omega_app.include_router(retrieval_router, prefix="/api/v1")
omega_app.include_router(management_router, prefix="/api/v1")
omega_app.include_router(agent_router)