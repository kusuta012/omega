from fastapi import FastAPI
from contextlib import asynccontextmanager
from omega.storage.postgres_session import db_pool
from omega.api_routes.sys_health import health_router
from omega.api_routes.ingestion_endpoints import ingestion_router
from omega.api_routes.retrieval_endpoints import retrieval_router

@asynccontextmanager
async def sys_lifespan(app: FastAPI):
    await db_pool.connect()
    yield
    await db_pool.disconnect()

omega_app = FastAPI(
    title="Omega",
    description="skibidi description not needed",
    lifespan=sys_lifespan
)

omega_app.include_router(health_router, prefix="/api/v1")
omega_app.include_router(ingestion_router, prefix="/api/v1")
omega_app.include_router(retrieval_router, prefix="/api/v1")