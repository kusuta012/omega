from fastapi import APIRouter
from omega.storage.postgres_session import db_pool

health_router = APIRouter()

@health_router.get("/health")
async def check_system_health():
    db_status = "disconnected"
    try:
        if db_pool.pool:
            async with db_pool.pool.acquire() as connection:
                result = await connection.fetchval("SELECT 1")
                if result == 1:
                    db_status = "connected"
    except Exception as error:
        db_status = f"error: {str(error)}"

    return {
        "status": "online",
        "database": db_status,
        "engine": "Omega"
    }