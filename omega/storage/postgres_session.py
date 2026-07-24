import asyncpg
from contextlib import asynccontextmanager
from omega.environment.conf_loader import omega_settings

class DatabasePool:
    def __init__(self):
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(dsn=omega_settings.database_url)

    async def disconnect(self):
        if self.pool:
            await self.pool.close()

    @asynccontextmanager
    async def acquire(self):
        if self.pool is None:
            raise RuntimeError("Database connection pool is not intialized")
        async with self.pool.acquire() as connection:
            yield connection

db_pool = DatabasePool()