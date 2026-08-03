import asyncpg
import asyncio
from contextlib import asynccontextmanager
from importlib import resources
from omega.environment.conf_loader import omega_settings

class DatabasePool:
    def __init__(self):
        self.pool = None
        self._lifecycle_lock = asyncio.Lock()

    async def connect(self):
        async with self._lifecycle_lock:
            if self.pool is None:
                pool = await asyncpg.create_pool(dsn=omega_settings.database_url)
                try:
                    schema = resources.files("omega").joinpath("resources", "schema.sql").read_text(encoding="utf-8")
                    async with pool.acquire() as connection:
                        await connection.execute(schema)
                except Exception:
                    await pool.close()
                    raise
                self.pool = pool

    async def disconnect(self):
        async with self._lifecycle_lock:
            if self.pool is not None:
                pool, self.pool = self.pool, None
                await pool.close()

    @asynccontextmanager
    async def acquire(self):
        if self.pool is None:
            raise RuntimeError("Database connection pool is not intialized")
        async with self.pool.acquire() as connection:
            yield connection

db_pool = DatabasePool()