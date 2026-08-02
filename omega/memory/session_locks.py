import asyncio
from contextlib import asynccontextmanager

class SessionTurnLocks:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self, session_id: str):
        async with self._registry_lock:
            lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            yield

session_turn_locks = SessionTurnLocks()