from __future__ import annotations
import asyncio
import subprocess
import sys
from omega.agent.agent_loop import AgentLoop
from omega.storage.postgres_session import db_pool

async def create_new_session() -> str:
    await db_pool.connect()
    try:
        agent = AgentLoop()
        return await agent.new_session()
    finally:
        await db_pool.disconnect()

def run_new_session() -> int:
    try:
        session_id = asyncio.run(create_new_session())
    except Exception as ex:
        print(f"Could not start a new session: {ex}")
        return 1
    print(f"Started a new session: {session_id}")
    return 0


def run_uninstall(*, yes: bool) -> int:
    if not yes:
        print("Refusing to uninstall without --yes")
        print("This removes the installed Omega package only; .env, omega_memory, and PostgreSQL data are preserved.")
        return 2
    
    command = [sys.executable, "-m", "pip", "uninstall", "--yes", "omega-agent"]
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as exc:
        print(f"Could not run the package installer: {exc}")
        return 1