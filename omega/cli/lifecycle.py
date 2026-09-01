from __future__ import annotations
import asyncio
import subprocess
import sys
from pathlib import Path
from omega.agent.agent_loop import AgentLoop
from omega.knowledge_ingestion import enqueue_knowledge_ingestion
from omega.storage.management_queries import (
    delete_item,
    get_item_detail,
    list_items_paginated,
    retry_failed_item,
)
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

async def create_knowledge_ingestion(
    source_type: str,
    source_ref: str | None,
    raw_content: str | None,
    title: str | None,
) -> dict:
    await db_pool.connect()
    try:
        return await enqueue_knowledge_ingestion(source_type, source_ref, raw_content, title)
    finally:
        await db_pool.disconnect()

def run_ingest(
    source_type: str,
    content: str | None,
    title: str | None,
    file_path: str | None,
) -> int:
    try:
        if source_type == "url":
            if file_path:
                raise ValueError("url ingestion does not accept --file")
            source_ref, raw_content = content, None
        else:
            if file_path and content:
                raise ValueError("provide content or --file, not both")
            if file_path:
                path = Path(file_path).expanduser()
                raw_content = path.read_text(encoding="utf-8")
                title = title or path.name
            else:
                raw_content = content
            source_ref = None
        result = asyncio.run(create_knowledge_ingestion(source_type, source_ref, raw_content, title))
    except (OSError, ValueError) as e:
        print(f"Could not ingest knowledge: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Could not ingest knowledge: {e}", file=sys.stderr)
        return 1

    if result["duplicate"]:
        print(f"Knowledge-base item already exists: {result['item_id']}")
    else:
        print(f"Queued knowledge-base item: {result['item_id']} (job {result['job_id']})")
    return 0

async def manage_knowledge_base(command: str, item_id: str | None, limit: int, status: str | None) -> dict:
    if command != "list" and not item_id:
        raise ValueError("an item ID is required")
    assert command == "list" or item_id is not None
    await db_pool.connect()
    try:
        if command == "list":
            items, total = await list_items_paginated(page=1, page_size=limit, status=status)
            return {"items": items, "total": total}
        if command == "status":
            return {"item": await get_item_detail(item_id)}
        if command == "retry":
            return await retry_failed_item(item_id)
        if command == "remove":
            return {"deleted": await delete_item(item_id)}
        raise ValueError(f"unknown knowledge-base command: {command}")
    finally:
        await db_pool.disconnect()

def run_kb_command(
    command: str,
    item_id: str | None,
    limit: int,
    status: str | None,
    confirmed: bool,
) -> int:
    if command == "remove" and not confirmed:
        print("Refusing to remove a knowledge-base item without --yes", file=sys.stderr)
        return 2
    try:
        result = asyncio.run(manage_knowledge_base(command, item_id, limit, status))
    except Exception as err:
        print(f"Could not manage knowledge base: {err}", file=sys.stderr)
        return 1

    if command == "list":
        for item in result["items"]:
            print(f"{item['id']} {item['status']:<7}  {item['source_type']:<4}  {item['title'] or 'Untitled'}")
        print(f"Showing {len(result['items'])} of {result['total']} item(s)")
        return 0
    if command == "status":
        item_detail = result["item"]
        if not item_detail:
            print("Knowledge-base item not found", file=sys.stderr)
            return 1
        item = item_detail["item"]
        print(f"{item['id']}  {item['status']}  {item['source_type']}  {item['title'] or 'Untitled'}")
        for job in item_detail["jobs"]:
            print(f"job {job['id']}  {job['status']} attempts={job['attempts']}")
        return 0
    if command == "retry":
        if "error" in result:
            print(result["message"], file=sys.stderr)
            return 1
        print(result["message"])
        return 0

    deleted = result["deleted"]
    if not deleted:
        print("Knowledge-base item not found", file=sys.stderr)
        return 1
    print(f"Removed knowledge-base item: {deleted['id']}")
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

def run_worker() -> int:
    from omega.background_tasks.queue_worker import main as worker_main
    return worker_main()