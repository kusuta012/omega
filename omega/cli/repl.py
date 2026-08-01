from __future__ import annotations
import asyncio
from asyncio.exceptions import CancelledError
import logging
from contextlib import suppress
from omega.agent.agent_loop import AgentLoop
from omega.cli.commands import COMMAND_HELP, handle_command, is_command

logger = logging.getLogger(__name__)

async def _render_turn(agent: AgentLoop, user_message: str, resume: bool) -> None:
    answer_started = False
    try:
        async for event in agent.process_stream(user_message, resume=resume):
            if event.type == "text_delta" and event.text:
                if not answer_started:
                    print("\nOmega: ", end="", flush=True)
                    answer_started = True
                print(event.text, end="", flush=True)
            elif event.type == "tool_started" and event.tool_name:
                if answer_started:
                    print()
                    answer_started = False
                print(f"[tool] {event.tool_name}...", flush=True)
            elif event.type == "tool_completed" and event.tool_name:
                summary = event.summary or "complete"
                print(f"[tool] {event.tool_name}: {summary}", flush=True)
        if answer_started:
            print()
    except asyncio.CancelledError:
        if answer_started:
            print()
        print("[cancelled] response stopped; partial agent output was not saved")
        raise

async def _run_turn_with_interrupt(agent: AgentLoop, user_message: str, resume: bool) -> None:
    turn_task = asyncio.create_task(_render_turn(agent, user_message, resume))
    try:
        await asyncio.shield(turn_task)
    except asyncio.CancelledError:
        turn_task.cancel()
        with suppress(asyncio.CancelledError):
            await turn_task
        print()
    except Exception as exc:
        logger.exception("Interactive turn failed")
        print(f"\n[error] {exc}")

async def run_repl(*, continue_session: bool = False) -> int:
    agent = AgentLoop()
    resume_next_turn = continue_session
    
    print("Omega CLI")
    print("Type /commands for commands. Ctrl+C cancels a response, /quit exits.")
    if continue_session:
        print("The next message will resume the latest available session")

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except EOFError:
            print("\nGoodbye.")
            return 0
        except KeyboardInterrupt:
            print("\nType /quit to exit")
            continue

        if not user_message:
            continue
        if is_command(user_message):
            try:
                result = await handle_command(user_message, agent)
            except Exception as e:
                logger.exception("interactive command failed")
                print(f"[error] {e}")
                continue
            if result.message:
                print(result.message)
            if result.should_exit:
                return 0
            continue

        await _run_turn_with_interrupt(agent, user_message, resume_next_turn)
        resume_next_turn = False