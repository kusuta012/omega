from __future__ import annotations
from multiprocessing import Value
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING
from omega.environment.conf_loader import omega_settings
from omega.memory.consolidation import get_consolidation_job
from omega.memory.core import read_memory_md, read_user_md

if TYPE_CHECKING:
    from omega.agent.agent_loop import AgentLoop

COMMAND_HELP = """Available commands:
  /new                 Start a new session.
  /status              Show active-session and model details.
  /memory              Print Omega's current core-memory files.
  /compress            Compress the current session now.
  /consolidate         Run memory consolidation now.
  /commands, /help     Show this command list.
  /quit, /exit         Leave Omega.

Ctrl+C cancels an active response without saving partial agent output."""

@dataclass(frozen=True)
class CommandResult:
    message: str
    should_exit: bool = False

def is_command(text: str) -> bool:
    return text.lstrip().startswith("/")

async def handle_command(text: str, agent: "AgentLoop") -> CommandResult:
    try:
        parts = shlex.split(text.strip())
    except ValueError as e:
        return CommandResult(f"Invalid command syntax: {e}")
    if not parts:
        return CommandResult("")

    command = parts[0].lower()
    if command in {"/help", "/commands"}:
        return CommandResult(COMMAND_HELP)
    if command in {"/quit", "/exit"}:
        return CommandResult("Goodbye :)", should_exit=True)
    if command == "/new":
        session_id = await agent.new_session()
        return CommandResult(f"Started a new session: {session_id}")
    if command == "/status":
        session_id = agent.session_manager.active_session_id or "not started"
        return CommandResult(
            "\n".join([
                f"Session: {session_id}",
                f"Provider: {omega_settings.llm_provider}",
                f"Model: {omega_settings.llm_model}",
                f"Tool-round limit: {omega_settings.max_tool_rounds_per_turn}",
                f"Memory directory: {omega_settings.memory_dir}",
            ])
        )
    if command == "/memory":
        memory = read_memory_md().strip() or "(MEMORY.md is empty)"
        profile = read_user_md().strip() or "(USER.md is empty)"
        return CommandResult(f"# MEMORY.md\n{memory}\n\n# USER.md\n{profile}")
    if command == "/compress":
        compressed = await agent.session_manager.compress_now()
        if compressed:
            return CommandResult("Current session completed")
        return CommandResult("Nothing to compress yet; the session needs at least two messages")
    if command == "/consolidate":
        await get_consolidation_job().run()
        return CommandResult("Memory consolidate completed")
    
    return CommandResult(f"unknown command: {parts[0]}. Type /commands for help")