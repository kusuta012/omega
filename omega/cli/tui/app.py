from __future__ import annotations
import asyncio
import logging
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from omega.agent.agent_loop import AgentLoop
from omega.cli.commands import handle_command, is_command
from omega.cli.tui.widgets import DynamicLine, Transcript
from omega.environment.conf_loader import omega_settings

logger = logging.getLogger(__name__)

_STYLE = Style.from_dict(
    {
        "header": "bg:#202a44 #f1f5f9 bold",
        "status": "bg:#172033 #a7f3d0",
        "input": "bg:#0f172a #f8fafc",
        "transcript": "bg:#0b1020 #e2e8f0",
    }
)



class OmegaTui:
    def __init__(self, *, continue_session: bool = False) -> None:
        self.agent = AgentLoop()
        self.continue_next_turn = continue_session
        self.turn_task: asyncio.Task[None] | None = None
        self.state = "idle"
        self.tool_round = 0
        self.session_usage: dict[str, int] = {}
        self.transcript = Transcript()
        self.input = TextArea(
            prompt="> ",
            multiline=False,
            style="class:input",
            accept_handler=self._accept_input,
        )

        key_bindings = KeyBindings()

        @key_bindings.add("c-c")
        def _interrupt(event) -> None:
            self.interrupt_turn()

        self.application = Application(
            layout=Layout(
                HSplit(
                    [
                        DynamicLine(self._header_text, style="class:header"),
                        self.transcript.area,
                        DynamicLine(self._status_text, style="class:status"),
                        self.input,
                    ]
                ),
                focused_element=self.input,
            ),
            key_bindings=key_bindings,
            full_screen=True,
            mouse_support=False,
            style=_STYLE,
        )

    def _header_text(self) -> str:
        session = self.agent.session_manager.active_session_id
        session_label = str(session)[:8] if session else "new"
        token_count = sum(self.session_usage.values())
        return (
            f" Omega  -  {omega_settings.llm_provider}/{omega_settings.llm_model}"
            f"  -  {token_count:,} streamed tokens  -  session {session_label}"
        )

    def _status_text(self) -> str:
        if self.turn_task and not self.turn_task.done():
            return (
                f" generating - tools this turn {self.tool_round}"
                " - Ctrl + C cancels - /commands for help"
            )
        return " idle - Enter sends - Ctrl+C exits - /commands for help"

    def _accept_input(self, buffer) -> bool:
        text = buffer.text.strip()
        buffer.reset()
        if text and (self.turn_task is None or self.turn_task.done()):
            self.turn_task = asyncio.create_task(self._handle_input(text))
        return True

    async def _handle_input(self, text: str) -> None:
        try:
            if is_command(text):
                await self._handle_command(text)
            else:
                await self._run_agent_turn(text)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("TUI input handling failed")
            self.transcript.line(f"[error] {e}")
        finally:
            self.state = "idle"
            self.input.buffer.reset()
            self.application.invalidate()
    
    async def _handle_command(self, text: str) -> None:
        result = await handle_command(text, self.agent)
        if text.strip().split(maxsplit=1)[0].lower() == "/new" and not result.should_exit:
            self.session_usage.clear()
            self.tool_round = 0
        if result.message:
            self.transcript.line(result.message)
        if result.should_exit:
            self.application.exit(result=0)
        self.application.invalidate()

    async def _run_agent_turn(self, user_message: str) -> None:
        self.state = "generating"
        self.tool_round = 0
        self.transcript.line(f"You: {user_message}")
        self.transcript.append("Omega: ")
        wrote_answer = False
        resume = self.continue_next_turn
        self.continue_next_turn = False
        self.application.invalidate()

        try:
            async for event in self.agent.process_stream(user_message, resume=resume):
                if event.type == "text_delta" and event.text:
                    self.transcript.append(event.text)
                    wrote_answer = True
                elif event.type == "tool_started" and event.tool_name:
                    self.tool_round += 1
                    if wrote_answer:
                        self.transcript.line()
                    self.transcript.line(f"[tool] {event.tool_name}...")
                    wrote_answer = False
                elif event.type == "tool_completed" and event.tool_name:
                    self.transcript.line(
                        f"[tool] {event.tool_name}: {event.summary or 'complete'}"
                    )
                elif event.type == "turn_complete":
                    for key, value in event.usage.items():
                        self.session_usage[key] = self.session_usage.get(key, 0) + value
                self.application.invalidate()
        except asyncio.CancelledError:
            if wrote_answer:
                self.transcript.line()
            self.transcript.line("[cancelled] Response stopped; partial assistant output was not saved.")
            raise
        finally:
            self.transcript.line()
            self.application.invalidate()

    def interrupt_turn(self) -> None:
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            self.transcript.line("[interrupt requested]")
        else:
            self.transcript.line("Use /quit to exit Omega")
        self.application.invalidate()

    async def run(self) -> int:
        self.transcript.line("Omega")
        self.transcript.line("Type /commands for commands. Ctrl+C cancels a response, /quit exits")
        if self.continue_next_turn:
            self.transcript.line("The next message will resume the latest available session")
        self.transcript.line()
        try:
            return await self.application.run_async()
        finally:
            await self.agent.close_session()


async def run_tui(*, continue_session: bool = False) -> int:
    tui = OmegaTui(continue_session=continue_session)
    return await tui.run()
