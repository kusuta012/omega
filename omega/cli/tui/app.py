from __future__ import annotations
import asyncio
import logging
import time

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from omega.agent.agent_loop import AgentLoop
from omega.cli.commands import handle_command, is_command
from omega.cli.tui.widgets import DynamicLine, Transcript
from omega.environment.conf_loader import omega_settings

logger = logging.getLogger(__name__)

_BUSY_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

# I provided a color pallete to gpt and told him to generate this _STYLE code cause I'm lazy :)
_STYLE = Style.from_dict(
    {
        "header": "bg:#120f0a #f8e8c1",
        "header.brand": "bg:#120f0a #f6bd60 bold",
        "header.meta": "bg:#120f0a #c9a66b",
        "header.model": "bg:#120f0a #e59f4f",
        "status": "bg:#1a140b #e9c783",
        "status.busy": "bg:#1a140b #ffbf5b bold",
        "status.muted": "bg:#1a140b #af9465",
        "input": "bg:#151008 #f7e5bd",
        "input.prompt": "bg:#151008 #ffbf5b bold",
        "input.hint": "bg:#0e0b07 #826d4d",
        "transcript": "bg:#0b0907 #f1dfb7",
        "transcript.user_label": "bg:#0b0907 #e7ae55 bold",
        "transcript.omega_label": "bg:#0b0907 #ffcc70 bold",
        "transcript.tool_running": "bg:#0b0907 #e9b95c",
        "transcript.tool_done": "bg:#0b0907 #d5a661",
        "transcript.notice": "bg:#0b0907 #e39864",
        "transcript.heading": "bg:#0b0907 #ffcc70 bold",
        "transcript.bullet": "bg:#0b0907 #e7d1a0",
        "transcript.bold": "bg:#0b0907 #ffe1a0 bold",
        "transcript.italic": "bg:#0b0907 #e8c987 italic",
        "transcript.code": "bg:#22190e #ffd88e",
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
        self.turn_started_at: float | None = None
        self.transcript = Transcript()
        self.input = TextArea(
            prompt="› ",
            multiline=False,
            style="class:input",
            accept_handler=self._accept_input,
        )

        key_bindings = KeyBindings()

        @key_bindings.add("c-c")
        def _interrupt(event) -> None:
            self.interrupt_turn()

        @key_bindings.add("pageup")
        def _scroll_up(event) -> None:
            self.transcript.page_up()
            self.application.invalidate()

        @key_bindings.add("pagedown")
        def _scroll_down(event) -> None:
            self.transcript.page_down()
            self.application.invalidate()

        @key_bindings.add("c-end")
        def _follow_latest(event) -> None:
            self.transcript.follow_latest()
            self.application.invalidate()

        @key_bindings.add("c-home")
        def _scroll_to_top(event) -> None:
            self.transcript.scroll_to_top()
            self.application.invalidate()

        self.application = Application(
            layout=Layout(
                HSplit(
                    [
                        DynamicLine(self._header_top, style="class:header"),
                        DynamicLine(self._header_text, style="class:header"),
                        DynamicLine(self._header_bottom, style="class:header"),
                        self.transcript.area,
                        DynamicLine(self._status_text, style="class:status"),
                        Window(
                            content=FormattedTextControl(self._input_hint),
                            height=1,
                            style="class:input.hint",
                            dont_extend_height=True,
                        ),
                        self.input,
                    ]
                ),
                focused_element=self.input,
            ),
            key_bindings=key_bindings,
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.1,
            style=_STYLE,
        )

    def _terminal_width(self) -> int:
        return max(40, get_app().output.get_size().columns)

    def _header_top(self) -> str:
        prefix = "╭─ ☆ OMEGA "
        return f"{prefix}{'─' * max(1, self._terminal_width() - len(prefix) - 1)}╮"

    def _header_bottom(self) -> str:
        width = self._terminal_width()
        return f"╰{'─' * max(1, width - 2)}╯"

    def _header_text(self) -> StyleAndTextTuples:
        session = self.agent.session_manager.active_session_id
        session_label = str(session)[:8] if session else "new"
        provider = f"{omega_settings.llm_provider}/{omega_settings.llm_model}"
        width = self._terminal_width()
        left = f"│  local agent · session {session_label}"
        max_provider = max(12, width - len(left) - 3)
        if len(provider) > max_provider:
            provider = f"{provider[:max_provider - 1]}…"
        gap = max(1, width - len(left) - len(provider) - 3)
        return [
            ("class:header.brand", "│  ☁︎  "),
            ("class:header.meta", f"local agent · session {session_label}"),
            ("class:header.model", f"{' ' * gap}{provider} │"),
        ]

    def _status_text(self) -> StyleAndTextTuples:
        if self.turn_task and not self.turn_task.done():
            elapsed = int(time.monotonic() - (self.turn_started_at or time.monotonic()))
            frame = _BUSY_FRAMES[int(time.monotonic() * 10) % len(_BUSY_FRAMES)]
            return [
                ("class:status.busy", f" {frame} generating · {elapsed:02}s"),
                ("", f" {self.tool_round} tool{'s' if self.tool_round != 1 else ''} this turn"),
                ("class:status.muted", "  ·  Ctrl+C stops generation"),
            ]
        return [
            ("class:status.busy", " ● ready"),
            ("class:status.muted", " Enter sends  ·  /commands for help"),
        ]

    @staticmethod
    def _input_hint() -> StyleAndTextTuples:
        return [
            ("class:input.hint", "  Enter  "),
            ("class:status.muted", "send    "),
            ("class:input.hint", "Ctrl+C "),
            ("class:status.muted", "stop generation    "),
            ("class:input.hint", "/ "),
            ("class:status.muted", "commands"),
        ]

    def _accept_input(self, buffer) -> bool:
        text = buffer.text.strip()
        buffer.reset()
        if text and (self.turn_task is None or self.turn_task.done()):
            self.transcript.start_turn()
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
            self.transcript.notice(f"error · {e}", marker="!")
        finally:
            self.state = "idle"
            self.turn_started_at = None
            self.application.invalidate()
    
    async def _handle_command(self, text: str) -> None:
        result = await handle_command(text, self.agent)
        if text.strip().split(maxsplit=1)[0].lower() == "/new" and not result.should_exit:
            self.session_usage.clear()
            self.tool_round = 0
        if result.message:
            self.transcript.notice(result.message, marker="·")
        if result.should_exit:
            self.application.exit(result=0)
        self.application.invalidate()

    async def _run_agent_turn(self, user_message: str) -> None:
        self.state = "generating"
        self.turn_started_at = time.monotonic()
        self.tool_round = 0
        self.transcript.user_message(user_message)
        wrote_answer = False
        assistant_started = False
        resume = self.continue_next_turn
        self.continue_next_turn = False
        self.application.invalidate()

        try:
            async for event in self.agent.process_stream(user_message, resume=resume):
                if event.type == "text_delta" and event.text:
                    if not assistant_started:
                        self.transcript.assistant_start()
                        assistant_started = True
                    self.transcript.assistant_append(event.text)
                    wrote_answer = True
                elif event.type == "tool_started" and event.tool_name:
                    self.tool_round += 1
                    if wrote_answer:
                        self.transcript.line()
                        wrote_answer = False
                    self.transcript.tool_started(event.tool_name)
                elif event.type == "tool_completed" and event.tool_name:
                    self.transcript.tool_completed(event.tool_name, event.summary)
                elif event.type == "turn_complete":
                    for key, value in event.usage.items():
                        self.session_usage[key] = self.session_usage.get(key, 0) + value
                self.application.invalidate()
        except asyncio.CancelledError:
            if wrote_answer:
                self.transcript.line()
            self.transcript.notice("cancelled · partial assistant output was not saved", marker="×")
            raise
        finally:
            self.transcript.line()
            self.application.invalidate()

    def interrupt_turn(self) -> None:
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()
            self.transcript.notice("interrupt requested", marker="×")
        else:
            self.transcript.notice("Use /quit to exit Omega", marker="·")
        self.application.invalidate()

    async def run(self) -> int:
        self.transcript.welcome(resume=self.continue_next_turn)
        try:
            return await self.application.run_async()
        finally:
            await self.agent.close_session()


async def run_tui(*, continue_session: bool = False) -> int:
    tui = OmegaTui(continue_session=continue_session)
    return await tui.run()
