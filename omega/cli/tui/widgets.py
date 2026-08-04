from __future__ import annotations
import re
from collections.abc import Callable

from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import TextArea

_INLINE_MARKDOWN = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\n]+\*)")

class MarkdownTranscriptLexer(Lexer):
    def lex_document(self, document: Document):
        def get_line(line_number: int) -> StyleAndTextTuples:
            line = document.lines[line_number]
            if line == "● ":
                return [("class:transcript.user_label", line)]
            if line == "⚕ Omega":
                return [("class:transcript.omega_label", line)]
            if line.lstrip().startswith("◌"):
                return [("class:transcript.tool_running", line)]
            if line.lstrip().startswith("●"):
                return [("class:transcript.tool_done", line)]
            if line.startswith(("!  ", "×  ")):
                return [("class:transcript.notice", line)]
            if line.startswith("   #"):
                return [("class:transcript.heading", line)]
            if line.startswith("   - ") or line.startswith("   * "):
                return [("class:transcript.bullet", line)]

            fragments: StyleAndTextTuples = []
            cursor = 0
            for match in _INLINE_MARKDOWN.finditer(line):
                if match.start() > cursor:
                    fragments.append(("class:transcript", line[cursor:match.start()]))
                token = match.group(0)
                if token.startswith("**"):
                    fragments.append(("class:transcript.bold", token[2:-2]))
                elif token.startswith("`"):
                    fragments.append(("class:transcript.code", token[1:-1]))
                else:
                    fragments.append(("class:transcript.italic", token[1:-1]))
                cursor = match.end()
            if cursor < len(line) or not fragments:
                fragments.append(("class:transcript", line[cursor:]))
            return fragments

        return get_line

class Transcript:
    def __init__(self) -> None:
        self.following_output = True
        self.area = TextArea(
            text="",
            read_only=False,
            scrollbar=True,
            focusable=False,
            wrap_lines=True,
            height=Dimension(weight=1),
            lexer=MarkdownTranscriptLexer(),
            get_line_prefix=self._continuation_prefix,
            style="class:transcript",
        )
        self.area.window._scroll_up = lambda: self.scroll(-3)
        self.area.window._scroll_down = lambda: self.scroll(3)

    def _continuation_prefix(self, line_number: int, wrap_count: int) -> StyleAndTextTuples:
        if wrap_count <= 0:
            return []
        line = self.area.buffer.document.lines[line_number]
        return [("class:transcript", "   ")] if line.startswith("   ") else []

    def append(self, text: str) -> None:
        if not text:
            return
        self.area.buffer.text += text
        if self.following_output:
            self.area.buffer.cursor_position = len(self.area.buffer.text)

    def scroll(self, amount: int) -> None:
        document = self.area.buffer.document
        target_row = max(0, min(len(document.lines) - 1, document.cursor_position_row + amount))
        self.following_output = False
        self.area.buffer.cursor_position = document.translate_row_col_to_index(target_row, 0)

    def page_up(self) -> None:
        document = self.area.buffer.document
        render_info = self.area.window.render_info
        first_visible = render_info.first_visible_line() if render_info else document.cursor_position_row
        target_row = max(0, min(first_visible, document.cursor_position_row - 1))
        self.following_output = False
        self.area.window.vertical_scroll = 0
        self.area.buffer.cursor_position = document.translate_row_col_to_index(target_row, 0)

    def page_down(self) -> None:
        document = self.area.buffer.document
        render_info = self.area.window.render_info
        last_visible = render_info.last_visible_line() if render_info else document.cursor_position_row
        target_row = min(len(document.lines) - 1, max(last_visible, document.cursor_position_row + 1))
        self.following_output = False
        self.area.window.vertical_scroll = target_row
        self.area.buffer.cursor_position = document.translate_row_col_to_index(target_row, 0)

    def follow_latest(self) -> None:
        self.following_output = True
        self.area.buffer.cursor_position = len(self.area.buffer.text)

    def start_turn(self) -> None:
        self.follow_latest()

    def scroll_to_top(self) -> None:
        self.following_output = False
        self.area.window.vertical_scroll = 0
        self.area.buffer.cursor_position = 0

    def line(self, text: str = "") -> None:
        self.append(f"{text}\n")

    def welcome(self, *, resume: bool) -> None:
        self.line("⚕ Omega")
        self.line("   Ready. Personal memory and saved knowledge stay separate.")
        if resume:
            self.line("   The next message will resume your latest available session.")
        self.line()

    def user_message(self, text: str) -> None:
        self.line("● ")
        self.line(f"   {text.replace(chr(10), chr(10) + '   ')}")
        self.line()

    def assistant_start(self) -> None:
        self.line("⚕ Omega")
        self.append("   ")

    def assistant_append(self, text: str) -> None:
        self.append(text.replace("\n", "\n   "))

    def tool_started(self, name: str) -> None:
        self.line()
        self.line(f"   ◌  {name}  ·  working")

    def tool_completed(self, name: str, summary: str | None) -> None:
        self.line(f"   ●  {name}  ·  {summary or 'complete'}")

    def notice(self, text: str, *, marker: str = "·") -> None:
        self.line(f"{marker}  {text}")

class DynamicLine(Window):
    def __init__(self, get_text: Callable[[], str | StyleAndTextTuples], *, style: str) -> None:
        super().__init__(
            content=FormattedTextControl(get_text),
            height=1,
            style=style,
            dont_extend_height=True,
        )