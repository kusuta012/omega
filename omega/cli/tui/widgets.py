from __future__ import annotations
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.widgets import TextArea

class Transcript:

    def __init__(self) -> None:
        self.area = TextArea(
            text="",
            read_only=False,
            scrollbar=True,
            focusable=False,
            wrap_lines=True,
            style="class:transcript",
        )

    def append(self, text: str) -> None:
        if not text:
            return
        self.area.buffer.text += text
        self.area.buffer.cursor_position = len(self.area.buffer.text)

    def line(self, text: str = "") -> None:
        self.append(f"{text}\n")

class DynamicLine(Window):
    def __init__(self, get_text, *, style: str) -> None:
        super().__init__(
            content=FormattedTextControl(get_text),
            height=1,
            style=style,
            dont_extend_height=True,
        )