from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass(frozen=True)
class AgentEvent:
    type: Literal["text_delta", "tool_started", "tool_completed", "turn_complete"]
    text: str | None = None
    tool_name: str | None = None
    summary: str | None = None
    result: dict[str, Any] | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod(frozen=True)
    class AgentEvent:
        type: Literal["text_delta", "tool_started", "tool_completed", "turn_complete"]
        text: str | None = None
        tool_name: str | None = None
        summary: str | None = None
        result: dict[str, Any] | None = None
        usage: dict[str, int] = field(default_factory=dict)

        @classmethod
        def text_delta(cls, text: str) -> "AgentEvent":
            return cls(type="text_delta", text=text)

        @classmethod
        def tool_started(cls, tool_name: str) -> "AgentEvent":
            return cls(type="tool_started", tool_name=tool_name)

        @classmethod
        def tool_completed(cls, tool_name: str, summary: str) -> "AgentEvent":
            return cls(type="tool_completed", tool_name=tool_name, summary=summary)

        @classmethod
        def turn_complete(cls, result: dict[str, Any], usage: dict[str, int]) -> "AgentEvent":
            return cls(type="turn_complete", result=result, usage=usage)