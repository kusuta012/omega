from __future__ import annotations
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger("LLMStreaming")

class StreamProtocolError(ValueError):
    pass

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class StreamEvent:
    type: Literal["text_delta", "tool_call", "message_end"]
    text: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def text_delta(cls, text: str) -> "StreamEvent":
        return cls(type="text_delta", text=text)

    @classmethod
    def completed_tool_call(cls, tool_call: ToolCall) -> "StreamEvent":
        return cls(type="tool_call", tool_call=tool_call)

    @classmethod
    def message_end(cls, finish_reason: str | None, usage: dict[str, int] | None = None) -> "StreamEvent":
        return cls(type="message_end", finish_reason=finish_reason, usage=usage or {})


@dataclass
class _ToolCallFragments:
    index: int
    id: str = ""
    name: str = ""
    argument_fragments: list[str] = field(default_factory=list)
    input_object: dict[str, Any] | None = None

class ToolCallAccumulator:
    def __init__(self) -> None:
        self._calls: dict[int, _ToolCallFragments] = {}

    def add_openai_delta(self, payload: dict[str, Any]) -> None:
        index = payload.get("index")
        if not isinstance(index, int):
            raise StreamProtocolError("Openai stream tool call didn't include an integer index")

        fragments = self._calls.setdefault(index, _ToolCallFragments(index=index))
        if  payload.get("id"):
            fragments.id = str(payload["id"])

        function = payload.get("function") or {}
        if function.get("name"):
            fragments.name += str(function["name"])
        if function.get("arguments"):
            fragments.argument_fragments.append(str(function["arguments"]))

    def start_anthropic_tool(self, index: int, block: dict[str, Any]) -> None:
        fragments = self._calls.setdefault(index, _ToolCallFragments(index))
        if block.get("id"):
            fragments.id = str(block["id"])
        if block.get("name"):
            fragments.name = str(block["name"])
        initial_input = block.get("input")
        if initial_input is not None:
            if not isinstance(initial_input, dict):
                raise StreamProtocolError("Anthropic tool input must be a JSON object")
            fragments.input_object = initial_input

    def add_anthropic_json_delta(self, index: int, partial_json: str) -> None:
        fragments = self._calls.get(index)
        if fragments is None:
            raise StreamProtocolError("Anthropic sent tool JSON before starting a tool block")
        fragments.argument_fragments.append(partial_json)

    def finalize(self) -> list[ToolCall]:
        completed: list[ToolCall] = []
        for _, fragments in sorted(self._calls.items()):
            if not fragments.id or not fragments.name:
                raise StreamProtocolError("Provider stream ended with an incomplete tool identifier or name")

            if fragments.argument_fragments:
                raw_arguments = "".join(fragments.argument_fragments)
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as e:
                    raise StreamProtocolError(
                        f"Provided stream ended with invalid JSON for tool {fragments.name}: {e.msg}"
                    ) from e
            elif fragments.input_object is not None:
                arguments = fragments.input_object
            else:
                arguments = {}

            if not isinstance(arguments, dict):
                raise StreamProtocolError(
                    f"Provider stream supplied non-object arguments for tool {fragments.name}"
                )
            completed.append(ToolCall(fragments.id, fragments.name, arguments))
        return completed

async def iter_sse_data(response: Any) -> AsyncIterator[str]:
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield "\n".join(data_lines)


def usage_from_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(item, int)}