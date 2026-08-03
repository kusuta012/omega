from __future__ import annotations
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("LoopDetector")

@dataclass(frozen=True)
class ToolCallDecision:
    allowed: bool
    code: str | None = None
    message: str | None = None


_ALLOWED = ToolCallDecision(allowed=True)


class LoopDetector:
    def __init__(self, max_calls_per_turn: int = 8, repeat_threshold: int = 2):
        self.max_calls_per_turn = max_calls_per_turn
        self.repeat_threshold = repeat_threshold
        self.call_history: list[tuple[str, str]] = []

    def reset(self):
        self.call_history.clear()

    @staticmethod
    def _signature(tool_name: str, arguments: dict) -> tuple[str, str]:
        return tool_name, json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def record_call(self, tool_name: str, arguments: dict) -> ToolCallDecision:
        call_signature = self._signature(tool_name, arguments)
        self.call_history.append(call_signature)

        if len(self.call_history) > self.max_calls_per_turn:
            message = (
                "Tool call blocked: the per-turn tool-call limit has been reached"
                "Use the information already available or answer with the limitation"
            )
            logger.warning(
                f"tool-call budget reached: calls={len(self.call_history)}, cap={self.max_calls_per_turn}"
            )
            return ToolCallDecision(False, "call_budget", message)

        signature_count = self.call_history.count(call_signature)
        if signature_count >= self.repeat_threshold:
            message = (
                "Tool call blocked: this exact call was already attempted in this turn. "
                "Use its existing result or choose a meaningfully different action"
            )
            logger.warning(
                f"duplicate tool call blocked: tool={tool_name}, count={signature_count}, threshold={self.repeat_threshold}"
            )
            return ToolCallDecision(False, "duplicate_call", message)

        tool_count = sum(1 for recorded_name, _ in self.call_history if recorded_name == tool_name)
        if tool_count > self.repeat_threshold:
            message = (
                f"Tool call blocked: {tool_name} has already been used repeatedly in this turn"
                "Use the information already available or choose a different tool"
            )
            logger.warning(
                f"repeated tool blocked: tool={tool_name}, count={tool_count}, threshold={self.repeat_threshold}"
            )
            return ToolCallDecision(False, "repeated_tool", message)

        return _ALLOWED