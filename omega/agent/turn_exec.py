from __future__ import annotations
from dataclasses import dataclass, field

_MAX_STATE_ITEMS = 6
_MAX_STATE_TEXT = 320

def _compact(text: object) -> str:
    value = " ".join(str(text).split())
    if len(value) <= _MAX_STATE_TEXT:
        return value
    return value[: _MAX_STATE_TEXT - 1] + "..."

@dataclass
class TurnExecutionState:
    user_goal: str
    completed_actions: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    open_needs: list[str] = field(default_factory=list)

    def record_tool_result(self, tool_name: str, result: dict) -> None:
        summary = _compact(result.get("result_summary") or result.get("answer") or result.get("error") or "No result")
        action = f"{tool_name}: {summary}"
        self.completed_actions.append(action)

        if result.get("success", False):
            self.findings.append(action)
            return

        self.failures.append(action)
        self.open_needs.append(f"Resolve the missing information or limitation from {tool_name} before answering")

    def decision_context(self) -> str:
        sections: list[str] = [f"User goal: {_compact(self.user_goal)}"]
        if self.findings:
            sections.append("Complete findings:\n" + "\n".join(f"- {item}" for item in self.findings[-_MAX_STATE_ITEMS:]))
        if self.failures:
            sections.append("Failures or limitations:\n" + "\n".join(f"- {item}" for item in self.failures[-_MAX_STATE_ITEMS:]))
        if self.open_needs:
            sections.append("Outstanding needs:\n" + "\n".join(f"- {item}" for item in self.open_needs[-_MAX_STATE_ITEMS:]))
        return "\n".join(sections)