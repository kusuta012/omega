from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TurnExecutionState:
    user_goal: str
    max_items: int = 6
    max_text: int = 320
    completed_actions: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    open_needs: list[str] = field(default_factory=list)
    working_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_items < 1 or self.max_text < 1:
            raise ValueError("turn-execution limits must be positive")

    def _compact(self, text: object) -> str:
        value = " ".join(str(text).split())
        if len(value) <= self.max_text:
            return value
        return value[: self.max_text - 1] + "..."

    def record_tool_result(self, tool_name: str, result: dict) -> None:
        summary = self._compact(result.get("result_summary") or result.get("answer") or result.get("error") or "No result")
        action = f"{tool_name}: {summary}"
        self.completed_actions.append(action)

        if result.get("success", False):
            self.findings.append(action)
            return

        self.failures.append(action)
        self.open_needs.append(f"Resolve the missing information or limitation from {tool_name} before answering")

    def record_working_note(self, note: object) -> bool:
        compact_note = self._compact(note)
        if not compact_note or compact_note in self.working_notes:
            return False
        self.working_notes.append(compact_note)
        del self.working_notes[:-self.max_items]
        return True

    def decision_context(self) -> str:
        sections: list[str] = [f"User goal: {self._compact(self.user_goal)}"]
        if self.findings:
            sections.append("Complete findings:\n" + "\n".join(f"- {item}" for item in self.findings[-self.max_items:]))
        if self.failures:
            sections.append("Failures or limitations:\n" + "\n".join(f"- {item}" for item in self.failures[-self.max_items:]))
        if self.open_needs:
            sections.append("Outstanding needs:\n" + "\n".join(f"- {item}" for item in self.open_needs[-self.max_items:]))
        if self.working_notes:
            sections.append("Working notes:\n" + "\n".join(f"- {item}" for item in self.working_notes[-self.max_items:]))
        return "\n".join(sections)