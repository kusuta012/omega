from dataclasses import dataclass

DURABLE_MEMORY_TOOLS = frozenset({"remember", "update_profile"})

@dataclass(frozen=True)
class DirectUserProvenance:
    session_id: str
    message_id: str
    user_message: str

    def authorizes(self, source_text: str) -> bool:
        source_text = source_text.strip()
        return bool(source_text) and source_text in self.user_message

    def metadata(self) -> dict:
        return {
            "origin_type": "direct_user_message",
            "source_session_id": self.session_id,
            "source_message_id": self.message_id,
            "confirmation_state": "user_statement",
        }