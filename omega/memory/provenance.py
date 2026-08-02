from dataclasses import dataclass

DURABLE_MEMORY_TOOLS = frozenset({"remeber", "update_profile"})

@dataclass(frozen=True)
class DirectUserProvenance:
    session_id: str
    message_id: str
    user_messsage: str

    def authorizes(self, source_text: str) -> bool:
        source_text = source_text.strip()
        return bool(source_text) and source_text in self.user_messsage

    def metadata(self) -> dict:
        return {
            "orgin_type": "direct_user_message",
            "source_session_id": self.session_id,
            "source_message_id": self.message_id,
            "confirmation_state": "user_statement",
        }