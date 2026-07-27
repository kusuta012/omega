import logging
logger = logging.getLogger("LoopDetector")

class LoopDetector:
    def __init__(self, max_calls_per_turn: int = 8, repeat_threshold: int = 2):
        self.max_calls_per_turn = max_calls_per_turn
        self.repeat_threshold = repeat_threshold
        self.call_history = []

    def reset(self):
        self.call_history = []

    def record_call(self, tool_name: str, arguments: dict) -> bool:
        call_sig = f"{tool_name}:{sorted(arguments.items())}"
        self.call_history.append(call_sig)

        if len(self.call_history) >= self.max_calls_per_turn:
            logger.warning(
                f"Loop detected: {len(self.call_history)} tool calls in a single turn"
                f"(cap={self.max_calls_per_turn})"
            )
            return True

        sig_count = self.call_history.count(call_sig)
        if sig_count >= self.repeat_threshold:
            logger.warning(
                f"Loop detected: tool '{tool_name}' called {sig_count} times "
                f"with same arguments (threshold={self.repeat_threshold})"
            )
            return True

        tool_count = sum(1 for c in self.call_history if c.startswith(f"{tool_name}"))
        if tool_count >= self.repeat_threshold + 1:
            logger.warning(
                f"loop detected: tool '{tool_name}' called {tool_count} times in this turn"
            )
            return True

        return False