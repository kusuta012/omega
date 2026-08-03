from unittest import result
import uuid
import logging

logger = logging.getLogger("TurnContextManager")

class TurnContextManager:
    def __init__(self, chunk_size: int = 2000, max_cached_chars: int = 16000):
        if chunk_size < 1 or max_cached_chars < 0:
            raise ValueError("turn-context limits must be non-negative")
        self.chunk_size = chunk_size
        self.max_cached_chars = max_cached_chars
        self.cached_chars = 0
        self.overflow_cache: dict[str, list[str]] = {}

    def truncate_and_cache(self, tool_name: str, content: str) -> str:
        if len(content) <= self.chunk_size:
            return content

        first_chunk = content[:self.chunk_size]
        remaining_content = content[self.chunk_size:]
        cache_capacity = max(0, self.max_cached_chars - self.cached_chars)
        cached_content = remaining_content[:cache_capacity]
        omitted_chars = len(remaining_content) - len(cached_content)

        if not cached_content:
            logger.warning(
                f"tool_result truncated without overflow cache: tool={tool_name} remaining_chars={len(remaining_content)}"
            )
            return first_chunk + (
                "\n\n[TOOL RESULT TRUNCATED: additional content was not retained because "
                "this turn's overflow budget is exhausted]"
            ) 

        result_id = f"res_{uuid.uuid4().hex[:8]}"
        chunks = [first_chunk]
        chunks.extend(
            cached_content[index:index + self.chunk_size]
            for index in range(0, len(cached_content), self.chunk_size)
        )
        self.overflow_cache[result_id] = chunks
        self.cached_chars += len(cached_content)
        continuation_count = len(chunks) - 1
        omission_note = ""
        if omitted_chars:
            omission_note = f" {omitted_chars} additional characters were not retained for this turn."

        logger.info(
            f"tool result truncated: tool={tool_name} cached_chars={len(cached_content), self.max_cached_chars - self.cached_chars}"
        )
        return first_chunk + (
            "\n\n[TOOL RESULT TRUNCATED: "
            f"result_id='{result_id}' has {continuation_count} cached continuation chunk(s)"
            f"with chunk_index values 1 through {continuation_count}"
            f"Use read_overflow with result_id='{result_id}' and a chunk_index to read one.]"
            f"{omission_note}"
        )

    def read_chunk(self, result_id: str, chunk_index: int) -> dict:
        chunks = self.overflow_cache[result_id]
        if chunks is None:
            return {
                "success": False,
                "error": f"result_id '{result_id}' was not retained for this turn",
            }
        if chunk_index < 1 or chunk_index >= len(chunks):
            return {"success": False, "error": f"chunk_index {chunk_index} is out of bounds (1 to {len(chunks) - 1})."}

        chunk_data = chunks[chunk_index]
        if chunk_index == len(chunks) - 1:
            footer = "\n\n[End of cached result.]"
        else:
            footer =(f"\n\n[Cached Chunk {chunk_index} of {len(chunks)-1}. Use read_overflow with result_id='{result_id}' and chunk_index={chunk_index + 1} for the next cached chunk.]")
        return {
            "success": True,
            "answer": chunk_data + footer
        }