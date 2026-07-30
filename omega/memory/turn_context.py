import uuid
import math
import logging

logger = logging.getLogger("TurnContextManager")

class TurnContextManager:
    def __init__(self, chunk_size: int = 2000):
        self.chunk_size = chunk_size
        self.overflow_cache = {}

    def truncate_and_cache(self, tool_name: str, content: str) -> str:
        if len(content) <= self.chunk_size:
            return content

        result_id = f"res_{uuid.uuid4().hex[:6]}"

        chunks = []
        for i in range(0, len(content), self.chunk_size):
            chunks.append(content[i:i + self.chunk_size])

        self.overflow_cache[result_id] = chunks
        remaining_chars = len(content) - self.chunk_size
        total_chunks = len(chunks)

        notice = (
            f"\n\n[SYSTEM: result truncated to protect context window"
            f"{remaining_chars} characters remaining across {total_chunks - 1} more chunks. "
            f"Use the `read_overflow` tool with result_id='{result_id} and chunk_index=1 to {total_chunks - 1} to read the rest ]"
        )

        logger.info(f"Context protected: {tool_name} returned {len(content)} chars. Cached as {result_id}")

        return chunks[0] + notice

    def read_chunk(self, result_id: str, chunk_index: int) -> dict:
        if result_id not in self.overflow_cache:
            return {"success": False, "error": f"result_id '{result_id}' not found or expired"}

        chunks = self.overflow_cache[result_id]
        if chunk_index < 1 or chunk_index >= len(chunks):
            return {"success": False, "error": f"chunk_index {chunk_index} is out of bounds (1 to {len(chunks) - 1})."}

        chunk_data = chunks[chunk_index]

        is_last = chunk_index == len(chunks) - 1
        footer = "\n\n[End of result.]" if is_last else f"\n\n[SYSTEM: Chunk {chunk_index} of {len(chunks)-1}. Use read_overflow for next chunk]"

        return {
            "success": True,
            "answer": chunk_data + footer
        }