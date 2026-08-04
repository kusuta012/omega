import logging
import re

logger = logging.getLogger("ChunkSplitter")


class ChunkSplitter:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_chunks(self, text: str) -> list[str]:
        return [chunk["content"] for chunk in self.split_document(text)]

    def split_document(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []

        page_markers = [
            (match.start(), int(match.group(1)))
            for match in re.finditer(r"^--- Page (\d+) ---\s*$", text, re.MULTILINE)
        ]
        chunks = []
        start = self._skip_whitespace(text, 0)
        text_length = len(text)

        while start < text_length:
            end = self._find_boundary(text, start)
            content_start, content_end = self._trim_bounds(text, start, end)
            if content_start == content_end:
                start = self._skip_whitespace(text, end)
                continue
            chunks.append({
                "content": text[content_start:content_end],
                "start_offset": content_start,
                "end_offset": content_end,
                "page_start": self._page_at(page_markers, content_start),
                "page_end": self._page_at(page_markers, max(content_start, content_end -1)),
            })
            if end >= text_length:
                break
            start = self._next_start(text, start, end)

        logger.info(f"Split {len(text)} characters into {len(chunks)} chunks")
        return chunks

    def _find_boundary(self, text: str, start: int) -> int:
        limit = min(len(text), start + self.chunk_size)
        if limit == len(text):
            return limit
        window = text[start:limit]
        paragraph_breaks = list(re.finditer(r"\n\s*\n", window))
        if paragraph_breaks:
            return start + paragraph_breaks[-1].end()
        sentence_breaks = list(re.finditer(r"[.!?](?=\s|$)", window))
        if sentence_breaks:
            return start + sentence_breaks[-1].end()
        whitespace = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
        if whitespace > 0:
            return start + whitespace + 1
        return limit

    def _next_start(self, text: str, current_start: int, end: int) -> int:
        if end - current_start <= self.chunk_overlap:
            return self._skip_whitespace(text, end)
        overlap_start = end - self.chunk_overlap
        while overlap_start < end and text[overlap_start].isspace():
            overlap_start += 1
        if overlap_start >= end:
            return self._skip_whitespace(text, end)
        return overlap_start

    @staticmethod
    def _skip_whitespace(text: str, start: int) -> int:
        while start < len(text) and text[start].isspace():
            start += 1
        return start

    @staticmethod
    def _trim_bounds(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end -1].isspace():
            end -= 1
        return start, end

    @staticmethod
    def _page_at(page_markers: list[tuple[int, int]], offset: int) -> int | None:
        page = None
        for marker_offset, marker_page in page_markers:
            if marker_offset > offset:
                break
            page = marker_page
        return page