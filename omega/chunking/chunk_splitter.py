import logging

logger = logging.getLogger("ChunkSplitter")


class ChunkSplitter:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_chunks(self, text: str) -> list[str]:
        if not text or not text.strip():
            return[]

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                lines = paragraph.split("\n")
                for line in lines:
                    if len(current_chunk) + len(line) + 1 <= self.chunk_size:
                        current_chunk += ("\n" if current_chunk else "") + line
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        
                        overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                        current_chunk = current_chunk[overlap_start:] + ("\n" if current_chunk else "") + line
            else:
                if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                    current_chunk += ("\n\n" if current_chunk else "") + paragraph
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + ("\n\n" if current_chunk else "") + paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        logger.info(f"Split {len(text)} characters into {len(chunks)} chunks")
        return chunks

