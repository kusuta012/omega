import logging
from omega.embeddings.embedding_service import EmbeddingService
from omega.storage.retrieval_queries import search_hybrid_chunks
from omega.llm.client import GroqClient

logger = logging.getLogger("Synthesis")

SYSTEM_PROMPT = """
You are Omega, a strictly factual Personal Assistant.
Your task is to answer the user's question using ONLY the provided context chunks.

RULES:
1. If the provided context does not contain the answer, say "I don't have enough information in your saved items to answer this" Do not Guess
2. Always cite your sources explicitly at the end of sentence using [Source X].
3. Do not rely on your pre-training data.
"""

class Synthesis:
    def __init__(self):
        self.embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
        self.llm_client = GroqClient()

    async def search_knowledge(self, query: str, top_k: int = 5) -> list[dict]:
        query_vector = self.embedding_service.generate_single_embedding(query)
        return await search_hybrid_chunks(query, query_vector, top_k)

    async def answer_question(self, query: str, top_k: int = 5) -> dict:
        relevant_chunks = await self.search_knowledge(query, top_k)

        if not relevant_chunks:
            return {
                "question": query,
                "answer": "No relevant items found in your knowledge base",
                "sources": []
            }
        
        context_passages = []
        sources_meta = []

        for idx, chunk in enumerate(relevant_chunks, start=1):
            source_label = f"Source {idx}"
            source_desc = f"{chunk['source_title']} ({chunk['source_ref'] or 'Raw Text'})"
            context_passages.append(f"[{source_label}: {source_desc}]\n{chunk['chunk_text']}")
            sources_meta.append({
                "label": source_label,
                "title": chunk['source_title'],
                "score": round(chunk['rrf_score'], 4)
            })

        compiled_context = "\n\n".join(context_passages)
        user_prompt = f"QUESTION:\n{query}\n\nCONTEXT:\n{compiled_context}\n\nANSWER:"

        logger.info(f"Synthesizing answer for query: '{query}'")
        answer = await self.llm_client.generate_answer(SYSTEM_PROMPT, user_prompt)

        return {
            "question": query,
            "answer": answer,
            "sources": sources_meta
        }