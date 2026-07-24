from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger("EmbeddingService")

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading local embedding model {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("local embedding model loaded successfully")
    
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [embedding.tolist() for embedding in embeddings]

    def generate_single_embedding(self, text: str) -> list[float]:
        return self.generate_embeddings([text])[0]