from sentence_transformers import SentenceTransformer
import logging
from omega.environment.conf_loader import omega_settings

logger = logging.getLogger("EmbeddingService")

class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        model_name = model_name or omega_settings.embedding_model
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

_embedding_service: EmbeddingService | None = None

def get_embedding_service(model_name: str | None = None) -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name=model_name)
    return _embedding_service