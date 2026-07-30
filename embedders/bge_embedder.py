# embedders/bge_embedder.py

from sentence_transformers import SentenceTransformer
from core import Embedder


class BGEEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        self.model      = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dimension  = self.model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        # BGE models perform better with this prefix on queries
        vectors = self.model.encode(
            texts,
            batch_size        = self.batch_size,
            normalize_embeddings = True,
            show_progress_bar = False,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        # BGE uses a query prefix for asymmetric retrieval
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed([prefixed])[0]