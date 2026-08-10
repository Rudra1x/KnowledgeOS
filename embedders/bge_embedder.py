# embedders/bge_embedder.py

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from core import Embedder


class BGEEmbedder(Embedder):
    """
    BGE (BAAI General Embedding) family embedder.

    Supported models:
    - BAAI/bge-small-en-v1.5  (384-dim, fastest)
    - BAAI/bge-base-en-v1.5   (768-dim, balanced)
    - BAAI/bge-large-en-v1.5  (1024-dim, best quality)

    Asymmetric retrieval: queries use a prefix; passages do not.
    All vectors are L2-normalized at embed time (cosine = dot product).

    Metadata contract:
    - NAME: str        class-level identifier used in benchmarks
    - dimension: int   embedding size (set at load time)
    - model_name: str  HuggingFace model path

    BENCHMARK FIELDS (set after each call to embed/embed_query):
    - last_embed_ms: float   time for the last embed() call in milliseconds
    """

    NAME          = "bge"
    QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model      = SentenceTransformer(model_name)
        self.dimension  = self.model.get_embedding_dimension()
        self.last_embed_ms: float = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0      = time.perf_counter()
        vectors = self.model.encode(
            texts,
            batch_size           = self.batch_size,
            normalize_embeddings = True,
            show_progress_bar    = False,
        )
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        prefixed = f"{self.QUERY_PREFIX}{query}"
        t0       = time.perf_counter()
        vector   = self.model.encode(
            [prefixed],
            normalize_embeddings = True,
            show_progress_bar    = False,
        )[0]
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vector.tolist()

    def embed_numpy(self, texts: list[str]) -> np.ndarray:
        """Returns a float32 numpy array directly — skips list conversion for FAISS."""
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        t0      = time.perf_counter()
        vectors = self.model.encode(
            texts,
            batch_size           = self.batch_size,
            normalize_embeddings = True,
            show_progress_bar    = False,
        )
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vectors.astype("float32")