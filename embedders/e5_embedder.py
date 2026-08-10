# embedders/e5_embedder.py

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from core import Embedder


class E5Embedder(Embedder):
    """
    Microsoft E5 embedding family.

    Supported models:
    - intfloat/e5-small-v2   (384-dim, fastest)
    - intfloat/e5-base-v2    (768-dim, balanced)
    - intfloat/e5-large-v2   (1024-dim, best quality)

    Key difference from BGE:
    - Both queries AND passages receive prefixes
    - Query prefix:   'query: '
    - Passage prefix: 'passage: '

    This makes the embedding space explicitly asymmetric on both sides,
    vs BGE which only prefixes queries.

    All vectors are L2-normalized. Inner product = cosine similarity.
    """

    NAME           = "e5"
    QUERY_PREFIX   = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(self, model_name: str = "intfloat/e5-small-v2", batch_size: int = 32):
        self.model_name    = model_name
        self.batch_size    = batch_size
        self.model         = SentenceTransformer(model_name)
        self.dimension     = self.model.get_embedding_dimension()
        self.last_embed_ms: float = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (with passage prefix)."""
        if not texts:
            return []
        prefixed = [f"{self.PASSAGE_PREFIX}{t}" for t in texts]
        t0       = time.perf_counter()
        vectors  = self.model.encode(
            prefixed,
            batch_size           = self.batch_size,
            normalize_embeddings = True,
            show_progress_bar    = False,
        )
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed query (with query prefix)."""
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
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        prefixed = [f"{self.PASSAGE_PREFIX}{t}" for t in texts]
        t0       = time.perf_counter()
        vectors  = self.model.encode(
            prefixed,
            batch_size           = self.batch_size,
            normalize_embeddings = True,
            show_progress_bar    = False,
        )
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vectors.astype("float32")