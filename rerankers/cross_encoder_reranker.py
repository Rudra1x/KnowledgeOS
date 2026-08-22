# rerankers/cross_encoder_reranker.py

import time
import numpy as np
from core import Chunk, Reranker


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using sentence-transformers.

    Cross-encoders see query + chunk simultaneously — unlike bi-encoders
    which encode each independently. This joint encoding allows the model
    to reason about specific query-chunk interactions.

    Trade-off:
    - Better ranking quality than bi-encoder cosine similarity
    - No pre-indexing possible — must run inference on each candidate
    - Only feasible on small candidate sets (top 10-50 from retrieval)

    Recommended models (small → large):
    - cross-encoder/ms-marco-MiniLM-L-6-v2  (22MB, fast, good quality)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (33MB, balanced)
    - cross-encoder/ms-marco-electra-base   (110MB, best quality)

    Parameters
    ----------
    model_name : str   HuggingFace model identifier
    batch_size : int   pairs per inference batch
    max_length : int   max tokens per query+chunk pair
    """

    NAME = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder
        self.model_name      = model_name
        self.batch_size      = batch_size
        self.max_length      = max_length
        self.last_rerank_ms: float = 0.0

        print(f"  [CrossEncoder] Loading {model_name}...")
        self.model = CrossEncoder(
            model_name,
            max_length = max_length,
        )
        print(f"  [CrossEncoder] Ready.")

    def rerank(
        self,
        query:      str,
        chunks:     list[Chunk],
        top_k:      int | None = None,
    ) -> list[Chunk]:
        """
        Rerank chunks by cross-encoder relevance score.

        Returns chunks sorted by relevance (highest first).
        If top_k is set, returns only the top_k chunks.
        """
        if not chunks:
            return []

        t0 = time.perf_counter()

        # Build (query, chunk_content) pairs
        pairs  = [(query, chunk.content[:self.max_length]) for chunk in chunks]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        self.last_rerank_ms = (time.perf_counter() - t0) * 1000

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.metadata["rerank_score"]  = float(score)
            chunk.metadata["rerank_model"]  = self.model_name
            chunk.metadata["original_rank"] = chunks.index(chunk) + 1

        ranked = sorted(chunks, key=lambda c: c.metadata["rerank_score"],
                        reverse=True)

        return ranked[:top_k] if top_k else ranked