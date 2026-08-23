# rerankers/similarity_reranker.py

import time
import numpy as np
from core import Chunk, Reranker, Embedder


class SimilarityReranker(Reranker):
    """
    Reranker that re-scores candidates using a (potentially different)
    embedder and/or query expansion.

    Use cases:
    - Retrieve with a fast small embedder (BGE-small), rerank with a
      better embedder (E5, Instructor) for higher quality final scores
    - Apply query expansion before re-embedding to capture more facets
    - Re-normalize scores from different retrieval backends

    Parameters
    ----------
    embedder       : Embedder   embedder for re-scoring (can differ from retrieval)
    query_expansion: list[str]  additional query variants to average into score
    score_weight   : float      weight for re-score vs original score (0=only new, 1=only old)
    """

    NAME = "similarity_reranker"

    def __init__(
        self,
        embedder:        Embedder,
        query_expansion: list[str] = None,
        score_weight:    float     = 0.0,   # 0 = fully replace, 1 = fully keep original
    ):
        self.embedder        = embedder
        self.query_expansion = query_expansion or []
        self.score_weight    = score_weight
        self.last_rerank_ms: float = 0.0

    def rerank(
        self,
        query:  str,
        chunks: list[Chunk],
        top_k:  int | None = None,
    ) -> list[Chunk]:
        if not chunks:
            return []

        t0 = time.perf_counter()

        # Build query embedding (average over original + expansions)
        all_queries  = [query] + self.query_expansion
        query_vecs   = np.array(
            [self.embedder.embed_query(q) for q in all_queries],
            dtype="float32"
        )
        query_vec    = query_vecs.mean(axis=0)
        norm         = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Re-score each chunk
        for i, chunk in enumerate(chunks):
            if chunk.embedding:
                chunk_vec  = np.array(chunk.embedding, dtype="float32")
                new_score  = float(np.dot(query_vec, chunk_vec))
            else:
                # No embedding — embed on the fly
                chunk_vec  = np.array(
                    self.embedder.embed_query(chunk.content), dtype="float32"
                )
                new_score  = float(np.dot(query_vec, chunk_vec))

            old_score = chunk.metadata.get("score", new_score)
            final     = (self.score_weight * old_score +
                         (1 - self.score_weight) * new_score)

            chunk.metadata["rerank_score"]    = final
            chunk.metadata["original_score"]  = old_score
            chunk.metadata["new_score"]       = new_score
            chunk.metadata["rerank_model"]    = self.embedder.NAME
            chunk.metadata["original_rank"]   = i + 1

        self.last_rerank_ms = (time.perf_counter() - t0) * 1000

        ranked = sorted(chunks, key=lambda c: c.metadata["rerank_score"],
                        reverse=True)
        return ranked[:top_k] if top_k else ranked