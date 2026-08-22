# rerankers/bge_reranker.py

import time
from core import Chunk, Reranker


class BGEReranker(Reranker):
    """
    BGE Reranker — cross-encoder from the BGE family.

    Trained on diverse retrieval data including technical and academic
    content. Often outperforms MS-MARCO models on domain-specific corpora.

    Models (small → large):
    - BAAI/bge-reranker-base   (278MB, good balance)
    - BAAI/bge-reranker-large  (560MB, best quality)
    - BAAI/bge-reranker-v2-m3  (568MB, multilingual)

    Same cross-encoder interface as CrossEncoderReranker.
    Interchangeable — swap model_name only.

    Parameters
    ----------
    model_name : str
    batch_size : int
    max_length : int
    """

    NAME = "bge_reranker"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 16,
        max_length: int = 512,
    ):
        from sentence_transformers import CrossEncoder
        self.model_name      = model_name
        self.batch_size      = batch_size
        self.max_length      = max_length
        self.last_rerank_ms: float = 0.0

        print(f"  [BGEReranker] Loading {model_name}...")
        self.model = CrossEncoder(model_name, max_length=max_length)
        print(f"  [BGEReranker] Ready.")

    def rerank(
        self,
        query:  str,
        chunks: list[Chunk],
        top_k:  int | None = None,
    ) -> list[Chunk]:
        if not chunks:
            return []

        t0     = time.perf_counter()
        pairs  = [(query, c.content[:self.max_length]) for c in chunks]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        self.last_rerank_ms = (time.perf_counter() - t0) * 1000

        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            chunk.metadata["rerank_score"]  = float(score)
            chunk.metadata["rerank_model"]  = self.model_name
            chunk.metadata["original_rank"] = i + 1

        ranked = sorted(chunks, key=lambda c: c.metadata["rerank_score"],
                        reverse=True)
        return ranked[:top_k] if top_k else ranked