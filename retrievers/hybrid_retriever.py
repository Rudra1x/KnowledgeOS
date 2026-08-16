# retrievers/hybrid_retriever.py

from collections import defaultdict
from core import Chunk, Retriever, Embedder
from indexes.bm25_index  import BM25Index
from indexes.faiss_index import FaissFlatIndex


class HybridRetriever(Retriever):
    """
    Hybrid retrieval: BM25 (sparse) + Dense (vector) merged via RRF.

    Reciprocal Rank Fusion formula:
        RRF(chunk) = 1/(k + rank_bm25) + 1/(k + rank_dense)

    Why RRF over score normalization:
    - BM25 and cosine scores live in different scales
    - Normalizing is unstable (one outlier shifts everything)
    - RRF only uses ranks — robust, scale-agnostic, no hypertuning

    Parameters
    ----------
    bm25_index    : BM25Index
    dense_index   : FaissFlatIndex (or any FAISS variant)
    embedder      : Embedder   for query embedding
    k             : int        RRF constant (default 60 per original paper)
    bm25_weight   : float      weight for BM25 leg (1.0 = equal weight)
    dense_weight  : float      weight for dense leg (1.0 = equal weight)
    fetch_k       : int        candidates per leg before fusion
    """

    NAME = "hybrid_rrf"

    def __init__(
        self,
        bm25_index:   BM25Index,
        dense_index:  FaissFlatIndex,
        embedder:     Embedder,
        k:            int   = 60,
        bm25_weight:  float = 1.0,
        dense_weight: float = 1.0,
        fetch_k:      int   = 20,
    ):
        self.bm25_index   = bm25_index
        self.dense_index  = dense_index
        self.embedder     = embedder
        self.k            = k
        self.bm25_weight  = bm25_weight
        self.dense_weight = dense_weight
        self.fetch_k      = fetch_k

    def retrieve(
        self,
        query:     str,
        top_k:     int = 5,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        # BM25 leg — returns ranked list by keyword relevance
        bm25_results = self.bm25_index.search_text(
            query, top_k=self.fetch_k, tenant_id=tenant_id
        )

        # Dense leg — returns ranked list by semantic similarity
        qvec = self.embedder.embed_query(query)
        dense_results = self.dense_index.search(
            qvec, top_k=self.fetch_k, tenant_id=tenant_id
        )

        # RRF fusion
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map:  dict[str, Chunk] = {}

        for rank, chunk in enumerate(bm25_results, start=1):
            rrf_scores[chunk.chunk_id] += self.bm25_weight * (1.0 / (self.k + rank))
            chunk_map[chunk.chunk_id]   = chunk

        for rank, chunk in enumerate(dense_results, start=1):
            rrf_scores[chunk.chunk_id] += self.dense_weight * (1.0 / (self.k + rank))
            chunk_map[chunk.chunk_id]   = chunk

        # Sort by RRF score descending
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, rrf_score in ranked[:top_k]:
            chunk = chunk_map[chunk_id]
            chunk.metadata["score"]       = rrf_score
            chunk.metadata["score_type"]  = "rrf"
            results.append(chunk)

        return results