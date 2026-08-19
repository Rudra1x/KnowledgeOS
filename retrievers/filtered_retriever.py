# retrievers/filtered_retriever.py

from core import Chunk, Retriever, Embedder, Index


class FilteredRetriever(Retriever):
    """
    Retriever with metadata filtering support.

    Three filter modes:
    - 'pre':    apply filter before semantic search (restrict candidate set)
    - 'post':   apply filter after semantic search (rank then restrict)
    - 'boost':  apply filter as score multiplier (soft boost for matching chunks)

    Filter is a dict of {metadata_key: expected_value | list[expected_values]}.

    Examples:
        filter={"file_type": "pdf"}
        filter={"content_type": ["text", "table"]}
        filter={"heading_level": 1}

    Also supports keyword search: if keyword is set, re-ranks results
    by keyword overlap with the query (simple term matching boost).

    Parameters
    ----------
    embedder    : Embedder
    index       : Index (must be FAISS or Qdrant for pre-filter)
    mode        : 'pre' | 'post' | 'boost'
    filter      : dict[str, any]  metadata conditions
    boost_factor: float  score multiplier for matching chunks (boost mode)
    keyword_weight: float  weight for keyword overlap bonus
    """

    NAME = "filtered"

    def __init__(
        self,
        embedder:       Embedder,
        index:          Index,
        mode:           str   = "post",
        filter:         dict  = None,
        boost_factor:   float = 2.0,
        keyword_weight: float = 0.3,
    ):
        self.embedder       = embedder
        self.index          = index
        self.mode           = mode
        self.filter         = filter or {}
        self.boost_factor   = boost_factor
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        query:     str,
        top_k:     int = 5,
        tenant_id: str = "default",
        filter:    dict = None,        # per-call override
    ) -> list[Chunk]:
        active_filter = filter or self.filter
        qvec          = self.embedder.embed_query(query)

        if self.mode == "pre":
            return self._pre_filter(qvec, query, top_k, tenant_id, active_filter)
        if self.mode == "boost":
            return self._boost_filter(qvec, query, top_k, tenant_id, active_filter)
        return self._post_filter(qvec, query, top_k, tenant_id, active_filter)

    # ------------------------------------------------------------------

    def _post_filter(self, qvec, query, top_k, tenant_id, active_filter) -> list[Chunk]:
        """Fetch more candidates, filter by metadata, return top_k survivors."""
        fetch_k   = top_k * 5   # over-fetch to account for filtering
        retrieved = self.index.search(qvec, top_k=fetch_k, tenant_id=tenant_id)
        filtered  = [c for c in retrieved if self._matches(c, active_filter)]
        return filtered[:top_k]

    def _pre_filter(self, qvec, query, top_k, tenant_id, active_filter) -> list[Chunk]:
        """
        Pre-filter: restrict the index candidates first.
        For FAISS (no native pre-filter), we simulate by fetching all and
        filtering before returning — same semantic as pre-filter but
        costs full FAISS search. True pre-filter requires Qdrant/Chroma.
        Returns top_k filtered results.
        """
        # Fetch a large candidate set from index
        fetch_k   = min(top_k * 10, 200)
        retrieved = self.index.search(qvec, top_k=fetch_k, tenant_id=tenant_id)
        filtered  = [c for c in retrieved if self._matches(c, active_filter)]
        return filtered[:top_k]

    def _boost_filter(self, qvec, query, top_k, tenant_id, active_filter) -> list[Chunk]:
        """
        Boost: metadata match multiplies the score.
        Non-matching chunks still appear but rank lower.
        """
        fetch_k   = top_k * 3
        retrieved = self.index.search(qvec, top_k=fetch_k, tenant_id=tenant_id)

        query_terms = set(query.lower().split())
        scored      = []
        for chunk in retrieved:
            score = chunk.metadata.get("score", 0.0)

            # Boost matching metadata
            if self._matches(chunk, active_filter):
                score *= self.boost_factor

            # Keyword overlap bonus
            chunk_terms  = set(chunk.content.lower().split())
            overlap      = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            score       += overlap * self.keyword_weight

            chunk.metadata["score_filtered"] = score
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for score, chunk in scored[:top_k]:
            chunk.metadata["score"] = score
            result.append(chunk)
        return result

    def _matches(self, chunk: Chunk, active_filter: dict) -> bool:
        """Check if a chunk matches all filter conditions."""
        for key, expected in active_filter.items():
            val = chunk.metadata.get(key)
            if isinstance(expected, list):
                if val not in expected:
                    return False
            else:
                if val != expected:
                    return False
        return True