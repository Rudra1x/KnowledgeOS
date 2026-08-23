# rerankers/metadata_reranker.py

import time
from datetime import datetime
from core import Chunk, Reranker


class MetadataReranker(Reranker):
    """
    Reranker that adjusts scores based on chunk metadata.

    Applies multiplicative boosts or penalties to the base retrieval score.
    No model inference — pure score manipulation.

    Boost rules:
    - recency_boost:   recent documents rank higher (exponential decay by age)
    - source_boost:    dict mapping source names to multipliers
    - content_type_boost: dict mapping content_type values to multipliers
    - keyword_boost:   boost chunks containing specific terms

    Parameters
    ----------
    recency_boost      : float  max recency multiplier (1.0 = no boost)
    recency_field      : str    metadata field containing date string
    recency_half_life  : int    days after which recency boost is halved
    source_boost       : dict   {source_name: multiplier}
    content_type_boost : dict   {content_type: multiplier}
    keyword_boost      : dict   {keyword: multiplier}
    """

    NAME = "metadata_reranker"

    def __init__(
        self,
        recency_boost:       float = 1.0,
        recency_field:       str   = "date",
        recency_half_life:   int   = 30,
        source_boost:        dict  = None,
        content_type_boost:  dict  = None,
        keyword_boost:       dict  = None,
    ):
        self.recency_boost      = recency_boost
        self.recency_field      = recency_field
        self.recency_half_life  = recency_half_life
        self.source_boost       = source_boost       or {}
        self.content_type_boost = content_type_boost or {}
        self.keyword_boost      = keyword_boost      or {}
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

        for i, chunk in enumerate(chunks):
            base   = chunk.metadata.get("score", 1.0)
            boost  = 1.0

            # Recency boost
            if self.recency_boost > 1.0:
                boost *= self._recency_multiplier(chunk)

            # Source boost
            source = chunk.metadata.get("source", chunk.doc_id)
            for src_pattern, multiplier in self.source_boost.items():
                if src_pattern.lower() in str(source).lower():
                    boost *= multiplier
                    break

            # Content type boost
            ctype = chunk.metadata.get("content_type", "text")
            if ctype in self.content_type_boost:
                boost *= self.content_type_boost[ctype]

            # Keyword boost
            content_lower = chunk.content.lower()
            for kw, multiplier in self.keyword_boost.items():
                if kw.lower() in content_lower:
                    boost *= multiplier

            final = base * boost
            chunk.metadata["rerank_score"]   = final
            chunk.metadata["boost_applied"]  = round(boost, 4)
            chunk.metadata["original_score"] = base
            chunk.metadata["rerank_model"]   = "metadata"
            chunk.metadata["original_rank"]  = i + 1

        self.last_rerank_ms = (time.perf_counter() - t0) * 1000

        ranked = sorted(chunks, key=lambda c: c.metadata["rerank_score"],
                        reverse=True)
        return ranked[:top_k] if top_k else ranked

    def _recency_multiplier(self, chunk: Chunk) -> float:
        """Exponential decay: newer = higher multiplier."""
        date_str = chunk.metadata.get(self.recency_field)
        if not date_str:
            return 1.0
        try:
            date    = datetime.fromisoformat(str(date_str))
            age_days = max(0, (datetime.now() - date).days)
            import math
            decay   = math.exp(-0.693 * age_days / self.recency_half_life)
            return 1.0 + (self.recency_boost - 1.0) * decay
        except (ValueError, TypeError):
            return 1.0