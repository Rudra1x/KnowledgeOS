# indexes/bm25_index.py

import re
import math
from collections import defaultdict, Counter
from core import Chunk, Index


class BM25Index(Index):
    """
    BM25 (Best Match 25) index implemented from scratch.

    Improvements over TF-IDF:
    1. TF saturation via k1: repeated terms have diminishing returns
    2. Document length normalization via b: longer docs aren't unfairly penalized

    Parameters
    ----------
    k1 : float
        TF saturation. k1=0 → binary (term present/absent).
        k1=1.5 (default) → standard production value.
        k1→∞ → approaches raw TF (no saturation).

    b : float
        Length normalization. b=0 → no normalization. b=1 → full.
        b=0.75 (default) → standard production value.

    Formula
    -------
    BM25(q, d) = Σ IDF(t) × tf(t,d)×(k1+1) / (tf(t,d) + k1×(1 - b + b×|d|/avgdl))

    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
             [Robertson-Sparck Jones IDF — avoids negative IDF for very common terms]
    """

    NAME = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b

        # Inverted index: {term → {chunk_id: bm25_score}}
        # Scores are pre-computed at index time (IDF depends on full corpus)
        self.inverted_index: dict[str, dict[str, float]] = defaultdict(dict)
        self._chunks:        dict[str, Chunk]            = {}
        self.doc_freq:       dict[str, int]              = defaultdict(int)
        self.doc_lengths:    dict[str, int]              = {}
        self.n_docs:         int                         = 0
        self.avgdl:          float                       = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        # Pass 1: tokenize + collect doc lengths + doc frequencies
        chunk_data = {}
        for chunk in chunks:
            terms = self._tokenize(chunk.content)
            if not terms:
                continue
            tf_raw = Counter(terms)
            chunk_data[chunk.chunk_id] = (chunk, tf_raw, len(terms))
            self._chunks[chunk.chunk_id] = chunk
            self.doc_lengths[chunk.chunk_id] = len(terms)
            for term in tf_raw:
                self.doc_freq[term] += 1

        self.n_docs += len(chunk_data)

        # Recompute avgdl over all indexed docs
        if self.doc_lengths:
            self.avgdl = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        # Pass 2: compute BM25 scores
        # Note: we pre-compute scores at index time for O(1) lookup at query time
        # This works because IDF is now stable (full corpus known after add())
        self.inverted_index.clear()
        for chunk_id, (chunk, tf_raw, doc_len) in chunk_data.items():
            for term, tf_count in tf_raw.items():
                idf = self._idf(term)
                tf_norm = self._tf_norm(tf_count, doc_len)
                self.inverted_index[term][chunk_id] = idf * tf_norm

    def search(
        self,
        query_vector: list[float],   # ignored — BM25 is text-based
        top_k: int,
        tenant_id: str = "default",
        query_text: str = "",
    ) -> list[Chunk]:
        if not query_text:
            return []

        query_terms = self._tokenize(query_text)
        scores: dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            for chunk_id, bm25_score in self.inverted_index[term].items():
                chunk = self._chunks.get(chunk_id)
                if chunk and chunk.tenant_id == tenant_id:
                    scores[chunk_id] += bm25_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, score in ranked[:top_k]:
            chunk = self._chunks[chunk_id]
            chunk.metadata["score"]      = score
            chunk.metadata["score_type"] = "bm25"
            results.append(chunk)
        return results

    def search_text(self, query: str, top_k: int, tenant_id: str = "default") -> list[Chunk]:
        return self.search([], top_k, tenant_id, query_text=query)

    # ------------------------------------------------------------------
    # BM25 internals
    # ------------------------------------------------------------------

    def _idf(self, term: str) -> float:
        """Robertson-Sparck Jones IDF — avoids negative values for very common terms."""
        df  = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def _tf_norm(self, tf: int, doc_len: int) -> float:
        """TF saturation + length normalization."""
        return (tf * (self.k1 + 1)) / (
            tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def stats(self) -> dict:
        return {
            "n_docs":     self.n_docs,
            "vocab_size": len(self.inverted_index),
            "avgdl":      round(self.avgdl, 1),
            "k1":         self.k1,
            "b":          self.b,
        }