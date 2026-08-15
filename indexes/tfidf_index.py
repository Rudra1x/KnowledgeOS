# indexes/tfidf_index.py

import re
import math
import uuid
from collections import defaultdict, Counter
from core import Chunk, Index


class TFIDFIndex(Index):
    """
    TF-IDF index implemented from scratch.

    TF(t, d)  = count(t in d) / total_terms(d)
    IDF(t)    = log((1 + N) / (1 + df(t))) + 1   [smoothed]
    Score(q, d) = sum of TF-IDF(t, d) for t in query

    Retrieval: for each query term, look up which docs contain it,
    accumulate TF-IDF scores, return top-k.

    Data structures:
    - inverted_index: {term → {chunk_id: tf_idf_score}}
    - _chunks:        {chunk_id → Chunk}
    - doc_freq:       {term → count of chunks containing it}

    This is a learning implementation — not optimized for scale.
    At scale you'd use Elasticsearch, OpenSearch, or BM25s.
    """

    NAME = "tfidf"

    def __init__(self):
        self.inverted_index: dict[str, dict[str, float]] = defaultdict(dict)
        self._chunks:        dict[str, Chunk]            = {}
        self.doc_freq:       dict[str, int]              = defaultdict(int)
        self.n_docs:         int                         = 0

    def add(self, chunks: list[Chunk]) -> None:
        # First pass: build raw term frequencies and doc frequencies
        chunk_terms = {}
        for chunk in chunks:
            terms = self._tokenize(chunk.content)
            if not terms:
                continue
            tf_raw      = Counter(terms)
            chunk_terms[chunk.chunk_id] = (chunk, tf_raw, len(terms))
            self._chunks[chunk.chunk_id] = chunk
            for term in tf_raw:
                self.doc_freq[term] += 1

        self.n_docs += len(chunk_terms)

        # Second pass: compute TF-IDF scores now that doc_freq is complete
        for chunk_id, (chunk, tf_raw, total_terms) in chunk_terms.items():
            for term, count in tf_raw.items():
                tf  = count / total_terms
                idf = math.log((1 + self.n_docs) / (1 + self.doc_freq[term])) + 1
                self.inverted_index[term][chunk_id] = tf * idf

    def search(
        self,
        query_vector: list[float],    # ignored — TF-IDF is query-string based
        top_k: int,
        tenant_id: str = "default",
        query_text: str = "",         # TF-IDF uses the raw query string
    ) -> list[Chunk]:
        if not query_text:
            return []

        query_terms = self._tokenize(query_text)
        scores: dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            for chunk_id, tfidf_score in self.inverted_index[term].items():
                chunk = self._chunks.get(chunk_id)
                if chunk and chunk.tenant_id == tenant_id:
                    scores[chunk_id] += tfidf_score

        # Sort by score descending, return top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, score in ranked[:top_k]:
            chunk = self._chunks[chunk_id]
            chunk.metadata["score"]        = score
            chunk.metadata["score_type"]   = "tfidf"
            results.append(chunk)

        return results

    def search_text(self, query: str, top_k: int, tenant_id: str = "default") -> list[Chunk]:
        """Convenience method — search by raw query string."""
        return self.search([], top_k, tenant_id, query_text=query)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, remove punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def vocab_size(self) -> int:
        return len(self.inverted_index)

    def stats(self) -> dict:
        return {
            "n_docs":     self.n_docs,
            "vocab_size": self.vocab_size(),
            "n_postings": sum(len(v) for v in self.inverted_index.values()),
        }