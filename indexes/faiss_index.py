# indexes/faiss_index.py

import faiss
import numpy as np
from core import Chunk, Index


class FaissFlatIndex(Index):
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index     = faiss.IndexFlatIP(dimension)  # Inner product = cosine on normalized vecs
        self._chunks: list[Chunk] = []                 # parallel list to FAISS internal ids

    def add(self, chunks: list[Chunk]) -> None:
        vectors = np.array([c.embedding for c in chunks], dtype="float32")
        self.index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_vector: list[float], top_k: int, tenant_id: str = "default") -> list[Chunk]:
        query = np.array([query_vector], dtype="float32")
        distances, indices = self.index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            if chunk.tenant_id != tenant_id:
                continue
            chunk.metadata["score"] = float(dist)
            results.append(chunk)

        return results