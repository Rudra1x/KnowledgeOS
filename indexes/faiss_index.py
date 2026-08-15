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

# Add to indexes/faiss_index.py (append to the existing file)

class FaissIVFIndex(Index):
    """
    FAISS IVF (Inverted File) approximate nearest neighbor index.

    Clusters vectors into nlist Voronoi cells at build time.
    At query time, searches only the nprobe nearest cells.

    Parameters
    ----------
    dimension      : int   embedding dimension
    nlist          : int   number of clusters (cells). Rule of thumb: sqrt(N) to 4*sqrt(N)
    nprobe         : int   cells to search at query time. Higher = better recall, slower.
    quantizer_type : str   'flat' (exact cell search) or 'hnsw' (approximate cell search)

    Build requirement: FAISS IVF needs to be trained on a sample of vectors
    before adding. Call train(vectors) before add(chunks).
    """

    NAME = "faiss_ivf"

    def __init__(self, dimension: int = 384, nlist: int = 10, nprobe: int = 3):
        import faiss
        self.dimension = dimension
        self.nlist     = nlist
        self.nprobe    = nprobe
        self._chunks: list[Chunk] = []
        self._trained = False

        quantizer  = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist,
                                        faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = nprobe

    def train(self, vectors: list[list[float]]) -> None:
        """Must be called before add(). Trains the k-means clustering."""
        import faiss
        import numpy as np
        vecs = np.array(vectors, dtype="float32")
        if len(vecs) < self.nlist:
            # Reduce nlist if we have fewer vectors than clusters
            self.nlist = max(1, len(vecs) // 2)
            quantizer  = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = self.nprobe
        self.index.train(vecs)
        self._trained = True

    def add(self, chunks: list[Chunk]) -> None:
        import numpy as np
        if not self._trained:
            # Auto-train on the first batch
            vectors = [c.embedding for c in chunks if c.embedding]
            self.train(vectors)

        vectors = np.array([c.embedding for c in chunks if c.embedding],
                           dtype="float32")
        self.index.add(vectors)
        self._chunks.extend(c for c in chunks if c.embedding)

    def search(self, query_vector: list[float], top_k: int,
               tenant_id: str = "default") -> list[Chunk]:
        import numpy as np
        query = np.array([query_vector], dtype="float32")
        distances, indices = self.index.search(query, top_k * 2)  # over-fetch for tenant filter

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            if chunk.tenant_id != tenant_id:
                continue
            chunk.metadata["score"]      = float(dist)
            chunk.metadata["score_type"] = "faiss_ivf"
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results


class FaissHNSWIndex(Index):
    """
    FAISS HNSW (Hierarchical Navigable Small World) index.

    Builds a layered graph for lightning-fast approximate search.
    State-of-the-art recall/speed trade-off for most production workloads.

    Parameters
    ----------
    dimension      : int   embedding dimension
    M              : int   connections per node (16-64). Higher = better recall, more memory.
    ef_construction: int   search depth at build time (100-500). Higher = better quality.
    ef_search      : int   search depth at query time. Higher = better recall, slower.

    Note: HNSW does not require training. Add vectors directly.
    Note: HNSW does not support direct vector removal — rebuild to delete.
    """

    NAME = "faiss_hnsw"

    def __init__(self, dimension: int = 384, M: int = 16,
                 ef_construction: int = 200, ef_search: int = 50):
        import faiss
        self.dimension = dimension
        self._chunks: list[Chunk] = []

        self.index = faiss.IndexHNSWFlat(dimension, M,
                                          faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch       = ef_search

    def add(self, chunks: list[Chunk]) -> None:
        import numpy as np
        vectors = np.array([c.embedding for c in chunks if c.embedding],
                           dtype="float32")
        if len(vectors) == 0:
            return
        self.index.add(vectors)
        self._chunks.extend(c for c in chunks if c.embedding)

    def search(self, query_vector: list[float], top_k: int,
               tenant_id: str = "default") -> list[Chunk]:
        import numpy as np
        query = np.array([query_vector], dtype="float32")
        distances, indices = self.index.search(query, top_k * 2)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            if chunk.tenant_id != tenant_id:
                continue
            chunk.metadata["score"]      = float(dist)
            chunk.metadata["score_type"] = "faiss_hnsw"
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results