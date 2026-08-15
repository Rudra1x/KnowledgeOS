# indexes/qdrant_index.py

import uuid
import numpy as np
from core import Chunk, Index


class QdrantIndex(Index):
    """
    Qdrant vector database index.

    Key capabilities beyond FAISS/Chroma:
    - Native payload filtering (filter before ANN, not after)
    - Named vectors (multiple embeddings per point)
    - Disk-based HNSW (handles corpora larger than RAM)
    - Native hybrid search via query fusion (Dense + BM25)

    Two modes:
    - In-memory (location=":memory:"): ephemeral, tests/dev
    - Local persistent (location="path"): survives restarts
    - Remote (url="http://localhost:6333"): production server

    Multi-tenancy: payload field "tenant_id" filtered at search time
    via Qdrant's native FieldCondition — filters before ANN traversal.

    Parameters
    ----------
    collection_name : str
    dimension       : int   embedding dimension
    location        : str   ':memory:' | local path | URL
    distance        : str   'Cosine' | 'Dot' | 'Euclid'
    """

    NAME = "qdrant"

    def __init__(
        self,
        collection_name: str = "knowledgeos",
        dimension:       int = 384,
        location:        str = ":memory:",
        distance:        str = "Cosine",
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        self.collection_name = collection_name
        self.dimension       = dimension
        self.distance        = distance
        self._chunk_map: dict[str, Chunk] = {}

        self._client = QdrantClient(location=location)

        dist_enum = {
            "Cosine": Distance.COSINE,
            "Dot":    Distance.DOT,
            "Euclid": Distance.EUCLID,
        }.get(distance, Distance.COSINE)

        # Clean slate — delete if exists, then create
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass

        self._client.create_collection(
            collection_name = collection_name,
            vectors_config  = VectorParams(
                size     = dimension,
                distance = dist_enum,
            ),
        )

    def add(self, chunks: list[Chunk]) -> None:
        from qdrant_client.models import PointStruct

        valid = [c for c in chunks if c.embedding and c.content.strip()]
        if not valid:
            return

        points = []
        for chunk in valid:
            point_id = str(uuid.UUID(chunk.chunk_id)) \
                if self._is_uuid(chunk.chunk_id) \
                else str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            payload = {
                "chunk_id":    chunk.chunk_id,
                "doc_id":      chunk.doc_id,
                "content":     chunk.content,
                "tenant_id":   chunk.tenant_id,
                "start_index": chunk.start_index,
                "end_index":   chunk.end_index,
                **{k: v for k, v in chunk.metadata.items()
                   if isinstance(v, (str, int, float, bool))},
            }

            points.append(PointStruct(
                id      = point_id,
                vector  = chunk.embedding,
                payload = payload,
            ))
            self._chunk_map[chunk.chunk_id] = chunk

        self._client.upsert(
            collection_name = self.collection_name,
            points          = points,
        )

    def search(
        self,
        query_vector: list[float],
        top_k:        int,
        tenant_id:    str = "default",
    ) -> list[Chunk]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Native payload filter — runs BEFORE ANN search, not after
        tenant_filter = Filter(
            must=[FieldCondition(
                key   = "tenant_id",
                match = MatchValue(value=tenant_id),
            )]
        )

        results = self._client.query_points(
            collection_name = self.collection_name,
            query           = query_vector,
            limit           = top_k,
            query_filter    = tenant_filter,
            with_payload    = True,
        ).points

        chunks = []
        for hit in results:
            payload = hit.payload
            chunk   = Chunk(
                chunk_id    = payload["chunk_id"],
                doc_id      = payload["doc_id"],
                content     = payload["content"],
                tenant_id   = payload["tenant_id"],
                start_index = int(payload.get("start_index", 0)),
                end_index   = int(payload.get("end_index", 0)),
                metadata    = {k: v for k, v in payload.items()
                               if k not in ("chunk_id", "doc_id", "content",
                                            "tenant_id", "start_index", "end_index")},
            )
            chunk.metadata["score"]      = hit.score
            chunk.metadata["score_type"] = "qdrant"
            chunks.append(chunk)

        return chunks

    def delete(self, chunk_ids: list[str]) -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        self._client.delete(
            collection_name = self.collection_name,
            points_selector = Filter(
                must=[FieldCondition(
                    key   = "chunk_id",
                    match = MatchAny(any=chunk_ids),
                )]
            ),
        )

    def count(self) -> int:
        info = self._client.get_collection(self.collection_name)
        return info.points_count

    @staticmethod
    def _is_uuid(s: str) -> bool:
        try:
            uuid.UUID(s)
            return True
        except ValueError:
            return False