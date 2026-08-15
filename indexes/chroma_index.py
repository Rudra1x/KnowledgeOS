# indexes/chroma_index.py

import uuid
import numpy as np
from core import Chunk, Index


class ChromaIndex(Index):
    """
    Chroma vector database index.

    Key differences from FaissFlatIndex:
    - Persistent by default (DuckDB + Parquet on disk)
    - Native metadata filtering (WHERE clause on any metadata field)
    - Built-in collection management (one collection per tenant)
    - Supports deletion by chunk_id

    Two modes:
    - In-memory (persist_directory=None): ephemeral, tests only
    - Persistent (persist_directory="path"): survives restarts

    Multi-tenancy: one Chroma collection per tenant_id. The collection
    name is the tenant_id. This gives native isolation with no
    post-retrieval filtering needed.

    Parameters
    ----------
    collection_name   : str   default collection (overridden per-tenant if multi-tenant=True)
    persist_directory : str | None   path to persist data; None = in-memory
    multi_tenant      : bool  if True, creates one collection per tenant_id
    distance_metric   : str   'cosine' | 'ip' | 'l2'
    """

    NAME = "chroma"

    def __init__(
        self,
        collection_name:   str        = "knowledgeos",
        persist_directory: str | None = None,
        multi_tenant:      bool       = False,
        distance_metric:   str        = "cosine",
    ):
        import chromadb
        from chromadb.config import Settings

        self.collection_name   = collection_name
        self.multi_tenant      = multi_tenant
        self.distance_metric   = distance_metric
        self._dimension: int | None = None

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()
        # Chroma requires collection names: 3-512 chars, [a-zA-Z0-9._-]
        if len(collection_name) < 3:
            raise ValueError(
                f"Chroma collection name must be >= 3 chars, got: '{collection_name}'"
            )
        # Default collection (used in single-tenant mode)
        self._collection = self._get_or_create_collection(collection_name)

    # ------------------------------------------------------------------
    # Index interface
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        if self.multi_tenant:
            # Group by tenant and add to per-tenant collections
            from collections import defaultdict
            by_tenant: dict[str, list[Chunk]] = defaultdict(list)
            for c in chunks:
                by_tenant[c.tenant_id].append(c)
            for tenant_id, tenant_chunks in by_tenant.items():
                coll = self._get_or_create_collection(tenant_id)
                self._add_to_collection(coll, tenant_chunks)
        else:
            self._add_to_collection(self._collection, chunks)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        if self.multi_tenant:
            coll = self._get_or_create_collection(tenant_id)
        else:
            coll = self._collection

        try:
            results = coll.query(
                query_embeddings = [query_vector],
                n_results        = top_k,
                include          = ["documents", "metadatas", "distances"],
            )
        except Exception as e:
            # Collection may be empty
            if "no documents" in str(e).lower() or "empty" in str(e).lower():
                return []
            raise

        chunks = []
        if not results["ids"] or not results["ids"][0]:
            return []

        for i, chunk_id in enumerate(results["ids"][0]):
            meta      = results["metadatas"][0][i]
            content   = results["documents"][0][i]
            distance  = results["distances"][0][i]

            # Chroma cosine returns distance (lower=better); convert to similarity
            score = 1.0 - distance if self.distance_metric == "cosine" else -distance

            chunk = Chunk(
                chunk_id    = chunk_id,
                doc_id      = meta.get("doc_id", ""),
                content     = content,
                tenant_id   = meta.get("tenant_id", tenant_id),
                start_index = int(meta.get("start_index", 0)),
                end_index   = int(meta.get("end_index", 0)),
                metadata    = {k: v for k, v in meta.items()
                               if k not in ("doc_id", "tenant_id",
                                            "start_index", "end_index")},
            )
            chunk.metadata["score"]      = score
            chunk.metadata["score_type"] = "chroma"
            chunks.append(chunk)

        return chunks

    def delete(self, chunk_ids: list[str]) -> None:
        """Delete chunks by ID — supported natively in Chroma."""
        self._collection.delete(ids=chunk_ids)

    def count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(self, name: str):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return self._client.get_or_create_collection(
            name      = name,
            metadata  = {"hnsw:space": self.distance_metric},
        )

    def _add_to_collection(self, collection, chunks: list[Chunk]) -> None:
        valid = [c for c in chunks if c.embedding and c.content.strip()]
        if not valid:
            return

        ids        = [c.chunk_id for c in valid]
        embeddings = [c.embedding for c in valid]
        documents  = [c.content for c in valid]
        metadatas  = [
            {
                "doc_id":      c.doc_id,
                "tenant_id":   c.tenant_id,
                "start_index": c.start_index,
                "end_index":   c.end_index,
                **{k: str(v) for k, v in c.metadata.items()
                   if isinstance(v, (str, int, float, bool))},
            }
            for c in valid
        ]

        collection.add(
            ids        = ids,
            embeddings = embeddings,
            documents  = documents,
            metadatas  = metadatas,
        )