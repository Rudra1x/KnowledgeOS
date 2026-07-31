# retrievers/vector_retriever.py

from core import Chunk, Retriever, Index, Embedder


class VectorRetriever(Retriever):
    def __init__(self, embedder: Embedder, index: Index):
        self.embedder = embedder
        self.index    = index

    def retrieve(self, query: str, top_k: int = 5, tenant_id: str = "default") -> list[Chunk]:
        query_vector = self.embedder.embed_query(query)
        return self.index.search(query_vector, top_k=top_k, tenant_id=tenant_id)