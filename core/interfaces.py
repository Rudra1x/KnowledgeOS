# core/interfaces.py

from abc import ABC, abstractmethod
from .models import Document, Chunk


class Loader(ABC):
    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Load from a source path or URL and return Documents."""


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a Document into Chunks."""


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class Index(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Index a list of Chunks."""

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int, tenant_id: str) -> list[Chunk]:
        """Return top_k Chunks nearest to query_vector."""


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int, tenant_id: str) -> list[Chunk]:
        """Return top_k relevant Chunks for a query."""


class Reranker(ABC):
    """Reorders a candidate set by relevance to a query."""

    @abstractmethod
    def rerank(
        self,
        query:  str,
        chunks: list[Chunk],
        top_k:  int | None = None,
    ) -> list[Chunk]:
        """Return chunks sorted by relevance, highest first."""
        ...


class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """Return a grounded answer with citations."""