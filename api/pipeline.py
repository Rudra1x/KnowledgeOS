# api/pipeline.py

import copy
from core import load_config, NormalizationPipeline
from loaders.router import LoaderRouter
from chunkers.recursive_chunker import RecursiveChunker
from embedders.bge_embedder import BGEEmbedder
from embedders.cache import CachedEmbedder
from indexes.faiss_index import FaissFlatIndex
from indexes.bm25_index import BM25Index
from retrievers.hybrid_retriever import HybridRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from generation.local_generator import LocalLLMGenerator
from generation.faithfulness_checker import FaithfulnessChecker
from generation.prompt_builder import extract_citations


class RAGPipeline:
    """
    Production RAG pipeline for one tenant.

    Instantiate once per tenant, reuse across requests.
    All components are stateful (index holds documents).

    Usage:
        pipeline = RAGPipeline(tenant_id="acme_corp")
        pipeline.ingest("path/to/docs/")
        result = pipeline.query("What is our refund policy?")
    """

    def __init__(
        self,
        tenant_id:     str   = "default",
        chunk_size:    int   = 300,
        chunk_overlap: int   = 0,
        top_k_fetch:   int   = 10,
        top_k_rerank:  int   = 3,
    ):
        self.tenant_id    = tenant_id
        self.top_k_fetch  = top_k_fetch
        self.top_k_rerank = top_k_rerank

        cfg        = load_config()
        normalizer = NormalizationPipeline()

        # Components
        self.loader     = LoaderRouter()
        self.chunker    = RecursiveChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        raw_embedder    = BGEEmbedder("BAAI/bge-small-en-v1.5")
        self.embedder   = CachedEmbedder(
            raw_embedder, cache_path=f"cache/{tenant_id}_embeddings.db"
        )
        self.dense_idx  = FaissFlatIndex(dimension=384)
        self.sparse_idx = BM25Index()
        self.reranker   = CrossEncoderReranker(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.generator  = LocalLLMGenerator(
            max_tokens=512, temperature=0.0
        )
        self.faith_checker = FaithfulnessChecker(
            strategy   = "nli",
            model_name = "cross-encoder/nli-MiniLM2-L6-H768",
            threshold  = 0.25,
        )
        self.normalizer = normalizer
        self._n_docs    = 0

    def ingest(self, path: str) -> dict:
        """Load, chunk, embed, and index documents from a path."""
        docs   = self.normalizer.apply_many(self.loader.load(path))
        chunks = []
        for doc in docs:
            chunks.extend(self.chunker.chunk(doc))

        # Set tenant_id on all chunks
        for c in chunks:
            c.tenant_id = self.tenant_id

        # Embed
        vecs = self.embedder.embed([c.content for c in chunks])
        for c, v in zip(chunks, vecs):
            c.embedding = v

        # Index
        self.dense_idx.add(copy.deepcopy(chunks))
        self.sparse_idx.add(copy.deepcopy(chunks))
        self._n_docs += len(chunks)

        return {
            "status":   "ok",
            "chunks":   len(chunks),
            "total":    self._n_docs,
            "tenant":   self.tenant_id,
        }

    def query(self, question: str, check_faithfulness: bool = True) -> dict:
        """Retrieve, rerank, generate, verify."""
        # Retrieve
        retriever  = HybridRetriever(
            bm25_index  = self.sparse_idx,
            dense_index = self.dense_idx,
            embedder    = self.embedder,
            fetch_k     = self.top_k_fetch,
        )
        candidates = retriever.retrieve(
            question, top_k=self.top_k_fetch, tenant_id=self.tenant_id
        )
        reranked   = self.reranker.rerank(
            question, copy.deepcopy(candidates), top_k=self.top_k_rerank
        )

        # Generate
        answer = self.generator.generate(question, reranked)

        # Citations
        cited  = extract_citations(answer, reranked)

        # Faithfulness (optional — skip for latency-sensitive paths)
        faith_result = None
        if check_faithfulness and reranked:
            faith_result = self.faith_checker.check(answer, reranked)

        return {
            "answer":      answer,
            "citations":   {
                str(n): {
                    "chunk_id": c.chunk_id,
                    "source":   c.metadata.get("source", c.doc_id),
                    "snippet":  c.content[:200],
                }
                for n, c in cited.items()
            },
            "faithfulness": faith_result["score"] if faith_result else None,
            "sources_used": len(reranked),
            "tenant_id":   self.tenant_id,
        }

    @property
    def stats(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "n_chunks":  self._n_docs,
            "cache":     self.embedder.stats(),
        }