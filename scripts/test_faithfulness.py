# scripts/test_faithfulness.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from retrievers.vector_retriever      import VectorRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from generation.local_generator       import LocalLLMGenerator
from generation.faithfulness_checker  import FaithfulnessChecker
from core                             import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
generator  = LocalLLMGenerator(max_tokens=5, temperature=0.0)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = VectorRetriever(embedder=embedder, index=index)
reranker  = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
checker   = FaithfulnessChecker(strategy="llm", generator=generator)

print("=" * 65)
print("TEST 1 — Faithful answer (fully supported by context)")
print("=" * 65)
query       = "What is BM25?"
candidates  = retriever.retrieve(query, top_k=5)
reranked    = reranker.rerank(query, copy.deepcopy(candidates), top_k=3)

faithful_answer = (
    "BM25 is a sparse retrieval algorithm based on term frequency "
    "and inverse document frequency. It scores documents by how many "
    "query terms appear, weighted by rarity."
)

result = checker.check(faithful_answer, reranked)
print(f"Answer:    {faithful_answer[:80]}...")
print(f"Faithful:  {result['faithful']}")
print(f"Score:     {result['score']:.3f}")
print(f"Supported: {len(result['supported'])}/{len(result['claims'])} claims")
print(f"Latency:   {checker.last_check_ms:.0f}ms\n")

print("=" * 65)
print("TEST 2 — Unfaithful answer (adds hallucinated facts)")
print("=" * 65)
hallucinated_answer = (
    "BM25 is a sparse retrieval algorithm. "
    "It was invented by Stephen Robertson in 1994 at Cambridge University. "
    "BM25 is the default algorithm in Elasticsearch and Apache Solr."
)

result2 = checker.check(hallucinated_answer, reranked)
print(f"Answer:      {hallucinated_answer[:80]}...")
print(f"Faithful:    {result2['faithful']}")
print(f"Score:       {result2['score']:.3f}")
print(f"Unsupported: {result2['unsupported']}")
print(f"Latency:     {checker.last_check_ms:.0f}ms\n")

print("=" * 65)
print("CLAIM DETAILS (Test 2)")
print("=" * 65)
for detail in result2["claim_details"]:
    mark = "✓" if detail["supported"] else "✗"
    print(f"  {mark} [{detail['score']:.2f}] {detail['claim'][:70]}")