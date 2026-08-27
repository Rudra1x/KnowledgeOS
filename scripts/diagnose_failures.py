# scripts/diagnose_failures.py

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

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = VectorRetriever(embedder=embedder, index=index)
reranker  = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
generator = LocalLLMGenerator(max_tokens=200, temperature=0.0)
checker   = FaithfulnessChecker(
    strategy="nli",
    model_name="cross-encoder/nli-MiniLM2-L6-H768",
    threshold=0.25,
)

FAILING_QUERIES = [
    # Retrieval failures (r@1=0)
    ("RETRIEVAL", "What is faithfulness in RAG evaluation?",
     "Faithfulness measures whether"),
    ("RETRIEVAL", "How do cross-encoder rerankers work?",
     "cross-encoder rerankers"),
    # Faithfulness failures (faith=0.00)
    ("FAITHFULNESS", "How is dense retrieval different from BM25?",
     "Dense retrieval uses neural embeddings"),
    ("FAITHFULNESS", "How does chunking affect retrieval quality?",
     "Chunking strategy has a large impact"),
]

print("=" * 70)
print("RETRIEVAL FAILURES — what ranks above the correct chunk?")
print("=" * 70)
for issue, query, relevant in FAILING_QUERIES:
    if issue != "RETRIEVAL":
        continue
    candidates = retriever.retrieve(query, top_k=5)
    print(f"\nQ: {query}")
    for i, c in enumerate(candidates[:3], 1):
        hit = "✓" if relevant[:20] in c.content else " "
        print(f"  {hit}[{i}] {c.content[:80].strip()}...")

print("\n" + "=" * 70)
print("FAITHFULNESS FAILURES — what claims does NLI flag?")
print("=" * 70)
for issue, query, relevant in FAILING_QUERIES:
    if issue != "FAITHFULNESS":
        continue
    candidates = retriever.retrieve(query, top_k=5)
    reranked   = reranker.rerank(query, copy.deepcopy(candidates), top_k=3)
    answer     = generator.generate(query, reranked)
    faith      = checker.check(answer, reranked)
    print(f"\nQ: {query}")
    print(f"A: {answer[:120]}...")
    for d in faith["claim_details"]:
        mark = "✓" if d["supported"] else "✗"
        print(f"  {mark} [{d['score']:.3f}] {d['claim'][:70]}")