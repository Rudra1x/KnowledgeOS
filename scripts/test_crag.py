# scripts/test_crag.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from retrievers.crag_retriever   import CRAGRetriever
from generation.local_generator  import LocalLLMGenerator
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
generator  = LocalLLMGenerator(max_tokens=30, temperature=0.0)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = CRAGRetriever(
    embedder        = embedder,
    index           = index,
    generator       = generator,
    max_corrections = 2,
)

QUERIES = [
    {
        "query":    "What is BM25 and how does it score documents?",
        "expected": "CORRECT",   # good chunks exist for BM25
    },
    {
        "query":    "What is the weather in Tokyo today?",
        "expected": "INCORRECT", # nothing relevant in corpus
    },
    {
        "query":    "How does retrieval quality get measured?",
        "expected": "CORRECT",   # evaluation metrics chunk exists
    },
]

print("=" * 70)
print("CRAG EVALUATION")
print("=" * 70)

for item in QUERIES:
    q        = item["query"]
    expected = item["expected"]
    print(f"\nQ: {q}")

    # First evaluate directly to show the decision
    qvec       = embedder.embed_query(q)
    candidates = index.search(qvec, top_k=4, tenant_id="default")
    evaluation = retriever._evaluate(q, candidates[:3])

    print(f"  Evaluation:  {evaluation}  (expected {expected})")
    print(f"  Top-1 chunk: {candidates[0].content[:60].strip()}..." if candidates else "  No candidates")

    # Full CRAG retrieval
    results = retriever.retrieve(q, top_k=3)
    if results:
        action  = results[0].metadata.get("crag_action", "?")
        attempt = results[0].metadata.get("crag_attempt", "?")
        print(f"  CRAG action: {action}  (attempt {attempt})")
        print(f"  Returned {len(results)} chunks")
    else:
        print("  Returned 0 chunks")