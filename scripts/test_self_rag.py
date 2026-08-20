# scripts/test_self_rag.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from retrievers.self_rag_retriever import SelfRAGRetriever
from generation.local_generator  import LocalLLMGenerator
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
generator  = LocalLLMGenerator(max_tokens=10, temperature=0.0)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index    = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = SelfRAGRetriever(
    embedder        = embedder,
    index           = index,
    generator       = generator,
    always_retrieve = False,
)

# --- Test 1: retrieve gate ---
GATE_TESTS = [
    ("What is 2 + 2?",                          False),  # general knowledge
    ("What is the capital of France?",           False),  # general knowledge
    ("What is BM25 in information retrieval?",   True),   # needs corpus
    ("How does FAISS index vectors?",            True),   # needs corpus
    ("What year was Python created?",            False),  # general knowledge
    ("What is faithfulness in RAG evaluation?",  True),   # needs corpus
]

print("=" * 65)
print("RETRIEVE GATE (should retrieve YES/NO?)")
print("=" * 65)
print(f"{'Query':<45} {'Expected':>9} {'Got':>6} {'OK':>4}")
print("-" * 65)

gate_correct = 0
for query, expected in GATE_TESTS:
    decision = retriever._should_retrieve(query)
    ok       = decision == expected
    if ok:
        gate_correct += 1
    mark = "✓" if ok else "✗"
    print(f"{query[:45]:<45} {'YES' if expected else 'NO':>9} "
          f"{'YES' if decision else 'NO':>6} {mark:>4}")

print(f"\nGate accuracy: {gate_correct}/{len(GATE_TESTS)}")

# --- Test 2: relevance filter ---
print("\n" + "=" * 65)
print("RELEVANCE FILTER")
print("=" * 65)

RELEVANCE_TESTS = [
    {
        "query":    "What is BM25 used for?",
        "relevant": True,
        "passage":  "BM25 is a sparse retrieval algorithm based on term frequency "
                    "and inverse document frequency. It scores documents by how many "
                    "query terms appear, weighted by rarity.",
    },
    {
        "query":    "What is BM25 used for?",
        "relevant": False,
        "passage":  "FAISS is a library for efficient similarity search on dense "
                    "vectors. It supports flat indexes and approximate indexes.",
    },
]

for t in RELEVANCE_TESTS:
    from core.models import Chunk
    import uuid
    chunk = Chunk(
        chunk_id  = str(uuid.uuid4()),
        doc_id    = "test",
        content   = t["passage"],
        tenant_id = "default",
        metadata  = {},
    )
    decision = retriever._is_relevant(t["query"], chunk)
    ok       = decision == t["relevant"]
    mark     = "✓" if ok else "✗"
    print(f"{mark} Relevant={t['relevant']} → Got={'YES' if decision else 'NO'}: "
          f"{t['passage'][:60]}...")

# --- Test 3: end-to-end ---
print("\n" + "=" * 65)
print("END-TO-END RETRIEVAL")
print("=" * 65)
for query, needs in [
    ("What is 2 + 2?",                  False),
    ("What is BM25 in RAG retrieval?",  True),
]:
    results = retriever.retrieve(query, top_k=3)
    print(f"\nQ: {query}")
    print(f"   needs_retrieval={needs}  →  returned {len(results)} chunks")
    if results:
        for r in results:
            rel = r.metadata.get("self_rag_relevant", "?")
            print(f"   relevant={rel}  {r.content[:60].strip()}...")