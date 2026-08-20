# scripts/test_multi_hop.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from retrievers.vector_retriever import VectorRetriever
from retrievers.multi_hop_retriever import MultiHopRetriever
from generation.local_generator  import LocalLLMGenerator
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
generator  = LocalLLMGenerator(max_tokens=100, temperature=0.0)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
baseline  = VectorRetriever(embedder=embedder, index=index)
multi_hop = MultiHopRetriever(embedder=embedder, index=index,
                               generator=generator,
                               max_hops=3, fetch_k=2, hop_decay=0.7)

# Multi-hop query: requires chaining two concepts
QUERIES = [
    {
        "query":    "What method combines the index that FAISS implements with BM25?",
        "expected": "hybrid",   # hop1: FAISS → dense vectors; hop2: combine dense+BM25 → hybrid
    },
    {
        "query":    "What metric measures where the first correct result ranks in retrieval?",
        "expected": "MRR",
    },
]

for item in QUERIES:
    q = item["query"]
    print("=" * 65)
    print(f"QUERY: {q}\n")

    # Baseline
    b_res = baseline.retrieve(q, top_k=3)
    print("BASELINE top-3:")
    for r in b_res:
        print(f"  score={r.metadata['score']:.4f}  {r.content[:70].strip()}...")

    # Multi-hop
    print("\nMULTI-HOP:")
    m_res = multi_hop.retrieve(q, top_k=5)
    hops_seen = set()
    for r in m_res:
        hop        = r.metadata.get("hop", "?")
        hop_q      = r.metadata.get("hop_query", "")[:50]
        score      = r.metadata.get("score", 0)
        if hop not in hops_seen:
            print(f"  --- Hop {hop}: '{hop_q}...' ---")
            hops_seen.add(hop)
        print(f"    score={score:.4f}  {r.content[:65].strip()}...")
    print()