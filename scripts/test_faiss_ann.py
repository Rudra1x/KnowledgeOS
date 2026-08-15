# scripts/test_faiss_ann.py

import sys, time
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

import numpy as np
from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex, FaissIVFIndex, FaissHNSWIndex
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)

import copy
docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

QUERY = "What is BM25 and when does it work well?"
qvec  = embedder.embed_query(QUERY)

INDEXES = [
    ("Flat (exact)",  FaissFlatIndex(dimension=384)),
    ("IVF  (approx)", FaissIVFIndex(dimension=384, nlist=4, nprobe=2)),
    ("HNSW (approx)", FaissHNSWIndex(dimension=384, M=16, ef_construction=100,
                                     ef_search=20)),
]

print(f"Corpus: {len(chunks)} chunks  |  Query: '{QUERY}'\n")
print(f"{'Index':<18} {'Top-1 content':<45} {'Score':>7}  {'ms':>6}")
print("-" * 85)

for name, index in INDEXES:
    run_chunks = copy.deepcopy(chunks)
    t0 = time.perf_counter()
    index.add(run_chunks)
    t_add = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    results = index.search(qvec, top_k=3, tenant_id="default")
    t_search = (time.perf_counter() - t0) * 1000

    top1 = results[0] if results else None
    content = top1.content[:45].strip() if top1 else "—"
    score   = top1.metadata.get("score", 0) if top1 else 0
    print(f"{name:<18} {content:<45} {score:>7.4f}  {t_search:>5.1f}ms")

print()
print("IVF/HNSW on 8 chunks has no speed advantage — the benefit shows at 100K+ vectors.")
print("All three should return the same top-1 on this small corpus.")

# --- nprobe sensitivity ---
print("\n\nnprobe SENSITIVITY  (IVF, query: 'BM25')")
print(f"{'nprobe':>8} {'top1_chunk':>45} {'score':>8}")
print("-" * 65)
for nprobe in [1, 2, 4]:
    idx = FaissIVFIndex(dimension=384, nlist=4, nprobe=nprobe)
    run_chunks = copy.deepcopy(chunks)
    idx.add(run_chunks)
    r = idx.search(qvec, top_k=1)
    if r:
        print(f"{nprobe:>8}  {r[0].content[:42].strip():>45}  {r[0].metadata['score']:>7.4f}")