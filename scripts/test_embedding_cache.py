# scripts/test_embedding_cache.py

import sys, os, time
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from embedders.bge_embedder import BGEEmbedder
from embedders.cache        import CachedEmbedder

TEXTS = [
    "BM25 is a sparse retrieval algorithm.",
    "Dense retrieval uses neural embeddings.",
    "Hybrid retrieval combines sparse and dense.",
    "FAISS is a library for similarity search.",
    "Rerankers refine initial retrieval results.",
]

# Clean slate every run
DB_PATH = "cache/test_embeddings.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

raw   = BGEEmbedder("BAAI/bge-small-en-v1.5")
cache = CachedEmbedder(raw, cache_path=DB_PATH)

print("=" * 55)
print("RUN 1 — cold cache (all misses, real embedding)")
print("=" * 55)
t0 = time.perf_counter()
v1 = cache.embed(TEXTS)
t1 = (time.perf_counter() - t0) * 1000
print(f"Time:    {t1:.1f}ms")
print(f"Vectors: {len(v1)} x {len(v1[0])}")
print(f"Stats:   {cache.stats()}\n")

print("=" * 55)
print("RUN 2 — warm L2 cache (DB hits, L1 populated)")
print("=" * 55)
# Clear L1 to simulate process restart — only DB persists
cache._mem.clear()
cache._mem_order.clear()
t0 = time.perf_counter()
v2 = cache.embed(TEXTS)
t2 = (time.perf_counter() - t0) * 1000
print(f"Time:    {t2:.1f}ms")
print(f"Speedup: {t1/t2:.1f}x faster than embedding")
print(f"L1 entries after: {cache.stats()['l1_entries']}  (populated from L2)")
print(f"Stats:   {cache.stats()}\n")

print("=" * 55)
print("RUN 3 — warm L1 cache (memory hits, fastest)")
print("=" * 55)
t0 = time.perf_counter()
v3 = cache.embed(TEXTS)
t3 = (time.perf_counter() - t0) * 1000
print(f"Time:    {t3:.1f}ms")
print(f"Speedup: {t1/t3:.1f}x faster than embedding")
print(f"Stats:   {cache.stats()}\n")

print("=" * 55)
print("RUN 4 — partial cache (3 cached + 2 new)")
print("=" * 55)
mixed = TEXTS[:3] + [
    "Semantic chunking detects topic shifts.",
    "Parent-child chunking has a 3x context multiplier.",
]
t0 = time.perf_counter()
v4 = cache.embed(mixed)
t4 = (time.perf_counter() - t0) * 1000
print(f"Time:    {t4:.1f}ms  (3 L1 hits + 2 embedded)")
print(f"Stats:   {cache.stats()}\n")

import numpy as np
max_diff = float(np.max(np.abs(np.array(v1) - np.array(v2))))
print(f"Correctness — Run 1 vs Run 2 max diff: {max_diff:.2e}  (expect 0.00)")

cache.close()