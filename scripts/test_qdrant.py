# scripts/test_qdrant.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from indexes.qdrant_index        import QdrantIndex
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

QUERY = "What is BM25 and when does it work well?"
qvec  = embedder.embed_query(QUERY)

print("=" * 60)
print("TEST 1 — Basic retrieval (Qdrant vs FAISS)")
print("=" * 60)

faiss = FaissFlatIndex(dimension=384)
faiss.add(copy.deepcopy(chunks))
faiss_res = faiss.search(qvec, top_k=3)

qdrant = QdrantIndex(collection_name="test_basic", dimension=384)
qdrant.add(copy.deepcopy(chunks))
qdrant_res = qdrant.search(qvec, top_k=3)

print("\nFAISS top-3:")
for r in faiss_res:
    print(f"  score={r.metadata['score']:.4f}  {r.content[:60].strip()}...")

print("\nQdrant top-3:")
for r in qdrant_res:
    print(f"  score={r.metadata['score']:.4f}  {r.content[:60].strip()}...")

same = (faiss_res[0].chunk_id == qdrant_res[0].chunk_id
        if faiss_res and qdrant_res else False)
print(f"\nSame top-1: {same}")

print("\n" + "=" * 60)
print("TEST 2 — Native payload filtering (tenant isolation)")
print("=" * 60)

qt = QdrantIndex(collection_name="test_tenant", dimension=384)
chunks_a = copy.deepcopy(chunks[:4])
chunks_b = copy.deepcopy(chunks[4:])
for c in chunks_a:
    c.tenant_id = "tenant_A"
for c in chunks_b:
    c.tenant_id = "tenant_B"

qt.add(chunks_a + chunks_b)
res_a = qt.search(qvec, top_k=5, tenant_id="tenant_A")
res_b = qt.search(qvec, top_k=5, tenant_id="tenant_B")

print(f"Tenant A: {len(res_a)} results  (all tenant_A: "
      f"{all(r.tenant_id == 'tenant_A' for r in res_a)})")
print(f"Tenant B: {len(res_b)} results  (all tenant_B: "
      f"{all(r.tenant_id == 'tenant_B' for r in res_b)})")

print("\n" + "=" * 60)
print("TEST 3 — Deletion")
print("=" * 60)
qd = QdrantIndex(collection_name="test_del", dimension=384)
qd.add(copy.deepcopy(chunks))
before = qd.count()
del_id = chunks[0].chunk_id
qd.delete([del_id])
after = qd.count()
print(f"Before: {before}  |  After: {after}  |  Deleted: {before - after == 1}")