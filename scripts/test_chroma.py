# scripts/test_chroma.py

import sys, os
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

import shutil
from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from indexes.chroma_index        import ChromaIndex
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
print("TEST 1 — Basic retrieval (Chroma vs FAISS)")
print("=" * 60)
import copy

faiss_idx  = FaissFlatIndex(dimension=384)
faiss_idx.add(copy.deepcopy(chunks))
faiss_res  = faiss_idx.search(qvec, top_k=3)

chroma_idx = ChromaIndex(collection_name="test", persist_directory=None)
chroma_idx.add(copy.deepcopy(chunks))
chroma_res = chroma_idx.search(qvec, top_k=3)

print(f"\nFAISS top-3:")
for r in faiss_res:
    print(f"  score={r.metadata['score']:.4f}  {r.content[:60].strip()}...")

print(f"\nChroma top-3:")
for r in chroma_res:
    print(f"  score={r.metadata['score']:.4f}  {r.content[:60].strip()}...")

same_top1 = (faiss_res[0].chunk_id == chroma_res[0].chunk_id
             if faiss_res and chroma_res else False)
print(f"\nSame top-1: {same_top1}")

# --- Test 2: Deletion ---
print("\n" + "=" * 60)
print("TEST 2 — Deletion (Chroma native, FAISS not supported)")
print("=" * 60)
before = chroma_idx.count()
del_id = chroma_res[0].chunk_id
chroma_idx.delete([del_id])
after  = chroma_idx.count()
print(f"Before: {before} chunks  |  After delete: {after} chunks")
print(f"Deleted chunk_id: {del_id[:16]}...")

# --- Test 3: Persistence ---
print("\n" + "=" * 60)
print("TEST 3 — Persistence (survives process restart)")
print("=" * 60)
PERSIST_PATH = "cache/chroma_test"
if os.path.exists(PERSIST_PATH):
    shutil.rmtree(PERSIST_PATH)

p1 = ChromaIndex(collection_name="persist_test", persist_directory=PERSIST_PATH)
p1.add(copy.deepcopy(chunks))
count_before = p1.count()
del p1   # simulate process restart

p2 = ChromaIndex(collection_name="persist_test", persist_directory=PERSIST_PATH)
count_after = p2.count()
print(f"Written: {count_before}  |  Reloaded: {count_after}  |  Persisted: {count_before == count_after}")

# --- Test 4: Multi-tenancy ---
print("\n" + "=" * 60)
print("TEST 4 — Multi-tenancy (one collection per tenant)")
print("=" * 60)
import copy, uuid

mt_index = ChromaIndex(collection_name="multi_tenant", persist_directory=None, multi_tenant=True)
chunks_a = copy.deepcopy(chunks[:4])
chunks_b = copy.deepcopy(chunks[4:])
for c in chunks_a:
    c.tenant_id = "tenant_A"
for c in chunks_b:
    c.tenant_id = "tenant_B"

mt_index.add(chunks_a + chunks_b)
res_a = mt_index.search(qvec, top_k=3, tenant_id="tenant_A")
res_b = mt_index.search(qvec, top_k=3, tenant_id="tenant_B")

print(f"Tenant A results: {len(res_a)}  (all should have tenant_A)")
for r in res_a:
    print(f"  tenant={r.tenant_id}  {r.content[:50].strip()}...")
print(f"Tenant B results: {len(res_b)}  (all should have tenant_B)")

# Cleanup
shutil.rmtree(PERSIST_PATH, ignore_errors=True)