# scripts/test_batch_processor.py

import sys, uuid
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

import numpy as np
from core.models                  import Chunk
from embedders.bge_embedder       import BGEEmbedder
from embedders.cache              import CachedEmbedder
from embedders.batch_processor    import BatchEmbedder, normalize_vectors, verify_normalization


# --- Normalization utilities ---
print("=" * 55)
print("NORMALIZATION UTILITIES")
print("=" * 55)

good = np.random.randn(5, 384).astype("float32")
good = good / np.linalg.norm(good, axis=1, keepdims=True)
print(f"Pre-normalized vectors:  {verify_normalization(good)}")

bad  = np.random.randn(5, 384).astype("float32")  # NOT normalized
print(f"Un-normalized vectors:   {verify_normalization(bad)}")

fixed = normalize_vectors(bad.copy())
print(f"After normalize_vectors: {verify_normalization(fixed)}")
print()

# --- Build test chunks ---
CORPUS = [
    "BM25 is a sparse retrieval algorithm based on term frequency.",
    "Dense retrieval uses neural embeddings to capture semantic meaning.",
    "Hybrid retrieval combines sparse and dense methods with RRF.",
    "Rerankers refine retrieval using cross-encoder models.",
    "Evaluation metrics include recall@k, MRR, and nDCG.",
    "FAISS supports flat and approximate indexes like HNSW and IVF.",
    "Semantic chunking splits text at topic shift boundaries.",
    "Parent-child chunking indexes small chunks but returns large context.",
    "Adaptive chunking uses content density to set target chunk sizes.",
    "Metadata-aware chunking treats tables as atomic units.",
]

chunks = [
    Chunk(
        chunk_id  = str(uuid.uuid4()),
        doc_id    = "test-doc",
        content   = text,
        metadata  = {"chunk_index": i},
        tenant_id = "default",
    )
    for i, text in enumerate(CORPUS)
]

# Add one empty chunk to test skip logic
chunks.insert(5, Chunk(
    chunk_id  = str(uuid.uuid4()),
    doc_id    = "test-doc",
    content   = "   ",   # whitespace only
    metadata  = {"chunk_index": 99},
    tenant_id = "default",
))

print("=" * 55)
print(f"BATCH EMBEDDER  ({len(chunks)} chunks, 1 empty)")
print("=" * 55)

raw       = BGEEmbedder("BAAI/bge-small-en-v1.5")
cached    = CachedEmbedder(raw, cache_path="cache/batch_test.db")
processor = BatchEmbedder(cached, batch_size=4, show_progress=True, normalize=True)

result = processor.embed_chunks(chunks)

embedded = [c for c in result if c.embedding]
skipped  = [c for c in result if not c.embedding]

print(f"\nEmbedded: {len(embedded)}  Skipped: {len(skipped)}")
print(f"Sample embedding: {len(embedded[0].embedding)}-dim  "
      f"norm={np.linalg.norm(embedded[0].embedding):.6f}")
print(f"Cache stats: {cached.stats()}")
cached.close()