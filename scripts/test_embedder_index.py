# scripts/test_embedder_index.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader import TXTLoader
from chunkers.fixed_chunker import FixedChunker
from embedders.bge_embedder import BGEEmbedder
from indexes.faiss_index import FaissFlatIndex
from core import load_config

cfg      = load_config()
loader   = TXTLoader()
chunker  = FixedChunker(
    chunk_size    = cfg.get("chunker", "fixed", "chunk_size"),
    chunk_overlap = cfg.get("chunker", "fixed", "chunk_overlap"),
)
embedder = BGEEmbedder(
    model_name = cfg.get("embedder", "bge_small", "model_name"),
    batch_size = cfg.get("embedder", "bge_small", "batch_size"),
)
index    = FaissFlatIndex(dimension=cfg.get("index", "faiss_flat", "dimension"))

# --- ingest ---
docs     = loader.load("scripts/sample.txt")
chunks   = chunker.chunk(docs[0])
vectors  = embedder.embed([c.content for c in chunks])

for chunk, vec in zip(chunks, vectors):
    chunk.embedding = vec

index.add(chunks)
print(f"Indexed {len(chunks)} chunks  |  dimension: {embedder.dimension}")

# --- query ---
query        = "How does chunking affect retrieval quality?"
query_vector = embedder.embed_query(query)
results      = index.search(query_vector, top_k=2, tenant_id="default")

print(f"\nQuery: {query}\n")
for i, r in enumerate(results):
    print(f"  [{i+1}] score={r.metadata['score']:.4f}")
    print(f"       {r.content[:100].strip()}...")
    print()