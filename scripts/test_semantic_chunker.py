# scripts/test_semantic_chunker.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from chunkers.semantic_chunker   import SemanticChunker
from embedders.bge_embedder      import BGEEmbedder
from core                        import load_config

cfg      = load_config()
embedder = BGEEmbedder(
    model_name = cfg.get("embedder", "bge_small", "model_name"),
    batch_size = cfg.get("embedder", "bge_small", "batch_size"),
)

loader = TXTLoader()
docs   = loader.load("scripts/corpus.txt")
doc    = docs[0]

print(f"Document length: {len(doc.content)} chars\n")

# --- Recursive baseline ---
rc = RecursiveChunker(chunk_size=400, chunk_overlap=50)
rc_chunks = rc.chunk(doc)

# --- Semantic ---
sc = SemanticChunker(
    embedder              = embedder,
    breakpoint_percentile = 95.0,
    min_chunk_size        = 100,
    max_chunk_size        = 800,
)
sc_chunks = sc.chunk(doc)

print("=" * 60)
print(f"RECURSIVE:  {len(rc_chunks)} chunks")
print("=" * 60)
for i, c in enumerate(rc_chunks):
    print(f"[{i}] ({c.metadata['chunk_size_chars']} chars) {c.content[:80].strip()}...")

print("\n" + "=" * 60)
print(f"SEMANTIC:   {len(sc_chunks)} chunks")
print("=" * 60)
for i, c in enumerate(sc_chunks):
    print(f"[{i}] ({c.metadata['chunk_size_chars']} chars) {c.content[:80].strip()}...")

# --- Size stats ---
def stats(chunks, name):
    sizes = [c.metadata['chunk_size_chars'] for c in chunks]
    print(f"{name:12s} min={min(sizes)} max={max(sizes)} avg={sum(sizes)//len(sizes)} count={len(sizes)}")

print("\n" + "=" * 60)
print("SIZE DISTRIBUTION")
print("=" * 60)
stats(rc_chunks, "Recursive")
stats(sc_chunks, "Semantic")