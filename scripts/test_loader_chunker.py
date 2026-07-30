# scripts/test_loader_chunker.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader import TXTLoader
from chunkers.fixed_chunker import FixedChunker
from core import load_config

cfg     = load_config()
loader  = TXTLoader()
chunker = FixedChunker(
    chunk_size    = cfg.get("chunker", "fixed", "chunk_size"),
    chunk_overlap = cfg.get("chunker", "fixed", "chunk_overlap"),
)

docs   = loader.load("scripts/sample.txt")
chunks = chunker.chunk(docs[0])

print(f"Documents : {len(docs)}")
print(f"Chunks    : {len(chunks)}")
print()
for c in chunks:
    print(f"[{c.chunk_id[:8]}] chars {c.start_index}–{c.end_index}")
    print(f"  {c.content[:80].strip()}...")
    print()