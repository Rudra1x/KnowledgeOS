# scripts/test_recursive_chunker.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.fixed_chunker      import OverlappingChunker
from chunkers.recursive_chunker  import RecursiveChunker


loader = TXTLoader()
docs   = loader.load("scripts/corpus.txt")
doc    = docs[0]

print(f"Document length: {len(doc.content)} chars\n")

# --- Overlapping baseline ---
print("=" * 60)
print("OVERLAPPING CHUNKER (baseline)")
print("=" * 60)
op = OverlappingChunker(chunk_size=400, chunk_overlap=50)
op_chunks = op.chunk(doc)
print(f"Chunks: {len(op_chunks)}\n")
for i, c in enumerate(op_chunks):
    print(f"[{i}] chars {c.start_index}-{c.end_index} ({c.metadata['chunk_size_chars']} chars)")
    print(f"    starts: {repr(c.content[:60])}")
    print(f"    ends:   ...{repr(c.content[-60:])}")
    print()

# --- Recursive ---
print("=" * 60)
print("RECURSIVE CHUNKER")
print("=" * 60)
rc = RecursiveChunker(chunk_size=400, chunk_overlap=50)
rc_chunks = rc.chunk(doc)
print(f"Chunks: {len(rc_chunks)}\n")
for i, c in enumerate(rc_chunks):
    print(f"[{i}] chars {c.start_index}-{c.end_index} ({c.metadata['chunk_size_chars']} chars)")
    print(f"    starts: {repr(c.content[:60])}")
    print(f"    ends:   ...{repr(c.content[-60:])}")
    print()

# --- Boundary quality metric ---
def ends_at_natural_boundary(text: str) -> bool:
    """A chunk ends naturally if it ends with sentence/paragraph/word boundary."""
    tail = text.rstrip()
    if not tail:
        return False
    return tail.endswith(('.', '!', '?', '"', ')', ':', ';', '\n')) or tail[-1].isspace()

op_natural = sum(1 for c in op_chunks if ends_at_natural_boundary(c.content))
rc_natural = sum(1 for c in rc_chunks if ends_at_natural_boundary(c.content))

print("=" * 60)
print("BOUNDARY QUALITY")
print("=" * 60)
print(f"Overlapping: {op_natural}/{len(op_chunks)} chunks end at a natural boundary")
print(f"Recursive:   {rc_natural}/{len(rc_chunks)} chunks end at a natural boundary")