# scripts/test_adaptive_chunker.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader        import TXTLoader
from chunkers.adaptive_chunker import AdaptiveChunker


# Two very different corpora — same chunker, different behavior
SAMPLES = {
    "RAG prose (scripts/corpus.txt)": "scripts/corpus.txt",
}

loader = TXTLoader()

# Also create a dense synthetic sample inline
DENSE_TEXT = """
def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a)**0.5
    norm_b = sum(x**2 for x in b)**0.5
    return dot / (norm_a * norm_b)

class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def score(self, tf, df, N, dl, avgdl):
        idf = log((N - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
        return idf * tf_norm
"""

chunker = AdaptiveChunker(
    min_chunk_size  = 150,
    max_chunk_size  = 900,
    base_chunk_size = 450,
    overlap_chars   = 40,
)

# --- Corpus.txt (prose) ---
print("=" * 60)
print("PROSE CORPUS (sparse text)")
print("=" * 60)
docs   = loader.load("scripts/corpus.txt")
chunks = chunker.chunk(docs[0])
print(f"Chunks: {len(chunks)}\n")
for c in chunks:
    print(f"  density={c.metadata['density_score']:.3f}  "
          f"target={c.metadata['target_size']}  "
          f"actual={c.metadata['chunk_size_chars']}  "
          f"{c.content[:60].strip()}...")

# --- Dense code sample ---
print("\n" + "=" * 60)
print("DENSE CODE SAMPLE")
print("=" * 60)
from core.models import Document
import uuid
code_doc = Document(
    doc_id   = str(uuid.uuid4()),
    content  = DENSE_TEXT.strip(),
    source   = "synthetic",
    metadata = {"file_type": "txt"},
)
code_chunks = chunker.chunk(code_doc)
print(f"Chunks: {len(code_chunks)}\n")
for c in code_chunks:
    print(f"  density={c.metadata['density_score']:.3f}  "
          f"target={c.metadata['target_size']}  "
          f"actual={c.metadata['chunk_size_chars']}  "
          f"{c.content[:60].strip()}...")

# --- Summary ---
if chunks and code_chunks:
    prose_targets = [c.metadata["target_size"] for c in chunks]
    code_targets  = [c.metadata["target_size"] for c in code_chunks]
    print("\n" + "=" * 60)
    print("DENSITY SUMMARY")
    print("=" * 60)
    print(f"  Prose  avg target size: {sum(prose_targets)//len(prose_targets)}")
    print(f"  Code   avg target size: {sum(code_targets)//len(code_targets)}")
    print(f"\n  Higher target → chunker treats text as sparse (use larger chunks)")
    print(f"  Lower target  → chunker treats text as dense (use smaller chunks)")