# scripts/test_bm25.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from indexes.tfidf_index         import TFIDFIndex
from indexes.bm25_index          import BM25Index
from core                        import NormalizationPipeline

normalizer = NormalizationPipeline()
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
docs       = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks     = chunker.chunk(docs[0])

tfidf = TFIDFIndex()
bm25  = BM25Index(k1=1.5, b=0.75)

tfidf.add(chunks)
bm25.add(chunks)

print(f"TF-IDF: {tfidf.stats()}")
print(f"BM25:   {bm25.stats()}\n")

QUERIES = [
    "What is BM25 and when does it work well?",
    "How does chunking affect retrieval quality?",
    "What is faithfulness in RAG?",
    "What is FAISS used for?",
    "How does dense retrieval differ from keyword search?",
]

print(f"{'Query':<45} {'TF-IDF #1':>15} {'BM25 #1':>15} {'Match':>6}")
print("-" * 85)
for q in QUERIES:
    tr = tfidf.search_text(q, top_k=1)
    br = bm25.search_text(q, top_k=1)
    t_score = f"{tr[0].metadata['score']:.4f}" if tr else "—"
    b_score = f"{br[0].metadata['score']:.4f}" if br else "—"
    t_text  = tr[0].content[:30].strip() if tr else ""
    b_text  = br[0].content[:30].strip() if br else ""
    same    = "✓" if tr and br and tr[0].chunk_id == br[0].chunk_id else "✗"
    print(f"{q[:45]:<45} {t_score:>8} ({t_text[:12]}) {b_score:>8} ({b_text[:12]}) {same:>4}")

# --- BM25 parameter sensitivity ---
print("\n\nBM25 PARAMETER SENSITIVITY (query: 'What is BM25?')")
print(f"{'k1':>5} {'b':>5} {'top1_score':>12} {'top1_content'}")
print("-" * 60)
for k1 in [0.5, 1.2, 1.5, 2.0]:
    for b in [0.0, 0.5, 0.75]:
        bx = BM25Index(k1=k1, b=b)
        bx.add(chunks)
        r = bx.search_text("What is BM25?", top_k=1)
        if r:
            print(f"{k1:>5} {b:>5}  {r[0].metadata['score']:>10.4f}  {r[0].content[:40].strip()}")