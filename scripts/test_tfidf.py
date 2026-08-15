# scripts/test_tfidf.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader      import TXTLoader
from chunkers.recursive_chunker import RecursiveChunker
from indexes.tfidf_index     import TFIDFIndex
from core                    import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
loader     = TXTLoader()
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
index      = TFIDFIndex()

docs   = normalizer.apply_many(loader.load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
index.add(chunks)

print(f"Indexed {index.stats()}\n")

queries = [
    "What is BM25 and when does it work well?",
    "How does chunking affect retrieval quality?",
    "What is faithfulness in RAG?",
    "What is FAISS used for?",
]

for q in queries:
    results = index.search_text(q, top_k=3)
    print(f"Q: {q}")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r.metadata['score']:.4f}  {r.content[:80].strip()}...")
    print()