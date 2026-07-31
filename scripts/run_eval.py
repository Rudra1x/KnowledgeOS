# scripts/run_eval.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.fixed_chunker      import FixedChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from retrievers.vector_retriever import VectorRetriever
from generation.generator        import OpenRouterGenerator
from core import load_config
from eval  import evaluate, print_report


cfg       = load_config()
loader    = TXTLoader()
chunker   = FixedChunker(
    chunk_size    = cfg.get("chunker", "fixed", "chunk_size"),
    chunk_overlap = cfg.get("chunker", "fixed", "chunk_overlap"),
)
embedder  = BGEEmbedder(
    model_name = cfg.get("embedder", "bge_small", "model_name"),
    batch_size = cfg.get("embedder", "bge_small", "batch_size"),
)
index     = FaissFlatIndex(dimension=cfg.get("index", "faiss_flat", "dimension"))
retriever = VectorRetriever(embedder=embedder, index=index)
generator = OpenRouterGenerator(
    model       = cfg.get("generator", "openrouter", "model"),
    max_tokens  = cfg.get("generator", "openrouter", "max_tokens"),
    temperature = cfg.get("generator", "openrouter", "temperature"),
)

# --- ingest corpus ---
docs    = loader.load("scripts/corpus.txt")
chunks  = chunker.chunk(docs[0])
vectors = embedder.embed([c.content for c in chunks])
for chunk, vec in zip(chunks, vectors):
    chunk.embedding = vec
index.add(chunks)
print(f"Indexed {len(chunks)} chunks from corpus.\n")

# --- run eval (retrieval only first, faster) ---
print("Running RETRIEVAL evaluation (10 queries)...")
report = evaluate(retriever=retriever, k_values=[1, 3, 5])
print_report(report)

# --- run with generation too (slower, uses API) ---
print("\nRunning RETRIEVAL + GENERATION evaluation (uses API)...")
report_full = evaluate(retriever=retriever, generator=generator, k_values=[1, 3, 5], run_generation=True)
print_report(report_full)