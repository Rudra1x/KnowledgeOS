# scripts/test_pipeline.py

import sys, os
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader       import TXTLoader
from chunkers.fixed_chunker   import FixedChunker
from embedders.bge_embedder   import BGEEmbedder
from indexes.faiss_index      import FaissFlatIndex
from retrievers.vector_retriever import VectorRetriever
from generation.generator     import OpenRouterGenerator
from core import load_config

# --- build pipeline from config ---
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

# --- ingest ---
docs    = loader.load("scripts/sample.txt")
chunks  = chunker.chunk(docs[0])
vectors = embedder.embed([c.content for c in chunks])
for chunk, vec in zip(chunks, vectors):
    chunk.embedding = vec
index.add(chunks)
print(f"Ingested {len(chunks)} chunks\n")

# --- ask ---
def ask(query: str, top_k: int = 2) -> str:
    chunks  = retriever.retrieve(query, top_k=top_k)
    answer  = generator.generate(query, chunks)
    return answer

queries = [
    "What is RAG and how does it work?",
    "How does chunking affect retrieval quality?",
    "What happens during the retrieval step?",
]

for q in queries:
    print(f"Q: {q}")
    print(f"A: {ask(q)}")
    print()