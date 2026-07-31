# scripts/diagnose.py
import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.fixed_chunker      import FixedChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from retrievers.vector_retriever import VectorRetriever
from core import load_config

cfg       = load_config()
loader    = TXTLoader()
chunker   = FixedChunker(cfg.get("chunker","fixed","chunk_size"), cfg.get("chunker","fixed","chunk_overlap"))
embedder  = BGEEmbedder(cfg.get("embedder","bge_small","model_name"), cfg.get("embedder","bge_small","batch_size"))
index     = FaissFlatIndex(dimension=cfg.get("index","faiss_flat","dimension"))
retriever = VectorRetriever(embedder=embedder, index=index)

docs    = loader.load("scripts/corpus.txt")
chunks  = chunker.chunk(docs[0])
for c, v in zip(chunks, embedder.embed([c.content for c in chunks])):
    c.embedding = v
index.add(chunks)

results = retriever.retrieve("Which metrics evaluate retrieval?", top_k=5)
for i, r in enumerate(results, 1):
    print(f"[{i}] score={r.metadata['score']:.4f}")
    print(f"    {r.content[:150]}")
    print()