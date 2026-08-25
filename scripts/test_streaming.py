# scripts/test_streaming.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from retrievers.vector_retriever      import VectorRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from generation.streaming_generator   import StreamingGenerator
from core                             import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = VectorRetriever(embedder=embedder, index=index)
reranker  = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
streamer  = StreamingGenerator(max_tokens=200, temperature=0.0)

QUERIES = [
    "What is BM25 and how does it score documents?",
    "What evaluation metrics exist for RAG systems?",
]

for query in QUERIES:
    print("=" * 65)
    print(f"Q: {query}\n")
    print("A: ", end="", flush=True)

    candidates = retriever.retrieve(query, top_k=5)
    reranked   = reranker.rerank(query, copy.deepcopy(candidates), top_k=3)

    # Stream mode — tokens appear as they arrive
    result = streamer.collect(query, reranked)

    print(result["answer"])
    print(f"\nTTFT:      {result['ttft_ms']:.0f}ms")
    print(f"Total:     {result['total_ms']:.0f}ms")
    print(f"Tokens:    {result['n_tokens']} (~{result['tps']} tok/s)\n")