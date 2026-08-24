# scripts/test_context_compression.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from retrievers.vector_retriever      import VectorRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from generation.local_generator       import LocalLLMGenerator
from generation.context_compressor    import ContextCompressor
from core                             import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
generator  = LocalLLMGenerator(max_tokens=200, temperature=0.0)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = VectorRetriever(embedder=embedder, index=index)
reranker  = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Three compressors
sim_compressor = ContextCompressor(
    strategy      = "similarity",
    embedder      = embedder,
    top_sentences = 2,
)
llm_compressor = ContextCompressor(
    strategy  = "llm",
    generator = generator,
)
budget_compressor = ContextCompressor(
    strategy     = "budget",
    token_budget = 50,
)

QUERY = "What is BM25 and how does it score documents?"

candidates = retriever.retrieve(QUERY, top_k=5)
reranked   = reranker.rerank(QUERY, copy.deepcopy(candidates), top_k=3)

print(f"Query: {QUERY}\n")
print(f"Original top-1 length: {len(reranked[0].content.split())} words")
print(f"Original content:\n  {reranked[0].content[:200].strip()}...\n")

for name, compressor in [
    ("Similarity", sim_compressor),
    ("LLM",        llm_compressor),
    ("Budget",     budget_compressor),
]:
    run_chunks = copy.deepcopy(reranked)
    compressed = compressor.compress(QUERY, run_chunks)

    top1         = compressed[0]
    orig_len     = top1.metadata.get("original_length", "?")
    comp_len     = top1.metadata.get("compressed_length", "?")
    ratio        = top1.metadata.get("compression_ratio", "?")
    latency      = compressor.last_compress_ms

    print(f"--- {name} compressor ---")
    print(f"  {orig_len} words -> {comp_len} words  "
          f"(ratio={ratio}, {latency:.0f}ms)")
    print(f"  Compressed: {top1.content[:150].strip()}...")
    print()