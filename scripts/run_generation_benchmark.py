# scripts/run_generation_benchmark.py

import sys, copy, time
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from statistics import mean
from dotenv     import load_dotenv
load_dotenv()

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from retrievers.vector_retriever      import VectorRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from generation.local_generator       import LocalLLMGenerator
from generation.streaming_generator   import StreamingGenerator
from generation.context_compressor    import ContextCompressor
from generation.faithfulness_checker  import FaithfulnessChecker
from generation.answer_relevance      import AnswerRelevanceScorer
from generation.prompt_builder        import extract_citations
from eval.gold_set                    import GOLD_SET
from eval.metrics                     import recall_at_k
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

# Pipeline components
retriever  = VectorRetriever(embedder=embedder, index=index)
reranker   = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = ContextCompressor(
    strategy      = "similarity",
    embedder      = embedder,
    top_sentences = 3,
)
generator  = LocalLLMGenerator(max_tokens=200, temperature=0.0)
streamer   = StreamingGenerator(max_tokens=200, temperature=0.0)
faith_checker = FaithfulnessChecker(
    strategy   = "nli",
    model_name = "cross-encoder/nli-MiniLM2-L6-H768",
    threshold  = 0.25,
)
rel_scorer = AnswerRelevanceScorer(
    embedder    = embedder,
    generator   = LocalLLMGenerator(max_tokens=150, temperature=0.3),
    n_questions = 1,
)

print(f"Corpus: {len(docs[0].content)} chars | Chunks: {len(chunks)}\n")
print("Running generation benchmark (this will take a few minutes)...\n")

results = []

for item in GOLD_SET[:5]:   # first 5 for speed — full run takes ~10 mins on CPU
    q  = item["query"]
    rt = item["relevant_text"]

    t0 = time.perf_counter()

    # Stage 1: Retrieve + Rerank
    candidates = retriever.retrieve(q, top_k=5)
    reranked   = reranker.rerank(q, copy.deepcopy(candidates), top_k=3)
    retrieval_r1 = recall_at_k(reranked, rt, 1)

    # Stage 2: Compress
    compressed = compressor.compress(q, copy.deepcopy(reranked))
    comp_ratio = mean([
        c.metadata.get("compression_ratio", 1.0) for c in compressed
    ])

    # Stage 3: Generate
    gen_result = streamer.collect(q, compressed)
    answer     = gen_result["answer"]
    ttft_ms    = gen_result["ttft_ms"]

    # Stage 4: Faithfulness
    faith = faith_checker.check(answer, reranked)

    # Stage 5: Answer relevance
    rel = rel_scorer.score(q, answer)

    total_ms = (time.perf_counter() - t0) * 1000

    results.append({
        "query":          q[:45],
        "retrieval_r1":   retrieval_r1,
        "comp_ratio":     comp_ratio,
        "ttft_ms":        ttft_ms,
        "faithful":       faith["faithful"],
        "faith_score":    faith["score"],
        "relevance":      rel["relevance_score"],
        "verdict":        rel["verdict"],
        "total_ms":       total_ms,
        "answer_snippet": answer[:80],
    })

    print(f"Q: {q[:55]}")
    print(f"  r@1={retrieval_r1:.0f}  "
          f"compress={comp_ratio:.2f}  "
          f"ttft={ttft_ms:.0f}ms  "
          f"faithful={faith['faithful']}({faith['score']:.2f})  "
          f"relevance={rel['relevance_score']:.2f}({rel['verdict']})")
    print(f"  A: {answer[:80].strip()}...")
    print()

# --- Summary table ---
print("=" * 78)
print(f"{'Query':<38} {'r@1':>4} {'Comp':>5} {'Faith':>6} {'Rel':>5} {'TTFT':>6}")
print("=" * 78)
for r in results:
    print(f"{r['query']:<38} {r['retrieval_r1']:>4.0f} "
          f"{r['comp_ratio']:>5.2f} "
          f"{r['faith_score']:>6.2f} "
          f"{r['relevance']:>5.2f} "
          f"{r['ttft_ms']:>5.0f}ms")

print("=" * 78)
print(f"{'MEAN':<38} {mean(r['retrieval_r1'] for r in results):>4.2f} "
      f"{mean(r['comp_ratio'] for r in results):>5.2f} "
      f"{mean(r['faith_score'] for r in results):>6.2f} "
      f"{mean(r['relevance'] for r in results):>5.2f} "
      f"{mean(r['ttft_ms'] for r in results):>5.0f}ms")

# --- RESULTS.md ---
md_rows = "\n".join(
    f"| {r['query']:<38} | {r['retrieval_r1']:.0f} | "
    f"{r['faith_score']:.2f} | {r['relevance']:.2f} | "
    f"{r['ttft_ms']:.0f}ms |"
    for r in results
)
md_block = f"""
## M7 Generation Benchmark

**Pipeline:** Vector retrieval → MS-MARCO rerank → Similarity compress → Qwen2.5-3B generate → NLI faithfulness → Relevance score
**Corpus:** corpus.txt | **LLM:** qwen2.5:3b-instruct (Ollama)

| Query | r@1 | Faithfulness | Relevance | TTFT |
|-------|-----|-------------|-----------|------|
{md_rows}

**Mean:** r@1={mean(r['retrieval_r1'] for r in results):.2f} | \
faith={mean(r['faith_score'] for r in results):.2f} | \
relevance={mean(r['relevance'] for r in results):.2f} | \
ttft={mean(r['ttft_ms'] for r in results):.0f}ms
"""
with open("RESULTS.md", "a", encoding="utf-8") as f:
    f.write(md_block)
print("\nResults appended to RESULTS.md")