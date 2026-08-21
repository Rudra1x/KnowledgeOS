# scripts/run_retriever_benchmark.py

import sys, copy, time
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from statistics import mean
from dotenv     import load_dotenv
load_dotenv()

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from indexes.bm25_index               import BM25Index
from retrievers.vector_retriever      import VectorRetriever
from retrievers.hybrid_retriever      import HybridRetriever
from retrievers.filtered_retriever    import FilteredRetriever
from retrievers.query_rewriting_retriever import QueryRewritingRetriever
from retrievers.multi_query_retriever import MultiQueryRetriever
from retrievers.self_rag_retriever    import SelfRAGRetriever
from retrievers.crag_retriever        import CRAGRetriever
from retrievers.agentic_retriever     import AgenticRetriever
from generation.local_generator       import LocalLLMGenerator
from eval.gold_set                    import GOLD_SET
from eval.metrics                     import recall_at_k, mean_reciprocal_rank
from core                             import NormalizationPipeline, load_config


cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder","bge_small","model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
generator  = LocalLLMGenerator(max_tokens=150, temperature=0.0)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

# Build indexes once
dense_idx  = FaissFlatIndex(dimension=384)
sparse_idx = BM25Index()
dense_idx.add(copy.deepcopy(chunks))
sparse_idx.add(copy.deepcopy(chunks))

print(f"Corpus: {len(docs[0].content)} chars | Chunks: {len(chunks)}\n")

# --- Retriever portfolio ---
RETRIEVERS = [
    ("vector",         VectorRetriever(embedder=embedder, index=dense_idx)),
    ("hybrid_rrf",     HybridRetriever(bm25_index=sparse_idx,
                                       dense_index=dense_idx,
                                       embedder=embedder, fetch_k=8)),
    ("filtered_boost", FilteredRetriever(embedder=embedder, index=dense_idx,
                                         mode="boost")),
    ("query_rewrite",  QueryRewritingRetriever(embedder=embedder,
                                               index=dense_idx,
                                               mode="reformulate",
                                               generator=generator)),
    ("multi_query",    MultiQueryRetriever(embedder=embedder,
                                           index=dense_idx,
                                           generator=generator,
                                           n_variants=3, fetch_k=5)),
    ("self_rag",       SelfRAGRetriever(embedder=embedder,
                                        index=dense_idx,
                                        generator=generator)),
    ("crag",           CRAGRetriever(embedder=embedder,
                                     index=dense_idx,
                                     generator=generator,
                                     max_corrections=1)),
    ("agentic",        AgenticRetriever(embedder=embedder,
                                        dense_index=dense_idx,
                                        sparse_index=sparse_idx,
                                        generator=generator,
                                        max_steps=2, fetch_k=3)),
]

results = []
for name, retriever in RETRIEVERS:
    print(f"Evaluating {name}...")
    t0     = time.perf_counter()
    r1s, r3s, mrrs = [], [], []

    for item in GOLD_SET:
        q   = item["query"]
        rt  = item["relevant_text"]
        res = retriever.retrieve(q, top_k=5, tenant_id="default")
        r1s.append(recall_at_k(res, rt, 1))
        r3s.append(recall_at_k(res, rt, 3))
        mrrs.append(mean_reciprocal_rank(res, rt))

    elapsed = time.perf_counter() - t0
    results.append({
        "name":     name,
        "recall@1": mean(r1s),
        "recall@3": mean(r3s),
        "mrr":      mean(mrrs),
        "time_s":   round(elapsed, 1),
    })
    print(f"  r@1={mean(r1s):.3f}  r@3={mean(r3s):.3f}  "
          f"mrr={mean(mrrs):.3f}  time={elapsed:.1f}s")

# --- Ranked table ---
ranked = sorted(results, key=lambda x: (x["recall@1"], x["mrr"]), reverse=True)
print("\n" + "=" * 75)
print(f"{'RANK':<5} {'RETRIEVER':<18} {'r@1':>7} {'r@3':>7} "
      f"{'MRR':>7} {'TIME(s)':>8}")
print("=" * 75)
for rank, r in enumerate(ranked, 1):
    marker = " <- WINNER" if rank == 1 else ""
    print(f"{rank:<5} {r['name']:<18} {r['recall@1']:>7.3f} "
          f"{r['recall@3']:>7.3f} {r['mrr']:>7.3f} "
          f"{r['time_s']:>8.1f}{marker}")

# --- RESULTS.md ---
md_rows = "\n".join(
    f"| {r['name']:<18} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | "
    f"{r['mrr']:.3f} | {r['time_s']:.1f}s |"
    for r in ranked
)
md_block = f"""
## M5 Retriever Benchmark

**Corpus:** corpus.txt | **Chunker:** Recursive | **Embedder:** BGE-small
**LLM:** qwen2.5:3b-instruct (Ollama local)

| Retriever | recall@1 | recall@3 | MRR | time/10q |
|-----------|----------|----------|-----|----------|
{md_rows}
"""
with open("RESULTS.md", "a", encoding="utf-8") as f:
    f.write(md_block)
print("\nResults appended to RESULTS.md")