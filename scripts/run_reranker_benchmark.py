# scripts/run_reranker_benchmark.py

import sys, copy, time
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from statistics import mean
from dotenv     import load_dotenv
load_dotenv()

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from embedders.e5_embedder            import E5Embedder
from indexes.faiss_index              import FaissFlatIndex
from indexes.bm25_index               import BM25Index
from retrievers.vector_retriever      import VectorRetriever
from retrievers.hybrid_retriever      import HybridRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from rerankers.bge_reranker           import BGEReranker
from rerankers.llm_reranker           import LLMReranker
from rerankers.metadata_reranker      import MetadataReranker
from generation.local_generator       import LocalLLMGenerator
from eval.gold_set                    import GOLD_SET
from eval.metrics                     import recall_at_k, mean_reciprocal_rank
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

dense_idx  = FaissFlatIndex(dimension=384)
sparse_idx = BM25Index()
dense_idx.add(copy.deepcopy(chunks))
sparse_idx.add(copy.deepcopy(chunks))

retriever_v = VectorRetriever(embedder=embedder, index=dense_idx)
retriever_h = HybridRetriever(
    bm25_index  = sparse_idx,
    dense_index = dense_idx,
    embedder    = embedder,
    fetch_k     = 8,
)

generator = LocalLLMGenerator(max_tokens=10, temperature=0.0)

# Load rerankers once
print("Loading rerankers...")
msmarco  = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
bge_rr   = BGEReranker("BAAI/bge-reranker-base")
llm_rr   = LLMReranker(generator=generator, mode="score")
meta_rr  = MetadataReranker(
    keyword_boost = {"retrieval": 1.1, "BM25": 1.2, "hybrid": 1.1}
)
print("All rerankers ready.\n")

def evaluate(retriever, reranker, name, fetch_k=5, rerank_k=3):
    """Retrieve fetch_k, rerank, keep top rerank_k. Return metrics + timing."""
    r1s, r3s, mrrs = [], [], []
    t0 = time.perf_counter()

    for item in GOLD_SET:
        q   = item["query"]
        rt  = item["relevant_text"]

        candidates = retriever.retrieve(q, top_k=fetch_k)

        if reranker:
            final = reranker.rerank(q, copy.deepcopy(candidates), top_k=rerank_k)
        else:
            final = candidates[:rerank_k]

        r1s.append(recall_at_k(final, rt, 1))
        r3s.append(recall_at_k(final, rt, 3))
        mrrs.append(mean_reciprocal_rank(final, rt))

    elapsed = time.perf_counter() - t0
    return {
        "name":     name,
        "recall@1": mean(r1s),
        "recall@3": mean(r3s),
        "mrr":      mean(mrrs),
        "time_s":   round(elapsed, 1),
    }

results = []

# --- Baselines (no reranking) ---
print("Evaluating baselines...")
results.append(evaluate(retriever_v, None,    "Vector (no rerank)"))
results.append(evaluate(retriever_h, None,    "Hybrid (no rerank)"))

# --- Vector + rerankers ---
print("Evaluating vector + rerankers...")
results.append(evaluate(retriever_v, msmarco, "Vector + MS-MARCO"))
results.append(evaluate(retriever_v, bge_rr,  "Vector + BGE"))
results.append(evaluate(retriever_v, llm_rr,  "Vector + LLM"))
results.append(evaluate(retriever_v, meta_rr, "Vector + Metadata"))

# --- Hybrid + best reranker ---
print("Evaluating hybrid + rerankers...")
results.append(evaluate(retriever_h, msmarco, "Hybrid + MS-MARCO"))
results.append(evaluate(retriever_h, bge_rr,  "Hybrid + BGE"))

# --- Ranked table ---
ranked = sorted(results,
                key=lambda x: (x["recall@1"], x["mrr"]),
                reverse=True)

print("\n" + "=" * 72)
print(f"{'RANK':<5} {'PIPELINE':<22} {'r@1':>7} {'r@3':>7} "
      f"{'MRR':>7} {'TIME(s)':>8}")
print("=" * 72)
for rank, r in enumerate(ranked, 1):
    marker = " <- WINNER" if rank == 1 else ""
    print(f"{rank:<5} {r['name']:<22} {r['recall@1']:>7.3f} "
          f"{r['recall@3']:>7.3f} {r['mrr']:>7.3f} "
          f"{r['time_s']:>8.1f}{marker}")

# --- Key comparison ---
print("\n" + "=" * 72)
print("KEY INSIGHT: does reranking on weak retriever beat strong retriever?")
print("=" * 72)
vec_base  = next(r for r in results if r["name"] == "Vector (no rerank)")
vec_msm   = next(r for r in results if r["name"] == "Vector + MS-MARCO")
hyb_base  = next(r for r in results if r["name"] == "Hybrid (no rerank)")
print(f"  Vector alone:           r@1={vec_base['recall@1']:.3f}  "
      f"{vec_base['time_s']:.1f}s")
print(f"  Vector + MS-MARCO:      r@1={vec_msm['recall@1']:.3f}  "
      f"{vec_msm['time_s']:.1f}s")
print(f"  Hybrid alone:           r@1={hyb_base['recall@1']:.3f}  "
      f"{hyb_base['time_s']:.1f}s")

# --- RESULTS.md update ---
md_rows = "\n".join(
    f"| {r['name']:<22} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | "
    f"{r['mrr']:.3f} | {r['time_s']:.1f}s |"
    for r in ranked
)
md_block = f"""
## M6 Reranker Benchmark

**Corpus:** corpus.txt | **Retriever:** Vector (fetch_k=5) + Hybrid
**Reranker models:** MS-MARCO-MiniLM-L6, BGE-reranker-base, LLM (Qwen2.5-3B), Metadata

| Pipeline | recall@1 | recall@3 | MRR | time/10q |
|----------|----------|----------|-----|----------|
{md_rows}
"""
with open("RESULTS.md", "a", encoding="utf-8") as f:
    f.write(md_block)
print("\nResults appended to RESULTS.md")