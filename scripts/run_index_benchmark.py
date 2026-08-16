# scripts/run_index_benchmark.py

import sys, time, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from statistics import mean
from dotenv     import load_dotenv
load_dotenv()

from loaders.txt_loader              import TXTLoader
from chunkers.recursive_chunker      import RecursiveChunker
from embedders.bge_embedder          import BGEEmbedder
from indexes.faiss_index             import FaissFlatIndex
from indexes.bm25_index              import BM25Index
from indexes.raptor_index            import RAPTORIndex
from retrievers.vector_retriever     import VectorRetriever
from retrievers.hybrid_retriever     import HybridRetriever
from generation.generator            import OpenRouterGenerator
from eval.gold_set                   import GOLD_SET
from eval.metrics                    import recall_at_k, mean_reciprocal_rank
from core                            import NormalizationPipeline, load_config


cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)

# Load + normalise + chunk once
docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

print(f"Corpus: {len(docs[0].content)} chars  |  Chunks: {len(chunks)}\n")


def run_eval(retriever, name: str) -> dict:
    r1, r3, mrr_scores = [], [], []
    for item in GOLD_SET:
        retrieved = retriever.retrieve(item["query"], top_k=5, tenant_id="default")
        r1.append(recall_at_k(retrieved, item["relevant_text"], 1))
        r3.append(recall_at_k(retrieved, item["relevant_text"], 3))
        mrr_scores.append(mean_reciprocal_rank(retrieved, item["relevant_text"]))
    return {
        "name":     name,
        "recall@1": mean(r1),
        "recall@3": mean(r3),
        "mrr":      mean(mrr_scores),
    }


results = []

# --- 1. BM25 only ---
print("Building BM25 index...")
bm25_idx = BM25Index(k1=1.5, b=0.75)
bm25_idx.add(copy.deepcopy(chunks))

class BM25RetrieverWrapper:
    """Wrap BM25Index as a Retriever for eval."""
    def retrieve(self, query, top_k=5, tenant_id="default"):
        return bm25_idx.search_text(query, top_k=top_k, tenant_id=tenant_id)

results.append(run_eval(BM25RetrieverWrapper(), "BM25 (sparse)"))
print(f"  recall@1={results[-1]['recall@1']:.3f}")

# --- 2. Dense only (FAISS Flat) ---
print("Building FAISS flat index...")
faiss_idx = FaissFlatIndex(dimension=384)
faiss_idx.add(copy.deepcopy(chunks))
dense_retriever = VectorRetriever(embedder=embedder, index=faiss_idx)
results.append(run_eval(dense_retriever, "Dense (FAISS)"))
print(f"  recall@1={results[-1]['recall@1']:.3f}")

# --- 3. Hybrid (BM25 + Dense via RRF) ---
print("Building Hybrid (BM25 + Dense + RRF)...")
bm25_h  = BM25Index(k1=1.5, b=0.75)
dense_h = FaissFlatIndex(dimension=384)
bm25_h.add(copy.deepcopy(chunks))
dense_h.add(copy.deepcopy(chunks))

hybrid = HybridRetriever(
    bm25_index  = bm25_h,
    dense_index = dense_h,
    embedder    = embedder,
    k           = 60,
    fetch_k     = 10,
)
results.append(run_eval(hybrid, "Hybrid (RRF)"))
print(f"  recall@1={results[-1]['recall@1']:.3f}")

# --- 4. RAPTOR + Dense ---
print("\nBuilding RAPTOR tree (makes API calls for summarization)...")
generator = OpenRouterGenerator(
    model       = "openrouter/free",
    max_tokens  = 300,
    temperature = 0.0,
    reasoning   = False,
)
raptor = RAPTORIndex(
    embedder         = embedder,
    generator        = generator,
    n_clusters       = 3,
    max_levels       = 2,
    min_cluster_size = 2,
)
raptor.add(copy.deepcopy(chunks))
print(f"  Tree: {raptor.tree_stats()}")

class RAPTORRetrieverWrapper:
    def retrieve(self, query, top_k=5, tenant_id="default"):
        qvec = embedder.embed_query(query)
        return raptor.search(qvec, top_k=top_k, tenant_id=tenant_id)

results.append(run_eval(RAPTORRetrieverWrapper(), "RAPTOR"))
print(f"  recall@1={results[-1]['recall@1']:.3f}")

# --- Ranked table ---
ranked = sorted(results, key=lambda x: (x["recall@1"], x["mrr"]), reverse=True)

print("\n" + "=" * 65)
print(f"{'RANK':<5} {'INDEX':<20} {'r@1':>7} {'r@3':>7} {'MRR':>7}")
print("=" * 65)
for rank, r in enumerate(ranked, 1):
    marker = " ← WINNER" if rank == 1 else ""
    print(f"{rank:<5} {r['name']:<20} {r['recall@1']:>7.3f} "
          f"{r['recall@3']:>7.3f} {r['mrr']:>7.3f}{marker}")

# --- RESULTS.md update ---
md_rows = "\n".join(
    f"| {r['name']:<20} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['mrr']:.3f} |"
    for r in ranked
)
md_block = f"""
## M4 Index Benchmark

**Corpus:** corpus.txt  |  **Chunker:** RecursiveChunker  |  **Embedder:** BGE-small

| Index | recall@1 | recall@3 | MRR |
|-------|----------|----------|-----|
{md_rows}
"""
with open("RESULTS.md", "a", encoding="utf-8") as f:
    f.write(md_block)
print("\nResults appended to RESULTS.md")