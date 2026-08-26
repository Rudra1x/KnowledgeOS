# scripts/run_retrieval_eval.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from indexes.bm25_index               import BM25Index
from retrievers.vector_retriever      import VectorRetriever
from retrievers.hybrid_retriever      import HybridRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from eval.retrieval_evaluator         import RetrievalEvaluator
from eval.gold_set_v2                 import GOLD_SET_V2
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

reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Wrap hybrid + reranker as a single retriever
class HybridReranked:
    def retrieve(self, query, top_k=5, tenant_id="default"):
        from retrievers.hybrid_retriever import HybridRetriever
        h = HybridRetriever(
            bm25_index=sparse_idx, dense_index=dense_idx,
            embedder=embedder, fetch_k=10,
        )
        candidates = h.retrieve(query, top_k=10, tenant_id=tenant_id)
        return reranker.rerank(query, copy.deepcopy(candidates), top_k=top_k)

RETRIEVERS = [
    ("Vector",           VectorRetriever(embedder=embedder, index=dense_idx)),
    ("Hybrid+Rerank",    HybridReranked()),
]

print(f"Gold set: {len(GOLD_SET_V2)} queries "
      f"({sum(1 for q in GOLD_SET_V2 if q['relevant_text'])} positive, "
      f"{sum(1 for q in GOLD_SET_V2 if not q['relevant_text'])} negative)\n")

all_results = {}
for name, retriever in RETRIEVERS:
    print(f"Evaluating {name}...")
    evaluator = RetrievalEvaluator(retriever=retriever, top_k=5)
    results   = evaluator.evaluate(GOLD_SET_V2)
    all_results[name] = results

    agg = results["aggregate"]
    print(f"  recall@1={agg['recall@1']:.3f}  "
          f"recall@3={agg['recall@3']:.3f}  "
          f"nDCG@3={agg['ndcg@3']:.3f}  "
          f"MRR={agg['mrr']:.3f}  "
          f"neg_acc={agg['neg_accuracy']}")

    print("  By type:")
    for qt, metrics in results["by_type"].items():
        print(f"    {qt:<12} r@1={metrics['recall@1']:.3f}  "
              f"nDCG@3={metrics['ndcg@3']:.3f}  n={metrics['n']}")
    print()

# --- Comparison table ---
print("=" * 72)
print(f"{'Retriever':<18} {'r@1':>6} {'r@3':>6} {'nDCG@3':>8} "
      f"{'MRR':>6} {'neg_acc':>8}")
print("=" * 72)
for name, results in all_results.items():
    agg = results["aggregate"]
    print(f"{name:<18} {agg['recall@1']:>6.3f} {agg['recall@3']:>6.3f} "
          f"{agg['ndcg@3']:>8.3f} {agg['mrr']:>6.3f} "
          f"{str(agg['neg_accuracy']):>8}")