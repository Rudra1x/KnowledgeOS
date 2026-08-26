# scripts/run_pipeline_eval.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from indexes.bm25_index               import BM25Index
from retrievers.hybrid_retriever      import HybridRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from generation.local_generator       import LocalLLMGenerator
from generation.faithfulness_checker  import FaithfulnessChecker
from generation.answer_relevance      import AnswerRelevanceScorer
from eval.pipeline_evaluator          import PipelineEvaluator
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

reranker  = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

class HybridReranked:
    def retrieve(self, query, top_k=5, tenant_id="default"):
        h = HybridRetriever(
            bm25_index=sparse_idx, dense_index=dense_idx,
            embedder=embedder, fetch_k=10,
        )
        cands = h.retrieve(query, top_k=10, tenant_id=tenant_id)
        return reranker.rerank(query, copy.deepcopy(cands), top_k=top_k)

faith_checker = FaithfulnessChecker(
    strategy="nli",
    model_name="cross-encoder/nli-MiniLM2-L6-H768",
    threshold=0.25,
)
rel_scorer = AnswerRelevanceScorer(
    embedder    = embedder,
    generator   = LocalLLMGenerator(max_tokens=100, temperature=0.3),
    n_questions = 1,
)

evaluator = PipelineEvaluator(
    retriever        = HybridReranked(),
    generator        = LocalLLMGenerator(max_tokens=200, temperature=0.0),
    faith_checker    = faith_checker,
    relevance_scorer = rel_scorer,
    retrieval_top_k  = 5,
    generation_top_k = 3,
)

print(f"Running pipeline eval: {len(GOLD_SET_V2)} queries\n")
results = evaluator.evaluate(GOLD_SET_V2, run_name="hybrid_rerank_v1")

# --- Summary ---
agg = results["aggregate"]
print(f"\n{'='*65}")
print(f"PIPELINE EVALUATION: {results['run_name']}")
print(f"{'='*65}")
print(f"  Retrieval  r@1={agg['recall@1']:.3f}  "
      f"r@3={agg['recall@3']:.3f}  "
      f"nDCG@3={agg['ndcg@3']:.3f}  "
      f"MRR={agg['mrr']:.3f}")
print(f"  Generation faith={agg['faithfulness']:.3f}  "
      f"cov={agg['citation_coverage']:.3f}  "
      f"rel={agg['relevance']:.3f}")
print(f"  Negatives  decline_rate={agg['decline_rate']}")
print(f"  Latency    {agg['mean_latency_ms']:.0f}ms/query")

print(f"\nBY TYPE:")
for qt, m in sorted(results["by_type"].items()):
    print(f"  {qt:<12} r@1={m['recall@1']:.3f}  "
          f"faith={m['faithfulness']:.3f}  "
          f"rel={m['relevance']:.3f}  n={m['n']}")

print(f"\nBY DIFFICULTY:")
for d, m in sorted(results["by_difficulty"].items()):
    print(f"  {d:<8} r@1={m['recall@1']:.3f}  "
          f"faith={m['faith']:.3f}  "
          f"rel={m['rel']:.3f}  n={m['n']}")

# Append to RESULTS.md
import json
with open("RESULTS.md", "a", encoding="utf-8") as f:
    f.write(f"""
## M8 Pipeline Evaluation: {results['run_name']}

**Gold set:** {results['n_queries']} queries (v2) | **Total time:** {results['total_s']}s

| Metric | Score |
|--------|-------|
| recall@1 | {agg['recall@1']} |
| recall@3 | {agg['recall@3']} |
| nDCG@3 | {agg['ndcg@3']} |
| MRR | {agg['mrr']} |
| Faithfulness | {agg['faithfulness']} |
| Citation coverage | {agg['citation_coverage']} |
| Answer relevance | {agg['relevance']} |
| Negative decline rate | {agg['decline_rate']} |
""")
print("\nResults appended to RESULTS.md")