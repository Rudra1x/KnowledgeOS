# scripts/run_generation_eval.py

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
from eval.generation_evaluator        import GenerationEvaluator
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

# Pipeline
reranker  = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

class HybridReranked:
    def retrieve(self, query, top_k=5, tenant_id="default"):
        h = HybridRetriever(
            bm25_index=sparse_idx, dense_index=dense_idx,
            embedder=embedder, fetch_k=10,
        )
        candidates = h.retrieve(query, top_k=10, tenant_id=tenant_id)
        return reranker.rerank(query, copy.deepcopy(candidates), top_k=top_k)

generator = LocalLLMGenerator(max_tokens=200, temperature=0.0)
faith_checker = FaithfulnessChecker(
    strategy   = "nli",
    model_name = "cross-encoder/nli-MiniLM2-L6-H768",
    threshold  = 0.25,
)
rel_scorer = AnswerRelevanceScorer(
    embedder    = embedder,
    generator   = LocalLLMGenerator(max_tokens=100, temperature=0.3),
    n_questions = 1,
)

evaluator = GenerationEvaluator(
    retriever        = HybridReranked(),
    generator        = generator,
    faith_checker    = faith_checker,
    relevance_scorer = rel_scorer,
    top_k            = 3,
)

print(f"Running generation eval on {len(GOLD_SET_V2)} queries...\n")
results = evaluator.evaluate(GOLD_SET_V2)

# --- Summary ---
agg = results["aggregate"]
print("\n" + "=" * 60)
print("AGGREGATE RESULTS")
print("=" * 60)
print(f"  Faithfulness:      {agg['faithfulness']:.3f}")
print(f"  Citation coverage: {agg['citation_coverage']:.3f}")
print(f"  Answer relevance:  {agg['relevance']:.3f}")
print(f"  Decline rate:      {agg['decline_rate']}")
print(f"  Mean latency:      {agg['mean_latency_ms']:.0f}ms")

print("\nBY TYPE:")
for qt, metrics in results["by_type"].items():
    print(f"  {qt:<12} faith={metrics['faithfulness']:.3f}  "
          f"cov={metrics['citation_coverage']:.3f}  "
          f"rel={metrics['relevance']:.3f}  n={metrics['n']}")

# --- Per-query detail ---
print("\nPER-QUERY DETAIL:")
print(f"{'Query':<42} {'Faith':>6} {'Cov':>5} {'Rel':>5}")
print("-" * 60)
for q in results["per_query"]:
    if q["is_negative"]:
        status = "DECLINED" if q.get("declined") else "WRONG"
        print(f"{q['query'][:42]:<42} {'[NEG:'+status+']':>18}")
    else:
        print(f"{q['query'][:42]:<42} "
              f"{q['faithfulness']:>6.3f} "
              f"{q['citation_coverage']:>5.2f} "
              f"{q['relevance']:>5.2f}")