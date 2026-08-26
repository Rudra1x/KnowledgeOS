# scripts/generate_eval_report.py

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
from eval.report_generator            import generate_report
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

class HybridReranked:
    def retrieve(self, query, top_k=5, tenant_id="default"):
        h = HybridRetriever(
            bm25_index=sparse_idx, dense_index=dense_idx,
            embedder=embedder, fetch_k=10,
        )
        cands = h.retrieve(query, top_k=10, tenant_id=tenant_id)
        return reranker.rerank(query, copy.deepcopy(cands), top_k=top_k)

evaluator = PipelineEvaluator(
    retriever        = HybridReranked(),
    generator        = LocalLLMGenerator(max_tokens=200, temperature=0.0),
    faith_checker    = FaithfulnessChecker(
        strategy="nli",
        model_name="cross-encoder/nli-MiniLM2-L6-H768",
        threshold=0.25,
    ),
    relevance_scorer = AnswerRelevanceScorer(
        embedder    = embedder,
        generator   = LocalLLMGenerator(max_tokens=100, temperature=0.3),
        n_questions = 1,
    ),
    retrieval_top_k  = 5,
    generation_top_k = 3,
)

print(f"Running evaluation ({len(GOLD_SET_V2)} queries)...\n")
results  = evaluator.evaluate(GOLD_SET_V2, run_name="hybrid_rerank_v1")
out_path = generate_report(results, "eval_report.html")
print(f"\nReport saved: {out_path}")