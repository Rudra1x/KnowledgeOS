# eval/pipeline_evaluator.py

import time
import copy
from core import Retriever
from generation.local_generator       import LocalLLMGenerator
from generation.faithfulness_checker  import FaithfulnessChecker
from generation.answer_relevance      import AnswerRelevanceScorer
from eval.retrieval_evaluator         import RetrievalEvaluator
from eval.generation_evaluator        import GenerationEvaluator


class PipelineEvaluator:
    """
    End-to-end pipeline evaluator combining retrieval + generation metrics.

    Runs a single pass over the gold set, collecting:
    - Retrieval: recall@k, MRR, nDCG@k, negative accuracy
    - Generation: faithfulness, citation coverage, answer relevance, decline rate
    - Combined: per-query table with all metrics together

    Parameters
    ----------
    retriever        : Retriever
    generator        : LocalLLMGenerator
    faith_checker    : FaithfulnessChecker
    relevance_scorer : AnswerRelevanceScorer
    retrieval_top_k  : int   candidates to fetch for retrieval eval
    generation_top_k : int   chunks passed to generator
    tenant_id        : str
    """

    def __init__(
        self,
        retriever:        Retriever,
        generator:        LocalLLMGenerator,
        faith_checker:    FaithfulnessChecker,
        relevance_scorer: AnswerRelevanceScorer,
        retrieval_top_k:  int = 5,
        generation_top_k: int = 3,
        tenant_id:        str = "default",
    ):
        self.retriever        = retriever
        self.generator        = generator
        self.faith_checker    = faith_checker
        self.relevance_scorer = relevance_scorer
        self.retrieval_top_k  = retrieval_top_k
        self.generation_top_k = generation_top_k
        self.tenant_id        = tenant_id

    def evaluate(
        self,
        gold_set:    list[dict],
        run_name:    str = "pipeline",
        verbose:     bool = True,
    ) -> dict:
        """
        Full pipeline evaluation on a gold set.

        Returns unified results dict with retrieval + generation metrics.
        """
        from eval.metrics import (
            recall_at_k, mean_reciprocal_rank, ndcg_at_k
        )
        from generation.prompt_builder import extract_citations

        per_query  = []
        t_start    = time.perf_counter()

        for i, item in enumerate(gold_set):
            query         = item["query"]
            relevant_text = item.get("relevant_text", "")
            query_type    = item.get("query_type", "factoid")
            difficulty    = item.get("difficulty", "medium")
            is_negative   = not relevant_text

            if verbose:
                print(f"  [{i+1:2d}/{len(gold_set)}] [{query_type:10s}] "
                      f"{query[:45]}...", end=" ", flush=True)

            t0 = time.perf_counter()

            # --- Retrieval ---
            candidates = self.retriever.retrieve(
                query,
                top_k     = self.retrieval_top_k,
                tenant_id = self.tenant_id,
            )
            reranked_for_gen = candidates[:self.generation_top_k]

            # --- Generation ---
            answer = self.generator.generate(query, reranked_for_gen)

            latency_ms = (time.perf_counter() - t0) * 1000

            if is_negative:
                declined = self._is_decline(answer)
                row = {
                    "query":          query,
                    "query_type":     query_type,
                    "difficulty":     difficulty,
                    "is_negative":    True,
                    "answer":         answer,
                    "declined":       declined,
                    "latency_ms":     round(latency_ms, 1),
                    # Retrieval metrics — N/A for negative
                    "recall@1": None, "recall@3": None,
                    "mrr": None,      "ndcg@3": None,
                    # Generation metrics — N/A for negative
                    "faithfulness": None, "citation_coverage": None,
                    "relevance": None,
                }
                if verbose:
                    status = "DECLINED" if declined else "ANSWERED(wrong)"
                    print(f"neg={status}")

            else:
                # Retrieval metrics
                r1    = recall_at_k(candidates, relevant_text, 1)
                r3    = recall_at_k(candidates, relevant_text, 3)
                mrr   = mean_reciprocal_rank(candidates, relevant_text)
                ndcg3 = ndcg_at_k(candidates, relevant_text, 3)

                # Faithfulness
                faith = self.faith_checker.check(answer, reranked_for_gen)

                # Citation coverage
                cited = extract_citations(answer, reranked_for_gen)
                cov   = len(cited) / max(len(reranked_for_gen), 1)

                # Answer relevance
                rel   = self.relevance_scorer.score(query, answer)

                row = {
                    "query":             query,
                    "query_type":        query_type,
                    "difficulty":        difficulty,
                    "is_negative":       False,
                    "answer":            answer,
                    "latency_ms":        round(latency_ms, 1),
                    # Retrieval
                    "recall@1":          r1,
                    "recall@3":          r3,
                    "mrr":               mrr,
                    "ndcg@3":            ndcg3,
                    # Generation
                    "faithfulness":      faith["score"],
                    "unsupported":       faith["unsupported"],
                    "citation_coverage": round(cov, 3),
                    "citations_used":    len(cited),
                    "relevance":         rel["relevance_score"],
                    "relevance_verdict": rel["verdict"],
                }
                if verbose:
                    print(f"r@1={r1:.0f}  "
                          f"faith={faith['score']:.2f}  "
                          f"rel={rel['relevance_score']:.2f}")

            per_query.append(row)

        total_s = time.perf_counter() - t_start

        return {
            "run_name":    run_name,
            "total_s":     round(total_s, 1),
            "n_queries":   len(gold_set),
            "aggregate":   self._aggregate(per_query),
            "by_type":     self._by_type(per_query),
            "by_difficulty": self._by_difficulty(per_query),
            "per_query":   per_query,
        }

    # ------------------------------------------------------------------

    NOT_IN_CONTEXT_PHRASES = [
        "not in context", "does not contain", "no information",
        "not mentioned",  "cannot find",      "not found",
        "don't have",     "no relevant",
    ]

    def _is_decline(self, answer: str) -> bool:
        return any(p in answer.lower() for p in self.NOT_IN_CONTEXT_PHRASES)

    def _aggregate(self, rows: list[dict]) -> dict:
        from statistics import mean
        pos = [r for r in rows if not r["is_negative"]]
        neg = [r for r in rows if r["is_negative"]]

        def avg(items, key):
            vals = [i[key] for i in items
                    if i.get(key) is not None]
            return round(mean(vals), 4) if vals else None

        return {
            "recall@1":          avg(pos, "recall@1"),
            "recall@3":          avg(pos, "recall@3"),
            "mrr":               avg(pos, "mrr"),
            "ndcg@3":            avg(pos, "ndcg@3"),
            "faithfulness":      avg(pos, "faithfulness"),
            "citation_coverage": avg(pos, "citation_coverage"),
            "relevance":         avg(pos, "relevance"),
            "decline_rate":      (mean(1.0 if r["declined"] else 0.0
                                       for r in neg) if neg else None),
            "mean_latency_ms":   avg(rows, "latency_ms"),
        }

    def _by_type(self, rows: list[dict]) -> dict:
        from statistics import mean
        pos    = [r for r in rows if not r["is_negative"]]
        types  = {r["query_type"] for r in pos}
        result = {}
        for qt in types:
            sub = [r for r in pos if r["query_type"] == qt]
            def avg(key):
                vals = [r[key] for r in sub if r.get(key) is not None]
                return round(mean(vals), 4) if vals else None
            result[qt] = {
                "n": len(sub),
                "recall@1":     avg("recall@1"),
                "ndcg@3":       avg("ndcg@3"),
                "faithfulness": avg("faithfulness"),
                "relevance":    avg("relevance"),
            }
        return result

    def _by_difficulty(self, rows: list[dict]) -> dict:
        from statistics import mean
        pos    = [r for r in rows if not r["is_negative"]]
        diffs  = {r["difficulty"] for r in pos}
        result = {}
        for d in diffs:
            sub = [r for r in pos if r["difficulty"] == d]
            def avg(key):
                vals = [r[key] for r in sub if r.get(key) is not None]
                return round(mean(vals), 4) if vals else None
            result[d] = {
                "n":        len(sub),
                "recall@1": avg("recall@1"),
                "ndcg@3":   avg("ndcg@3"),
                "faith":    avg("faithfulness"),
                "rel":      avg("relevance"),
            }
        return result