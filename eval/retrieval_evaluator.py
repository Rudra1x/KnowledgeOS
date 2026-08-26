# eval/retrieval_evaluator.py

import time
import copy
from statistics import mean
from core import Retriever
from eval.metrics import recall_at_k, mean_reciprocal_rank, ndcg_at_k


class RetrievalEvaluator:
    """
    Formal retrieval evaluation framework.

    Evaluates any Retriever on a typed gold set.
    Reports per-query and aggregate metrics:
    - recall@1, recall@3, recall@5
    - MRR (Mean Reciprocal Rank)
    - nDCG@3, nDCG@5
    - Per-type breakdown (factoid / comparison / thematic / negative)
    - Negative query handling (precision at returning nothing)

    Parameters
    ----------
    retriever  : Retriever   any retriever from M5 portfolio
    top_k      : int         max candidates to retrieve
    tenant_id  : str
    """

    def __init__(
        self,
        retriever:  Retriever,
        top_k:      int = 5,
        tenant_id:  str = "default",
    ):
        self.retriever  = retriever
        self.top_k      = top_k
        self.tenant_id  = tenant_id

    def evaluate(self, gold_set: list[dict]) -> dict:
        """
        Run evaluation on a gold set.

        Returns a results dict with per-query details and aggregate metrics.
        """
        per_query = []

        for item in gold_set:
            query         = item["query"]
            relevant_text = item.get("relevant_text", "")
            query_type    = item.get("query_type", "factoid")
            is_negative   = not relevant_text

            t0        = time.perf_counter()
            retrieved = self.retriever.retrieve(
                query, top_k=self.top_k, tenant_id=self.tenant_id
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            if is_negative:
                # For negative queries: correct = returning nothing
                # Wrong = returning chunks (no answer in corpus)
                correct_negative = len(retrieved) == 0
                per_query.append({
                    "query":             query,
                    "query_type":        query_type,
                    "is_negative":       True,
                    "correct_negative":  correct_negative,
                    "n_retrieved":       len(retrieved),
                    "latency_ms":        round(latency_ms, 1),
                    "recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0,
                    "mrr": 0.0, "ndcg@3": 0.0, "ndcg@5": 0.0,
                })
            else:
                per_query.append({
                    "query":       query,
                    "query_type":  query_type,
                    "is_negative": False,
                    "recall@1":    recall_at_k(retrieved, relevant_text, 1),
                    "recall@3":    recall_at_k(retrieved, relevant_text, 3),
                    "recall@5":    recall_at_k(retrieved, relevant_text, 5),
                    "mrr":         mean_reciprocal_rank(retrieved, relevant_text),
                    "ndcg@3":      ndcg_at_k(retrieved, relevant_text, 3),
                    "ndcg@5":      ndcg_at_k(retrieved, relevant_text, 5),
                    "latency_ms":  round(latency_ms, 1),
                    "n_retrieved": len(retrieved),
                })

        return self._aggregate(per_query)

    def _aggregate(self, per_query: list[dict]) -> dict:
        """Compute aggregate + per-type metrics."""
        positive = [q for q in per_query if not q["is_negative"]]
        negative = [q for q in per_query if q["is_negative"]]

        def agg(items, metric):
            vals = [i[metric] for i in items if metric in i]
            return round(mean(vals), 4) if vals else 0.0

        # Per-type breakdown
        types       = {q["query_type"] for q in positive}
        by_type     = {}
        for qt in types:
            subset   = [q for q in positive if q["query_type"] == qt]
            by_type[qt] = {
                "n":        len(subset),
                "recall@1": agg(subset, "recall@1"),
                "recall@3": agg(subset, "recall@3"),
                "mrr":      agg(subset, "mrr"),
                "ndcg@3":   agg(subset, "ndcg@3"),
            }

        # Negative query handling
        neg_accuracy = (
            mean(1.0 if q["correct_negative"] else 0.0 for q in negative)
            if negative else None
        )

        return {
            "aggregate": {
                "n_positive":  len(positive),
                "n_negative":  len(negative),
                "recall@1":    agg(positive, "recall@1"),
                "recall@3":    agg(positive, "recall@3"),
                "recall@5":    agg(positive, "recall@5"),
                "mrr":         agg(positive, "mrr"),
                "ndcg@3":      agg(positive, "ndcg@3"),
                "ndcg@5":      agg(positive, "ndcg@5"),
                "mean_latency_ms": agg(per_query, "latency_ms"),
                "neg_accuracy": neg_accuracy,
            },
            "by_type":   by_type,
            "per_query": per_query,
        }