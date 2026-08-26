# eval/generation_evaluator.py

import time
import copy
from statistics import mean
from core import Retriever
from generation.local_generator      import LocalLLMGenerator
from generation.prompt_builder       import build_prompt, extract_citations
from generation.faithfulness_checker import FaithfulnessChecker
from generation.answer_relevance     import AnswerRelevanceScorer


class GenerationEvaluator:
    """
    Automated generation quality evaluation.

    Metrics per query:
    - faithfulness_score : float  fraction of claims supported by context
    - citation_coverage  : float  fraction of retrieved chunks actually cited
    - answer_relevance   : float  RAGAS reverse-question similarity
    - answered_negative  : bool   did the system correctly decline negative queries?
    - latency_ms         : float  generation time

    Aggregate metrics by query type:
    - mean faithfulness / relevance / citation coverage
    - negative query decline rate

    Parameters
    ----------
    retriever         : Retriever
    generator         : LocalLLMGenerator
    faith_checker     : FaithfulnessChecker
    relevance_scorer  : AnswerRelevanceScorer
    top_k             : int   chunks passed to generator
    """

    NOT_IN_CONTEXT_PHRASES = [
        "not in context",
        "does not contain",
        "no information",
        "not mentioned",
        "cannot find",
        "not found",
        "don't have information",
        "no relevant",
    ]

    def __init__(
        self,
        retriever:        Retriever,
        generator:        LocalLLMGenerator,
        faith_checker:    FaithfulnessChecker,
        relevance_scorer: AnswerRelevanceScorer,
        top_k:            int = 3,
    ):
        self.retriever        = retriever
        self.generator        = generator
        self.faith_checker    = faith_checker
        self.relevance_scorer = relevance_scorer
        self.top_k            = top_k

    def evaluate(self, gold_set: list[dict]) -> dict:
        """Run generation evaluation on a typed gold set."""
        per_query = []

        for i, item in enumerate(gold_set):
            query         = item["query"]
            relevant_text = item.get("relevant_text", "")
            query_type    = item.get("query_type", "factoid")
            is_negative   = not relevant_text

            print(f"  [{i+1:2d}/{len(gold_set)}] {query[:50]}...", end=" ", flush=True)
            t0 = time.perf_counter()

            # Retrieve
            candidates = self.retriever.retrieve(
                query, top_k=self.top_k * 2, tenant_id="default"
            )
            reranked = candidates[:self.top_k]

            # Generate
            answer = self.generator.generate(query, reranked)

            latency_ms = (time.perf_counter() - t0) * 1000

            if is_negative:
                # Did the system correctly decline?
                declined = self._is_decline(answer)
                per_query.append({
                    "query":            query,
                    "query_type":       query_type,
                    "is_negative":      True,
                    "answer":           answer,
                    "declined":         declined,
                    "latency_ms":       round(latency_ms, 1),
                    "faithfulness":     None,
                    "citation_coverage": None,
                    "relevance":        None,
                })
                status = "DECLINED" if declined else "WRONG (answered)"
                print(f"neg={status}")

            else:
                # Faithfulness
                faith  = self.faith_checker.check(answer, reranked)

                # Citation coverage: what fraction of retrieved chunks were cited?
                cited  = extract_citations(answer, reranked)
                cov    = len(cited) / max(len(reranked), 1)

                # Answer relevance
                rel    = self.relevance_scorer.score(query, answer)

                per_query.append({
                    "query":             query,
                    "query_type":        query_type,
                    "is_negative":       False,
                    "answer":            answer,
                    "faithfulness":      faith["score"],
                    "faith_claims":      len(faith["claims"]),
                    "unsupported":       faith["unsupported"],
                    "citation_coverage": round(cov, 3),
                    "citations_used":    len(cited),
                    "relevance":         rel["relevance_score"],
                    "relevance_verdict": rel["verdict"],
                    "latency_ms":        round(latency_ms, 1),
                })
                print(f"faith={faith['score']:.2f}  "
                      f"cov={cov:.2f}  "
                      f"rel={rel['relevance_score']:.2f}")

        return self._aggregate(per_query)

    def _is_decline(self, answer: str) -> bool:
        """Check if the answer correctly declines a negative query."""
        answer_lower = answer.lower()
        return any(phrase in answer_lower
                   for phrase in self.NOT_IN_CONTEXT_PHRASES)

    def _aggregate(self, per_query: list[dict]) -> dict:
        positive = [q for q in per_query if not q["is_negative"]]
        negative = [q for q in per_query if q["is_negative"]]

        def agg(items, key):
            vals = [i[key] for i in items if i.get(key) is not None]
            return round(mean(vals), 4) if vals else None

        # Per-type breakdown
        types   = {q["query_type"] for q in positive}
        by_type = {}
        for qt in types:
            subset = [q for q in positive if q["query_type"] == qt]
            by_type[qt] = {
                "n":                len(subset),
                "faithfulness":     agg(subset, "faithfulness"),
                "citation_coverage": agg(subset, "citation_coverage"),
                "relevance":        agg(subset, "relevance"),
            }

        decline_rate = (
            mean(1.0 if q["declined"] else 0.0 for q in negative)
            if negative else None
        )

        return {
            "aggregate": {
                "n_positive":        len(positive),
                "n_negative":        len(negative),
                "faithfulness":      agg(positive, "faithfulness"),
                "citation_coverage": agg(positive, "citation_coverage"),
                "relevance":         agg(positive, "relevance"),
                "decline_rate":      decline_rate,
                "mean_latency_ms":   agg(per_query, "latency_ms"),
            },
            "by_type":   by_type,
            "per_query": per_query,
        }