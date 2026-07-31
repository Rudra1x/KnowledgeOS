# eval/runner.py

import time
from statistics import mean
from core import Retriever, Generator
from .gold_set import GOLD_SET
from .metrics  import recall_at_k, mean_reciprocal_rank, faithfulness_check


def evaluate(
    retriever:      Retriever,
    generator:      Generator | None = None,
    k_values:       list[int] = [1, 3, 5],
    run_generation: bool      = False,
    gen_delay_sec:  float     = 2.0,
) -> dict:
    results = {f"recall@{k}": [] for k in k_values}
    results["mrr"]          = []
    results["faithfulness"] = []

    per_query = []

    for item in GOLD_SET:
        query         = item["query"]
        relevant_text = item["relevant_text"]

        retrieved = retriever.retrieve(query, top_k=max(k_values), tenant_id="default")

        row = {"query": query}
        for k in k_values:
            score = recall_at_k(retrieved, relevant_text, k)
            results[f"recall@{k}"].append(score)
            row[f"r@{k}"] = score

        mrr = mean_reciprocal_rank(retrieved, relevant_text)
        results["mrr"].append(mrr)
        row["mrr"] = mrr

        if run_generation and generator is not None:
            time.sleep(gen_delay_sec)
            try:
                answer = generator.generate(query, retrieved[:3])
                faith  = faithfulness_check(answer, retrieved[:3])
                results["faithfulness"].append(faith)
                row["faith"] = faith
            except Exception as e:
                print(f"  [generation failed]: {str(e)[:100]}")
                row["faith"] = None

        per_query.append(row)

    aggregated = {metric: mean(scores) if scores else 0.0
                  for metric, scores in results.items()}

    return {"aggregated": aggregated, "per_query": per_query}


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print("PER-QUERY RESULTS")
    print("=" * 60)
    for row in report["per_query"]:
        print(row)

    print("\n" + "=" * 60)
    print("AGGREGATED METRICS")
    print("=" * 60)
    for metric, score in report["aggregated"].items():
        print(f"  {metric:15s} : {score:.3f}")
    print()