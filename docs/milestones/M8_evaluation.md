# Milestone 8 — Evaluation Harness

**Status:** ✅ Complete
**Duration:** 5 checkpoints
**Deliverable:** A complete automated evaluation framework — typed gold set, formal retrieval metrics (nDCG), generation quality metrics, unified pipeline evaluator, and a standalone HTML dashboard. The platform is now self-evaluating.

---

## 1. Milestone summary

### Goal
Build the automated evaluation infrastructure that makes KnowledgeOS continuously measurable. Replace ad-hoc benchmark scripts with a formal, typed, reproducible evaluation harness.

### Why this milestone matters
Without a formal evaluation harness, every architectural change requires manual inspection. With it, every change has a measurable impact: "switching from Vector to Hybrid+Rerank improved comparison query recall@1 from 0.333 to 0.667." The harness is what makes the platform scientifically defensible.

### What "done" looks like
- Typed gold set (factoid/comparison/thematic/negative) with difficulty annotations
- RetrievalEvaluator with nDCG, per-type breakdown, negative accuracy
- GenerationEvaluator with faithfulness, citation coverage, relevance, decline rate
- PipelineEvaluator unifying both in a single pass
- HTML report generator producing a shareable dashboard
- Final benchmark: r@1=0.800, faith=0.692, rel=0.822, decline=1.0

---

## 2. Architecture recap

### The evaluation stack

```
Gold set (typed, annotated)
         │
         ▼
PipelineEvaluator
         │
    ┌────┴────────────────────────┐
    │                            │
RetrievalEvaluator         GenerationEvaluator
    │                            │
recall@k, MRR,             faithfulness (NLI),
nDCG@k,                    citation coverage,
neg_accuracy,              answer relevance,
per-type breakdown         decline rate,
                           per-type breakdown
    │                            │
    └────────────┬───────────────┘
                 │
                 ▼
          Results dict
                 │
                 ▼
     HTML Report (standalone)
```

### The gold set schema

```python
{
    "query":           str   # the user's question
    "query_type":      str   # factoid | comparison | thematic | negative
    "relevant_text":   str   # substring of correct chunk ("" for negative)
    "expected_chunks": int   # how many chunks needed for complete answer
    "difficulty":      str   # easy | medium | hard
    "notes":           str   # what makes this query interesting
}
```

---

## 3. Technical deep dive

### 3.1 Query typing — the diagnosis layer

| Type | What it tests | Architecture implication of failure |
|------|--------------|-------------------------------------|
| Factoid | Basic retrieval — find one chunk | Embedding quality, chunk boundaries |
| Comparison | Cross-concept retrieval | Reranking, multi-hop, chunk size |
| Thematic | Synthesis across chunks | RAPTOR, larger context window |
| Negative | Robustness — don't hallucinate | System prompt, Self-RAG gate |

Without query typing, all failures look the same. With it, each failure points to a specific architectural fix.

### 3.2 nDCG — the position-aware metric

```
Recall@3: rank 1 = rank 3 = 1.0 (ignores position)
nDCG@3:   rank 1 = 1.000, rank 2 = 0.630, rank 3 = 0.500
```

The formula:
```
DCG@k  = Σ_{i=1}^{k} rel_i / log2(i+1)
IDCG@k = 1/log2(2) = 1.0  (perfect: relevant doc at rank 1)
nDCG@k = DCG@k / IDCG@k
```

For binary relevance (0 or 1): each query either finds the relevant chunk in top-k or doesn't. nDCG rewards early discovery — finding the relevant chunk at rank 1 scores 1.0, at rank 2 scores 0.630, at rank 3 scores 0.500.

**Why nDCG matters:** MRR already captures position (1/rank). nDCG is the standard for academic IR benchmarks (TREC, BEIR, MS MARCO) — using it makes KnowledgeOS results directly comparable to published work.

### 3.3 The three faithfulness failure modes

**Mode 1 — NLI false positive on paraphrased claims (most common):**
Answer: "Dense retrieval uses vector similarity to find semantically related content."
Source: "Dense retrieval uses neural embeddings to capture semantic meaning."
NLI: flags as unfaithful (paraphrase, not verbatim)
Word overlap: "Dense retrieval" + "semantic" overlap → catches this if ≥85%

**Mode 2 — Genuine hallucination (rare with good system prompt):**
Answer: "BM25 was developed by Robertson in 1994."
Source: BM25 chunk (doesn't mention Robertson or 1994)
NLI: correctly flags as unsupported (0.00)
Word overlap: "Robertson" not in source → catches this

**Mode 3 — Correct but not verbatim (hardest to handle):**
Answer: "Rerankers improve precision by re-scoring initial results."
Source: "Cross-encoder rerankers compare query and passage jointly to produce a relevance score."
NLI: low score (different vocabulary)
Word overlap: low (different vocabulary)
→ Both mechanisms fail → appears as false positive

Mode 3 requires LLM confirmation as a second pass.

### 3.4 The decline rate — testing negative robustness

```python
NOT_IN_CONTEXT_PHRASES = [
    "not in context", "does not contain", "no information",
    "not mentioned", "cannot find", "not found",
    "don't have", "no relevant",
]
```

Pattern matching works because the grounded system prompt produces predictable decline phrasings. When the model correctly follows "if not in context, say so," it uses standard phrases that this list catches reliably.

**Decline rate by architecture:**
- System prompt only: 1.0 for clean off-topic queries
- Without system prompt: ~0.0 (model answers from parametric memory)
- Self-RAG retrieve gate: 1.0 for parametric AND corpus-adjacent queries

### 3.5 The per-difficulty paradox explained

```
hard:   r@1=1.000  faith=0.000  rel=0.950
medium: r@1=0.667  faith=0.708  rel=0.853
easy:   r@1=1.000  faith=0.889  rel=0.719
```

**Hard queries:** cross-encoder reranker handles the hard comparison query perfectly (r@1=1.000). The comparative answer is rich and on-topic (rel=0.950). But the NLI checker flags the paraphrased comparative claim as unfaithful (faith=0.000). The system works; the evaluator is imperfect.

**Medium queries:** two retrieval misses where relevant content is embedded within larger paragraphs. The retrieval algorithm correctly finds the chunk but the answer to the specific question is not the primary focus of that chunk. Fix: smaller chunk size.

**Easy queries:** perfect retrieval, solid faithfulness, but lowest relevance (0.719). Simple one-sentence answers generate simple reverse questions that score lower on embedding similarity to the original. Relevance scorer underestimates quality of concise answers.

### 3.6 HTML report design principles

**Self-contained:** all CSS in `<style>`, no external resources. Works offline, firewall-safe.

**Color semantics:**
- Green (≥0.85): meeting production quality bar
- Yellow (0.60-0.85): acceptable but improvable
- Red (<0.60): requires attention

**Information hierarchy:** summary cards (highest abstraction) → by-type table → by-difficulty table → per-query detail (lowest abstraction). A hiring manager reads top-down; a debugger reads bottom-up.

**Dark theme:** professional appearance in portfolio context, reduces eye strain for extended review sessions.

---

## 4. Design decisions and trade-offs

### 4.1 12 queries vs 100 queries
12 queries is sufficient for comparing configurations on this corpus. With 12 queries and binary relevance, confidence intervals are wide — one query flip changes aggregate metrics by 10%. For production, 100+ queries with human-labeled relevance provides stable estimates. The architecture scales; the current gold set is representative, not rigorous.

### 4.2 nDCG with binary relevance vs graded relevance
We use binary relevance (0 or 1 per chunk) — a chunk either matches `relevant_text` or it doesn't. True graded relevance would label each retrieved chunk on a 0-3 scale (not relevant, marginally relevant, relevant, highly relevant). Binary is simpler and sufficient for single-answer queries; graded would add significant annotation cost for this project.

### 4.3 Synchronous pipeline evaluation
The evaluator runs each query sequentially: retrieve → generate → faith → relevance. GPU + async would reduce 16s/query to ~2s/query. For a portfolio benchmark running once, sequential is fine. For CI/CD on a 100-query gold set, async becomes necessary — the 100-query sequential run would take 26 minutes.

### 4.4 HTML report vs notebook
A notebook (`.ipynb`) would offer interactive charts and code cells for drill-down. But notebooks require Jupyter to render, have rendering inconsistencies, and embed large base64 images. The HTML report is simpler, portable, and self-contained. The per-query table in HTML is more scannable than a notebook cell output.

### 4.5 Phrase matching vs LLM classifier for decline detection
Phrase matching is O(1), deterministic, and sufficient for the standard grounded system prompt phrasings. An LLM classifier would handle edge cases (creative decline phrasings) but adds 2-3s latency per negative query. For 2 negative queries, phrase matching is clearly the right choice. For a larger negative query set with varied model outputs, an LLM classifier would be worth it.

---

## 5. Common pitfalls

1. **Not typing queries in the gold set** → aggregate metrics hide failure modes. Per-type breakdown is the most actionable insight from the evaluation harness.
2. **Using only recall@k without nDCG** → ignores retrieval position. A system that finds the right chunk at rank 3 every time looks identical to one that finds it at rank 1.
3. **Not including negative queries** → system appears perfect on questions the corpus can answer. Negative queries test the other half of robustness.
4. **Single-run LLM-based metrics** → high variance with n_questions=1. Report mean ± std from 3 runs for production quality claims.
5. **Not separating NLI false positives from genuine hallucinations** → faithfulness score becomes misleading. Track both "NLI flags" and "confirmed hallucinations" separately.
6. **Treating faithfulness=0.0 as definitive** → check relevance. If relevance is high and faithfulness is 0.0, it's likely an NLI false positive, not hallucination.
7. **Using absolute thresholds across domains** → medical RAG needs faithfulness >0.95; customer support RAG may tolerate 0.70. Calibrate thresholds to your domain and risk tolerance.
8. **Not versioning gold sets** → gold_set_v1 vs gold_set_v2 changes are invisible without version tracking. Always tag the gold set version in the evaluation results.
9. **Evaluating only the happy path** → add medium/hard difficulty and negative queries from the start. Gold sets that only contain easy queries produce misleadingly high scores.
10. **Running the full pipeline evaluator in CI for every commit** → 16s/query × 100 queries = 26 minutes. Use a fast retrieval-only eval (r@1 on 20 queries, <5s) for CI; run the full pipeline eval nightly.

---

## 6. Final benchmark results

### M8 Pipeline Evaluation: hybrid_rerank_v1

| Metric | Score | Interpretation |
|--------|-------|----------------|
| recall@1 | 0.800 | 2 medium-difficulty misses |
| recall@3 | 0.800 | Same misses — chunk boundary issue |
| nDCG@3 | 0.800 | Position-aware score matches recall |
| MRR | 0.800 | Mean reciprocal rank |
| Faithfulness | 0.692 | NLI false positives on paraphrased claims |
| Citation coverage | 0.300 | Model cites ~1 of 3 chunks |
| Answer relevance | 0.822 | Answers are clearly on-topic |
| Decline rate | 1.000 | Perfect negative query handling |

**By type:**
| Type | recall@1 | Faithfulness | Relevance |
|------|----------|-------------|-----------|
| factoid | 0.800 | 0.733 | 0.767 |
| comparison | 0.667 | 0.667 | 0.855 |
| thematic | 1.000 | 0.625 | 0.912 |

**Key interpretations:**
- The two retrieval misses are both medium-difficulty — relevant content embedded within larger paragraphs rather than being primary chunk topics. Fix: smaller chunk_size (256 instead of 512).
- Faithfulness false positives cluster on comparative and synthesis answers. These answers are correct but paraphrased — NLI underscores paraphrase. Fix: calibrate NLI threshold or add LLM second-pass.
- Answer relevance is strong across all types — the grounded prompt produces on-topic answers.
- Decline rate=1.0 confirms the system prompt handles negative queries without Self-RAG overhead.

---

## 7. Interview mock exam

### Section A — Fundamentals (10 questions)

1. What are the four query types in your gold set and what does each test?
2. What is nDCG and how does it differ from recall@k and MRR?
3. What is the formula for nDCG@k with binary relevance?
4. What does citation coverage measure and what does 0.366 mean?
5. What is decline rate and how do you detect it automatically?
6. Why do negative queries require special handling in a retrieval evaluator?
7. What three things does a single row in the PipelineEvaluator per-query table tell you?
8. Why is HTML the right format for a portfolio evaluation report?
9. What is the role of difficulty annotations in a gold set?
10. What's the difference between the RetrievalEvaluator and PipelineEvaluator?

### Section B — Applied Understanding (15 questions)

11. Hard queries scored r@1=1.000 but faith=0.000. Is the system working or broken?
12. Medium queries have the worst retrieval (r@1=0.667). What's the root cause?
13. Your neg_acc=0.0 for deterministic retrievers. Is this a failure?
14. Comparison queries scored r@1=0.333 with Vector and 0.667 with Hybrid+Rerank. What explains the improvement?
15. Thematic queries have the best relevance (0.912) and worst faithfulness (0.500). Why?
16. Easy queries have perfect retrieval (r@1=1.000) but lowest relevance (0.719). Why?
17. Your citation coverage is 0.366 — 1 of 3 chunks cited. Is this appropriate for factoid queries?
18. The decline detection uses phrase matching instead of an LLM classifier. When would you upgrade to LLM?
19. nDCG@3=0.800 and recall@3=0.800 are identical. What does that tell you about where the correct chunks are ranked?
20. You run the pipeline evaluator twice and get faith=0.692 vs 0.667. What causes the variance?
21. A query fails on retrieval (r@1=0) and has high relevance (0.850). How is that possible?
22. You add a new retriever to the benchmark. What changes in the evaluation code?
23. Your BEIR benchmark uses nDCG@10. Your gold set uses nDCG@3. Why the difference?
24. Two queries have identical r@1=0 but different nDCG@3 (0.500 vs 0.000). What happened?
25. The HTML report shows faithfulness=yellow for comparison queries. What does that mean operationally?

### Section C — Design and Trade-offs (10 questions)

26. Design a CI/CD evaluation pipeline that runs on every pull request without taking 26 minutes.
27. How would you expand the gold set to 100 queries while maintaining type balance and difficulty distribution?
28. A client says "your faithfulness score is 69% — that's unacceptable." Design a remediation plan.
29. Compare your evaluation harness to the RAGAS framework. What does RAGAS add?
30. Design a multi-configuration A/B evaluation: Vector vs Hybrid+Rerank. What's your decision criterion?
31. Your gold set has 0 hard thematic queries. How does this affect the validity of your thematic evaluation?
32. Design a continuous monitoring system that runs the pipeline evaluator daily and alerts on quality regression.
33. Your faithfulness false positive rate is 30%. Design a calibration procedure to reduce it to <10%.
34. A stakeholder wants a single number summarizing KnowledgeOS quality. How do you compute it?
35. You're onboarding a new corpus (50K legal documents). How do you build the gold set for it?

### Section D — Whiteboard Coding (5 questions)

36. Implement `ndcg_at_k(retrieved: list, relevant_text: str, k: int) -> float` for binary relevance.
37. Implement `RetrievalEvaluator._aggregate(per_query) -> dict` — compute mean metrics across positive and negative queries.
38. Implement `PipelineEvaluator._is_decline(answer: str) -> bool` using phrase matching with the NOT_IN_CONTEXT_PHRASES list.
39. Write pseudocode for a CI evaluation script that compares current results to a baseline JSON and fails if any metric drops >5%.
40. Implement `generate_report(results: dict, output_path: str) -> str` — describe the HTML structure and the minimum sections required.

---

## 8. Project walkthrough scripts

### 8.1 The 30-second pitch

> "M8 added a formal evaluation harness to KnowledgeOS — typed gold set with factoid, comparison, thematic, and negative queries; RetrievalEvaluator with nDCG and per-type breakdown; GenerationEvaluator measuring faithfulness, citation coverage, and answer relevance; a unified PipelineEvaluator; and an HTML dashboard for sharing results. Final benchmark: recall@1=0.800, faithfulness=0.692, relevance=0.822, decline_rate=1.0. The platform is now self-evaluating — every architecture change has a measurable impact."

### 8.2 The 2-minute technical walkthrough

> "M8 has three main pieces. The typed gold set: 12 queries across factoid, comparison, thematic, and negative types with easy/medium/hard difficulty annotations. Query typing is the most valuable addition — it turns 'recall@1=0.800' into 'factoid r@1=0.800, comparison r@1=0.667, thematic r@1=1.000' — completely different diagnostic information.
>
> The evaluation metrics: nDCG@3 alongside recall and MRR — it rewards finding the correct chunk at rank 1 over rank 3, matching academic IR benchmarks. Faithfulness via NLI (176ms), citation coverage (fraction of chunks cited), answer relevance via RAGAS reverse-question generation, and decline rate for negative queries.
>
> The unified pipeline evaluator runs one pass: retrieve + generate + faith + relevance per query. The hard query paradox was the key finding: r@1=1.000 but faith=0.000 on the dense/BM25 comparison query — perfect retrieval, most relevant answer (0.950), but NLI false positive on paraphrased comparative claim. Without per-difficulty breakdown, this would be invisible. The HTML report makes all of this shareable — standalone, color-coded, no dependencies."

### 8.3 The 5-minute deep walkthrough

> "Let me walk through four areas: query typing, the nDCG metric, the failure mode analysis, and the HTML report design.
>
> **Query typing.** The original gold set was 10 identical factoid queries. You can get recall@1=0.900 on 10 factoid queries and still have a system that completely fails on comparison or thematic queries. The four-type gold set forces the evaluation to cover different retrieval challenges. Factoid tests basic retrieval — can you find the right chunk? Comparison tests cross-concept retrieval — can you find chunks about two related concepts and contrast them? Thematic tests synthesis. Negative tests whether you know what you don't know. Each type points to a different architecture fix on failure.
>
> **nDCG.** nDCG@3 = DCG@3 / IDCG@3 where DCG@3 = Σ rel_i / log2(i+1). For binary relevance, the formula simplifies: if the correct chunk is at rank 1, nDCG=1.0; rank 2 = 0.630; rank 3 = 0.500. It's the standard metric in TREC, BEIR, and MS MARCO benchmarks — using it makes KnowledgeOS results directly comparable to published work.
>
> **The failure mode analysis.** Three findings from the per-type and per-difficulty breakdowns. First: comparison queries r@1=0.333 → 0.667 with Hybrid+Rerank — the cross-encoder reranker specifically helps cross-concept queries. Second: the hard query paradox — r@1=1.000, rel=0.950, faith=0.000. Perfect retrieval, richest answer, NLI false positive. The paradox is only visible in the per-difficulty table. Third: medium-difficulty retrieval failures (r@1=0.667) are both chunk boundary issues — the relevant content is embedded within larger paragraphs. Fix: chunk_size=256 instead of 512.
>
> **The HTML report.** Zero dependencies — all CSS inline, no JavaScript frameworks. Color coding encodes domain knowledge: green ≥0.85 (production quality), yellow 0.60-0.85 (acceptable), red <0.60 (needs work). Dark theme for professional appearance. Information hierarchy: summary cards → type/difficulty tables → per-query detail. The report is the artifact that makes the project demonstrable without running code — send it to a hiring manager or client and they can evaluate KnowledgeOS quality without installation."

---

## 9. Further reading

- **Clarke et al., 2009** — "Overview of the TREC 2009 Web Track." TREC evaluation methodology including nDCG.
- **Thakur et al., 2021** — "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." The 18-dataset retrieval benchmark that uses nDCG@10.
- **Es et al., 2023** — "RAGAS: Automated Evaluation of Retrieval Augmented Generation." The framework our answer relevance scorer is based on.
- **Järvelin & Kekäläinen, 2002** — "Cumulated Gain-based Evaluation of IR Techniques." The original nDCG paper.
- **Tenney et al., 2019** — "What do you learn from context? Probing for sentence structure in contextualized word representations." Background on why NLI models struggle with near-verbatim hypotheses.

---

## Milestone status

- [x] 8.1 — Gold set v2 (12 typed queries, difficulty annotations, negative queries)
- [x] 8.2 — RetrievalEvaluator (nDCG, per-type, neg_accuracy)
- [x] 8.3 — GenerationEvaluator (faithfulness, citation coverage, relevance, decline rate)
- [x] 8.4 — PipelineEvaluator (unified single-pass, per-type + per-difficulty)
- [x] 8.5 — HTML report generator (standalone dashboard, color-coded, shareable)

**Resume line (updated with M8, complete project):**

> *Completed KnowledgeOS M8 evaluation harness: typed gold set (factoid/comparison/thematic/negative × easy/medium/hard), RetrievalEvaluator with nDCG@k and per-type breakdown, GenerationEvaluator with NLI faithfulness + RAGAS relevance + citation coverage + negative decline detection, PipelineEvaluator unifying both in a single pass, and standalone HTML dashboard. Final benchmark: recall@1=0.800, nDCG@3=0.800, faithfulness=0.692, answer relevance=0.822, decline_rate=1.000. Key findings: comparison query r@1 improved 0.333→0.667 with cross-encoder reranking; medium-difficulty retrieval failures trace to chunk boundary artifacts; hard query paradox (r@1=1.000, rel=0.950, faith=0.000) reveals NLI false positive on paraphrased comparative claims. Eight-milestone RAG platform complete.*
