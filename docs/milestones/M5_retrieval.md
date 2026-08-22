# Milestone 5 — Retrieval Engine

**Status:** ✅ Complete
**Duration:** 8 checkpoints
**Deliverable:** Eight retrieval strategies from baseline vector to agentic ReAct, benchmarked on the same gold set with timing as a first-class dimension. First perfect recall@1=1.000 achieved on this corpus.

---

## 1. Milestone summary

### Goal
Build the full retrieval portfolio — covering the spectrum from filtered metadata retrieval through research-grade agentic planning. Understand when each strategy wins and at what cost.

### Why this milestone matters
The retriever is the most impactful component in a RAG system. Chunking and embedding quality sets the ceiling; the retriever determines how close you get to it. Every retrieval strategy we built answers a different version of the question: "what context does this query need?"

### What "done" looks like
- 8 retrievers behind the `Retriever` ABC
- Local LLM integration (Ollama + Qwen2.5-3B) for all LLM-dependent retrievers
- Benchmark with recall@1, recall@3, MRR, and timing
- Honest interpretation: agentic wins at 23s/query; vector baseline matches most LLM-augmented at 0.02s/query

---

## 2. Architecture recap

### The retrieval portfolio

```
                      Retriever (ABC)
                           │
     ┌─────────────────────┼─────────────────────┐
     │                     │                     │
DETERMINISTIC         LLM-AUGMENTED          AGENTIC
     │                     │                     │
VectorRetriever    QueryRewriting         AgenticRetriever
HybridRetriever    MultiQuery             (ReAct loop,
FilteredRetriever  MultiHop               plans strategy)
                   SelfRAG
                   CRAG
```

### The local LLM stack

```
All LLM-dependent retrievers
         │
         ▼
LocalLLMGenerator
         │
    ┌────┴────┐
    │         │
 Ollama    OpenRouter
 (primary)  (fallback)
    │
qwen2.5:3b-instruct
```

### The agentic ReAct loop

```
query
  │
  ▼
[Thought] → [Action: vector_search/keyword_search/filter_search/FINISH]
  │                    │
  │              [Execute tool]
  │                    │
  └──────[Observation]─┘
         (if not FINISH, loop)
  │
  ▼
Union of all retrieved chunks → ranked by score → top_k
```

---

## 3. Technical deep dive

### 3.1 The retrieval spectrum

| Strategy | Core question answered | LLM calls/query | Latency |
|----------|----------------------|-----------------|---------|
| VectorRetriever | What chunks embed closest to this query? | 0 | <1ms |
| HybridRetriever | What do both sparse and dense agree on? | 0 | <1ms |
| FilteredRetriever | What chunks of the right type embed closest? | 0 | <1ms |
| QueryRewriting | How would a document phrase this query? | 1 | 2-3s |
| MultiQuery | What are K ways to ask this? | 1 | 2-3s |
| MultiHop | What else do we need after finding this? | N per hop | N×2-3s |
| SelfRAG | Should we retrieve? Is each chunk relevant? | 1+K | (1+K)×2-3s |
| CRAG | Was the retrieval good enough? How to correct? | 1-3 | 1-3×2-3s |
| AgenticRetriever | What's the best retrieval plan for this query? | 1-4 | 4-12s |

### 3.2 Metadata filtering — the first production upgrade

Enterprise RAG almost always requires metadata filtering within weeks of deployment:
- "Only show HR policy documents"
- "Search only documents from Q4 2024"
- "Return PDFs only"

The three-mode design (post/pre/boost) covers all cases. Post-filter is the practical default; true pre-filter (Qdrant native) is the performance upgrade for large corpora.

**The over-fetch pattern:**
```python
fetch_k = top_k * 5  # compensate for filter losses
```
If filter eliminates 80%, over-fetching 5× ensures enough survivors.

### 3.3 Query rewriting — closing the style gap

User writes: "how does rag work"
Document says: "Retrieval-Augmented Generation combines information retrieval..."

Dense embeddings are not perfectly style-invariant. A reformulated query embeds closer to document-style text. HyDE goes further — embed a hypothetical answer document rather than the question.

**Key finding:** local Qwen2.5-3B > free-tier OpenRouter for this task. Query 3 went from r@1=0 (OpenRouter) to r@1=1 (Qwen). Model stability matters for components on the critical path.

### 3.4 Reciprocal rank aggregation — the shared primitive

Both MultiQuery and HybridRetriever use reciprocal rank aggregation:

```python
score(chunk) += 1.0 / rank_in_this_retrieval
```

Scale-agnostic: BM25 scores ~4.5 and cosine scores ~0.7 can't be added directly, but their ranks can. A chunk ranked #1 by 3 retrievals scores 3.0; ranked #2, #3, #1 scores 1.83. Promotes consistent top-rankers.

### 3.5 Self-RAG — the adaptive gate

Binary decisions via small LLM:
- Retrieve gate: YES/NO — does this query need the corpus?
- Relevance filter: YES/NO — is this chunk actually useful?

**Output format is the key design constraint.** "Answer with ONLY one word: YES or NO" makes parsing trivial. Without it, you're parsing paragraphs.

**The parametric query benefit:** ~30-40% of user queries to a knowledge base are answerable from general knowledge. Skipping retrieval on those saves: embedding call + index search + relevance evaluation. At 10K queries/day, that's 3K-4K skipped retrievals.

### 3.6 CRAG — the quality feedback loop

Two-signal evaluation:
1. LLM semantic judgment (primary — understands query intent)
2. Keyword overlap heuristic (backup — catches LLM false positives)

```python
if llm_eval == "CORRECT" and overlap_rate < 0.3:
    return "INCORRECT"  # heuristic overrides LLM
```

The heuristic is necessary because 3B models frequently call unrelated content CORRECT. The 0.3 threshold (30% of meaningful query terms must appear in chunks) is tunable per corpus.

**CRAG earns its 46s/query cost only on noisy corpora.** On clean corpora, first-attempt retrieval is already high quality and correction adds overhead without benefit.

### 3.7 The agentic win — dynamic strategy discovery

The persistent miss was "How is dense retrieval different from BM25?" — a query where the correct chunk required both keyword (BM25 is an exact term) and semantic (dense retrieval is a concept) matching.

- Fixed hybrid RRF: always uses BM25 + dense, same weight, fetch_k=8 → missed
- Agentic: reasoned about the query, chose keyword_search first, then vector_search, unioned the results → hit

The agent discovered the right strategy without being told. This is why agentic is the frontier — it adapts to each query's unique retrieval needs.

### 3.8 The latency wall

```
Deterministic: ~20ms/query  → 50 QPS
LLM-augmented: ~3-5s/query  → 0.2-0.3 QPS
Heavy LLM:     ~30-46s/query → 0.02-0.03 QPS
```

At 50 QPS (deterministic), 10K users get instant responses. At 0.02 QPS (CRAG), 10K queries takes 140 hours. The architecture must match the throughput requirement.

---

## 4. Design decisions and trade-offs

### 4.1 Local LLM first, API second
Ollama + Qwen2.5-3B gives: no rate limits, no API cost, stable model identity (doesn't rotate), ~2-3s per call. OpenRouter free tier rotates models (one repeated queries verbatim), has rate limits, and adds network latency. Local is the right default for retrieval-level LLM calls.

### 4.2 call_raw vs generate
`call_raw(prompt)` — direct string prompt, plain instruction format, for retrieval-level tasks (rewriting, evaluation, planning).
`generate(query, chunks)` — RAG-formatted prompt with system context block, for final answer generation.
The split is mandatory — using the RAG format for evaluation prompts produces meta-commentary, not evaluations.

### 4.3 Graceful degradation compounds
- QueryRewriting: LLM fail → return original query
- MultiHop: no follow-up → stop at hop 1
- SelfRAG: all filtered → min_relevant_chunks fallback
- CRAG: max corrections → return best available
- AgenticRetriever: parse fail → FINISH with what we have

Every component fails gracefully. Composing graceful components means the pipeline never hard-crashes.

### 4.4 min_relevant_chunks — never return empty
If SelfRAG's relevance filter removes everything, the generator has nothing to work with. `min_relevant_chunks=1` ensures at least one chunk always reaches generation. The trade-off: you return a potentially low-confidence chunk. Better than generating from nothing (which produces hallucinations).

### 4.5 Regex parser vs function calling
The ReAct agent uses regex to parse `Thought: ... Action: tool(args)`. This breaks on model output variations. Production upgrade: function calling or JSON mode. Qwen2.5 supports tool calling via Ollama — the structured output version is 10 lines of code change.

---

## 5. Common pitfalls

1. **Using RAG prompt format for retrieval-level LLM tasks** → model answers the instruction instead of evaluating/rewriting/planning. Use `call_raw` with plain prompts.
2. **Not windowing multi-hop context** → context window overflow at high hop counts. Window to last 3-5 snippets.
3. **No max_hops cap in multi-hop** → potential infinite loop if LLM never generates STOP.
4. **Using `or` with numpy arrays in cache** → ValueError on truth value. Use `is None` checks.
5. **Not including original query in multi-query** → risk that all variants miss the best phrasing.
6. **Over-aggressive relevance filter without fallback** → zero chunks returned, generator hallucinates.
7. **Unicode characters in PowerShell print** → UnicodeEncodeError (→ vs ->). Use ASCII in terminal-facing code.
8. **Trusting 3B LLM as sole relevance judge** → false positives. Add keyword overlap heuristic as backup signal.
9. **Benchmarking agentic with max_steps=4 on 10 gold queries** → 40+ LLM calls, ~5 min runtime. Use max_steps=2 for benchmarks.
10. **Not tracking crag_action/crag_attempt metadata** → can't diagnose whether corrections helped.
11. **Temperature=0.0 for multi-query variants** → near-identical variants, no union benefit. Use 0.3.

---

## 6. Benchmark results

| Rank | Retriever | recall@1 | recall@3 | MRR | time/10q | Notes |
|------|-----------|----------|----------|-----|----------|-------|
| 1 | agentic | 1.000 | 1.000 | 1.000 | 230s | First perfect recall |
| 2 | vector | 0.900 | 1.000 | 0.950 | 0.2s | Instant baseline |
| 3 | hybrid_rrf | 0.900 | 1.000 | 0.950 | 0.2s | Same as vector here |
| 4 | filtered_boost | 0.900 | 1.000 | 0.950 | 0.2s | Format-aware |
| 5 | query_rewrite | 0.900 | 1.000 | 0.950 | 35s | 175× cost, no gain |
| 6 | crag | 0.900 | 1.000 | 0.950 | 465s | 2325× cost, no gain |
| 7 | self_rag | 0.900 | 0.900 | 0.900 | 307s | Filter penalizes r@3 |
| 8 | multi_query | 0.800 | 1.000 | 0.900 | 58s | Union noise confirmed |

**The central finding:** on a small clean corpus, the deterministic retrievers (instant) match the expensive LLM-augmented ones. Complexity adds cost without benefit. Agentic is the exception — it discovered a strategy that fixed the one miss. The conditions for each strategy to shine are documented per-retriever.

---

## 7. Interview mock exam

### Section A — Fundamentals (10 questions)

1. What problem does metadata filtering solve that semantic search cannot?
2. What are the three filter modes in FilteredRetriever and when would you use each?
3. What is the query-document language mismatch and how does QueryRewriting address it?
4. How does HyDE differ from query reformulation?
5. Why does multi-query retrieval use reciprocal rank aggregation instead of score averaging?
6. What is the stop condition in multi-hop retrieval and why are two mechanisms needed?
7. What are the two decision points in Self-RAG?
8. What are CRAG's three evaluation labels and what action does each trigger?
9. What is the ReAct pattern and what does it enable in agentic retrieval?
10. Why does the agentic retriever use a union across all steps rather than just the last step?

### Section B — Applied Understanding (15 questions)

11. Multi-query scored 0.800 consistently in two separate test runs. What does this reproducibility tell you?
12. Your boost filter with 2× multiplier returned only CSV chunks even though it's a "soft" preference. Why?
13. Qwen2.5-3B produced better query rewrites than the OpenRouter free tier. What explains this difference?
14. Multi-hop stopped at hop 1 for both test queries. Is this a failure? Explain.
15. Self-RAG's relevance filter returned `relevant=False` but still returned the chunk. What happened?
16. Why does CRAG need a keyword overlap heuristic in addition to LLM evaluation?
17. CRAG took 46s/query to match baseline at 0.900. When is this cost justified?
18. The agentic retriever fixed the "dense vs BM25" miss that hybrid_rrf could not. Explain why.
19. What is the difference between `call_raw` and `generate` in LocalLLMGenerator?
20. Self-RAG's recall@3 dropped to 0.900 while all others are 1.000. What caused this?
21. Your multi-hop retriever uses `hop_context[-3:]`. What does this prevent?
22. Over-fetching is used in both FilteredRetriever and SelfRAGRetriever. What problem does each solve?
23. Why must the original query always be included in multi-query retrieval?
24. CRAG's reformulation loop ran 3 times and fell back. What does this indicate about the query?
25. The regex parser in AgenticRetriever has a fallback "use original query." Why is a fallback necessary?

### Section C — Design and Trade-offs (10 questions)

26. Design a retrieval system for a legal research tool where wrong answers are worse than no answers.
27. A client has 1M documents and needs <200ms p99 retrieval. Which retrievers can you use?
28. Design a retrieval strategy for a query mix that's 40% exact entity lookups and 60% conceptual questions.
29. Your Self-RAG relevance filter has a 15% false negative rate (filters relevant chunks). How do you tune it?
30. Compare CRAG and Self-RAG architecturally. When would you choose each?
31. A multi-query retriever improves recall@3 from 0.85 to 0.95 but costs 60s/query. Propose a caching strategy.
32. Design an agentic retriever with function calling instead of regex parsing. What changes?
33. Your CRAG evaluator uses keyword overlap as a heuristic. What are the failure modes of this heuristic?
34. A client's RAG system serves 10K queries/day across 5 retrieval strategies of different costs. Design a query router.
35. The agentic retriever achieves perfect recall but 23s/query. Propose a hybrid deployment: fast path + slow path.

### Section D — Whiteboard Coding (5 questions)

36. Implement `FilteredRetriever._matches(chunk, filter)` — check if a chunk satisfies all filter conditions including list-valued conditions.
37. Implement `MultiQueryRetriever._aggregate_scores(all_results: list[list[Chunk]]) → dict[str, float]` — reciprocal rank aggregation across multiple retrieval lists.
38. Implement `SelfRAGRetriever._should_retrieve(query) → bool` using LocalLLMGenerator. Include the prompt, the call, and the parsing logic.
39. Implement the CRAG keyword overlap check: given a query string and a list of chunks, return True if >30% of meaningful query terms appear in the chunks.
40. Implement `AgenticRetriever._parse_response(response) → tuple[str, str, str]` — extract (thought, action_name, action_args) from a ReAct-format response.

---

## 8. Project walkthrough scripts

### 8.1 The 30-second pitch

> "M5 added eight retrieval strategies to KnowledgeOS — from filtered metadata retrieval through a full ReAct agentic retriever. The benchmark showed agentic retrieval achieving perfect recall at r@1=1.000, the first on this corpus, by discovering that hybrid keyword+vector search was needed for the persistent miss. The cost: 23s per query vs 20ms for the vector baseline. The deterministic retrievers matched most LLM-augmented ones at a fraction of the latency — complexity is only worth it on large, diverse, noisy corpora."

### 8.2 The 2-minute technical walkthrough

> "M5 spans three tiers. Deterministic: VectorRetriever, HybridRetriever, FilteredRetriever — all sub-20ms, no LLM calls. FilteredRetriever adds pre/post/boost metadata filtering — enterprise RAG always needs 'only show PDF docs' within the first week. The over-fetch pattern compensates for filter losses: fetch 5× top_k, then filter.
>
> LLM-augmented: QueryRewriting reformulates informal queries to document language using local Qwen2.5-3B. Multi-query generates K variants and unions via reciprocal rank aggregation — same principle as M4's hybrid RRF. Multi-hop iterates, each hop's results seeding the next query, with a STOP sentinel for early termination.
>
> Research-grade: Self-RAG adds two gates — retrieve? and relevant-per-chunk? — skipping retrieval for parametric queries and filtering noise before generation. CRAG evaluates batch retrieval quality using LLM + keyword overlap heuristic and corrects via reformulation + re-retrieval. The agentic retriever implements ReAct — reasoning about which tool to use, executing it, observing the results, and planning the next step.
>
> The benchmark revealed: agentic = perfect recall at 23s/query; everything else ≤ 0.900 at 0.02-465s. Agentic won by dynamically discovering hybrid retrieval. Multi-query consistently lost — union noise on 8 chunks. The latency/recall curve is the central production decision."

### 8.3 The 5-minute deep walkthrough

> "Let me walk through three things: the metadata filtering design, the LLM-augmented strategies and what we learned from benchmarking them, and the agentic retriever's win.
>
> **Metadata filtering.** Enterprises always want 'only show me documents of this type.' FilteredRetriever supports three modes. Post-filter: run full semantic search over-fetching 5×, then apply the metadata condition — flexible, never misses semantically relevant content in the target type. Pre-filter: restrict the candidate pool first — efficient, requires Qdrant native filtering for true pre-filter (FAISS simulates it by fetching more). Boost: multiply matching chunks' scores — soft preference that ranks matching content higher without excluding non-matching. Per-call filter overrides let users dynamically filter without rebuilding the retriever.
>
> **LLM-augmented strategies and honest benchmarking.** Query rewriting reformulated 'how does rag work' to proper technical language using local Qwen2.5-3B — measurably better than OpenRouter free tier which repeated the query verbatim. HyDE generates a hypothetical answer document and embeds that — embeds in the document subspace, not the query subspace. Multi-query generates K diverse variants at temperature=0.3 and unions via reciprocal rank. Multi-hop iterates with a STOP sentinel. Self-RAG adds a retrieve gate (does this query need the corpus?) and per-chunk relevance filter. CRAG evaluates batch quality with LLM + keyword heuristic and corrects via reformulation. The benchmark showed: on 8 clean chunks, all these LLM-augmented strategies matched or lost to the instant vector baseline. Multi-query consistently scored 0.800 — union noise is the documented failure mode. CRAG took 46s to match 0.900 — correction adds no value when the corpus is already clean.
>
> **The agentic win.** The ReAct pattern gave the agent three tools: vector_search, keyword_search, filter_search. For each query, the agent outputs Thought (reasoning) and Action (tool call), receives the Observation (retrieved chunks), and loops. The trace showed the agent correctly choosing keyword_search for the BM25 entity query and vector_search for the conceptual comparison. The persistent miss — 'How is dense retrieval different from BM25?' — was finally closed: the agent used keyword_search in one step and vector_search in another, and the union included the correct chunk. No other retriever achieved this because they all used fixed strategies. The agent discovered hybrid retrieval was needed without being told. The cost: 23s/query. The production deployment is a two-path system: instant vector for simple queries, agentic for complex ones, with a query classifier routing between them."

---

## 9. Further reading

- **Asai et al., 2023** — "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." The original Self-RAG paper with fine-tuned critic tokens.
- **Yan et al., 2024** — "CRAG: Comprehensive RAG Benchmark." CRAG architecture with web search correction.
- **Yao et al., 2022** — "ReAct: Synergizing Reasoning and Acting in Language Models." The ReAct pattern foundational paper.
- **Gao et al., 2022** — "Precise Zero-Shot Dense Retrieval without Relevance Labels." The HyDE paper — hypothetical document embeddings for dense retrieval.
- **Ma et al., 2023** — "Query Rewriting for Retrieval-Augmented Large Language Models." Production query rewriting techniques.
- **LangChain MultiQueryRetriever** — langchain.readthedocs.io — the production implementation for comparison.

---

## Milestone status

- [x] 5.1 — FilteredRetriever (pre/post/boost)
- [x] 5.2 — QueryRewritingRetriever (reformulate + HyDE) + LocalLLMGenerator
- [x] 5.3 — MultiQueryRetriever (K variants + RR aggregation)
- [x] 5.4 — MultiHopRetriever (iterative + STOP sentinel)
- [x] 5.5 — SelfRAGRetriever (retrieve gate + relevance filter)
- [x] 5.6 — CRAGRetriever (CORRECT/AMBIGUOUS/INCORRECT + reformulation)
- [x] 5.7 — AgenticRetriever (ReAct loop, 3 tools)
- [x] 5.8 — Retrieval benchmark (8-way comparison, timing, honest interpretation)

**Resume line (updated with M5):**

> *Built an eight-strategy retrieval portfolio for KnowledgeOS: metadata-filtered retrieval (pre/post/boost), query rewriting (reformulate + HyDE) with local Qwen2.5-3B, multi-query with reciprocal rank union, multi-hop with STOP-sentinel iteration, Self-RAG with retrieve gate (6/6 accuracy) and per-chunk relevance filter, CRAG with two-signal evaluation (LLM + keyword heuristic), and a ReAct agentic retriever with three tool types. Benchmark: agentic achieved r@1=1.000 (first perfect recall) at 23s/query vs vector baseline 0.900 at 0.02s/query — a 1150× latency/quality trade-off documented with per-retriever timing. Key finding: deterministic retrievers match most LLM-augmented strategies on small clean corpora; agentic wins by dynamically discovering hybrid retrieval strategies.*
