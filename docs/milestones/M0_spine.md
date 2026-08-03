# Milestone 0 — The Spine (Walking Skeleton)

**Status:** ✅ Complete
**Duration:** 6 checkpoints
**Deliverable:** A working, modular, benchmarked RAG pipeline — resume-worthy as-is

---

## 1. Milestone summary

### Goal
Build the thinnest possible vertical slice through the entire RAG stack — one loader, one chunker, one embedder, one index, one retriever, one generator, plus an evaluation harness — so that a question goes in and a grounded, cited answer comes out with a **measured** quality number attached.

### Why this milestone matters more than any other
Every subsequent milestone (M1–M12) is a *variation* on this spine. New chunkers plug into the same interface. New embedders swap behind the same ABC. New retrievers extend the same pattern. If the spine's abstractions are wrong, every future milestone inherits the mistake. If the spine has no eval harness, every future component gets added on vibes.

M0 done right = all future milestones are cheap. M0 done wrong = permanent tax.

### What "done" looks like
- 8 components (loader, chunker, embedder, index, retriever, prompt builder, generator, eval harness) each behind a clean ABC
- Config-driven pipeline assembly (change YAML → change behavior)
- 10-query gold set with hand-labeled ground truth
- Recorded baseline: `recall@1 = 0.800, recall@3 = 1.000, MRR = 0.900, faithfulness = 0.531`
- 6 git commits, one per checkpoint
- Full end-to-end demo: `python scripts/test_pipeline.py` returns cited answers

---

## 2. Architecture recap

### The data flow

```
                ┌─────────────┐
File / URL ────▶│   Loader    │────▶ Document (uuid, content, metadata, tenant_id)
                └─────────────┘
                       │
                       ▼
                ┌─────────────┐
                │   Chunker   │────▶ list[Chunk] (uuid, doc_id, content, span, metadata, tenant_id)
                └─────────────┘
                       │
                       ▼
                ┌─────────────┐
                │  Embedder   │────▶ list[float] per chunk (384-dim, normalized)
                └─────────────┘
                       │
                       ▼
                ┌─────────────┐
                │    Index    │────▶ FAISS storage + tenant-scoped search
                └─────────────┘
                       ▲
                       │  query_vector
                       │
     query ──▶ Embedder.embed_query() ──▶ Retriever ──▶ list[Chunk] (with score)
                                                 │
                                                 ▼
                                          Prompt Builder
                                                 │
                                                 ▼
                                            Generator ──▶ cited answer

                       ┌──────────────────────────────────┐
                       │       Evaluation Harness         │
                       │  gold set × pipeline → metrics   │
                       └──────────────────────────────────┘
```

### The plugin architecture

Seven ABCs define the contracts:

| Interface  | Responsibility                           | Concrete class in M0    |
|------------|------------------------------------------|-------------------------|
| Loader     | source → `list[Document]`                | `TXTLoader`             |
| Chunker    | `Document` → `list[Chunk]`               | `FixedChunker`          |
| Embedder   | `list[str]` → `list[list[float]]`        | `BGEEmbedder`           |
| Index      | store vectors + `search(vec, k, tenant)` | `FaissFlatIndex`        |
| Retriever  | `query` → `list[Chunk]`                  | `VectorRetriever`       |
| Reranker   | `query, chunks` → reordered chunks       | (none yet, M6)          |
| Generator  | `query, chunks` → answer                 | `OpenRouterGenerator`   |

Every future component in M1–M12 subclasses one of these seven. The interface layer *is* the architecture.

### The data contracts

```python
@dataclass
class Document:
    doc_id:    str        # UUID
    content:   str        # raw text
    source:   str         # file path or URL
    metadata: dict[str, Any]
    tenant_id: str = "default"
    language:  str = "en"

@dataclass
class Chunk:
    chunk_id:    str      # UUID
    doc_id:      str      # backreference to parent
    content:     str      # the chunk's text
    embedding:   list[float]
    metadata:    dict[str, Any]
    tenant_id:   str = "default"
    start_index: int = 0
    end_index:   int = 0
```

Two data types flow through every stage. Every component takes one and produces the other. **`tenant_id` is first-class from day one** — the difference between a demo and a product.

---

## 3. Technical deep dive

### 3.1 Chunking: fixed strategy

Slides a window across text at regular character intervals, with overlap between adjacent chunks. Formula:
```
step = chunk_size - chunk_overlap
chunks = [text[i*step : i*step + chunk_size] for i in range(...)]
```

**Why overlap matters**: without it, a sentence split at the boundary is lost — it doesn't fully appear in either chunk, and neither will match a query about that sentence.

**Fixed vs alternatives**:
- **Fixed** (current): deterministic, fast, ignores meaning. Baseline.
- **Recursive** (M2): splits on `\n\n`, then `\n`, then `. `, then falls back to fixed. Respects natural boundaries.
- **Semantic** (M2): embeds sentences, detects topic shifts by cosine drop. Adaptive but slow (extra embeddings).
- **Parent-child** (M2): coarse chunks for retrieval breadth, fine chunks for generation precision.
- **Adaptive** (M2): variable chunk size based on content density.

**Field consensus**: chunking often has larger quality impact than embedding choice. The Anthropic contextual retrieval work (2024) showed prepending a chunk-summary sentence to each chunk gave a 35% recall improvement — chunking augmentation is a live research area.

### 3.2 Embeddings: dense representations

BGE-small maps text into a 384-dim vector space where semantic similarity ≈ cosine similarity. Trained by BAAI on a mixture of retrieval, classification, and clustering tasks with contrastive loss.

**Bi-encoder architecture**: query and document embedded independently, enabling pre-computation of document vectors. Query-time cost = one embedding + N dot products, not N transformer forward passes.

**Asymmetric retrieval**: BGE was trained with distinct objectives for queries vs passages. The query prefix — `"Represent this sentence for searching relevant passages: "` — signals which role the text plays. Skipping it drops recall by 5–15% in MTEB benchmarks.

**Normalization + inner product = cosine**: if vectors have unit L2 norm, cosine similarity equals dot product. Set `normalize_embeddings=True` at embed time, use `IndexFlatIP` at search time, get cosine for the price of a dot product.

**Embedding families to know**:
- **BGE (BAAI)**: strong MTEB scores, small/base/large tiers, English + multilingual variants
- **E5 (Microsoft)**: also asymmetric, uses `"query: "` and `"passage: "` prefixes
- **Instructor (HKUST)**: instruction-tuned; you specify the task
- **Jina**: long-context (8k) English models
- **OpenAI ada-002 / text-embedding-3**: API-only, no self-hosting

### 3.3 Indexing: FAISS internals

FAISS (Meta, 2017) is a library for efficient similarity search on dense vectors.

**Index types**:
- `IndexFlatIP` / `IndexFlatL2`: exact brute force. Perfect recall, O(N) per query. Fine to ~1M vectors.
- `IndexIVF`: partitions vectors into `nlist` clusters via k-means. At query time, only searches the `nprobe` nearest clusters. ~10-100× faster, tunable recall loss.
- `IndexHNSW`: hierarchical navigable small-world graph. State-of-the-art for large-scale. ~100× faster than flat with high recall (~0.95+ with proper tuning).
- `IndexPQ`: product quantization — compresses vectors 8-32× at moderate recall cost.

**When to graduate from flat**:
- Flat: <1M vectors, latency <10ms per query
- IVF: 1M–10M vectors, latency <50ms
- HNSW: 10M+ vectors, want <50ms with high recall
- PQ / IVFPQ: memory-constrained deployments, or beyond 100M vectors

### 3.4 Retrieval strategies (preview of M5)

Our M0 retriever is pure dense (bi-encoder + top-k). Not the only game in town:

| Strategy         | Query representation      | Strength                            |
|------------------|---------------------------|-------------------------------------|
| Dense            | vector                    | paraphrase, synonym, semantic       |
| Sparse (BM25)    | term-frequency scores     | exact keyword, entities, rare terms |
| Hybrid           | dense + sparse merged     | robust across query types           |
| Multi-query      | LLM rewrites into k forms | recall from diverse phrasings       |
| Multi-hop        | iterative refinement      | multi-step reasoning                |
| GraphRAG         | walks a knowledge graph   | structured, relational queries      |
| Self-RAG / CRAG  | LLM decides retrieval     | adaptive, avoids irrelevant fetches |

M5 is where this whole zoo becomes real.

### 3.5 Generation: grounding and citations

Three prompt constraints make an LLM into a RAG generator:

1. **Restrict to context**: "Answer using ONLY the provided context"
2. **Enforce citations**: "cite the chunk number like [1] or [2] at the end of every claim"
3. **Refuse when insufficient**: "if context is insufficient, say 'I don't have enough context'"

Citations are prompt-driven, not code-driven. We don't parse output to inject markers — the model produces them because the prompt asks. This is cheap and portable across LLMs.

**Lost in the middle** (Liu et al., 2023): LLMs attend more strongly to context at the start and end than the middle. Order chunks by relevance (best first) at minimum; sophisticated approaches bookend (best at start + end, weaker in middle) or reorder based on query relevance.

**Context assembly trade-offs**:
- More chunks → better recall, but dilutes attention and increases cost
- Fewer chunks → sharper attention, but may miss the relevant one
- Sweet spot: top 3–5 for most tasks, longer for multi-hop
- Compression (M7): summarize each chunk before including → more chunks in same token budget

### 3.6 Evaluation: measuring in the dark

**Retrieval metrics**:
- **Recall@k**: fraction of queries where a relevant chunk was in top-k. Binary per query.
- **Precision@k**: of top-k retrieved, fraction that were relevant. Matters for multi-answer queries.
- **MRR**: mean reciprocal rank of first relevant. Rewards top-1 accuracy.
- **nDCG**: discounted cumulative gain — rewards putting relevant results high. Standard for graded relevance.
- **Hit rate**: same as recall@k for binary relevance.

**Generation metrics**:
- **Faithfulness**: does the answer stay in the context? 4-gram overlap (ours), NLI models, LLM-as-judge.
- **Groundedness**: closely related — every claim traceable to a source.
- **Answer correctness**: does it answer the question well? Requires ground-truth answers or LLM-as-judge.
- **Citation accuracy**: do cited chunks actually contain the claim?

**RAGAS framework** (industry standard, M8): faithfulness, answer_relevancy, context_precision, context_recall — all LLM-as-judge based.

### 3.7 Multi-tenancy design

Even in a learning project, tenancy is baked in from M0:
- `tenant_id` on `Document` and `Chunk` dataclasses
- `Retriever.retrieve(query, top_k, tenant_id)` — signature enforces it
- `FaissFlatIndex.search()` filters by tenant post-retrieval

Production upgrade path (M9): per-tenant sub-indexes for scale, or use a vector DB with native metadata filtering (Qdrant, Milvus, Weaviate). Retrofitting tenancy to a single-tenant system is a rewrite; designing for it costs zero effort upfront.

---

## 4. Design decisions and trade-offs

### 4.1 Why we chose YAML for config over Python dicts
- Human-editable without code changes → safe for non-developers
- Version-controllable and diff-able
- Language-agnostic → future CLI, web UI, API can all consume it
- Downside: no computation in config (fine — configs shouldn't have logic)

### 4.2 Why we chose direct HTTP over the openai SDK for OpenRouter
- One less dependency
- Full visibility into wire format, error responses, and retries
- No lock-in to a client that might drift as OpenAI evolves
- Downside: manual error handling — but that's a feature for a learning project

### 4.3 Why we chose FAISS Flat instead of Chroma / Qdrant
- Lowest dependencies, fastest to spin up
- Learning goal: understand vector search primitives, not compare DB UX
- Production goal (M4): all the majors get wired for comparison
- Downside: no metadata filtering out of the box — done in wrapper code

### 4.4 Why we chose `list[float]` for embeddings instead of numpy arrays
- Dataclass serializes cleanly to JSON — needed for persistence and API responses
- Numpy conversion happens at the FAISS boundary, not in the data model
- Downside: less memory-efficient for very large batches — but this is a data-model choice, not a hot-path choice

### 4.5 Why we chose Python 3.11 instead of the newest (3.13/3.14)
- Binary-dependency wheels for torch, faiss, sentence-transformers lag new Python releases by 6–12 months
- Choosing bleeding-edge Python = fighting build errors instead of learning RAG
- 3.11 has structural pattern matching, better tracebacks, faster performance than 3.10

### 4.6 Why we chose 4-gram faithfulness instead of LLM-as-judge
- Deterministic, no API cost, runs on every commit
- Directionally useful even if absolute value is noisy
- LLM-as-judge is on the M8 roadmap when we go from "measure change" to "measure quality"

### 4.7 Why we chose to build eval in M0 instead of M8
- Every subsequent claim ("this new chunker is better") requires measurement
- Building components without a harness produces a bigger project that no one can tune
- Eval-first is the difference between engineering and vibes

---

## 5. Common pitfalls (real production failure modes)

1. **Chunking without overlap** → sentences split at boundaries are lost. Query matches nothing.
2. **Wrong query prefix for asymmetric embedders** → 5-15% recall loss. Silent, hard to detect.
3. **Un-normalized vectors with inner product** → similarity scores are meaningless. Get normalization right or use L2 with the correct interpretation.
4. **Missing tenant filter** → data leak between customers. Retrofitting is a rewrite.
5. **`temperature > 0` in eval** → non-reproducible results, noisy A/B comparisons.
6. **No rate-limit handling** → the 47th query in a 100-query benchmark crashes the whole run.
7. **Trusting HTTP 200 = success** → OpenRouter and others return errors as 200 with an `error` field in the body.
8. **Chunk size > embedder context window** → silently truncated, missing content in embedding.
9. **Not seeding UUID sources for tests** → non-reproducible test runs.
10. **Zero-score queries assumed to be regressions** → often an eval bug (see: `Evaluation → etrics`).
11. **Gold set contamination**: gold text also present in training data of embedder → inflated metrics. Real gold sets should be held out and ideally domain-specific.
12. **No feedback loop from production** → gold set stays static while user queries drift. Refresh quarterly.

---

## 6. Benchmarks and results

### Baseline (from `RESULTS.md`)

**Pipeline:** TXT loader → Fixed chunker (512/50) → BGE-small → FAISS flat → Vector top-k → Nemotron/Ling generator

**Corpus:** 10-paragraph RAG primer, chunked into 6 chunks
**Gold set:** 10 hand-labeled query→relevant-text pairs

| Metric        | Score |
|---------------|-------|
| recall@1      | 0.800 |
| recall@3      | 1.000 |
| recall@5      | 1.000 |
| MRR           | 0.900 |
| faithfulness  | 0.531 |

### Interpretation

- **Retrieval is near-ceiling on this small corpus.** Real degradation will show up when we grow to hundreds of docs in M1.
- **recall@1 = 0.8 vs recall@3 = 1.0 → 20% gap.** The correct chunk is always in top-3 but sometimes at rank 2 or 3. This is exactly the failure mode reranking (M6) is designed to close.
- **Faithfulness = 0.53 is the weak link.** With retrieval near-perfect, the generator is where quality is being lost. Two failure modes visible in per-query data: model paraphrases heavily (mechanical 4-gram penalty, not real hallucination) or model wanders beyond context (real hallucination). Distinguishing these needs LLM-as-judge (M8).
- **MRR = 0.9** confirms retrieval quality is high, with rank-1 hit rate = 0.8 explaining the gap from 1.0.

### What we'd expect after future milestones

| After | Expected recall@1 | Expected faithfulness |
|-------|-------------------|-----------------------|
| M2 semantic chunker  | 0.85 (chunk quality) | 0.55 |
| M4 hybrid indexing   | 0.90 (BM25 catches keyword matches) | 0.55 |
| M5 multi-query       | 0.92 | 0.55 |
| M6 cross-encoder rerank | 0.95 (rank-1 lift) | 0.60 |
| M7 context compression + reorder | 0.95 | 0.70 |
| M8 LLM-as-judge eval | recompute all | recompute all |

These are hypotheses — the M0 baseline is what we'll actually compare against.

---

## 7. Interview mock exam

Practice these until you can answer any of them cold in under a minute. Difficulty ramps from definitional (easy) to system design (hard).

### Section A — Fundamentals (Easy, 10 questions)

1. What does RAG stand for and what problem does it solve?
2. What is an embedding, in one sentence?
3. What's the difference between a Document and a Chunk in your system?
4. What is cosine similarity and how is it related to inner product?
5. What is FAISS and what does the "Flat" in `IndexFlatIP` mean?
6. What are the two data types that flow through every RAG stage?
7. What does `top_k` mean in a retrieval call?
8. Why do you normalize embeddings before storing them?
9. What's the purpose of the system prompt in your generator?
10. What is a gold set and why does the RAG project need one?

### Section B — Applied Understanding (Medium, 15 questions)

11. Why do you use a query prefix with BGE but wouldn't with MiniLM?
12. Why is `tenant_id` on the dataclass from day one instead of added later?
13. Your chunker uses fixed size with overlap. Why overlap, and how do you choose the overlap size?
14. What's the difference between a bi-encoder and a cross-encoder, and where does each belong in a RAG pipeline?
15. You measure recall@1 = 0.8 and recall@3 = 1.0. What does that tell you about your system and what would you do next?
16. What is "lost in the middle" and how does it affect prompt design?
17. Explain the difference between `IndexFlatIP` and `IndexFlatL2` and when to prefer each.
18. Your faithfulness metric is 4-gram overlap. What are two failure modes of this metric?
19. Why is generation eval separated from retrieval eval in your runner?
20. Your OpenRouter response returned HTTP 200 with an error inside. Why doesn't `response.raise_for_status()` catch it?
21. What's the difference between MRR and recall@k, and when would you care about one vs the other?
22. How does your prompt enforce citations without any output parsing code?
23. Your system prompt tells the LLM to say "I don't have enough context" — why is this important?
24. If you swapped BGE-small for a 1536-dim OpenAI embedding, what other component needs a change?
25. How would you extend this system to handle a 10-million-document corpus?

### Section C — Design and Trade-offs (Hard, 10 questions)

26. Walk through your plugin architecture. How would adding a new chunker require modifications to which files?
27. Design a multi-tenant RAG serving 1,000 customers with 1M documents each. What breaks in your current architecture and how would you fix it?
28. You're told the LLM is hallucinating. How do you diagnose whether the problem is retrieval or generation?
29. Your CEO says "make the RAG faster." Walk through where you'd measure and what you'd optimize.
30. Compare the trade-offs of FAISS Flat, IVF, and HNSW. When would you use each?
31. Why did you build the eval harness in M0 instead of after all components existed? Defend the choice.
32. Your gold set has 10 queries. A colleague argues for 10,000 auto-generated queries. What are the pros and cons?
33. Explain how you'd implement a hybrid retriever combining BM25 and dense vectors, and how you'd merge the two ranked lists.
34. Your product needs to answer questions from 100-page PDFs with tables and figures. What breaks in your current pipeline and what components need upgrades?
35. If you had to reduce your API costs by 80%, what would you change and how would you measure the quality trade-off?

### Section D — Whiteboard Coding (Hard, 5 questions)

36. Implement `recall_at_k(retrieved, relevant_id, k)` on the whiteboard. Assume `retrieved` is a list of chunks each with `.chunk_id`.
37. Implement fixed-size chunking with overlap in ~10 lines. Handle edge cases: empty text, chunk_size = 0.
38. Write pseudocode for Reciprocal Rank Fusion combining two ranked lists.
39. Implement cosine similarity for two vectors, without numpy.
40. Design a data structure for tenant-scoped vector search on top of a single global FAISS index. What are the operations and their complexities?

### Answer key patterns

- Section A: single-sentence, precise definitions
- Section B: 2–4 sentence answers, with the trade-off named
- Section C: 3–5 minute answers, structured as "the tension is X, in your case Y, I'd choose Z because"
- Section D: whiteboard code that compiles mentally, with edge cases named

---

## 8. Project walkthrough scripts

### 8.1 The 30-second pitch (for casual "what are you working on?")

> "I'm building KnowledgeOS — a modular RAG platform where every component is a swappable plugin. The idea is to implement every chunking strategy, retrieval algorithm, and reranker from scratch and benchmark them against each other with a proper evaluation harness. It's a learning project, but production-shaped: plugin architecture, tenant isolation, config-driven pipelines. Right now I've finished the spine — end-to-end retrieval and generation with a working eval framework."

### 8.2 The 2-minute technical walkthrough (for interviews)

> "The system is organized around seven abstract base classes — Loader, Chunker, Embedder, Index, Retriever, Reranker, and Generator — each defining one stage of the RAG pipeline. Concrete implementations live in dedicated packages, and a YAML config names which one to use at each stage, so swapping a chunker or embedder is a one-line change with no code edits.
>
> The M0 build has a TXT loader, a fixed-size chunker with configurable overlap, a BGE-small embedder that handles asymmetric retrieval via query prefixing, a FAISS flat index with tenant-scoped search, a vector retriever, and a generator that hits OpenRouter over plain HTTP. The prompt uses three constraints — restrict to context, cite chunk numbers, refuse when insufficient — to enforce grounded output.
>
> The evaluation harness is what makes it a benchmark platform, not just a toy. I have a 10-query hand-labeled gold set and metrics for recall@k, MRR, and a 4-gram faithfulness proxy. Retrieval eval runs without API calls — cheap enough for every commit — and generation eval runs opt-in with rate-limit handling.
>
> The current baseline is 0.8 recall@1, 1.0 recall@3, and 0.53 faithfulness. The gap between recall@1 and recall@3 is telling me my next high-value work is adding a reranker — that's exactly the kind of quantitative signal this architecture is designed to produce."

### 8.3 The 5-minute deep walkthrough (for senior interviews)

> "Let me start with the architecture, then walk through a query, then show you the eval.
>
> **Architecture.** Seven interfaces in `core/interfaces.py`, each ABC with a single required method. Every plugin subclasses one of them. This means when I add a new chunker in M2, it's one file that implements one method, and the rest of the system doesn't know or care. Data flows as two dataclasses — `Document` and `Chunk` — both carrying a `tenant_id` field from day one, because retrofitting multi-tenancy to a working system is a rewrite. Pipeline assembly is YAML-driven: `configs/default.yaml` names each component and its parameters. Change YAML, change behavior.
>
> **Query flow.** A query hits the vector retriever, which asks the embedder to embed it. Because BGE is an asymmetric retriever, the embedder applies a query prefix — `Represent this sentence for searching relevant passages: {query}` — that signals to the model which role the text plays. Vector is normalized to unit length. FAISS's `IndexFlatIP` computes dot products against every stored chunk vector; because vectors are unit-normalized, that's cosine similarity for free. Top-k results come back as integer indices which my wrapper translates to Chunk objects, filtering by `tenant_id` to enforce isolation. Those chunks flow into the prompt builder, which formats them as numbered blocks and constructs a two-message chat prompt with a system message enforcing citations and grounding. The generator hits OpenRouter, defensively checking the response body because OpenRouter sometimes returns errors as HTTP 200. The final answer contains numbered citations that the UI can resolve back to chunks.
>
> **Evaluation.** I have a 10-query gold set — hand-labeled query-to-relevant-text pairs. The runner iterates each query, calls the retriever, and computes recall@k, MRR, and a 4-gram faithfulness proxy. Retrieval eval runs without any API calls — deterministic and fast. Generation eval is opt-in, uses `time.sleep(2)` between calls for rate-limit safety, and wraps each call in try/except so partial failures don't kill the run.
>
> **The interesting number.** My baseline is recall@1 = 0.8, recall@3 = 1.0. That 20% gap tells me the correct chunk is always in the top-3 but not always at rank 1. That's exactly the failure mode a cross-encoder reranker fixes — retrieve broadly with a fast bi-encoder, rerank with a slower but more accurate model. My eval harness produced that signal automatically. That's why I built the harness in M0 rather than at the end: it's not something you add after the fact, it's the compass that tells you what to build next."

---

## 9. Further reading

Curated to what's worth reading for this milestone's material. Not exhaustive.

### Foundational papers
- **Karpukhin et al., 2020** — *Dense Passage Retrieval for Open-Domain Question Answering*. The DPR paper that established the modern dense retrieval pattern.
- **Lewis et al., 2020** — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. The original RAG paper.
- **Reimers & Gurevych, 2019** — *Sentence-BERT*. Bi-encoder architecture for retrieval.
- **Xiao et al., 2023** — *BGE Embedding: A Comprehensive Guide*. The technical background on BGE, its asymmetric training, and query prefixing.
- **Liu et al., 2023** — *Lost in the Middle: How Language Models Use Long Contexts*. The middle-of-context attention problem.

### Practical references
- **FAISS documentation** — [github.com/facebookresearch/faiss/wiki](https://github.com/facebookresearch/faiss/wiki). Especially the "Faiss indexes" and "Guidelines to choose an index" pages.
- **MTEB Leaderboard** — [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Standard benchmark for comparing embedders.
- **RAGAS** — [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas). Modern RAG evaluation framework (useful for M8 comparison).
- **Pinecone Learning Center** — good intro-level RAG articles, provider-agnostic.

### Engineering context
- **Anthropic — Contextual Retrieval** (2024). Prepending chunk summaries lifts recall by 35%. Modern chunking-augmentation technique.
- **LlamaIndex documentation on chunking** — a good survey of chunker types before we implement them in M2.

---

## Milestone status

- [x] Checkpoint 0.1 — Project scaffold
- [x] Checkpoint 0.2 — Interfaces, data models, config
- [x] Checkpoint 0.3 — TXT loader + fixed chunker
- [x] Checkpoint 0.4 — BGE embedder + FAISS index
- [x] Checkpoint 0.5 — Retriever + generator (end-to-end)
- [x] Checkpoint 0.6 — Evaluation harness + baseline

**Resume line unlocked:**

> *Built KnowledgeOS, a modular, plugin-based Retrieval-Augmented Generation platform in Python. Designed a seven-interface plugin architecture with YAML-driven pipeline assembly and multi-tenant isolation from day one. Implemented ingestion, fixed chunking with overlap, BGE-small dense embeddings with asymmetric query prefixing, FAISS flat indexing, vector retrieval, and grounded generation with prompt-enforced citation via OpenRouter. Built a scientific evaluation harness with recall@k, MRR, and 4-gram faithfulness metrics on a hand-labeled gold set; established a baseline (recall@1 = 0.8, recall@3 = 1.0, faithfulness = 0.53) that drives every subsequent architectural decision.*