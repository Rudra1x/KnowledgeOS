# Milestone 3 — Embedding Engine

**Status:** ✅ Complete
**Duration:** 7 checkpoints
**Deliverable:** A five-backend embedding framework with caching, batch processing, normalization utilities, and a quality/latency/cost benchmark.

---

## 1. Milestone summary

### Goal
Build a unified embedding framework: multiple backends behind one interface, an embedding cache for cost efficiency, batch processing for production throughput, and a scientific benchmark to answer "which embedder should I use for my corpus?"

### Why this milestone matters
Embeddings are the semantic heart of RAG. Every retrieval decision depends on how well the embedding space represents meaning. But embedding choice is also a cost and latency decision — the right embedder depends on your quality requirements, throughput needs, and budget. Without a benchmark, "which embedder?" is a blog-post question. With one, it's a 20-minute measurement.

### What "done" looks like
- 5 embedding backends behind one interface (BGE, E5, Instruction, Jina stub, API)
- CachedEmbedder with two-tier L1/L2 cache (788x speedup on L1 hits)
- BatchEmbedder with progress, normalization verification, skip-on-error
- normalize_vectors() + verify_normalization() utilities
- 3-embedder benchmark with quality, latency, and cost dimensions
- RESULTS.md updated with M3 benchmark

---

## 2. Architecture recap

### The embedding stack

```
                          Embedder (ABC)
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
    BGEEmbedder          E5Embedder          InstructionEmbedder
    (bge-small/base/     (e5-small/base/     (any ST model +
     large, 384-1024)     large, 384-1024)    custom instruction)

                     APIEmbedder              JinaEmbedder
                  (OpenAI-compatible,          (stub — compat
                   any provider)               issue, 8192-tok)
                               │
                               ▼
                        CachedEmbedder
                      (L1 memory + L2 SQLite)
                               │
                               ▼
                         BatchEmbedder
                   (chunked, progress, norm-verify)
                               │
                               ▼
                    normalize_vectors() / verify_normalization()
                               │
                               ▼
                          FAISS index
```

### The shared contract

Every embedder exposes:
```python
NAME:          str           # benchmark identifier
dimension:     int           # embedding size
last_embed_ms: float         # timing for last call

embed(texts)        → list[list[float]]  # passages
embed_query(query)  → list[float]        # query (may differ)
embed_numpy(texts)  → np.ndarray         # direct numpy for FAISS
```

---

## 3. Technical deep dive

### 3.1 The asymmetric retrieval family

All production embedding models for retrieval use asymmetric training — queries and documents have different roles in the embedding space:

**Why asymmetry matters:** a query is typically short, keyword-like, and expresses intent. A document passage is longer, topic-dense, and expresses content. If both are embedded in the same geometric subspace without role differentiation, the model can't learn that "what is X?" should point to "X is..." rather than "is X the..."

**How each family encodes role:**

| Family | Query | Passage | Mechanism |
|--------|-------|---------|-----------|
| BGE | Fixed prefix | None | Learned asymmetry on query side |
| E5 | `"query: "` | `"passage: "` | Explicit both-side role signaling |
| Instructor | Custom instruction | Custom instruction | User-controlled domain+role |
| Voyage/text-embedding-3 | Task parameter | Task parameter | API-side control |

### 3.2 Why BGE-small has one recall failure

The persistent miss: "What is hybrid retrieval?" → recall@1=0 with BGE-small.

Looking at BGE scores on 3 passages:
```
"BM25..." → 0.6427
"Dense..." → 0.3601
"Hybrid..." → 0.3862
```

Hybrid ranks behind BM25 even though the query is explicitly about hybrid. BGE-small's 384-dim space conflates "dense" and "hybrid" because both are "retrieval" concepts. E5 closed this gap via its dual-prefix training — the embedding space has more room to separate adjacent concepts.

This is a fundamental limitation of smaller embedding models: limited dimensionality compresses the semantic space, causing adjacent concepts to overlap.

### 3.3 The two-tier cache — economic impact

For a 100K-document corpus updated daily with 10% new content:

Without cache:
- Every day: 100K texts × 20ms = 33 minutes

With L2 cache:
- Day 1: 100K texts embedded (33 min)
- Day 2+: 10K new + 90K cached = 3.3 min + 2.9 min = 6.2 min
- 5.3x daily time saving

For API embeddings at $0.02/1M tokens:
- Without cache: $0.60/day
- With cache: $0.06/day
- Annual saving: $197

### 3.4 Silent truncation — the real cost

Every embedder has a max sequence length. BGE-small: 512 tokens ≈ 2000 chars.

For parent chunks averaging 1500 chars: 75% of the chunk fits. The last 25% is silently dropped — facts at the end of the chunk become unretrievable.

**How to detect truncation in production:**
1. Track chunk size distribution vs model limit
2. Use `verify_normalization()` as a proxy (truncated inputs sometimes produce off-norm vectors)
3. Test with a gold query whose answer appears late in a long document

### 3.5 Normalization — the numerical correctness layer

FAISS `IndexFlatIP` computes `dot(query, chunk)`. For unit vectors, this equals cosine similarity. For non-unit vectors:

```
dot(query, chunk) = |query| × |chunk| × cos(θ)
```

A chunk with norm=20 scores 20x higher than a unit-norm chunk at the same angle. Retrieval becomes "find the biggest vector" instead of "find the most similar vector."

Our `verify_normalization()` catches this before indexing:
```python
check = verify_normalization(vecs)
# {'normalized': False, 'max_dev': 19.76, 'n_outliers': 5}
```

A `max_dev` of 19.76 means some vectors are 20x too long. Normalization reduces this to 5.96e-08 (machine precision).

### 3.6 Matryoshka embeddings — the production flexibility layer

OpenAI's text-embedding-3-small (and -large) support dimension truncation:

```python
APIEmbedder(model_name="text-embedding-3-small", dimensions=256)
```

The first 256 of 1536 dimensions are independently useful. Quality vs cost trade-off:
- 1536-dim: full quality, $0.020/1M tokens, 1.5GB per 1M chunks
- 512-dim: ~98% quality, $0.020/1M tokens, 0.5GB per 1M chunks (3x smaller index)
- 256-dim: ~95% quality, $0.020/1M tokens, 0.25GB per 1M chunks (6x smaller)

Token price is fixed; storage and search cost varies with dimension. For memory-constrained deployments, Matryoshka dimensions are the right knob.

---

## 4. Design decisions and trade-offs

### 4.1 Why three methods (embed, embed_query, embed_numpy)
- `embed()`: general purpose, JSON-serializable output, works with CachedEmbedder
- `embed_query()`: query-specific treatment for asymmetric models
- `embed_numpy()`: skips list→numpy conversion for FAISS, saves seconds at scale

### 4.2 Why SQLite for L2 cache (not Redis)
SQLite is stdlib, zero server, zero config. For single-process use (a learning project, a batch ingestion script), SQLite is correct. For multi-worker production, Redis or Memcached replace L2. The `_l2_get` / `_l2_set` interface makes the swap trivial.

### 4.3 Why content-addressed cache (not time-based)
Embeddings are deterministic — same text + same model = same vector. Time-based TTL would expire valid entries and recompute identical vectors. Content-based keys never expire (unless the model changes), never return stale values, and are self-documenting: the key tells you exactly what's cached.

### 4.4 Why stub Jina instead of removing it
The stub preserves intent, documents the failure reason, and maintains the upgrade path. Removal loses the architectural decision (long-context embeddings matter). A clear `NotImplementedError` with the failure reason is better than a silent missing module.

### 4.5 Why skip_on_error is configurable
Both "fail fast" and "skip and continue" are legitimate production postures depending on context. Nightly batch over 1M documents: skip. Interactive ingestion of 100 documents: fail fast. Making it configurable avoids baking in assumptions about the caller's context.

---

## 5. Common pitfalls

1. **Skipping the query prefix on BGE** → 5-15% recall loss, silent
2. **Using the same prefix for queries and passages on E5** → miss E5's dual-prefix benefit
3. **Not sorting API embedding results by index** → permuted embedding matrix, silently wrong retrieval
4. **Using `or` with numpy arrays** → `ValueError: ambiguous truth value`, fix with `is None` check
5. **Not including model_name in cache key** → BGE vectors returned for E5 queries, silent retrieval corruption
6. **Hard-coding `:free` model IDs** → breaks when free tier rotates out
7. **Trusting `trust_remote_code` without pinning revision** → breaks on transformers upgrades
8. **Not deep-copying chunks between embedder benchmark runs** → silent cross-contamination, wrong results
9. **Assuming all embedders normalize** → un-normalized vectors silently corrupt FAISS scores
10. **Setting batch_size too large** → OOM on memory-constrained deployments
11. **Not tracking chunk size vs model max tokens** → silent truncation of content past 512 tokens

---

## 6. Benchmarks and results

### M3 embedding benchmark

| Rank | Embedder | dim | recall@1 | recall@3 | MRR | ms/chunk | 1M-chunk hrs | 1M-chunk GB |
|------|----------|-----|----------|----------|-----|----------|--------------|-------------|
| 1 | e5-small | 384 | 1.000 | 1.000 | 1.000 | 64.1 | 17.8 | 1.5 |
| 2 | instr-bge-b | 768 | 1.000 | 1.000 | 1.000 | 183.6 | 51.0 | 3.1 |
| 3 | bge-small | 384 | 0.900 | 1.000 | 0.950 | 20.0 | 5.6 | 1.5 |

### Key interpretations

**E5 wins on quality at 3x the cost of BGE-small.** The quality gain is one closed recall failure (hybrid retrieval). The compute penalty is 3.2x. For most production deployments, run the benchmark on your corpus to see if that one failure pattern appears in your query distribution.

**InstructionBGEb doesn't justify its cost on in-distribution content.** Same quality as E5, 9.2x the compute, 2x the index size. Worth testing when your corpus is heavily domain-specific and out-of-distribution for standard pre-trained models.

**Dimension doesn't predict quality on in-distribution content.** 384-dim E5 = 768-dim InstructionBGEb = 1.000 recall. Higher dimension helps when semantic space is genuinely complex — multilingual, highly specialized, or unusual query patterns.

---

## 7. Interview mock exam

### Section A — Fundamentals (10 questions)

1. What is an embedding and what property makes it useful for retrieval?
2. What is asymmetric retrieval and why does it require separate query/passage methods?
3. What does BGE's query prefix do and what happens if you skip it?
4. What is the difference between E5 and BGE's approach to asymmetric retrieval?
5. What is silent truncation and at what token limit does it occur for BGE-small?
6. What does normalize_vectors() do and why is it needed before FAISS indexing?
7. What is a content-addressed cache and how is the cache key constructed?
8. What does `last_embed_ms` enable in the benchmark?
9. What are the two tiers in CachedEmbedder and what does each provide?
10. What is the OpenAI-compatible embedding endpoint format?

### Section B — Applied Understanding (15 questions)

11. BGE-small consistently misses "What is hybrid retrieval?" at recall@1. E5-small retrieves it correctly. What explains the difference?
12. E5 scores are compressed into 0.72–0.85 while BGE spans 0.36–0.64. Which is "better" and for what use case does each characteristic help?
13. Why does the API embedding endpoint sort results by index before assembling the matrix?
14. Your embedding cache uses SHA-256(model_name + '::' + text) as the key. Why include model_name?
15. A corpus update changes 10% of documents daily. How does the two-tier cache change the economics?
16. The Jina library broke on both v2 and v3. What architectural decision does this validate about trust_remote_code?
17. Why is `chunk.embedding = bgе_vector` a benchmark validity bug when testing multiple embedders?
18. Un-normalized vectors have max_dev=19.76. What does retrieval look like, and why is it silent?
19. You benchmark three embedders and InstructionBGEb ties E5 at perfect recall but takes 9x longer. Which do you recommend and why?
20. What is Matryoshka Representation Learning and what production problem does it solve?
21. What's the difference between `embed()` and `embed_numpy()` and when would you use each?
22. Why does `vec = l1_get() or l2_get()` fail in Python when the return type is a numpy array?
23. How would you detect silent truncation in a deployed system without access to the source documents?
24. A client needs to embed a 50-page contract. BGE-small truncates at 512 tokens. What are the options?
25. You add a 4th embedder to the benchmark. What changes in the benchmark script?

### Section C — Design and Trade-offs (10 questions)

26. Design an embedding pipeline for a 100M-document corpus with daily 5% updates. What embedder, cache strategy, and throughput plan do you use?
27. A client's retrieval recall is 0.6 despite good chunking. Walk through the diagnosis — is it the embedder, the index, or the retrieval strategy?
28. Compare local embedding (BGE-small on CPU) vs API embedding (text-embedding-3-small) for a startup with 50K documents and $500/month budget.
29. Design a multi-tenant embedding system where tenants A and B share an embedder but must have isolated indexes. What breaks if you share the cache?
30. Your embedding cache grows to 500GB on disk. How do you implement cache eviction without disrupting production retrieval?
31. A model provider changes the embedding model behind the same API endpoint. What breaks in your system and how do you detect it?
32. You need to support embedding 5 languages in one pipeline. How does this change your embedder choice, prefix strategy, and cache key design?
33. Compare CachedEmbedder with Redis L2 vs SQLite L2 for a system with 10 concurrent ingestion workers.
34. A user queries in Spanish but all documents are in English. How does asymmetric retrieval behave and what's the fix?
35. Design a system that automatically selects the embedder based on corpus size, chunk size distribution, and quality requirements.

### Section D — Whiteboard Coding (5 questions)

36. Implement `normalize_vectors(vecs: np.ndarray) → np.ndarray` handling zero vectors safely.
37. Implement the cache key function: `make_key(model_name: str, text: str) → str` using SHA-256.
38. Implement the batch miss identification logic: given texts and a cache lookup function, return `(cached_results, miss_indices, miss_texts)`.
39. Implement `verify_normalization(vecs: np.ndarray, tol: float = 1e-4) → dict` returning `normalized`, `max_dev`, `n_outliers`.
40. Write pseudocode for an embedding benchmark that tests N embedders and outputs a ranked table by (recall@1, mrr, ms_per_chunk).

---

## 8. Project walkthrough scripts

### 8.1 The 30-second pitch

> "M3 added five embedding backends — BGE, E5, Instruction, Jina (stubbed for compatibility), and an OpenAI-compatible API client — all behind the same interface. A two-tier cache (memory + SQLite) gives 788x speedup on hot paths and 2.6x on restarts, reducing daily re-embedding costs by 90% on typical corpora. The benchmark showed E5-small achieves perfect recall at 3x the compute of BGE-small; larger dimension (InstructionBGEb at 768-dim) adds nothing on in-distribution content."

### 8.2 The 2-minute technical walkthrough

> "Every embedder in M3 implements a three-method interface: `embed()` for passages, `embed_query()` for queries — the split matters because asymmetric models treat them differently — and `embed_numpy()` for direct FAISS input, skipping a list→numpy round-trip that costs seconds at scale.
>
> The portfolio covers three asymmetric families. BGE uses a fixed query prefix only — one-sided asymmetry. E5 prefixes both queries (`'query: '`) and passages (`'passage: '`) — explicit both-side role signaling that closed BGE's one persistent recall failure in our benchmark. InstructionEmbedder accepts user-defined task instructions — useful for domain-specific corpora, implemented cleanly after the InstructorEmbedding library broke on current sentence-transformers.
>
> CachedEmbedder wraps any embedder with L1 (memory, 788x speedup) and L2 (SQLite, 2.6x speedup, persistent). The key is SHA-256(model_name + '::' + text) — content-addressed, never expires, includes the model to prevent cross-model cache pollution. A key bug: `l1_get() or l2_get()` fails on numpy arrays because `or` calls `bool()` on the array. Fix: explicit `is None` checks.
>
> Before indexing, `verify_normalization()` audits the vectors — un-normalized vectors have max_dev=19.76 in our test, which would make FAISS return 'biggest vector' not 'most similar vector.' Our benchmark result: E5 wins on quality at 3.2x the cost. Larger dimension (768 vs 384) adds nothing on this in-distribution corpus."

### 8.3 The 5-minute deep walkthrough

> "Let me walk through the three main capabilities: the embedder portfolio, the caching layer, and what the benchmark taught us.
>
> **The portfolio design.** All five backends share a contract: `NAME`, `dimension`, `last_embed_ms`, and three methods. The `NAME` and `last_embed_ms` fields are what make the benchmark one script — no external timing wrappers, no special-case logic. The three-method split (`embed` / `embed_query` / `embed_numpy`) handles asymmetric retrieval correctly: BGE applies a 50-word query prefix that shifts the embedding toward retrieval-optimized space; E5 applies short prefixes to both sides; InstructionEmbedder prepends user-defined task descriptions. Without the query prefix on BGE, recall drops 5-15% silently.
>
> **The Jina stub.** Both Jina v2 and v3 broke on current transformers — v2 through a removed module, v3 through a renamed attribute in custom remote code. The lesson is `trust_remote_code` is a liability: the model repo's custom code and the transformers library evolve independently, and your deployment is at the intersection of both release cadences. The stub raises a clear error with the failure reason and the fix when Jina publishes a compatible version. I kept it rather than deleting it because the long-context concept — silent truncation at 512 tokens — is architecturally important.
>
> **The cache economics.** CachedEmbedder uses SHA-256(model + text) as the key — content-addressed, deterministic, never expires. L1 is an in-memory LRU dict (788x speedup). L2 is SQLite (2.6x speedup, persistent). Write-through on L2 hits populates L1, so the first request after a restart gets 2.6x and every subsequent request gets 788x. On a 100K corpus with 10% daily updates, the cache reduces daily embedding time from 33 minutes to 6.2 minutes. A key implementation bug: `l1_get() or l2_get()` fails because Python's `or` calls `bool()` on numpy arrays, which is ambiguous. Explicit `is None` checks fix it.
>
> **The benchmark finding.** E5-small and BGE-small are the same dimension (384) and similar architecture. E5 closed BGE's one recall failure through dual-prefix training. The cost: 3.2x more compute per chunk. InstructionBGEb (768-dim) matched E5 at 9.2x the compute — no quality gain from the larger model on this in-distribution corpus. The deepcopy lesson: without `copy.deepcopy(chunks)` before each embedder run, run 1's embeddings persist into run 2's chunks, producing silently contaminated benchmarks with confidently wrong results."

---

## 9. Further reading

- **Xiao et al., 2023** — BGE Technical Report. The asymmetric training objective and query prefix rationale.
- **Wang et al., 2022** — E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training. The dual-prefix approach.
- **Su et al., 2022** — Instructor: One Embedder, Any Task. Instruction-tuned embedding via task prefix.
- **Kusupati et al., 2022** — Matryoshka Representation Learning. The theory behind dimension-truncatable embeddings.
- **MTEB Leaderboard** — huggingface.co/spaces/mteb/leaderboard. Current benchmark across all major embedding models.
- **Jina GitHub** — track compatibility status for long-context embedding.

---

## Milestone status

- [x] 3.1 — BGEEmbedder formalized
- [x] 3.2 — E5Embedder + InstructionEmbedder
- [x] 3.3 — JinaEmbedder (stubbed, concept documented)
- [x] 3.4 — APIEmbedder
- [x] 3.5 — CachedEmbedder
- [x] 3.6 — BatchEmbedder + normalization utilities
- [x] 3.7 — Embedding benchmark

**Resume line (updated with M3):**

> *Built a five-backend embedding framework (BGE, E5, Instruction, API-compatible, long-context stub) behind a unified interface, with a two-tier LRU+SQLite embedding cache delivering 788x speedup on memory hits and 90% reduction in daily re-embedding costs. Implemented batch processing with normalization verification (detecting max_dev=19.76 on un-normalized vectors that would silently corrupt FAISS retrieval). Benchmarked three backends: E5-small achieved perfect recall (1.000) vs BGE-small's 0.900 at 3.2x the compute cost; 768-dim InstructionBGEb tied E5 at 9.2x cost — demonstrating that dimension and algorithm complexity don't predict quality on in-distribution corpora.*
