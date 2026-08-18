# Milestone 4 — Indexing Engine

**Status:** ✅ Complete
**Duration:** 7 checkpoints
**Deliverable:** A seven-index portfolio (TF-IDF, BM25, FAISS Flat/IVF/HNSW, Chroma, Qdrant, RAPTOR) plus a HybridRetriever implementing RRF from scratch, all benchmarked against the same gold set.

---

## 1. Milestone summary

### Goal
Build a portfolio of indexing strategies spanning the full spectrum from sparse (keyword) to dense (semantic) to research-grade (multi-level summary trees). Implement Reciprocal Rank Fusion from scratch to enable hybrid retrieval. Benchmark all approaches and understand *when* each wins.

### Why this milestone matters
The index determines the trade-offs your system makes at query time: speed vs recall, keyword precision vs semantic generality, memory vs disk, single-granularity vs multi-granularity. Understanding the full index portfolio — and having built each from scratch — means you can choose the right tool and defend that choice with evidence.

### What "done" looks like
- TF-IDF and BM25 implemented from scratch — no sklearn for the core algorithm
- FAISS IVF and HNSW wired with correct training/parameter guidance
- Chroma and Qdrant wired with persistence, deletion, and multi-tenancy verified
- RAPTOR tree built, empirically proving thematic queries → summary nodes
- HybridRetriever with RRF from scratch
- 4-way benchmark with per-query diagnostic and honest interpretation
- RESULTS.md updated

---

## 2. Architecture recap

### The index portfolio

```
                        Index (ABC)
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   SPARSE                  DENSE                 RESEARCH
        │                    │                    │
  TFIDFIndex           FaissFlatIndex         RAPTORIndex
  BM25Index            FaissIVFIndex          (multi-level
                        FaissHNSWIndex          summary tree)
                        ChromaIndex
                        QdrantIndex
```

### The hybrid retrieval architecture

```
query
  │
  ├──→ BM25Index.search_text(query, top_k=20)
  │           ↓
  │    BM25 ranked list  [rank 1..20]
  │
  └──→ embed(query) → FaissFlatIndex.search(vec, top_k=20)
              ↓
       Dense ranked list  [rank 1..20]
              │
              ▼
    RRF: score(chunk) = 1/(60 + rank_bm25) + 1/(60 + rank_dense)
              │
              ▼
       Final ranked list  [top_k]
```

### The RAPTOR tree

```
Level 2 (root):  [1 global summary]
                        │
Level 1 (clusters): [3 cluster summaries] ← LLM-generated
                        │
Level 0 (leaves): [8 original chunks]  ← raw text

All levels indexed in one flat FAISS index.
Queries match at whichever level is most similar.
```

---

## 3. Technical deep dive

### 3.1 Sparse retrieval — the term-matching family

**TF-IDF:**
```
score(t, d) = TF(t,d) × IDF(t)
TF(t,d)     = count(t,d) / total_terms(d)
IDF(t)      = log((1+N)/(1+df(t))) + 1   [smoothed]
```

**BM25:**
```
score(t, d) = IDF_RSJ(t) × TF_BM25(t, d)
IDF_RSJ(t)  = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
TF_BM25     = tf × (k1+1) / (tf + k1 × (1 - b + b × |d| / avgdl))
```

Key differences:
- BM25 TF is *saturating* (bounded by k1+1) vs TF-IDF's linear growth
- BM25 normalizes by document length via `b`; TF-IDF uses raw length normalization
- BM25 uses Robertson-Sparck Jones IDF (always positive); original IDF can go negative

Both require two-pass indexing: collect `df` in pass 1, compute scores in pass 2.

Both are lexical — they cannot match synonyms or paraphrases. That's the gap dense retrieval fills.

### 3.2 ANN indexing — the speed-recall frontier

**The exact-to-approximate spectrum:**

| Index | Recall | Speed (vs flat) | Memory | Training |
|-------|--------|-----------------|--------|----------|
| Flat | 100% | 1× | Low | None |
| IVF (nprobe=10) | ~90% | 10-100× | Low | Yes (39×nlist vectors) |
| HNSW (ef=50) | ~95% | 50-500× | High | None |
| IVFPQ | ~85% | 100-1000× | Minimal | Yes |

**IVF nlist sizing:**
```
nlist ≈ sqrt(N) to 4×sqrt(N)
Training points ≥ 39 × nlist
```

For N=1M: nlist=1000-4000, training points=39K-156K.

**HNSW parameter guide:**
- M=16: good default for recall/memory. M=32 for higher recall, M=8 for memory savings.
- ef_construction=200: higher = better graph quality, slower build.
- ef_search=50: higher = better query recall, slower queries. Start at 50, tune for your SLA.

### 3.3 Managed vector DBs — the production layer

**Chroma:**
- DuckDB + Parquet storage (automatic persistence)
- Collections = isolated namespaces (one per tenant)
- Native deletion by ID
- Distance semantics: returns `distance`, convert to similarity with `1 - distance`
- Collection name constraints: 3-512 chars, alphanumeric

**Qdrant:**
- Pre-filter via payload FieldCondition (before ANN traversal)
- Payload stored atomically with vector (no parallel list)
- Native hybrid search (sparse + dense in one query)
- Disk-based HNSW (RAM not required)
- `query_points().points` (modern API, deprecated `search()`)
- `delete + create` pattern (deprecated `recreate_collection`)

**When to choose:**
- Development/small production: Chroma (simpler, no server required)
- Large production: Qdrant (pre-filter scales, disk-based, hybrid support)
- Extreme scale/distributed: Milvus, Weaviate, Pinecone

### 3.4 Reciprocal Rank Fusion — the fusion formula

```
RRF(d) = Σ_i weight_i / (k + rank_i(d))
```

Where k=60 is the dampening constant from Cormack et al. (2009).

**Why k=60:** without the constant, rank-1 in one list scores `weight/1 = weight`, dominating all other results regardless of what the other list says. k=60 ensures rank-1 scores `weight/61`, which still wins but doesn't completely dominate.

**Why RRF over score normalization:**
- Score scales differ: BM25 ~[0, 5], cosine ~[0, 1]
- Min-max normalization: one outlier shifts the whole scale
- Z-score normalization: requires knowing the distribution in advance
- RRF: only uses ranks — no calibration needed, no scale assumptions

**Hybrid advantage requires divergent failures:** if BM25 and dense fail on the same queries, RRF cannot rescue them. The advantage appears when: BM25 excels on exact entity queries that dense misses; dense excels on paraphrase queries that BM25 misses. Corpus diversity determines whether hybrid wins.

### 3.5 RAPTOR — the multi-granularity solution

**Why standard RAG fails on thematic queries:**
Standard RAG embeds individual chunks. No chunk represents "what is this document about?" — that's a summary-level representation that doesn't exist in the index.

**How RAPTOR solves it:**
1. Cluster chunks in embedding space (K-Means or GMM)
2. Summarize each cluster with an LLM
3. The summaries become new indexable nodes
4. Repeat recursively until one root summary

**The score hierarchy:**
- Specific query "What is BM25?" → leaf matches at 0.71 (exact content)
- Thematic query "How does retrieval work?" → summary matches at 0.84 (aggregate theme)

The summary scores *higher* because its embedding explicitly encodes the cluster theme, not scattered individual facts.

**Production costs:**
- Index build: O(N_clusters × levels) LLM calls
- For 1000 chunks, 3 clusters, 2 levels: ~1000 LLM calls
- Rebuild required on corpus update (cluster assignments may shift)

---

## 4. Design decisions and trade-offs

### 4.1 Pre-computing BM25 scores vs computing at query time
We pre-compute during `add()` — fast queries, slow indexing. Alternative: store raw TF, compute BM25 at query time — fast indexing, slower queries. For read-heavy RAG (many queries, rare updates), pre-computing is the right trade.

### 4.2 One collection per tenant in Chroma vs single collection with metadata filter
One collection per tenant: native isolation, no post-filter, clean API. Single collection with metadata filter: simpler management, but Chroma's filtering is less efficient than Qdrant's native pre-filter. For serious multi-tenancy, one collection per tenant is the right architecture.

### 4.3 RAPTOR with K-Means vs GMM
RAPTOR's original paper uses Gaussian Mixture Models (GMM) — soft cluster assignments where a chunk can belong to multiple clusters with different probabilities. We used K-Means (hard assignments) for simplicity and to avoid scipy dependency. GMM gives slightly better cluster quality; K-Means is simpler and faster. For most applications, K-Means is adequate.

### 4.4 Flat FAISS for RAPTOR's node index
RAPTOR has at most N + N/3 + N/9... = 1.5N nodes total. For N=1000, that's 1500 nodes — well within Flat's efficient range. No need for ANN complexity in the RAPTOR index itself. If N=1M, then 1.5M nodes would warrant HNSW, but at that scale you'd also have more RAPTOR levels.

### 4.5 Honest benchmark reporting
The benchmark showed all four indexes tied. We could have framed this as "hybrid works as well as sparse and dense on this corpus" — technically true but misleading. The honest interpretation — "the corpus is too small and uniform to reveal differences; here's exactly why" — is more valuable. Benchmark honesty is professional discipline.

---

## 5. Common pitfalls

1. **IVF with too few training points** → FAISS warns but auto-adapts; quality degrades. Check: n_training_points ≥ 39 × nlist.
2. **Not converting Chroma's distance to similarity** → retrieval returns least-similar chunks. Fix: `score = 1 - distance`.
3. **Using `client.search()` in new Qdrant** → AttributeError. Fix: `client.query_points(...).points`.
4. **Using `recreate_collection` in new Qdrant** → deprecated. Fix: explicit delete + create.
5. **RAPTOR summarization through RAG prompt builder** → model answers the instruction instead of summarizing. Fix: direct API call with plain summarization prompt.
6. **Hybrid RRF on divergent-failure queries** → both legs fail, hybrid cannot rescue. The fix is upstream (chunking, corpus quality).
7. **Collection name < 3 chars in Chroma** → InvalidArgumentError at runtime. Fix: validate at init.
8. **Not over-fetching for FAISS post-filter** → fewer than top_k results returned after tenant filter. Fix: fetch top_k × 2 and filter.
9. **TFIDFIndex computing IDF before all documents added** → IDF is wrong (corpus incomplete). Fix: two-pass indexing.
10. **HNSW efConstruction vs efSearch confusion** → construction is for graph quality at build time; search is for recall at query time. Higher construction doesn't help at query time if efSearch is low.
11. **Assuming hybrid always beats sparse or dense** → only on corpora where failure modes diverge. Always benchmark.

---

## 6. Benchmarks and results

### M4 index benchmark

| Rank | Index | recall@1 | recall@3 | MRR |
|------|-------|----------|----------|-----|
| 1 (tie) | BM25 (sparse) | 0.900 | 1.000 | 0.950 |
| 1 (tie) | Dense (FAISS) | 0.900 | 1.000 | 0.950 |
| 1 (tie) | Hybrid (RRF) | 0.900 | 1.000 | 0.950 |
| 1 (tie) | RAPTOR | 0.900 | 1.000 | 0.950 |

**One miss:** "How is dense retrieval different from BM25?" — chunk boundary artifact, all methods fail identically.

**Honest interpretation:** The corpus (8 chunks, uniform length, clean text) and gold set (10 specific factoid queries) are too uniform to reveal index differences. All methods fail on the same one query. The conditions that differentiate methods:
- Hybrid advantage requires: diverse query types (keyword + semantic) AND large corpus with varied vocabulary
- RAPTOR advantage requires: thematic queries in the gold set
- BM25 > dense requires: exact entity queries, rare terms
- Dense > BM25 requires: paraphrase queries, synonym queries

**What RAPTOR did prove (outside the benchmark):** thematic queries ("How does retrieval work?") retrieved L1 summary nodes at score=0.843 vs specific queries retrieving L0 leaves at score=0.709. Multi-granularity indexing works empirically.

---

## 7. Interview mock exam

### Section A — Fundamentals (10 questions)

1. What is an inverted index and how does it enable fast text search?
2. What are the two improvements BM25 makes over TF-IDF?
3. What does the k1 parameter in BM25 control?
4. What is the nprobe parameter in IVF and what does it trade?
5. What is HNSW and why does it require no training?
6. What is the key advantage of Chroma over FAISS?
7. What does Qdrant's native pre-filtering do that FAISS cannot?
8. What is RAPTOR and what problem does it solve?
9. What is Reciprocal Rank Fusion and why not add scores directly?
10. What is the k=60 constant in RRF and why does it matter?

### Section B — Applied Understanding (15 questions)

11. TF-IDF and BM25 produced identical rankings on your 8-chunk corpus. Why?
12. Your IVF index warns "clustering 8 points to 4 centroids: please provide at least 156 training points." What does this mean and how do you fix it?
13. Chroma returns `distance=0.29`. What is the cosine similarity score?
14. You call `client.search()` on Qdrant and get `AttributeError`. What happened?
15. RAPTOR thematic query scored 0.84 vs leaf's 0.71. Why did the summary score higher?
16. Your hybrid retriever returned the same results as BM25-only. Diagnose the root cause.
17. How does Qdrant's pre-filter reduce search cost from O(N) to O(N/tenants)?
18. Why did your RAPTOR summarization fail when passed through generator.generate()?
19. A query for "automobile" fails in your BM25 index but retrieves correctly with dense. Why?
20. Your Chroma collection name is "mt" and it crashes with InvalidArgumentError. Fix it.
21. Compare HNSW ef_construction and ef_search. What does each control?
22. How many LLM calls does RAPTOR make for 1000 chunks, 3 clusters, 2 levels?
23. Your IVF index has nlist=100, nprobe=10. What fraction of vectors does each query search?
24. Why does RAPTOR require a full rebuild when new documents are added?
25. BM25 and dense fail on the same query. Will hybrid help?

### Section C — Design and Trade-offs (10 questions)

26. Design an index architecture for a 100M-document corpus with 10K tenants, daily updates, and <50ms p99 query latency.
27. A client needs GDPR right-to-erasure (delete user data within 24 hours). Which indexes support this? Which don't?
28. Compare FAISS Flat, IVF, and HNSW for a corpus that grows from 10K to 10M vectors over 2 years.
29. Your BM25 index takes 2 hours to rebuild daily. How would you make it incremental?
30. Design a hybrid retrieval system that automatically weights BM25 vs dense based on query type (entity vs semantic).
31. Your RAPTOR summaries are too generic (the LLM summarizes broadly instead of preserving key facts). How do you fix the prompts?
32. A client has documents in 5 languages. How does this change your index architecture?
33. Compare Chroma (persistent local) vs Qdrant (pre-filter + disk HNSW) for a 50K-document single-tenant RAG system.
34. You're migrating from FAISS to Qdrant. What data needs to be migrated and what can be reconstructed?
35. Design a benchmark that would reveal hybrid retrieval's advantage on your corpus. What query types do you add?

### Section D — Whiteboard Coding (5 questions)

36. Implement `BM25._idf(term: str) → float` using Robertson-Sparck Jones IDF. You have `self.n_docs` and `self.doc_freq`.
37. Implement `HybridRetriever._rrf_fuse(bm25_results, dense_results, k=60) → dict[str, float]` mapping chunk_id to RRF score.
38. Implement `TFIDFIndex._tokenize(text: str) → list[str]` — lowercase, strip punctuation, filter short tokens.
39. Write pseudocode for RAPTOR's `_build_level(nodes, level)` — cluster, summarize, embed, return summary chunks.
40. Implement `verify_tenant_isolation(index, qvec, tenant_a, tenant_b)` — a test function that asserts no cross-tenant results.

---

## 8. Project walkthrough scripts

### 8.1 The 30-second pitch

> "M4 added seven index types to KnowledgeOS — TF-IDF and BM25 from scratch, three FAISS variants, Chroma, and Qdrant. Plus a hybrid retriever using Reciprocal Rank Fusion from scratch, and a RAPTOR multi-level summary tree. The benchmark showed all four methods tied on the small corpus — the honest finding is that hybrid's advantage requires divergent failure modes between the sparse and dense legs, which our uniform corpus doesn't have. RAPTOR correctly routes thematic queries to summary nodes, proven empirically."

### 8.2 The 2-minute technical walkthrough

> "The indexing layer spans sparse, dense, and research-grade. BM25 I built from scratch — two passes over the corpus: first collect document frequencies, then compute scores using TF saturation (k1=1.5 bounds TF at 2.5) and length normalization (b=0.75 penalizes longer documents). TF-IDF is simpler but linear — BM25's improvements matter on corpora with length variance and repeated terms.
>
> For dense ANN, FAISS IVF clusters vectors with k-means and searches only nearby clusters at query time — 10-100x faster than exact at small recall cost, controlled by nprobe. HNSW builds a layered graph — state-of-the-art recall/speed, no training required, but doesn't support deletion.
>
> For managed vector DBs, Chroma adds persistence and deletion over FAISS, but its filtering is post-retrieval. Qdrant adds pre-filtering — the payload filter runs before ANN traversal, so you search only the matching tenant's vectors. On 1M vectors with 1000 tenants, that's 1000x less work than FAISS post-filter.
>
> The hybrid retriever uses Reciprocal Rank Fusion: `1/(60 + rank_bm25) + 1/(60 + rank_dense)`. Rank-based fusion is scale-agnostic — BM25 scores around 4.5 and cosine around 0.7 can't be added directly, but their ranks can. The k=60 prevents rank-1 dominance.
>
> The benchmark showed all four methods tied at 0.900. The honest interpretation: hybrid needs divergent failure modes, which our 8-chunk uniform corpus doesn't have. The infrastructure is correct; the test conditions don't exercise the interesting cases."

### 8.3 The 5-minute deep walkthrough

> "Let me walk through the three main areas: sparse retrieval from scratch, managed vector DBs, and the hybrid benchmark finding.
>
> **BM25 from scratch.** Standard TF-IDF has two problems that BM25 fixes. First, TF grows linearly — a document with 'retrieval' 100 times scores 100x more than one with it once. BM25 saturates TF at k1+1=2.5 via a rational function: the 10th mention contributes much less than the 1st. Second, TF-IDF's length normalization is crude. BM25 has a dedicated parameter b=0.75 that partially normalizes — a chunk 33% longer than average has TF counts penalized by ~25%. Robertson's IDF always stays positive, preventing common terms from hurting queries. Building it from scratch required two-pass indexing: you can't compute IDF until you've seen all documents. This is why streaming ingestion requires approximate IDF. On our uniform 8-chunk corpus, BM25 and TF-IDF ranked identically — the improvements only manifest with length variance and high TF terms.
>
> **Qdrant vs Chroma.** The headline difference is pre-filter vs post-filter. FAISS and Chroma retrieve globally then filter by tenant — on 1M vectors with 1000 tenants, you search 1M and discard 99.9%. Qdrant applies a `FieldCondition` on `tenant_id` before ANN traversal — you search only that tenant's ~1000 vectors. The API evolution matters here: `client.search()` was replaced by `client.query_points().points`, and `recreate_collection` by explicit delete+create. Both are pattern-level lessons: always check return types when SDKs upgrade, and always read deprecation notices before upgrading.
>
> **The honest benchmark finding.** All four methods tied at 0.900 — one query fails identically across BM25, dense, hybrid, and RAPTOR. The diagnostic showed it's a chunk boundary artifact where the most discriminative tokens appear in the overlap region of the chunk. Hybrid cannot rescue a query both legs fail on — RRF sums two zeros. The conditions for hybrid to win are: diverse query types (keyword + semantic) and a corpus with varied vocabulary where failure modes diverge. Our 8-chunk uniform corpus has neither. Reporting this honestly, with the per-query breakdown showing which query fails and why, is more valuable than finding a synthetic test case that shows hybrid winning."

---

## 9. Further reading

- **Cormack et al., 2009** — "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods." The original RRF paper with k=60 derivation.
- **Robertson & Sparck Jones, 1976** — BM25 IDF. The Robertson-Sparck Jones IDF formula and its probabilistic derivation.
- **Sarthi et al., 2024** — "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." Stanford. The original RAPTOR paper.
- **FAISS documentation** — github.com/facebookresearch/faiss/wiki — "Guidelines to choose an index" is the canonical nlist/nprobe/M/ef guide.
- **BEIR benchmark** — github.com/beir-cellar/beir — 18 diverse retrieval tasks. Use this to validate that hybrid beats sparse/dense on a realistic corpus.
- **Qdrant documentation** — qdrant.tech/documentation — the filtering and payload indexing sections.

---

## Milestone status

- [x] 4.1 — TFIDFIndex from scratch
- [x] 4.2 — BM25Index from scratch
- [x] 4.3 — FaissIVFIndex + FaissHNSWIndex
- [x] 4.4 — ChromaIndex
- [x] 4.5 — QdrantIndex
- [x] 4.6 — RAPTORIndex
- [x] 4.7 — Index benchmark + HybridRetriever (RRF)

**Resume line (updated with M4):**

> *Extended KnowledgeOS with a seven-index portfolio: TF-IDF and BM25 from scratch (inverted index, IDF weighting, TF saturation, length normalization), FAISS IVF and HNSW (ANN with nprobe/ef_search tuning), Chroma (persistence + deletion), Qdrant (pre-filter payload isolation), and RAPTOR (K-Means clustering + LLM summarization + multi-level tree). Implemented Reciprocal Rank Fusion (RRF) from scratch for hybrid sparse+dense retrieval. Empirically proved RAPTOR routes thematic queries to summary nodes (score=0.84) vs leaves for specific queries (score=0.71). Honest benchmark finding: hybrid advantage requires divergent failure modes — documented why the small uniform corpus doesn't exhibit them and what corpus characteristics would.*
