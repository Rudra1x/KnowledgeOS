
## Multi-format (M1)

**Pipeline:** LoaderRouter (TXT/MD/CSV/PDF/EML) → NormalizationPipeline → FixedChunker → BGE-small → FAISS flat → VectorRetriever

**Corpus:** Mixed formats — 32 chunks across 5 file types

| Metric   | Score |
| -------- | ----- |
| recall@1 | 0.714 |
| recall@3 | 0.857 |
| MRR      | 0.821 |

**Per-format recall@1:**

| Format     | Score | Note                                  |
| ---------- | ----- | ------------------------------------- |
| faq.csv    | 1.000 | Field-value templating wins           |
| sample.eml | 1.000 | Subject-in-content pays off           |
| corpus.txt | 0.500 | Duplicate content elsewhere in corpus |
| sample.md  | 0.500 | recall@3=1.0 — reranker signal (M6)  |

**Observations:**

- Cross-format retrieval works fluidly; no format was silently broken
- Two "failures" are gold-set staleness (FAQ answered BM25 better than TXT) and rank-order (right chunk at rank 2/3, not 1)
- Reconfirms the M0 signal: recall@1 vs recall@3 gap → reranker is the highest-value next lever
- Slight overall drop from M0 baseline (0.8 → 0.71) — expected as gold set breadth outpaced audit

## M2 Chunking Benchmark

**Corpus:** corpus.txt (2478 chars, 10 paragraphs)
**Gold set:** 10 queries
**Embedder:** BGE-small-en-v1.5
**Index:** FAISS flat

| Chunker | recall@1 | recall@3 | MRR | chunks | avg_size |
|---------|----------|----------|-----|--------|----------|
| parent_child       | 1.000 | 1.000 | 1.000 |  10 |  279 |
| adaptive           | 1.000 | 1.000 | 1.000 |  10 |  291 |
| metadata_aware     | 1.000 | 1.000 | 1.000 |   6 |  454 |
| overlapping        | 0.900 | 1.000 | 0.950 |   6 |  454 |
| recursive          | 0.900 | 1.000 | 0.950 |   8 |  353 |
| semantic           | 0.800 | 0.900 | 0.850 |   5 |  493 |

## M3 Embedding Benchmark

**Corpus:** corpus.txt (2478 chars)
**Chunker:** RecursiveChunker (512/50)
**Gold set:** 10 queries

| Embedder | dim | recall@1 | recall@3 | MRR | ms/chunk |
|----------|-----|----------|----------|-----|----------|
| e5-small         |   384 | 1.000 | 1.000 | 1.000 | 64.1 |
| instr-bge-b      |   768 | 1.000 | 1.000 | 1.000 | 183.6 |
| bge-small        |   384 | 0.900 | 1.000 | 0.950 | 20.0 |

## M4 Index Benchmark

**Corpus:** corpus.txt  |  **Chunker:** RecursiveChunker  |  **Embedder:** BGE-small

| Index | recall@1 | recall@3 | MRR |
|-------|----------|----------|-----|
| BM25 (sparse)        | 0.900 | 1.000 | 0.950 |
| Dense (FAISS)        | 0.900 | 1.000 | 0.950 |
| Hybrid (RRF)         | 0.900 | 1.000 | 0.950 |
| RAPTOR               | 0.900 | 1.000 | 0.950 |

## M5 Retrieval (partial — Checkpoints 5.1–5.3)

**Corpus:** corpus.txt | **Chunker:** Recursive | **Embedder:** BGE-small

| Retriever | recall@1 | MRR | Notes |
|-----------|----------|-----|-------|
| Baseline (VectorRetriever) | 0.900 | 0.950 | M0 baseline |
| FilteredRetriever (CSV boost) | — | — | Format-specific routing |
| QueryRewriting (reformulate) | 0.900 | 0.950 | Qwen2.5 rewrites |
| QueryRewriting (HyDE) | 0.800 | — | HyDE hurts on small corpus |
| MultiQuery (n=3) | 0.800 | 0.900 | Union introduces noise on 8 chunks |
## M5 Retriever Benchmark

**Corpus:** corpus.txt | **Chunker:** Recursive | **Embedder:** BGE-small
**LLM:** qwen2.5:3b-instruct (Ollama local)

| Retriever | recall@1 | recall@3 | MRR | time/10q |
|-----------|----------|----------|-----|----------|
| agentic            | 1.000 | 1.000 | 1.000 | 230.3s |
| vector             | 0.900 | 1.000 | 0.950 | 0.2s |
| hybrid_rrf         | 0.900 | 1.000 | 0.950 | 0.2s |
| filtered_boost     | 0.900 | 1.000 | 0.950 | 0.2s |
| query_rewrite      | 0.900 | 1.000 | 0.950 | 34.8s |
| crag               | 0.900 | 1.000 | 0.950 | 464.5s |
| self_rag           | 0.900 | 0.900 | 0.900 | 306.8s |
| multi_query        | 0.800 | 1.000 | 0.900 | 58.1s |

## M6 Reranker Benchmark

**Corpus:** corpus.txt | **Retriever:** Vector (fetch_k=5) + Hybrid
**Reranker models:** MS-MARCO-MiniLM-L6, BGE-reranker-base, LLM (Qwen2.5-3B), Metadata

| Pipeline | recall@1 | recall@3 | MRR | time/10q |
|----------|----------|----------|-----|----------|
| Vector + MS-MARCO      | 1.000 | 1.000 | 1.000 | 3.2s |
| Vector + BGE           | 1.000 | 1.000 | 1.000 | 16.6s |
| Hybrid + MS-MARCO      | 1.000 | 1.000 | 1.000 | 1.9s |
| Hybrid + BGE           | 1.000 | 1.000 | 1.000 | 10.8s |
| Vector (no rerank)     | 0.900 | 1.000 | 0.950 | 0.5s |
| Hybrid (no rerank)     | 0.900 | 1.000 | 0.950 | 0.6s |
| Vector + LLM           | 0.900 | 1.000 | 0.950 | 360.0s |
| Vector + Metadata      | 0.700 | 1.000 | 0.850 | 0.5s |

## M7 Generation Benchmark

**Pipeline:** Vector retrieval → MS-MARCO rerank → Similarity compress → Qwen2.5-3B generate → NLI faithfulness → Relevance score
**Corpus:** corpus.txt | **LLM:** qwen2.5:3b-instruct (Ollama)

| Query | r@1 | Faithfulness | Relevance | TTFT |
|-------|-----|-------------|-----------|------|
| What is RAG?                           | 1 | 1.00 | 0.57 | 7581ms |
| How does chunking affect retrieval quality? | 1 | 1.00 | 0.79 | 7115ms |
| What is BM25 and when does it work well? | 1 | 1.00 | 0.71 | 7635ms |
| How is dense retrieval different from BM25? | 1 | 0.00 | 0.86 | 7526ms |
| What is hybrid retrieval?              | 1 | 1.00 | 0.74 | 7231ms |

**Mean:** r@1=1.00 | faith=0.80 | relevance=0.73 | ttft=7417ms

## M8 Pipeline Evaluation: hybrid_rerank_v1

**Gold set:** 12 queries (v2) | **Total time:** 261.7s

| Metric | Score |
|--------|-------|
| recall@1 | 0.8 |
| recall@3 | 0.8 |
| nDCG@3 | 0.8 |
| MRR | 0.8 |
| Faithfulness | 0.6917 |
| Citation coverage | 0.2997 |
| Answer relevance | 0.8223 |
| Negative decline rate | 1.0 |
