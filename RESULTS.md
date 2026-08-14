
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
