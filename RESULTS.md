
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
