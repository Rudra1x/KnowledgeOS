# KnowledgeOS — Benchmark Results

## Baseline (M0 spine)

**Pipeline:** TXT loader → Fixed chunker (512/50) → BGE-small → FAISS flat → Vector retriever → Nemotron/Ling generator

**Corpus:** 10-paragraph RAG primer, 6 chunks
**Gold set:** 10 hand-labeled queries

| Metric       | Score |
| ------------ | ----- |
| recall@1     | 0.800 |
| recall@3     | 1.000 |
| recall@5     | 1.000 |
| MRR          | 0.900 |
| faithfulness | 0.531 |

**Observations:**

- Retrieval is near-ceiling on this corpus size — expect it to degrade on larger corpora
- Faithfulness is the weak link; reranking + tighter prompting are likely wins
- Gap between recall@1 (0.8) and recall@3 (1.0) motivates M6 reranker
