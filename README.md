# KnowledgeOS

A modular, benchmark-driven **Retrieval Intelligence Platform** built from scratch — eight milestones from raw documents to cited, grounded, quality-verified answers.

Every component implemented from first principles, benchmarked against the same evaluation harness, documented with 40-question interview mock exams.

---

## What this is

KnowledgeOS is a full-stack RAG (Retrieval-Augmented Generation) platform. Every stage is an ABC with multiple interchangeable implementations, swappable via YAML config. Built with production discipline: multi-tenant isolation from day one, local-first LLM strategy, scientific evaluation at every milestone.

---

## Status: Complete (M0–M8)

| Milestone | Deliverable | Key finding |
|-----------|-------------|-------------|
| M0 — Spine | End-to-end RAG + eval harness | Baseline recall@1=0.900 |
| M1 — Ingestion | 8-format loader + normalization | Structured formats hit recall=1.000 |
| M2 — Chunking | 6-strategy portfolio + benchmark | Semantic (most complex) ranked last |
| M3 — Embedding | 5-backend + cache + benchmark | E5 closes gap at 3× cost; 788× cache speedup |
| M4 — Indexing | TF-IDF, BM25, FAISS, Chroma, Qdrant, RAPTOR, Hybrid | All tied; hybrid needs divergent failures |
| M5 — Retrieval | 8 strategies from filtered to agentic | Agentic=1.000 at 23s; vector=0.900 at 0.02s |
| M6 — Reranking | Cross-encoder, BGE, LLM, Metadata | Hybrid+MS-MARCO=1.000 at 0.19s (72× faster than agentic) |
| M7 — Generation | Grounded prompts, NLI faithfulness, streaming | TTFT=7.4s CPU; faith=0.800; rel=0.815 |
| M8 — Evaluation | Typed gold set, nDCG, HTML report | r@1=0.800; decline_rate=1.0 |

---

## Architecture

```
Documents → LoaderRouter (8 formats)
         → NormalizationPipeline
         → Chunker ×6
         → Embedder ×5 + CachedEmbedder (788× speedup)
         → Index ×7 (TF-IDF, BM25, FAISS Flat/IVF/HNSW, Chroma, Qdrant, RAPTOR)
         → Retriever ×8 (Vector, Hybrid, Filtered, QueryRewrite,
                          MultiQuery, MultiHop, SelfRAG, CRAG, Agentic)
         → Reranker ×4 (MS-MARCO, BGE, LLM, Metadata)
         → ContextCompressor (similarity/llm/budget)
         → StreamingGenerator (Ollama → OpenRouter fallback)
         → extract_citations() → {N: Chunk} traceability
         → FaithfulnessChecker (NLI + word overlap, 176ms)
         → AnswerRelevanceScorer (RAGAS-inspired)
         → PipelineEvaluator → HTML report
```

---

## Benchmark results

### M6 — Reranking (production-ready pipeline)

| Pipeline | recall@1 | time/10q |
|----------|----------|----------|
| **Hybrid + MS-MARCO** | **1.000** | **1.9s** |
| Vector + MS-MARCO | 1.000 | 3.2s |
| Vector (no rerank) | 0.900 | 0.5s |
| Vector + LLM reranker | 0.900 | 360s |

### M8 — Full pipeline evaluation (12 typed queries)

| Metric | Score |
|--------|-------|
| recall@1 | 0.800 |
| nDCG@3 | 0.800 |
| Faithfulness (NLI) | 0.692 |
| Answer relevance | 0.822 |
| Negative decline rate | 1.000 |

**Per query type:**
| Type | recall@1 | Faithfulness | Relevance |
|------|----------|-------------|-----------|
| factoid | 0.800 | 0.733 | 0.767 |
| comparison | 0.667 | 0.667 | 0.855 |
| thematic | 1.000 | 0.625 | 0.912 |

Full results in `RESULTS.md`.

---

## Setup

```bash
conda create -n knowledgeos python=3.11 -y
conda activate knowledgeos

pip install torch faiss-cpu sentence-transformers pyyaml python-dotenv \
            pypdf pdfplumber python-docx trafilatura beautifulsoup4 \
            lxml pandas langdetect requests openai \
            youtube-transcript-api transformers einops \
            chromadb qdrant-client scikit-learn

# Local LLM (required for M5-M8)
ollama pull qwen2.5:3b-instruct
ollama serve   # keep running

# Optional: OPENROUTER_API_KEY=... in .env
```

---

## Run the evaluation

```bash
# Full pipeline evaluation + HTML report
python scripts/generate_eval_report.py
# → eval_report.html (open in browser)

# Retrieval benchmark only (fast)
python scripts/run_retrieval_eval.py

# All retriever strategies
python scripts/run_retriever_benchmark.py

# Reranking benchmark
python scripts/run_reranker_benchmark.py
```

---

## What I learned building this

- **Complexity ≠ quality.** Semantic chunking ranked last. 768-dim embeddings didn't beat 384-dim. Multi-query hurt on small corpora. The benchmark always reveals the conditions under which sophistication helps.
- **recall@3 is retrieval; recall@1 is reranking.** A 22MB cross-encoder closes the recall gap that took a 23s agentic retriever, at 72× lower latency.
- **NLI model selection is non-trivial.** `nli-deberta-v3-small` scored 0.001 on clear entailment; `nli-MiniLM2-L6-H768` scored 0.796. Always validate on your domain.
- **Query typing reveals failure modes aggregate metrics hide.** "recall@1=0.800" → "comparison queries fail at 0.667; thematic queries succeed at 1.000." Completely different fixes.
- **System prompt handles negative queries for free.** decline_rate=1.0 without Self-RAG overhead — the grounded prompt's "if not in context, say so" is sufficient for clean off-topic queries.
- **TTFT is the UX metric, TPS is the hardware metric.** CPU TTFT=7.4s, GPU TTFT=~500ms. The streaming code is hardware-agnostic.
- **The evaluation harness is the most valuable artifact.** Every architecture change now has a measurable impact. The platform is self-evaluating.

---

## Documentation

Every milestone has a deep-dive document with architecture recap, per-checkpoint technical notes, 40-question interview mock exam, and 30/120/300-second walkthrough scripts. In `docs/milestones/` and `docs/checkpoints/`.

---

## Author

**Rudraksh Sharma** — Data Scientist at AIONOS, Technical Lead at Beerantum.
Qiskit Advocate · Berlin Quantum Hackathon 2026 (3rd place) · QIntern 2025 First Team Award.

[GitHub: Rudra1x](https://github.com/Rudra1x/KnowledgeOS)

---

*Eight milestones. Stop-safe at any point — M0 alone is demoable, M6 alone is production-ready.*