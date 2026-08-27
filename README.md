# KnowledgeOS

A modular, benchmark-driven **Retrieval Intelligence Platform** built from scratch — eight milestones from raw documents to cited, grounded, quality-verified answers.

Every component implemented from first principles, benchmarked against the same evaluation harness, documented with 40-question interview mock exams.

---

## What this is

KnowledgeOS is a full-stack RAG (Retrieval-Augmented Generation) platform. Every stage is an ABC with multiple interchangeable implementations, swappable via YAML config. Built with production discipline: multi-tenant isolation from day one, local-first LLM strategy, scientific evaluation at every milestone.

---

## Status: Complete (M0–M8) — Production-Ready

| Milestone | Deliverable | Key finding |
|-----------|-------------|-------------|
| M0 — Spine | End-to-end RAG + eval harness | Baseline recall@1=0.900 |
| M1 — Ingestion | 8-format loader + normalization | Structured formats hit recall=1.000 |
| M2 — Chunking | 6-strategy portfolio + benchmark | Semantic (most complex) ranked last |
| M3 — Embedding | 5-backend + cache + benchmark | E5 closes gap at 3× cost; 788× cache speedup |
| M4 — Indexing | TF-IDF, BM25, FAISS, Chroma, Qdrant, RAPTOR, Hybrid | All tied; hybrid needs divergent failures |
| M5 — Retrieval | 8 strategies from filtered to agentic | Agentic=1.000 at 23s; vector=0.900 at 0.02s |
| M6 — Reranking | Cross-encoder, BGE, LLM, Metadata | Hybrid+MS-MARCO=1.000 at 0.19s (72× faster than agentic) |
| M7 — Generation | Grounded prompts, NLI faithfulness, streaming | TTFT=7.4s CPU; faith=0.955; rel=0.840 |
| M8 — Evaluation | Typed gold set, nDCG, HTML report | r@1=0.909; nDCG@3=0.955; decline=1.0 |

---

## Architecture

```
Documents → LoaderRouter (8 formats)
         → NormalizationPipeline
         → Chunker ×6 (chunk_size=300, chunk_overlap=0)
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

## Final benchmark results

### Production pipeline: Hybrid + MS-MARCO reranker

| Metric | Score | Notes |
|--------|-------|-------|
| **recall@1** | **0.909** | 1 known hard query misses at rank 1, rank 3 correct |
| **recall@3** | **1.000** | Perfect — right chunk always in top 3 |
| **nDCG@3** | **0.955** | Strong position-aware ranking |
| **MRR** | **0.939** | Near-perfect mean reciprocal rank |
| **Faithfulness** | **0.955** | NLI-verified grounding, 176ms |
| **Answer relevance** | **0.840** | RAGAS reverse-question scoring |
| **Negative decline** | **1.000** | Perfect off-topic rejection |
| TTFT (CPU) | 7.4s | GPU target: ~500ms |

**Per query type (13 queries, typed gold set):**

| Type | recall@1 | Faithfulness | Relevance | n |
|------|----------|-------------|-----------|---|
| factoid | 1.000 | 0.900 | 0.790 | 5 |
| comparison | 0.750 | 1.000 | 0.861 | 4 |
| thematic | 1.000 | 1.000 | 0.926 | 2 |
| negative | — | — | — | 2 (decline=1.0) |

**Known limitation:** "How is dense retrieval different from BM25?" retrieves at rank 3 (not rank 1) — correctly classified as `difficulty: hard`. The generated answer is correct, faithful (1.000), and highly relevant (0.920). Documented, not hidden.

**How faithfulness went from 0.692 → 0.955:** corpus rebuild with clean paragraph boundaries and zero chunk overlap eliminated boundary artifact sentences that were confusing the NLI checker. The artifacts ("ses embedding similarity to detect topic shifts. BM25 is a...") were poisoning NLI scores across the pipeline from M2 onwards.

Full benchmark history in `RESULTS.md`.

---

## LLM setup

| Layer | Primary | Fallback |
|-------|---------|---------|
| Generation | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |
| Retrieval (rewriting, CRAG, SelfRAG, Agentic) | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |
| Faithfulness / Relevance | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |

Start Ollama before running: `ollama serve`

---

## Project structure

```
KnowledgeOS/
├── core/                    # ABCs, dataclasses, config, normalizers
├── loaders/                 # 8 format loaders + router
├── chunkers/                # 6 chunking strategies
├── embedders/               # 5 backends + CachedEmbedder + BatchEmbedder
├── indexes/                 # 7 types (TF-IDF, BM25, FAISS×3, Chroma, Qdrant, RAPTOR)
├── retrievers/              # 8 strategies + HybridRetriever
├── rerankers/               # CrossEncoder, BGE, LLM, Similarity, Metadata
├── generation/
│   ├── generator.py         # OpenRouterGenerator
│   ├── local_generator.py   # LocalLLMGenerator (Ollama primary)
│   ├── streaming_generator.py  # SSE streaming, TTFT measurement
│   ├── prompt_builder.py    # build_prompt(), extract_citations()
│   ├── context_compressor.py   # similarity/llm/budget compression
│   ├── faithfulness_checker.py # NLI + word overlap, 176ms
│   └── answer_relevance.py  # RAGAS-inspired reverse-question scoring
├── eval/
│   ├── gold_set.py          # Original 10-query gold set
│   ├── gold_set_v2.py       # 13 typed queries (factoid/comparison/thematic/negative)
│   ├── metrics.py           # recall@k, MRR, nDCG, faithfulness
│   ├── retrieval_evaluator.py
│   ├── generation_evaluator.py
│   ├── pipeline_evaluator.py
│   └── report_generator.py  # Standalone HTML dashboard
├── configs/default.yaml     # chunk_size=300, chunk_overlap=0
├── scripts/                 # All benchmark and test scripts
├── docs/                    # Milestone + checkpoint documentation
│   ├── milestones/          # M0-M8 deep dives with mock exams
│   └── checkpoints/         # Per-checkpoint technical notes
├── RESULTS.md               # Complete benchmark history
└── eval_report.html         # Latest HTML dashboard (open in browser)
```

---

## Setup

```bash
# 1. Environment
conda create -n knowledgeos python=3.11 -y
conda activate knowledgeos

# 2. Dependencies
pip install torch faiss-cpu sentence-transformers pyyaml python-dotenv \
            pypdf pdfplumber python-docx trafilatura beautifulsoup4 \
            lxml pandas langdetect requests openai \
            youtube-transcript-api transformers einops \
            chromadb qdrant-client scikit-learn

# 3. Local LLM (required for M5-M8)
# Install Ollama: https://ollama.com
ollama pull qwen2.5:3b-instruct
ollama serve   # keep running in background

# 4. API keys (optional — used as fallback)
# Create .env: OPENROUTER_API_KEY=sk-or-v1-...

# 5. Baseline eval
python scripts/run_eval.py
```

---

## Run the evaluation

```bash
# Full pipeline evaluation + HTML report (~15 min on CPU)
python scripts/generate_eval_report.py
# → eval_report.html (open in browser)

# Retrieval benchmark only (fast, no LLM)
python scripts/run_retrieval_eval.py

# Retrieval strategies comparison
python scripts/run_retriever_benchmark.py

# Reranking benchmark
python scripts/run_reranker_benchmark.py

# Index benchmark
python scripts/run_index_benchmark.py
```

---

## What I learned building this

- **Complexity ≠ quality.** Semantic chunking ranked last. 768-dim didn't beat 384-dim. Multi-query hurt on small corpora. The benchmark reveals the conditions under which sophistication helps.
- **recall@3 is retrieval; recall@1 is reranking.** A 22MB cross-encoder closes the gap that took a 23s agentic retriever, at 72× lower latency.
- **Corpus quality propagates through the entire stack.** Boundary artifacts in chunks caused 26% faithfulness failures across M2-M8. Fixing chunk boundaries raised faithfulness from 0.692 → 0.955. Upstream quality determines downstream quality.
- **NLI model selection is non-trivial.** `nli-deberta-v3-small` scored 0.001 on clear entailment; `nli-MiniLM2-L6-H768` scored 0.796. Always validate on your domain before deploying.
- **Query typing reveals failure modes aggregate metrics hide.** "recall@1=0.800" becomes "comparison queries fail at 0.667; thematic queries succeed at 1.000." Completely different fixes.
- **System prompt handles negative queries for free.** decline_rate=1.0 without Self-RAG overhead — the grounded system prompt is sufficient for clean off-topic queries.
- **The evaluation harness is the most valuable artifact.** Every architecture change has a measurable impact. The platform is self-evaluating.
- **TTFT is the UX metric, TPS is the hardware metric.** CPU TTFT=7.4s, GPU TTFT=~500ms. The streaming code is hardware-agnostic.
- **Hard queries are honest, not embarrassing.** The one known miss is correctly classified as `difficulty: hard`, documented in the gold set, and has recall@3=1.0. Honest evaluation builds trust.

---

## Documentation

Every milestone has a deep-dive document with:
- Architecture recap and design decisions
- Per-checkpoint technical notes with code walkthroughs
- 40-question interview mock exam (fundamentals → applied → design → whiteboard)
- 30-second, 2-minute, and 5-minute walkthrough scripts

In `docs/milestones/` and `docs/checkpoints/`.

---

## Author

**Rudraksh Sharma** — Data Scientist at AIONOS, Technical Lead at Beerantum.
Qiskit Advocate · Berlin Quantum Hackathon 2026 (3rd place) · QIntern 2025 First Team Award.

[GitHub: Rudra1x](https://github.com/Rudra1x/KnowledgeOS)

---

*Eight milestones. Stop-safe at any point — M0 alone is demoable, M6 alone is production-ready, M8 is the complete self-evaluating platform.*