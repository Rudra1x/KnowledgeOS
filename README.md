# KnowledgeOS

A modular, benchmark-driven **Retrieval Intelligence Platform** built from scratch.

Every component is implemented from first principles and benchmarked against the same evaluation harness. The goal is to understand *why* each architectural decision works, not just that it does.

---

## What this is

KnowledgeOS is a full-stack RAG (Retrieval-Augmented Generation) platform covering the complete pipeline from raw documents to cited, grounded, quality-verified answers. Every stage is an ABC with multiple interchangeable implementations, swappable via a single YAML config line.

Built with production discipline: multi-tenant isolation from day one, local-first LLM strategy, scientific evaluation at every milestone, grounded generation with inline citations and faithfulness verification.

---

## Current status

| Milestone | Status | Deliverable |
|-----------|--------|-------------|
| M0 — Spine | ✅ Complete | End-to-end RAG with eval harness |
| M1 — Ingestion | ✅ Complete | 8-format loader + normalization pipeline |
| M2 — Chunking | ✅ Complete | 6-strategy portfolio + benchmark |
| M3 — Embedding | ✅ Complete | 5-backend portfolio + cache + benchmark |
| M4 — Indexing | ✅ Complete | TF-IDF, BM25, FAISS ANN, Chroma, Qdrant, RAPTOR, Hybrid RRF |
| M5 — Retrieval | ✅ Complete | 8 strategies from filtered to agentic ReAct |
| M6 — Reranking | ✅ Complete | Cross-encoder, BGE, LLM, metadata rerankers |
| M7 — Generation | ✅ Complete | Grounded prompts, compression, faithfulness NLI, relevance, streaming |
| M8 — Evaluation | 🔜 Planned | Full RAGAS-style harness |

---

## Architecture

```
Raw documents (8 formats)
        │
        ▼
LoaderRouter (TXT·PDF·DOCX·MD·HTML·CSV·Email·YouTube)
        │
        ▼
NormalizationPipeline (TextCleaner·LanguageDetector·MetadataEnricher)
        │
        ▼
Chunker × 6 (Overlapping·Recursive·Semantic·ParentChild·Adaptive·MetaAware)
        │
        ▼
Embedder × 5 + CachedEmbedder (788x L1 speedup) + BatchEmbedder
        │
        ▼
Index × 7 (TF-IDF·BM25·FAISS Flat/IVF/HNSW·Chroma·Qdrant·RAPTOR)
        │
        ▼
Retriever × 8 (Vector·Hybrid·Filtered·QueryRewrite·MultiQuery·MultiHop·SelfRAG·CRAG·Agentic)
        │ top-20 candidates
        ▼
Reranker × 4 (MS-MARCO·BGE·LLM·Metadata)
        │ top-3 reranked
        ▼
ContextCompressor (similarity/llm/budget, 0.63 mean ratio)
        │ compressed chunks
        ▼
build_prompt() → [1][2][3] numbered grounded context
        │
        ▼
StreamingGenerator (Ollama primary → OpenRouter fallback, SSE streaming)
        │ answer with inline [N] citations
        ▼
extract_citations() → {N: Chunk} traceability
        │ [async]
        ▼
FaithfulnessChecker (NLI + word overlap, 176ms)
        │ [async]
        ▼
AnswerRelevanceScorer (reverse question generation, RAGAS-inspired)
```

---

## LLM setup

| Layer | Primary | Fallback |
|-------|---------|---------|
| Generation | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |
| Retrieval (rewriting, CRAG, SelfRAG, Agentic) | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |
| Relevance scoring | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |

Start Ollama before running any LLM-dependent component: `ollama serve`

---

## Benchmark results

### M2 — Chunking
Semantic chunking ranked last (recall@1=0.800). Chunk size predicted recall better than algorithm complexity.

### M3 — Embedding
E5-small (recall@1=1.000) > BGE-small (0.900) at 3× cost. 768-dim didn't help on in-distribution content. CachedEmbedder: 788× L1 speedup.

### M4 — Indexing
All four methods tied at recall@1=0.900. RAPTOR correctly routes thematic queries to summary nodes (score=0.84 vs leaf 0.71). Hybrid RRF needs divergent failure modes.

### M5 — Retrieval

| Retriever | recall@1 | time/10q |
|-----------|----------|----------|
| Agentic | 1.000 | 230s |
| Vector/Hybrid/Filtered | 0.900 | 0.2s |
| MultiQuery | 0.800 | 58s |

### M6 — Reranking

| Pipeline | recall@1 | time/10q |
|----------|----------|----------|
| Hybrid + MS-MARCO | 1.000 | 1.9s |
| Vector + MS-MARCO | 1.000 | 3.2s |
| Vector (no rerank) | 0.900 | 0.5s |
| Vector + LLM | 0.900 | 360s |

**Key finding:** Hybrid + MS-MARCO = perfect recall at 190ms/query. 72× faster than agentic retrieval for the same result.

### M7 — Generation (5 queries, CPU, Qwen2.5-3B)

| Metric | Score |
|--------|-------|
| Retrieval recall@1 | 1.000 |
| Context compression ratio | 0.63 |
| Faithfulness (NLI) | 0.80 |
| Answer relevance | 0.73 |
| TTFT (CPU) | 7.4s |
| TTFT (GPU estimate) | ~500ms |

Full results in `RESULTS.md`.

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
│   ├── streaming_generator.py # StreamingGenerator (SSE, TTFT measurement)
│   ├── prompt_builder.py    # build_prompt(), extract_citations()
│   ├── context_compressor.py # ContextCompressor (similarity/llm/budget)
│   ├── faithfulness_checker.py # FaithfulnessChecker (NLI + word overlap)
│   └── answer_relevance.py  # AnswerRelevanceScorer (RAGAS-inspired)
├── eval/                    # Gold set, recall@k, MRR, faithfulness
├── configs/default.yaml
├── scripts/                 # All benchmark and test scripts
├── docs/                    # Milestone + checkpoint documentation
└── RESULTS.md               # All benchmark results, chronological
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

# 3. Local LLM (required for M5-M7)
# Install Ollama: https://ollama.com
ollama pull qwen2.5:3b-instruct
ollama serve   # keep running in background

# 4. API keys (optional — used as fallback)
# Create .env: OPENROUTER_API_KEY=sk-or-v1-...

# 5. Baseline eval
python scripts/run_eval.py
```

---

## Key design decisions

**Plugin architecture.** 7 ABCs — swap any component via one YAML line.

**Eval-first.** Gold set in M0, every component benchmarked against it. No vibes.

**`tenant_id` from day one.** Multi-tenant isolation in the dataclass, free at design time, painful to retrofit.

**Local LLM first.** Ollama + Qwen2.5-3B for all LLM calls. No rate limits, no rotating model IDs, no cost.

**Grounded generation.** System prompt with three rules + numbered passages + NLI faithfulness checking. Defense in depth.

**Async quality checks.** Return the answer immediately, run faithfulness + relevance in the background, flag retroactively.

---

## What I learned building this

- Semantic chunking ranked **last** — chunk size predicted recall better than algorithm sophistication.
- E5 closed BGE-small's gap at 3× cost. 768-dim didn't help on in-distribution content.
- Hybrid RRF needs divergent failure modes. RAPTOR routes thematic queries to summary nodes.
- Agentic retrieval achieved perfect recall by discovering hybrid retrieval dynamically — not because it was told to.
- **M6 central finding:** recall@3 is a retrieval problem; recall@1 is a reranking problem. A 22MB cross-encoder closes the gap that took a 23s agentic retriever, at 72× lower latency.
- NLI model selection is critical — nli-deberta-v3-small scored 0.001 on clear entailment; nli-MiniLM2-L6-H768 scored 0.796. Always validate on your domain.
- Near-verbatim claims score paradoxically low in NLI — word overlap fallback (≥0.85 → supported) is necessary.
- LLM faithfulness checking is 37s per answer; NLI is 176ms. NLI wins on every metric except custom criteria.
- TTFT=7.4s on CPU is prefill-dominated. GPU cuts to 500ms. The streaming architecture is hardware-agnostic.
- Query-agnostic metadata boosts hurt precision. Recency boost is unconditionally valid.
- Answer relevance threshold 0.85 is too strict — correct answers score 0.65-0.75 via reverse-question generation.

---

## Documentation

Every milestone has:
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

*Built milestone by milestone. Stop-safe at any point — M0 alone is a complete, demoable RAG system. M6 alone is a production-ready retrieval + reranking pipeline.*