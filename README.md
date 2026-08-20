# KnowledgeOS

A modular, benchmark-driven **Retrieval Intelligence Platform** built from scratch.

Every component is implemented from first principles and benchmarked against the same evaluation harness — not imported from a library and trusted blindly. The goal is to understand *why* each architectural decision works, not just that it does.

---

## What this is

KnowledgeOS is a full-stack RAG (Retrieval-Augmented Generation) platform covering the complete pipeline from raw documents to cited, grounded answers. It is structured as a portfolio of interchangeable plugins — every stage (loader, chunker, embedder, index, retriever, reranker, generator) is an ABC with multiple implementations that can be swapped via a single YAML config line.

Built as a learning project with production discipline: multi-tenant isolation from day one, config-driven assembly, scientific evaluation at every milestone.

---

## Current status

| Milestone | Status | Deliverable |
|-----------|--------|-------------|
| M0 — Spine | ✅ Complete | End-to-end RAG with eval harness |
| M1 — Ingestion | ✅ Complete | 8-format loader + normalization pipeline |
| M2 — Chunking | ✅ Complete | 6-strategy portfolio + benchmark |
| M3 — Embedding | ✅ Complete | 5-backend portfolio + cache + benchmark |
| M4 — Indexing | ✅ Complete | TF-IDF, BM25, FAISS ANN, Chroma, Qdrant, RAPTOR, Hybrid RRF |
| M5 — Retrieval | 🔄 In progress | FilteredRetriever, QueryRewriting, MultiQuery, MultiHop, Self-RAG, CRAG, Agentic |
| M6 — Reranking | 🔜 Planned | Cross-encoder, LLM reranker |
| M7 — Generation | 🔜 Planned | Context compression, citation, streaming |
| M8 — Evaluation | 🔜 Planned | Full RAGAS-style harness |
| M9–M12 | 🔜 Future | Enterprise, knowledge graph, UI, research |

---

## Architecture

```
Raw documents (8 formats)
        │
        ▼
┌─────────────────────────────────────────┐
│            LoaderRouter                 │
│  TXT · PDF · DOCX · MD · HTML          │
│  CSV · Email · YouTube                 │
└────────────────────┬────────────────────┘
                     │  list[Document]
                     ▼
┌─────────────────────────────────────────┐
│         NormalizationPipeline           │
│  TextCleaner · LanguageDetector        │
│  MetadataEnricher                      │
└────────────────────┬────────────────────┘
                     │  list[Document] (canonical)
                     ▼
┌─────────────────────────────────────────┐
│           Chunker (6 strategies)        │
│  Overlapping · Recursive · Semantic    │
│  ParentChild · Adaptive · MetaAware   │
└────────────────────┬────────────────────┘
                     │  list[Chunk]
                     ▼
┌─────────────────────────────────────────┐
│       Embedder + Cache + Batch         │
│  BGE · E5 · Instruction · API         │
│  CachedEmbedder · BatchEmbedder       │
└────────────────────┬────────────────────┘
                     │  vectors (float32, normalized)
                     ▼
┌─────────────────────────────────────────┐
│              Index (M4)                 │
│  TF-IDF · BM25 (sparse)               │
│  FAISS Flat/IVF/HNSW (dense ANN)      │
│  Chroma · Qdrant (managed vector DB)  │
│  RAPTOR (multi-level summary tree)    │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│           Retriever (M5)                │
│  Vector · Hybrid (RRF)                 │
│  Filtered · QueryRewriting             │
│  MultiQuery · MultiHop                 │
│  Self-RAG · CRAG · Agentic            │
└────────────────────┬────────────────────┘
                     │
                     ▼
              Reranker (M6)
                     │
                     ▼
              Generator → cited answer
              (Ollama local + OpenRouter fallback)
```

Every stage is behind an ABC. Swap implementations via `configs/default.yaml`.

---

## LLM setup

KnowledgeOS uses a **local-first LLM strategy**:

| Layer | Primary | Fallback |
|-------|---------|---------|
| Generation | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |
| Retrieval (rewriting, multi-query, CRAG, Self-RAG) | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |

Run locally: `ollama serve` (keep running in background).
No GPU required — Qwen2.5-3B runs on CPU in ~2-3s per call.

---

## Benchmark results

### M2 — Chunking

| Rank | Chunker | recall@1 | recall@3 | MRR | avg_size | ms |
|------|---------|----------|----------|-----|----------|----|
| 1 | parent_child | 1.000 | 1.000 | 1.000 | 279 | 114 |
| 2 | adaptive | 1.000 | 1.000 | 1.000 | 291 | 116 |
| 3 | metadata_aware | 1.000 | 1.000 | 1.000 | 454 | 119 |
| 4 | overlapping | 0.900 | 1.000 | 0.950 | 454 | 461 |
| 5 | recursive | 0.900 | 1.000 | 0.950 | 353 | 139 |
| 6 | semantic | 0.800 | 0.900 | 0.850 | 493 | 316 |

**Key finding:** semantic chunking (most complex) ranked last. Chunk size predicted recall@1 better than algorithm sophistication.

### M3 — Embedding

| Rank | Embedder | dim | recall@1 | recall@3 | MRR | ms/chunk |
|------|----------|-----|----------|----------|-----|----------|
| 1 | e5-small | 384 | 1.000 | 1.000 | 1.000 | 64.1 |
| 2 | instr-bge-b | 768 | 1.000 | 1.000 | 1.000 | 183.6 |
| 3 | bge-small | 384 | 0.900 | 1.000 | 0.950 | 20.0 |

**Key finding:** E5's dual-prefix training closes BGE-small's one recall gap at 3x compute. 768-dim didn't help on in-distribution content.

### M4 — Indexing

| Rank | Index | recall@1 | recall@3 | MRR |
|------|-------|----------|----------|-----|
| 1 (tie) | BM25 / Dense / Hybrid / RAPTOR | 0.900 | 1.000 | 0.950 |

**Key finding:** all tied on small uniform corpus. Hybrid RRF requires divergent failure modes. RAPTOR correctly routes thematic queries to summary nodes (score=0.84 vs leaf score=0.71).

### M5 — Retrieval (in progress)

| Retriever | recall@1 | MRR | Notes |
|-----------|----------|-----|-------|
| VectorRetriever (baseline) | 0.900 | 0.950 | M0 spine |
| FilteredRetriever (post, CSV) | — | — | Format-specific routing |
| QueryRewriting (reformulate) | 0.900 | 0.950 | Qwen2.5 rewrites |
| QueryRewriting (HyDE) | 0.800 | — | Hurts on small corpus |
| MultiQuery (n=3) | 0.800 | 0.900 | Union noise on 8 chunks |

Full results in `RESULTS.md`.

---

## Project structure

```
KnowledgeOS/
├── core/                   # ABCs, dataclasses, config, normalizers
│   ├── interfaces.py       # 7 ABCs: Loader, Chunker, Embedder, Index,
│   │                       #         Retriever, Reranker, Generator
│   ├── models.py           # Document, Chunk dataclasses
│   ├── config.py           # YAML → dataclass config system
│   └── normalizers.py      # NormalizationPipeline
│
├── loaders/                # 8 format loaders + router
│   ├── txt_loader.py
│   ├── pdf_loader.py       # pdfplumber, per-page, table extraction
│   ├── docx_loader.py      # heading-aware sections, .doc conversion
│   ├── md_loader.py        # heading-level section splitting
│   ├── html_loader.py      # trafilatura + BS4 strategies
│   ├── csv_loader.py       # row/file strategies, field-value templating
│   ├── email_loader.py     # MIME-aware, header metadata
│   ├── youtube_loader.py   # timestamped segments, deep-link citations
│   └── router.py           # auto-dispatch by extension/URL
│
├── chunkers/               # 6 chunking strategies
│   ├── fixed_chunker.py    # OverlappingChunker (baseline)
│   ├── recursive_chunker.py
│   ├── semantic_chunker.py # cosine similarity breakpoints
│   ├── parent_child_chunker.py
│   ├── adaptive_chunker.py # density-based sizing
│   └── metadata_aware_chunker.py
│
├── embedders/              # 5 embedding backends + utilities
│   ├── bge_embedder.py
│   ├── e5_embedder.py
│   ├── instructor_embedder.py
│   ├── jina_embedder.py    # stub (compatibility issue)
│   ├── api_embedder.py     # OpenAI-compatible endpoint
│   ├── cache.py            # CachedEmbedder (L1 memory + L2 SQLite)
│   └── batch_processor.py  # BatchEmbedder + normalization utilities
│
├── indexes/                # 7 index types
│   ├── faiss_index.py      # FaissFlatIndex, FaissIVFIndex, FaissHNSWIndex
│   ├── tfidf_index.py      # TF-IDF from scratch
│   ├── bm25_index.py       # BM25 from scratch
│   ├── chroma_index.py     # ChromaIndex (persistence, deletion, collections)
│   ├── qdrant_index.py     # QdrantIndex (pre-filter, payload, upsert)
│   └── raptor_index.py     # RAPTORIndex (multi-level summary tree)
│
├── retrievers/             # Retrieval portfolio (M5)
│   ├── vector_retriever.py    # Dense baseline
│   ├── hybrid_retriever.py    # BM25 + Dense + RRF
│   ├── filtered_retriever.py  # Pre/post/boost metadata filtering
│   ├── query_rewriting_retriever.py  # Reformulate + HyDE
│   └── multi_query_retriever.py      # K variants → union
│
├── rerankers/              # M6
├── generation/
│   ├── prompt_builder.py
│   ├── generator.py        # OpenRouterGenerator
│   └── local_generator.py  # LocalLLMGenerator (Ollama + OpenRouter fallback)
│
├── eval/                   # Evaluation harness
│   ├── gold_set.py         # 10-query hand-labeled ground truth
│   ├── metrics.py          # recall@k, MRR, faithfulness
│   ├── runner.py           # benchmark orchestration
│   └── multiformat_gold_set.py
│
├── configs/
│   └── default.yaml        # pipeline assembly config
│
├── scripts/                # runnable benchmarks and tests
├── docs/                   # checkpoint + milestone documentation
│   ├── checkpoints/        # per-checkpoint deep dives
│   └── milestones/         # milestone docs with interview mock exams
│
├── results/                # benchmark output artifacts
├── cache/                  # embedding cache (SQLite, git-ignored)
├── RESULTS.md              # all benchmark results, chronological
└── .env                    # API keys (git-ignored)
```

---

## Setup

```bash
# 1. Create environment
conda create -n knowledgeos python=3.11 -y
conda activate knowledgeos

# 2. Install dependencies
pip install torch faiss-cpu sentence-transformers pyyaml python-dotenv \
            pypdf pdfplumber python-docx trafilatura beautifulsoup4 \
            lxml pandas langdetect unicodedata2 requests openai \
            youtube-transcript-api transformers einops \
            chromadb qdrant-client scikit-learn

# 3. Install and start Ollama (for local LLM)
# Download from: https://ollama.com
ollama pull qwen2.5:3b-instruct
ollama serve   # keep running in background

# 4. Configure API keys
# Create .env at repo root:
# OPENROUTER_API_KEY=sk-or-v1-...   (optional — used as fallback)

# 5. Run the baseline eval
python scripts/run_eval.py
```

---

## Running benchmarks

```bash
# Chunking benchmark (all 6 strategies)
python scripts/make_corpus.py
python scripts/run_chunker_benchmark.py

# Embedding benchmark (all backends)
python scripts/run_embedder_benchmark.py

# Index benchmark (BM25 / Dense / Hybrid / RAPTOR)
python scripts/run_index_benchmark.py

# Full retrieval benchmark (M5 — when complete)
python scripts/run_retriever_benchmark.py

# Full pipeline (question answering)
python scripts/test_pipeline.py
```

---

## Key design decisions

**Plugin architecture over monolithic code.** 7 ABCs enforce contracts — swap any component by changing one YAML line.

**Eval-first.** The evaluation harness was built in M0, not M8. Every component is measured against the same gold set. No vibes-driven development.

**`tenant_id` from day one.** Multi-tenant isolation is in the dataclass and enforced at every retrieval boundary.

**Local LLM first.** Ollama + Qwen2.5-3B for all LLM calls inside the retrieval pipeline — no rate limits, no cost, instant. OpenRouter as fallback.

**Benchmark over blog post.** Every "which X is best?" question is answered empirically on the actual corpus.

**Fail loud at init, not at runtime.** Constructor guards catch impossible configurations at object creation.

**Graceful degradation compounds.** Query rewriting falls back to original query. Local LLM falls back to OpenRouter. Index returns empty list instead of crashing. Multiple layers of graceful degradation mean the pipeline never hard-crashes.

---

## What I learned building this

- Semantic chunking (most algorithmically complex) ranked **last** on technical reference corpora. Chunk size predicted recall@1 better than algorithm sophistication.
- E5's dual-prefix training closed BGE-small's one recall gap at 3x compute. Dimension (384 vs 768) didn't matter on in-distribution content.
- BM25 from scratch takes 50 lines. TF saturation (k1) and length normalization (b) only matter at scale with length variance — on 8 uniform chunks, BM25 and TF-IDF rank identically.
- Hybrid RRF requires divergent failure modes. When both BM25 and dense miss the same query, RRF cannot rescue it.
- RAPTOR correctly routes thematic queries to summary nodes (score=0.84) vs leaves for specific queries (score=0.71).
- Pre-filter (Qdrant native payload filter before ANN) vs post-filter (FAISS + manual filter after) is the architectural distinction that matters at 1M+ vectors.
- 788x L1 cache speedup vs 2.6x L2 — the two tiers serve very different use cases.
- Local Qwen2.5-3B produces measurably better query rewrites than free-tier cloud models. Query 3 went from r@1=0 (OpenRouter) to r@1=1 (Qwen).
- Multi-query union introduces noise on small corpora — variants retrieve from the same small pool with shifted emphases, reranking incorrectly. On large corpora, the union covers more of the relevant space.
- `trust_remote_code` is a production liability. Jina v2 and v3 both broke on consecutive transformers upgrades.

---

## Documentation

Every milestone has a deep-dive document with:
- Architecture recap and design decisions
- Per-checkpoint technical notes
- 40-question interview mock exam (easy → hard → whiteboard coding)
- 30-second, 2-minute, and 5-minute project walkthrough scripts

Located in `docs/milestones/` and `docs/checkpoints/`.

---

## Author

**Rudraksh Sharma** — Data Scientist at AIONOS, Technical Lead at Beerantum.

[GitHub: Rudra1x](https://github.com/Rudra1x/KnowledgeOS)

---

*Built milestone by milestone. Stop-safe at any point — M0 alone is a complete, demoable RAG system.*