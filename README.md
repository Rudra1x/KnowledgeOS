
# KnowledgeOS

A modular, benchmark-driven **Retrieval Intelligence Platform** built from scratch.

Every component is implemented from first principles and benchmarked against the same evaluation harness — not imported from a library and trusted blindly. The goal is to understand *why* each architectural decision works, not just that it does.

---

## What this is

KnowledgeOS is a full-stack RAG (Retrieval-Augmented Generation) platform covering the complete pipeline from raw documents to cited, grounded answers. Every stage is an ABC with multiple interchangeable implementations, swappable via a single YAML config line.

Built with production discipline: multi-tenant isolation from day one, config-driven assembly, scientific evaluation at every milestone, local-first LLM strategy.

---

## Current status

| Milestone        | Status      | Deliverable                                                 |
| ---------------- | ----------- | ----------------------------------------------------------- |
| M0 — Spine      | ✅ Complete | End-to-end RAG with eval harness                            |
| M1 — Ingestion  | ✅ Complete | 8-format loader + normalization pipeline                    |
| M2 — Chunking   | ✅ Complete | 6-strategy portfolio + benchmark                            |
| M3 — Embedding  | ✅ Complete | 5-backend portfolio + cache + benchmark                     |
| M4 — Indexing   | ✅ Complete | TF-IDF, BM25, FAISS ANN, Chroma, Qdrant, RAPTOR, Hybrid RRF |
| M5 — Retrieval  | ✅ Complete | 8 strategies from filtered to agentic ReAct                 |
| M6 — Reranking  | ✅ Complete | Cross-encoder, BGE, LLM, Metadata rerankers                 |
| M7 — Generation | 🔜 Planned  | Context compression, citation, streaming                    |
| M8 — Evaluation | 🔜 Planned  | Full RAGAS-style harness                                    |

---

## Architecture

```
Raw documents (8 formats)
        │
        ▼
┌──────────────────────────────────────┐
│  LoaderRouter                        │
│  TXT · PDF · DOCX · MD · HTML       │
│  CSV · Email · YouTube              │
└─────────────────┬────────────────────┘
                  │ list[Document]
                  ▼
┌──────────────────────────────────────┐
│  NormalizationPipeline               │
│  TextCleaner · LanguageDetector     │
│  MetadataEnricher                   │
└─────────────────┬────────────────────┘
                  │ list[Document] (canonical)
                  ▼
┌──────────────────────────────────────┐
│  Chunker (6 strategies)              │
│  Overlapping · Recursive · Semantic │
│  ParentChild · Adaptive · MetaAware │
└─────────────────┬────────────────────┘
                  │ list[Chunk]
                  ▼
┌──────────────────────────────────────┐
│  Embedder + Cache + Batch            │
│  BGE · E5 · Instruction · API       │
│  CachedEmbedder (788x L1 speedup)  │
└─────────────────┬────────────────────┘
                  │ vectors (float32, normalized)
                  ▼
┌──────────────────────────────────────┐
│  Index (7 types)                     │
│  TF-IDF · BM25 (sparse)            │
│  FAISS Flat/IVF/HNSW (ANN)         │
│  Chroma · Qdrant (managed vector DB)│
│  RAPTOR (multi-level summary tree)  │
└─────────────────┬────────────────────┘
                  │ top-20 candidates
                  ▼
┌──────────────────────────────────────┐
│  Retriever (8 strategies)            │
│  Vector · Hybrid (RRF)              │
│  Filtered · QueryRewriting          │
│  MultiQuery · MultiHop              │
│  SelfRAG · CRAG · Agentic          │
└─────────────────┬────────────────────┘
                  │ top-5 candidates (recall@5 ≈ 1.000)
                  ▼
┌──────────────────────────────────────┐
│  Reranker (4 types)                  │
│  CrossEncoder (MS-MARCO, 20ms)      │
│  BGEReranker (278MB, 116ms)         │
│  LLMReranker (custom criteria)      │
│  MetadataReranker (recency/source)  │
└─────────────────┬────────────────────┘
                  │ top-3 reranked (recall@1 = 1.000)
                  ▼
         Generator → cited answer
    (Ollama local + OpenRouter fallback)
```

---

## LLM setup

KnowledgeOS uses a **local-first LLM strategy**:

| Layer                                          | Primary                            | Fallback            |
| ---------------------------------------------- | ---------------------------------- | ------------------- |
| Retrieval (rewriting, CRAG, Self-RAG, Agentic) | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |
| Generation                                     | `qwen2.5:3b-instruct` via Ollama | `openrouter/free` |

Start Ollama before running LLM-dependent components: `ollama serve`

---

## Benchmark results

### M2 — Chunking

| Rank | Chunker        | recall@1 | MRR   | Key property                     |
| ---- | -------------- | -------- | ----- | -------------------------------- |
| 1    | parent_child   | 1.000    | 1.000 | 3× context window               |
| 2    | adaptive       | 1.000    | 1.000 | Density-based sizing             |
| 3    | metadata_aware | 1.000    | 1.000 | Atomic tables                    |
| 4    | overlapping    | 0.900    | 0.950 | Simple baseline                  |
| 5    | recursive      | 0.900    | 0.950 | Natural boundaries               |
| 6    | semantic       | 0.800    | 0.850 | Over-merges on technical content |

**Key finding:** semantic chunking ranked last. Chunk size predicted recall@1 better than algorithm sophistication.

### M3 — Embedding

| Rank | Embedder    | dim | recall@1 | ms/chunk |
| ---- | ----------- | --- | -------- | -------- |
| 1    | e5-small    | 384 | 1.000    | 64.1     |
| 2    | instr-bge-b | 768 | 1.000    | 183.6    |
| 3    | bge-small   | 384 | 0.900    | 20.0     |

**Key finding:** 768-dim didn't help over 384-dim on in-distribution content. 788× L1 cache speedup.

### M4 — Indexing

All indexes tied at recall@1=0.900. Hybrid RRF needs divergent failure modes. RAPTOR routes thematic queries to summary nodes (score=0.84 vs leaf 0.71).

### M5 — Retrieval

| Retriever              | recall@1 | time/10q |
| ---------------------- | -------- | -------- |
| Agentic                | 1.000    | 230s     |
| Vector/Hybrid/Filtered | 0.900    | 0.2s     |
| QueryRewriting/CRAG    | 0.900    | 35-465s  |
| SelfRAG                | 0.900    | 307s     |
| MultiQuery             | 0.800    | 58s      |

**Key finding:** Agentic = perfect recall at 23s/query. Deterministic retrievers = 0.900 at 0.02s/query.

### M6 — Reranking

| Rank | Pipeline           | recall@1 | time/10q |
| ---- | ------------------ | -------- | -------- |
| 1    | Vector + MS-MARCO  | 1.000    | 3.2s     |
| 2    | Hybrid + MS-MARCO  | 1.000    | 1.9s     |
| 3    | Vector + BGE       | 1.000    | 16.6s    |
| 4    | Hybrid + BGE       | 1.000    | 10.8s    |
| 5    | Vector (no rerank) | 0.900    | 0.5s     |
| 6    | Vector + LLM       | 0.900    | 360s     |
| 7    | Vector + Metadata  | 0.700    | 0.5s     |

**Key finding:** Hybrid + MS-MARCO achieves perfect recall at 190ms/query — 72× faster than agentic retrieval for the same result. Cross-encoder reranking is the highest ROI upgrade in any RAG system.

**The architectural principle proven across M0→M6:** recall@3 is a retrieval problem; recall@1 is a reranking problem.

Full results in `RESULTS.md`.

---

## Project structure

```
KnowledgeOS/
├── core/                   # ABCs, dataclasses, config, normalizers
├── loaders/                # 8 format loaders + router
├── chunkers/               # 6 chunking strategies
├── embedders/              # 5 backends + CachedEmbedder + BatchEmbedder
├── indexes/                # 7 types: TF-IDF, BM25, FAISS×3, Chroma, Qdrant, RAPTOR
├── retrievers/             # 8 strategies: Vector, Hybrid, Filtered, QueryRewrite,
│                           #   MultiQuery, MultiHop, SelfRAG, CRAG, Agentic
├── rerankers/              # 4 types: CrossEncoder, BGE, LLM, Metadata+Similarity
├── generation/
│   ├── generator.py        # OpenRouterGenerator
│   └── local_generator.py  # LocalLLMGenerator (Ollama primary)
├── eval/                   # Harness: gold_set, recall@k, MRR, faithfulness
├── configs/default.yaml
├── scripts/                # Benchmark scripts (run_*_benchmark.py)
├── docs/                   # Milestone + checkpoint documentation
├── RESULTS.md              # All benchmark results
└── .env                    # API keys (git-ignored)
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

# 3. Local LLM (required for M5-M6)
# Install Ollama from https://ollama.com
ollama pull qwen2.5:3b-instruct
ollama serve   # keep running in background

# 4. API keys (optional — used as fallback)
# OPENROUTER_API_KEY=sk-or-v1-... in .env

# 5. Baseline eval
python scripts/run_eval.py
```

---

## Key design decisions

**Plugin architecture.** 7 ABCs, swap any component via YAML. New implementations plug in without changing downstream code.

**Eval-first.** Gold set built in M0. Every component measured against it. No vibes-driven development.

**`tenant_id` from day one.** Multi-tenant isolation in the dataclass, enforced at every boundary. Free at design time; painful to retrofit.

**Local LLM first.** Ollama + Qwen2.5-3B for all LLM calls inside the retrieval pipeline. No rate limits, no cost, no rotating model IDs.

**Benchmark over blog post.** Every "which X is best?" question answered empirically on actual corpus data.

**Graceful degradation compounds.** Query rewriting falls back to original query. Local LLM falls back to OpenRouter. Index returns empty instead of crashing. The pipeline never hard-crashes.

---

## What I learned building this

- Semantic chunking ranked **last** on technical corpora — chunk size predicted recall better than algorithm complexity.
- E5 closed BGE-small's recall gap at 3× compute. Dimension (384 vs 768) didn't matter on in-distribution content.
- Hybrid RRF needs divergent failure modes — both legs failing on the same query produces no benefit.
- RAPTOR routes thematic queries to summary nodes (0.84) vs leaves for specific queries (0.71) — proven empirically.
- Agentic retrieval achieved perfect recall by discovering hybrid retrieval dynamically — not because it was told to use it.
- **The central finding of M6:** recall@3 is a retrieval problem; recall@1 is a reranking problem. A 22MB cross-encoder closes the recall gap that took a 23s/query agentic retriever in M5, at 72× lower latency.
- Local Qwen2.5-3B produces better query rewrites than free-tier cloud models — stability matters for components on the critical path.
- Pre-filter (Qdrant native) vs post-filter (FAISS + manual) is the architectural distinction at 1M+ vectors.
- Query-agnostic keyword boosts hurt precision. Recency boost is unconditionally valid. LLM reranking is 149× slower than cross-encoder for no quality gain on semantic ranking tasks.
- 788× L1 cache speedup vs 2.6× L2 — two tiers serve different use cases.

---

## Documentation

Every milestone has a deep-dive document with:

- Architecture recap and design decisions
- Per-checkpoint technical notes
- 40-question interview mock exam
- 30-second, 2-minute, and 5-minute walkthrough scripts

Located in `docs/milestones/` and `docs/checkpoints/`.

---

## Author

**Rudraksh Sharma** — Data Scientist at AIONOS, Technical Lead at Beerantum.
Qiskit Advocate · Berlin Quantum Hackathon 2026 (3rd place) · QIntern 2025 First Team Award.

[GitHub: Rudra1x](https://github.com/Rudra1x/KnowledgeOS)

---

*Built milestone by milestone. Stop-safe at any point — M0 alone is a complete, demoable RAG system.*
