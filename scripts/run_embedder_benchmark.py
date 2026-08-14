# scripts/run_embedder_benchmark.py

import sys, time, os
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from statistics  import mean
from dotenv      import load_dotenv
load_dotenv()

from loaders.txt_loader              import TXTLoader
from chunkers.recursive_chunker      import RecursiveChunker
from embedders.bge_embedder          import BGEEmbedder
from embedders.e5_embedder           import E5Embedder
from embedders.instructor_embedder   import InstructionEmbedder
from embedders.batch_processor       import BatchEmbedder, verify_normalization
from indexes.faiss_index             import FaissFlatIndex
from retrievers.vector_retriever     import VectorRetriever
from eval.gold_set                   import GOLD_SET
from eval.metrics                    import recall_at_k, mean_reciprocal_rank
from core                            import load_config, NormalizationPipeline
import numpy as np


cfg        = load_config()
normalizer = NormalizationPipeline()

# Use RecursiveChunker — clean natural boundaries, moderate chunk count
chunker = RecursiveChunker(
    chunk_size    = cfg.get("chunker", "fixed", "chunk_size"),
    chunk_overlap = cfg.get("chunker", "fixed", "chunk_overlap"),
)

# Load + normalise + chunk once — same chunks for every embedder
docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
print(f"Corpus: {len(docs[0].content)} chars  |  Chunks: {len(chunks)}\n")

# --- Embedder portfolio ---
EMBEDDERS = [
    ("bge-small",    BGEEmbedder("BAAI/bge-small-en-v1.5",  batch_size=32)),
    ("e5-small",     E5Embedder("intfloat/e5-small-v2",      batch_size=32)),
    ("instr-bge-b",  InstructionEmbedder(
        model_name        = "BAAI/bge-base-en-v1.5",
        batch_size        = 16,
        query_instruction = "Represent the question for retrieving relevant documents: ",
        embed_instruction = "Represent the technical document for retrieval: ",
    )),
]

results = []

for name, embedder in EMBEDDERS:
    print(f"--- {name} (dim={embedder.dimension}) ---")

    # Deep-copy chunks so embeddings don't bleed across runs
    import copy
    run_chunks = copy.deepcopy(chunks)

    # Embed with batch processor
    processor = BatchEmbedder(embedder, batch_size=32,
                              show_progress=False, normalize=True)

    t0        = time.perf_counter()
    processor.embed_chunks(run_chunks)
    t_embed   = (time.perf_counter() - t0) * 1000

    # Verify normalization
    vecs  = np.array([c.embedding for c in run_chunks if c.embedding], dtype="float32")
    check = verify_normalization(vecs)

    # Build fresh index
    index     = FaissFlatIndex(dimension=embedder.dimension)
    index.add(run_chunks)
    retriever = VectorRetriever(embedder=embedder, index=index)

    # Eval
    r1, r3, mrr_scores = [], [], []
    for item in GOLD_SET:
        retrieved = retriever.retrieve(item["query"], top_k=5, tenant_id="default")
        r1.append(recall_at_k(retrieved, item["relevant_text"], 1))
        r3.append(recall_at_k(retrieved, item["relevant_text"], 3))
        mrr_scores.append(mean_reciprocal_rank(retrieved, item["relevant_text"]))

    ms_per_chunk = t_embed / max(len(run_chunks), 1)

    results.append({
        "name":          name,
        "dim":           embedder.dimension,
        "chunks":        len(run_chunks),
        "recall@1":      mean(r1),
        "recall@3":      mean(r3),
        "mrr":           mean(mrr_scores),
        "embed_ms":      round(t_embed, 1),
        "ms_per_chunk":  round(ms_per_chunk, 1),
        "norm_ok":       check["normalized"],
        "max_norm_dev":  check["max_dev"],
    })

    print(f"    r@1={mean(r1):.3f}  r@3={mean(r3):.3f}  mrr={mean(mrr_scores):.3f}  "
          f"embed={t_embed:.0f}ms  {ms_per_chunk:.1f}ms/chunk  "
          f"norm={check['normalized']}\n")

# --- Ranked table ---
ranked = sorted(results, key=lambda x: (x["recall@1"], x["mrr"]), reverse=True)

print("=" * 80)
print(f"{'RANK':<5} {'EMBEDDER':<16} {'DIM':>5} {'r@1':>6} {'r@3':>6} "
      f"{'MRR':>6} {'ms/chunk':>9} {'NORM':>5}")
print("=" * 80)
for rank, r in enumerate(ranked, 1):
    marker = " ← WINNER" if rank == 1 else ""
    print(f"{rank:<5} {r['name']:<16} {r['dim']:>5} {r['recall@1']:>6.3f} "
          f"{r['recall@3']:>6.3f} {r['mrr']:>6.3f} "
          f"{r['ms_per_chunk']:>9.1f} {str(r['norm_ok']):>5}{marker}")

# --- Cost model ---
print("\n" + "=" * 80)
print("COST MODEL  (hypothetical 1M-chunk corpus)")
print("=" * 80)
for r in ranked:
    time_hrs = (r["ms_per_chunk"] * 1_000_000) / (1000 * 3600)
    print(f"  {r['name']:<16} → {time_hrs:.1f}h to embed  "
          f"| index size ≈ {r['dim']*4*1_000_000/1e9:.1f}GB float32")

# --- RESULTS.md update ---
md_rows = "\n".join(
    f"| {r['name']:<16} | {r['dim']:>5} | {r['recall@1']:.3f} | "
    f"{r['recall@3']:.3f} | {r['mrr']:.3f} | {r['ms_per_chunk']:.1f} |"
    for r in ranked
)
md_block = f"""
## M3 Embedding Benchmark

**Corpus:** corpus.txt ({len(docs[0].content)} chars)
**Chunker:** RecursiveChunker (512/50)
**Gold set:** 10 queries

| Embedder | dim | recall@1 | recall@3 | MRR | ms/chunk |
|----------|-----|----------|----------|-----|----------|
{md_rows}
"""
with open("RESULTS.md", "a", encoding="utf-8") as f:
    f.write(md_block)
print("\nResults appended to RESULTS.md")