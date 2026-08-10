# scripts/run_chunker_benchmark.py

import sys, time
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from statistics import mean
from dotenv     import load_dotenv
load_dotenv()

from loaders.txt_loader               import TXTLoader
from chunkers.fixed_chunker           import OverlappingChunker
from chunkers.recursive_chunker       import RecursiveChunker
from chunkers.semantic_chunker        import SemanticChunker
from chunkers.parent_child_chunker    import ParentChildChunker
from chunkers.adaptive_chunker        import AdaptiveChunker
from chunkers.metadata_aware_chunker  import MetadataAwareChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from retrievers.vector_retriever      import VectorRetriever
from eval.gold_set                    import GOLD_SET
from eval.metrics                     import recall_at_k, mean_reciprocal_rank
from core                             import load_config, NormalizationPipeline


cfg      = load_config()
embedder = BGEEmbedder(
    model_name = cfg.get("embedder", "bge_small", "model_name"),
    batch_size = cfg.get("embedder", "bge_small", "batch_size"),
)
normalizer = NormalizationPipeline()

# --- Chunker portfolio ---
CHUNKERS = [
    ("overlapping",     OverlappingChunker(chunk_size=512, chunk_overlap=50)),
    ("recursive",       RecursiveChunker(chunk_size=512, chunk_overlap=50)),
    ("semantic",        SemanticChunker(embedder=embedder,
                                        breakpoint_percentile=95,
                                        min_chunk_size=100,
                                        max_chunk_size=900)),
    ("parent_child",    ParentChildChunker(parent_size=1500, child_size=300, child_overlap=40)),
    ("adaptive",        AdaptiveChunker(min_chunk_size=200, max_chunk_size=900,
                                        base_chunk_size=512, overlap_chars=50)),
    ("metadata_aware",  MetadataAwareChunker(base_chunk_size=512, overlap_chars=50)),
]

# --- Load + normalise once ---
docs = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
doc  = docs[0]
print(f"Corpus: {len(doc.content)} chars\n")

# --- Run each chunker ---
results = []

for name, chunker in CHUNKERS:
    t0     = time.perf_counter()
    chunks = chunker.chunk(doc)
    t_chunk = time.perf_counter() - t0

    # Embed + index
    t1      = time.perf_counter()
    vectors = embedder.embed([c.content for c in chunks])
    for c, v in zip(chunks, vectors):
        c.embedding = v

    index     = FaissFlatIndex(dimension=cfg.get("index", "faiss_flat", "dimension"))
    index.add(chunks)
    retriever = VectorRetriever(embedder=embedder, index=index)
    t_index   = time.perf_counter() - t1

    # Eval
    r1_scores, r3_scores, mrr_scores = [], [], []
    for item in GOLD_SET:
        retrieved = retriever.retrieve(item["query"], top_k=5, tenant_id="default")
        r1_scores.append(recall_at_k(retrieved, item["relevant_text"], 1))
        r3_scores.append(recall_at_k(retrieved, item["relevant_text"], 3))
        mrr_scores.append(mean_reciprocal_rank(retrieved, item["relevant_text"]))

    # Chunk size stats
    sizes = [len(c.content) for c in chunks]

    results.append({
        "name":      name,
        "chunks":    len(chunks),
        "avg_size":  int(mean(sizes)),
        "min_size":  min(sizes),
        "max_size":  max(sizes),
        "recall@1":  mean(r1_scores),
        "recall@3":  mean(r3_scores),
        "mrr":       mean(mrr_scores),
        "t_chunk_ms": round(t_chunk * 1000, 1),
        "t_index_ms": round(t_index * 1000, 1),
    })

    print(f"[{name:16s}] chunks={len(chunks):>3}  "
          f"r@1={mean(r1_scores):.3f}  r@3={mean(r3_scores):.3f}  "
          f"mrr={mean(mrr_scores):.3f}  "
          f"avg_size={int(mean(sizes)):>4}  "
          f"t={round((t_chunk+t_index)*1000,1)}ms")

# --- Ranked summary ---
ranked = sorted(results, key=lambda x: (x["recall@1"], x["mrr"]), reverse=True)

print("\n" + "=" * 80)
print(f"{'RANK':<5} {'CHUNKER':<18} {'r@1':>6} {'r@3':>6} {'MRR':>6} "
      f"{'CHUNKS':>7} {'AVG_SIZE':>9} {'TOTAL_MS':>9}")
print("=" * 80)

for rank, r in enumerate(ranked, 1):
    marker = " ← WINNER" if rank == 1 else ""
    print(f"{rank:<5} {r['name']:<18} {r['recall@1']:>6.3f} {r['recall@3']:>6.3f} "
          f"{r['mrr']:>6.3f} {r['chunks']:>7} {r['avg_size']:>9} "
          f"{r['t_chunk_ms']+r['t_index_ms']:>9.1f}{marker}")

# --- Update RESULTS.md ---
md_rows = "\n".join(
    f"| {r['name']:<18} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | "
    f"{r['mrr']:.3f} | {r['chunks']:>3} | {r['avg_size']:>4} |"
    for r in ranked
)

md_block = f"""
## M2 Chunking Benchmark

**Corpus:** corpus.txt ({len(doc.content)} chars, 10 paragraphs)
**Gold set:** 10 queries
**Embedder:** BGE-small-en-v1.5
**Index:** FAISS flat

| Chunker | recall@1 | recall@3 | MRR | chunks | avg_size |
|---------|----------|----------|-----|--------|----------|
{md_rows}
"""

with open("RESULTS.md", "a", encoding="utf-8") as f:
    f.write(md_block)

print("\nResults appended to RESULTS.md")