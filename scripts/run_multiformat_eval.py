# scripts/run_multiformat_eval.py

import sys, os
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from collections import defaultdict, Counter
from statistics  import mean

from loaders.router               import LoaderRouter
from chunkers.fixed_chunker       import FixedChunker
from embedders.bge_embedder       import BGEEmbedder
from indexes.faiss_index          import FaissFlatIndex
from retrievers.vector_retriever  import VectorRetriever
from core                         import NormalizationPipeline, load_config
from eval.metrics                 import recall_at_k, mean_reciprocal_rank, is_relevant
from eval.multiformat_gold_set    import MULTIFORMAT_GOLD_SET


# --- Sources to ingest (mixed formats) ---
SOURCES = [
    "scripts/corpus.txt",
    "scripts/sample.md",
    "scripts/faq.csv",
    "scripts/sample.eml",
    "scripts/sample.pdf",     # will contribute per-page docs
]

# --- Build the pipeline ---
cfg          = load_config()
router       = LoaderRouter(loader_kwargs={"CSVLoader": {"strategy": "row"}})
normalizer   = NormalizationPipeline()
chunker      = FixedChunker(
    chunk_size    = cfg.get("chunker", "fixed", "chunk_size"),
    chunk_overlap = cfg.get("chunker", "fixed", "chunk_overlap"),
)
embedder     = BGEEmbedder(
    model_name = cfg.get("embedder", "bge_small", "model_name"),
    batch_size = cfg.get("embedder", "bge_small", "batch_size"),
)
index        = FaissFlatIndex(dimension=cfg.get("index", "faiss_flat", "dimension"))
retriever    = VectorRetriever(embedder=embedder, index=index)


# --- Ingest, normalize, chunk, index ---
all_chunks = []
format_stats = Counter()

for source in SOURCES:
    if not os.path.exists(source):
        print(f"[SKIP] {source} (file not found)")
        continue

    try:
        docs = router.load(source)
    except Exception as e:
        print(f"[FAIL] {source}: {e}")
        continue

    docs = normalizer.apply_many(docs)

    for d in docs:
        chunks = chunker.chunk(d)
        # Tag every chunk with its origin file (for per-format breakdown)
        for c in chunks:
            c.metadata["source_file"] = os.path.basename(source)
        all_chunks.extend(chunks)
        format_stats[d.metadata.get("file_type", "unknown")] += len(chunks)

print("\n=== INGESTION SUMMARY ===")
print(f"Total chunks: {len(all_chunks)}")
for fmt, n in format_stats.most_common():
    print(f"  {fmt:12s}: {n:>4} chunks")

# --- Embed and index ---
print("\nEmbedding...")
vectors = embedder.embed([c.content for c in all_chunks])
for c, v in zip(all_chunks, vectors):
    c.embedding = v
index.add(all_chunks)
print(f"Indexed {len(all_chunks)} chunks.\n")

# --- Run the multi-format gold set ---
print("=" * 70)
print("MULTI-FORMAT EVAL")
print("=" * 70)

per_format_recall = defaultdict(list)
per_format_mrr    = defaultdict(list)
overall_recall_1  = []
overall_recall_3  = []
overall_mrr       = []

for item in MULTIFORMAT_GOLD_SET:
    query          = item["query"]
    relevant_text  = item["relevant_text"]
    expected_src   = item["expected_source"]

    retrieved = retriever.retrieve(query, top_k=5, tenant_id="default")

    r1  = recall_at_k(retrieved, relevant_text, 1)
    r3  = recall_at_k(retrieved, relevant_text, 3)
    mrr = mean_reciprocal_rank(retrieved, relevant_text)

    overall_recall_1.append(r1)
    overall_recall_3.append(r3)
    overall_mrr.append(mrr)

    # Find the source file of the top retrieved chunk (for diagnostic)
    top_source = retrieved[0].metadata.get("source_file", "?") if retrieved else "?"
    correct    = "✓" if top_source == expected_src and r1 == 1.0 else " "

    print(f"{correct} r1={r1:.0f} r3={r3:.0f} mrr={mrr:.2f}  "
          f"[{expected_src:20s} → top:{top_source:20s}]  {query}")

    per_format_recall[expected_src].append(r1)
    per_format_mrr[expected_src].append(mrr)

# --- Aggregate ---
print("\n" + "=" * 70)
print("OVERALL")
print("=" * 70)
print(f"  recall@1 : {mean(overall_recall_1):.3f}")
print(f"  recall@3 : {mean(overall_recall_3):.3f}")
print(f"  MRR      : {mean(overall_mrr):.3f}")

print("\n" + "=" * 70)
print("PER-FORMAT recall@1")
print("=" * 70)
for src, scores in sorted(per_format_recall.items()):
    print(f"  {src:20s} recall@1 = {mean(scores):.3f}  ({len(scores)} queries)")