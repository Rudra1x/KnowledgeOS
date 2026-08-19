# scripts/test_filtered_retriever.py

import sys, copy, uuid
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from loaders.csv_loader          import CSVLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from retrievers.filtered_retriever import FilteredRetriever
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)

# --- Load corpus with two file types ---
txt_docs = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
csv_docs = normalizer.apply_many(CSVLoader(strategy="row").load("scripts/faq.csv"))

all_chunks = []
for docs, file_type in [(txt_docs, "txt"), (csv_docs, "csv")]:
    for doc in docs:
        doc.metadata["file_type"] = file_type
        chunks = chunker.chunk(doc)
        for c in chunks:
            c.metadata["file_type"] = file_type
        all_chunks.extend(chunks)

vecs = embedder.embed([c.content for c in all_chunks])
for c, v in zip(all_chunks, vecs):
    c.embedding = v

index = FaissFlatIndex(dimension=384)
index.add(all_chunks)

print(f"Indexed {len(all_chunks)} chunks "
      f"(txt: {sum(1 for c in all_chunks if c.metadata.get('file_type')=='txt')}, "
      f"csv: {sum(1 for c in all_chunks if c.metadata.get('file_type')=='csv')})\n")

QUERY = "How do I measure RAG quality?"

# --- No filter (baseline) ---
print("=" * 60)
print("BASELINE (no filter)")
print("=" * 60)
baseline = FilteredRetriever(embedder=embedder, index=index, mode="post", filter={})
results  = baseline.retrieve(QUERY, top_k=3)
for r in results:
    print(f"  [{r.metadata.get('file_type','?'):4s}] {r.content[:70].strip()}...")

# --- Post-filter: CSV only ---
print("\n" + "=" * 60)
print("POST-FILTER: file_type=csv only")
print("=" * 60)
csv_ret  = FilteredRetriever(embedder=embedder, index=index, mode="post",
                              filter={"file_type": "csv"})
results  = csv_ret.retrieve(QUERY, top_k=3)
for r in results:
    print(f"  [{r.metadata.get('file_type','?'):4s}] {r.content[:70].strip()}...")

# --- Boost: prefer CSV but don't exclude TXT ---
print("\n" + "=" * 60)
print("BOOST: prefer csv (2x score), show TXT too")
print("=" * 60)
boost_ret = FilteredRetriever(embedder=embedder, index=index, mode="boost", filter={"file_type": "csv"}, boost_factor=2.0)
results   = boost_ret.retrieve(QUERY, top_k=5)
for r in results:
    boosted = "★" if r.metadata.get("file_type") == "csv" else " "
    print(f"  {boosted}[{r.metadata.get('file_type','?'):4s}] "
          f"score={r.metadata['score']:.4f}  {r.content[:60].strip()}...")