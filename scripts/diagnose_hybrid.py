# scripts/diagnose_hybrid.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")
from dotenv import load_dotenv
load_dotenv()

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from indexes.bm25_index          import BM25Index
from retrievers.hybrid_retriever import HybridRetriever
from eval.gold_set               import GOLD_SET
from eval.metrics                import recall_at_k, is_relevant
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder","bge_small","model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

# Build both indexes
bm25  = BM25Index()
dense = FaissFlatIndex(dimension=384)
bm25.add(copy.deepcopy(chunks))
dense.add(copy.deepcopy(chunks))

hybrid = HybridRetriever(bm25_index=bm25, dense_index=dense,
                          embedder=embedder, fetch_k=8)

print(f"{'Query':<45} {'BM25':>6} {'Dense':>6} {'Hybrid':>7}")
print("-" * 70)
for item in GOLD_SET:
    q   = item["query"]
    rt  = item["relevant_text"]

    bm  = bm25.search_text(q, top_k=5)
    dv  = embedder.embed_query(q)
    dn  = dense.search(dv, top_k=5)
    hy  = hybrid.retrieve(q, top_k=5)

    b1  = recall_at_k(bm, rt, 1)
    d1  = recall_at_k(dn, rt, 1)
    h1  = recall_at_k(hy, rt, 1)

    # Mark queries where hybrid helps
    flag = " ← hybrid wins" if h1 > max(b1, d1) else \
           " ← all miss"    if h1 == 0            else ""
    print(f"{q[:45]:<45} {b1:>6.0f} {d1:>6.0f} {h1:>7.0f}{flag}")