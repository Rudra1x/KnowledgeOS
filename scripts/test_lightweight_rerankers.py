# scripts/test_lightweight_rerankers.py

import sys, copy, uuid
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader              import TXTLoader
from chunkers.recursive_chunker      import RecursiveChunker
from embedders.bge_embedder          import BGEEmbedder
from embedders.e5_embedder           import E5Embedder
from indexes.faiss_index             import FaissFlatIndex
from retrievers.vector_retriever     import VectorRetriever
from rerankers.similarity_reranker   import SimilarityReranker
from rerankers.metadata_reranker     import MetadataReranker
from eval.gold_set                   import GOLD_SET
from eval.metrics                    import recall_at_k, mean_reciprocal_rank
from core                            import NormalizationPipeline, load_config
from core.models                     import Chunk
from statistics                      import mean

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = VectorRetriever(embedder=embedder, index=index)

# Similarity reranker: re-score with E5 embedder
e5_embedder = E5Embedder("intfloat/e5-small-v2")
sim_reranker = SimilarityReranker(
    embedder        = e5_embedder,
    query_expansion = [],
    score_weight    = 0.0,
)

# Metadata reranker: boost chunks mentioning BM25/retrieval keywords
meta_reranker = MetadataReranker(
    keyword_boost = {
        "retrieval": 1.2,
        "BM25":      1.3,
        "hybrid":    1.2,
        "recall":    1.1,
    },
)

print("=" * 72)
print(f"{'Query':<42} {'Base':>5} {'Sim(E5)':>8} {'Meta':>5}")
print("=" * 72)

base_r1s, sim_r1s, meta_r1s = [], [], []

for item in GOLD_SET:
    q   = item["query"]
    rt  = item["relevant_text"]

    candidates = retriever.retrieve(q, top_k=5)
    base_r1    = recall_at_k(candidates, rt, 1)

    sim_ranked  = sim_reranker.rerank(q, copy.deepcopy(candidates), top_k=5)
    sim_r1      = recall_at_k(sim_ranked, rt, 1)

    meta_ranked = meta_reranker.rerank(q, copy.deepcopy(candidates), top_k=5)
    meta_r1     = recall_at_k(meta_ranked, rt, 1)

    base_r1s.append(base_r1)
    sim_r1s.append(sim_r1)
    meta_r1s.append(meta_r1)

    flag = ""
    if sim_r1 > base_r1 or meta_r1 > base_r1:
        flag = " <- wins"
    elif sim_r1 < base_r1 or meta_r1 < base_r1:
        flag = " <- loses"
    print(f"{q[:42]:<42} {base_r1:>5.0f} {sim_r1:>8.0f} {meta_r1:>5.0f}{flag}")

print("=" * 72)
print(f"{'MEAN':<42} {mean(base_r1s):>5.3f} {mean(sim_r1s):>8.3f} "
      f"{mean(meta_r1s):>5.3f}")
print(f"\nSimilarity reranker latency: {sim_reranker.last_rerank_ms:.1f}ms")
print(f"Metadata reranker latency:   {meta_reranker.last_rerank_ms:.1f}ms")

# --- Metadata recency test ---
print("\n" + "=" * 55)
print("METADATA RECENCY BOOST (synthetic test)")
print("=" * 55)

recent_chunk = Chunk(
    chunk_id  = str(uuid.uuid4()),
    doc_id    = "recent",
    content   = "This is identical content about BM25 retrieval.",
    tenant_id = "default",
    metadata  = {"score": 0.75, "date": "2025-08-01"},
)
old_chunk = Chunk(
    chunk_id  = str(uuid.uuid4()),
    doc_id    = "old",
    content   = "This is identical content about BM25 retrieval.",
    tenant_id = "default",
    metadata  = {"score": 0.75, "date": "2020-01-01"},
)

recency_reranker = MetadataReranker(
    recency_boost     = 2.0,
    recency_half_life = 180,
)
result = recency_reranker.rerank("BM25", [old_chunk, recent_chunk])
print(f"Recency reranker order (should be recent first):")
for r in result:
    doc_id = r.doc_id
    boost  = r.metadata.get("boost_applied", 1.0)
    score  = r.metadata.get("rerank_score", 0)
    print(f"  {doc_id:<10}  boost={boost:.3f}  final_score={score:.4f}")