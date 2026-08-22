# scripts/test_bge_reranker.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader              import TXTLoader
from chunkers.recursive_chunker      import RecursiveChunker
from embedders.bge_embedder          import BGEEmbedder
from indexes.faiss_index             import FaissFlatIndex
from retrievers.vector_retriever     import VectorRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from rerankers.bge_reranker          import BGEReranker
from eval.gold_set                   import GOLD_SET
from eval.metrics                    import recall_at_k, mean_reciprocal_rank
from core                            import NormalizationPipeline, load_config
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

# Load both rerankers
msmarco = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
bge     = BGEReranker("BAAI/bge-reranker-base")

print("\n" + "=" * 78)
print(f"{'Query':<42} {'Base':>5} {'MS-MARCO':>9} {'BGE':>5}")
print("=" * 78)

base_r1s, ms_r1s, bge_r1s = [], [], []
base_mrrs, ms_mrrs, bge_mrrs = [], [], []

for item in GOLD_SET:
    q   = item["query"]
    rt  = item["relevant_text"]

    candidates = retriever.retrieve(q, top_k=5)
    base_r1    = recall_at_k(candidates, rt, 1)
    base_mrr   = mean_reciprocal_rank(candidates, rt)

    ms_ranked  = msmarco.rerank(q, copy.deepcopy(candidates), top_k=5)
    ms_r1      = recall_at_k(ms_ranked, rt, 1)
    ms_mrr     = mean_reciprocal_rank(ms_ranked, rt)

    bge_ranked = bge.rerank(q, copy.deepcopy(candidates), top_k=5)
    bge_r1     = recall_at_k(bge_ranked, rt, 1)
    bge_mrr    = mean_reciprocal_rank(bge_ranked, rt)

    base_r1s.append(base_r1);  ms_r1s.append(ms_r1);  bge_r1s.append(bge_r1)
    base_mrrs.append(base_mrr); ms_mrrs.append(ms_mrr); bge_mrrs.append(bge_mrr)

    flag = ""
    if bge_r1 > ms_r1:   flag = " <- BGE wins"
    elif ms_r1 > bge_r1: flag = " <- MS-MARCO wins"
    print(f"{q[:42]:<42} {base_r1:>5.0f} {ms_r1:>9.0f} {bge_r1:>5.0f}{flag}")

print("=" * 78)
print(f"{'MEAN recall@1':<42} {mean(base_r1s):>5.3f} {mean(ms_r1s):>9.3f} "
      f"{mean(bge_r1s):>5.3f}")
print(f"{'MEAN MRR':<42} {mean(base_mrrs):>5.3f} {mean(ms_mrrs):>9.3f} "
      f"{mean(bge_mrrs):>5.3f}")
print(f"\nMS-MARCO latency: {msmarco.last_rerank_ms:.1f}ms")
print(f"BGE latency:      {bge.last_rerank_ms:.1f}ms")