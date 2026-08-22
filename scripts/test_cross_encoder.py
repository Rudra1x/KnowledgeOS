# scripts/test_cross_encoder.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader              import TXTLoader
from chunkers.recursive_chunker      import RecursiveChunker
from embedders.bge_embedder          import BGEEmbedder
from indexes.faiss_index             import FaissFlatIndex
from retrievers.vector_retriever     import VectorRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
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
reranker  = CrossEncoderReranker(
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    batch_size = 8,
)

print("\n" + "=" * 70)
print(f"{'Query':<45} {'Base r@1':>8} {'CE r@1':>7} {'Rank change':>12}")
print("=" * 70)

b_r1s, ce_r1s = [], []
b_mrrs, ce_mrrs = [], []

for item in GOLD_SET:
    q   = item["query"]
    rt  = item["relevant_text"]

    # Retrieve top-5 candidates
    candidates = retriever.retrieve(q, top_k=5)
    b_r1  = recall_at_k(candidates, rt, 1)
    b_mrr = mean_reciprocal_rank(candidates, rt)

    # Rerank
    reranked = reranker.rerank(q, copy.deepcopy(candidates), top_k=5)
    ce_r1    = recall_at_k(reranked, rt, 1)
    ce_mrr   = mean_reciprocal_rank(reranked, rt)

    b_r1s.append(b_r1);   ce_r1s.append(ce_r1)
    b_mrrs.append(b_mrr); ce_mrrs.append(ce_mrr)

    # Show rank movement for the relevant chunk
    orig_rank, new_rank = "?", "?"
    for i, c in enumerate(candidates, 1):
        if rt[:30] in c.content:
            orig_rank = i
            break
    for i, c in enumerate(reranked, 1):
        if rt[:30] in c.content:
            new_rank = i
            break

    flag = " <- CE wins" if ce_r1 > b_r1 else \
           " <- CE loses" if ce_r1 < b_r1 else ""
    print(f"{q[:45]:<45} {b_r1:>8.0f} {ce_r1:>7.0f} "
          f"  {orig_rank!s:>3}->{new_rank!s:<3}{flag}")

print("=" * 70)
print(f"{'MEAN':<45} {mean(b_r1s):>8.3f} {mean(ce_r1s):>7.3f}")
print(f"{'MRR':<45} {mean(b_mrrs):>8.3f} {mean(ce_mrrs):>7.3f}")
print(f"\nReranker latency: {reranker.last_rerank_ms:.1f}ms (last batch)")