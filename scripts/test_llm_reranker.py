# scripts/test_llm_reranker.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader              import TXTLoader
from chunkers.recursive_chunker      import RecursiveChunker
from embedders.bge_embedder          import BGEEmbedder
from indexes.faiss_index             import FaissFlatIndex
from retrievers.vector_retriever     import VectorRetriever
from rerankers.llm_reranker          import LLMReranker
from generation.local_generator      import LocalLLMGenerator
from eval.gold_set                   import GOLD_SET
from eval.metrics                    import recall_at_k, mean_reciprocal_rank
from core                            import NormalizationPipeline, load_config
from statistics                      import mean

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
generator  = LocalLLMGenerator(max_tokens=10, temperature=0.0)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = VectorRetriever(embedder=embedder, index=index)

score_reranker   = LLMReranker(generator=generator, mode="score")
compare_reranker = LLMReranker(
    generator = LocalLLMGenerator(max_tokens=30, temperature=0.0),
    mode      = "compare",
)

print("=" * 72)
print(f"{'Query':<42} {'Base':>5} {'Score':>7} {'Compare':>8}")
print("=" * 72)

base_r1s, sc_r1s, cmp_r1s = [], [], []

for item in GOLD_SET:
    q   = item["query"]
    rt  = item["relevant_text"]

    candidates = retriever.retrieve(q, top_k=5)
    base_r1    = recall_at_k(candidates, rt, 1)

    sc_ranked  = score_reranker.rerank(q, copy.deepcopy(candidates), top_k=5)
    sc_r1      = recall_at_k(sc_ranked, rt, 1)

    cmp_ranked = compare_reranker.rerank(q, copy.deepcopy(candidates), top_k=5)
    cmp_r1     = recall_at_k(cmp_ranked, rt, 1)

    base_r1s.append(base_r1)
    sc_r1s.append(sc_r1)
    cmp_r1s.append(cmp_r1)

    flag = ""
    if sc_r1 > base_r1 or cmp_r1 > base_r1:
        flag = " <- LLM wins"
    elif sc_r1 < base_r1 or cmp_r1 < base_r1:
        flag = " <- LLM loses"
    print(f"{q[:42]:<42} {base_r1:>5.0f} {sc_r1:>7.0f} {cmp_r1:>8.0f}{flag}")

print("=" * 72)
print(f"{'MEAN':<42} {mean(base_r1s):>5.3f} {mean(sc_r1s):>7.3f} "
      f"{mean(cmp_r1s):>8.3f}")
print(f"\nScore mode  latency: {score_reranker.last_rerank_ms:.0f}ms "
      f"(last query × 5 chunks)")
print(f"Compare mode latency: {compare_reranker.last_rerank_ms:.0f}ms "
      f"(last query, all chunks at once)")