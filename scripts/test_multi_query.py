# scripts/test_multi_query.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from retrievers.vector_retriever import VectorRetriever
from retrievers.multi_query_retriever import MultiQueryRetriever
from generation.local_generator  import LocalLLMGenerator
from eval.gold_set               import GOLD_SET
from eval.metrics                import recall_at_k, mean_reciprocal_rank
from core                        import NormalizationPipeline, load_config
from statistics                  import mean

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
generator  = LocalLLMGenerator(max_tokens=200, temperature=0.3)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index = FaissFlatIndex(dimension=384)
index.add(chunks)

baseline = VectorRetriever(embedder=embedder, index=index)
mq       = MultiQueryRetriever(embedder=embedder, index=index,
                                generator=generator, n_variants=3, fetch_k=5)

# --- Show variants for one query ---
print("=" * 65)
print("VARIANT GENERATION")
print("=" * 65)
sample_q  = "How does chunking affect retrieval quality?"
variants  = mq._generate_variants(sample_q)
print(f"Original:   {sample_q}")
for i, v in enumerate(variants, 1):
    print(f"Variant {i}:  {v}")

# --- Eval on gold set ---
print("\n" + "=" * 65)
print("GOLD SET EVAL (10 queries)")
print("=" * 65)
print(f"{'Query':<45} {'Base r@1':>8} {'MQ r@1':>7}")
print("-" * 65)

b_r1s, b_mrrs = [], []
m_r1s, m_mrrs = [], []

for item in GOLD_SET:
    q  = item["query"]
    rt = item["relevant_text"]

    b_res = baseline.retrieve(q, top_k=5)
    m_res = mq.retrieve(q, top_k=5)

    b_r1  = recall_at_k(b_res, rt, 1)
    m_r1  = recall_at_k(m_res, rt, 1)
    b_mrr = mean_reciprocal_rank(b_res, rt)
    m_mrr = mean_reciprocal_rank(m_res, rt)

    b_r1s.append(b_r1); b_mrrs.append(b_mrr)
    m_r1s.append(m_r1); m_mrrs.append(m_mrr)

    flag = " ← MQ wins" if m_r1 > b_r1 else \
           " ← MQ loses" if m_r1 < b_r1 else ""
    print(f"{q[:45]:<45} {b_r1:>8.0f} {m_r1:>7.0f}{flag}")

print("-" * 65)
print(f"{'MEAN':<45} {mean(b_r1s):>8.3f} {mean(m_r1s):>7.3f}")
print(f"{'MRR':<45} {mean(b_mrrs):>8.3f} {mean(m_mrrs):>7.3f}")