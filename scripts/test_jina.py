# scripts/test_jina.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

import numpy as np
from embedders.bge_embedder  import BGEEmbedder
from embedders.jina_embedder import JinaEmbedder

# Two test cases:
# 1. Short text — both models have full context
# 2. Long text — BGE truncates, Jina preserves

SHORT_TEXT = "BM25 is a sparse retrieval algorithm based on term frequency."

# Simulate a 1500-char parent chunk
LONG_TEXT = """Retrieval-Augmented Generation (RAG) combines information retrieval \
with language model generation. Instead of relying only on parametric knowledge \
stored in model weights, RAG fetches relevant documents at query time. \
The retrieval step embeds both the query and documents into a shared vector space. \
Nearest neighbor search then finds the most semantically similar chunks. \
These retrieved chunks are passed as context to the generator, which produces \
a cited, grounded response. Chunking strategy has a large impact on retrieval quality. \
Fixed chunking splits text at regular character intervals regardless of meaning. \
Recursive chunking splits on paragraph, sentence, and word boundaries hierarchically. \
Semantic chunking uses embedding similarity to detect topic shifts. \
BM25 is a sparse retrieval algorithm based on term frequency and inverse document frequency. \
It scores documents by how many query terms appear, weighted by rarity. \
Dense retrieval uses neural embeddings to capture semantic meaning. \
Unlike BM25, dense retrievers can match paraphrases and synonyms that share no surface words. \
Hybrid retrieval combines sparse and dense methods using Reciprocal Rank Fusion.""" * 2

QUERY = "What is BM25 and how does it score documents?"

print(f"Short text: {len(SHORT_TEXT)} chars")
print(f"Long text:  {len(LONG_TEXT)} chars (~{len(LONG_TEXT)//4} tokens)")
print(f"BGE-small max: ~2048 chars (~512 tokens)")
print(f"Jina max: ~32768 chars (8192 tokens)\n")

print("Loading BGE-small...")
bge = BGEEmbedder("BAAI/bge-small-en-v1.5")

print("Loading Jina v3 (downloads ~570MB on first run)...")
jina = JinaEmbedder("jinaai/jina-embeddings-v3")

for name, emb in [("BGE-small", bge), ("Jina-base", jina)]:
    print(f"\n{'='*55}")
    print(f"{name}  (dim={emb.dimension}, max≈{'512 tok' if name=='BGE-small' else '8192 tok'})")
    print(f"{'='*55}")

    # Short text
    t0   = __import__("time").perf_counter()
    sv   = emb.embed([SHORT_TEXT])[0]
    t_s  = (__import__("time").perf_counter() - t0) * 1000

    # Long text
    t0   = __import__("time").perf_counter()
    lv   = emb.embed([LONG_TEXT])[0]
    t_l  = (__import__("time").perf_counter() - t0) * 1000

    qv   = emb.embed_query(QUERY)

    sim_short = float(np.dot(qv, sv))
    sim_long  = float(np.dot(qv, lv))

    print(f"  Short text ({len(SHORT_TEXT):>5} chars) | sim={sim_short:.4f} | {t_s:.0f}ms")
    print(f"  Long text  ({len(LONG_TEXT):>5} chars) | sim={sim_long:.4f} | {t_l:.0f}ms")
    print(f"  Sim ratio long/short: {sim_long/sim_short:.3f}")
    print(f"  (BGE: ratio drops when long text gets truncated)")
    print(f"  (Jina: ratio stays high — full content preserved)")

    norm = abs(1.0 - float(np.dot(lv, lv)))
    print(f"  Norm deviation: {norm:.2e}")