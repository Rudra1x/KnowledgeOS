# scripts/test_api_embedder.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

import numpy as np
from embedders.api_embedder import APIEmbedder

TEXTS = [
    "BM25 is a sparse retrieval algorithm based on term frequency.",
    "Dense retrieval uses neural embeddings to capture semantic meaning.",
    "Hybrid retrieval combines sparse and dense methods.",
]
QUERY = "What is BM25 and when does it work well?"

print("Testing APIEmbedder with OpenAI text-embedding-3-small...\n")

try:
    emb = APIEmbedder(
        model_name  = "text-embedding-3-small",
        base_url    = "https://api.openai.com/v1",
        api_key_env = "OPENAI_API_KEY",
    )

    vecs  = emb.embed(TEXTS)
    qvec  = emb.embed_query(QUERY)
    scores = [float(np.dot(qvec, v)) for v in vecs]
    ranked = sorted(zip(scores, TEXTS), reverse=True)

    print(f"Dimension: {emb.dimension}")
    print(f"Embed time: {emb.last_embed_ms:.1f}ms\n")
    print(f"Ranking for: '{QUERY}'")
    for score, text in ranked:
        marker = " ← expected top" if "BM25" in text else ""
        print(f"  {score:.4f}  {text[:60]}{marker}")

    norms = [abs(1.0 - float(np.dot(v, v))) for v in vecs]
    print(f"\nMax norm deviation: {max(norms):.2e}")

except RuntimeError as e:
    print(f"[SKIPPED] {e}")
    print("\nNo OpenAI key — testing architecture only.")
    print("APIEmbedder is compatible with any OpenAI-compatible endpoint:")
    print("  - OpenAI:   base_url='https://api.openai.com/v1'")
    print("  - Voyage:   base_url='https://api.voyageai.com/v1'")
    print("  - Local:    base_url='http://localhost:8000/v1'")
    print("\nTo test: add OPENAI_API_KEY to your .env file.")