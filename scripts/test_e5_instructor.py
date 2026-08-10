# scripts/test_e5_instructor.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

import numpy as np
from embedders.bge_embedder        import BGEEmbedder
from embedders.e5_embedder         import E5Embedder
from embedders.instructor_embedder import InstructionEmbedder

TEXTS = [
    "BM25 is a sparse retrieval algorithm based on term frequency.",
    "Dense retrieval uses neural embeddings to capture semantic meaning.",
    "Hybrid retrieval combines sparse and dense methods.",
]
QUERY = "What is BM25 and when does it work well?"

print("Loading models...\n")

embedders = [
    ("BGE-small",         BGEEmbedder("BAAI/bge-small-en-v1.5")),
    ("E5-small",          E5Embedder("intfloat/e5-small-v2")),
    ("Instruction-BGEb",  InstructionEmbedder(
        model_name        = "BAAI/bge-base-en-v1.5",
        query_instruction = "Represent the question for retrieving relevant documents: ",
        embed_instruction = "Represent the technical document for retrieval: ",
    )),
]

for name, emb in embedders:
    print(f"{'='*55}")
    print(f"{name}  (dim={emb.dimension})")
    print(f"{'='*55}")

    vecs = emb.embed(TEXTS)
    print(f"  Passages embedded: {len(vecs)} x {len(vecs[0])} dims")
    print(f"  Embed time: {emb.last_embed_ms:.1f}ms")

    qvec = emb.embed_query(QUERY)
    print(f"  Query time: {emb.last_embed_ms:.1f}ms")

    scores = [float(np.dot(qvec, v)) for v in vecs]
    ranked = sorted(zip(scores, TEXTS), reverse=True)
    print(f"\n  Ranking for: '{QUERY}'")
    for score, text in ranked:
        marker = " ← expected top" if "BM25" in text else ""
        print(f"    {score:.4f}  {text[:60]}{marker}")

    norms = [abs(1.0 - float(np.dot(v, v))) for v in vecs]
    print(f"\n  Max norm deviation from 1.0: {max(norms):.2e}\n")