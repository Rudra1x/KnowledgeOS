# scripts/test_query_rewriting.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from retrievers.vector_retriever      import VectorRetriever
from retrievers.query_rewriting_retriever import QueryRewritingRetriever
from eval.metrics                     import recall_at_k
from core                             import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index = FaissFlatIndex(dimension=384)
index.add(chunks)

# --- Test queries — informal, abbreviated, ambiguous ---
# Each paired with the expected relevant text from gold set
TEST_CASES = [
    {
        "query":         "how does rag work?",
        "relevant_text": "Retrieval-Augmented Generation (RAG) combines information retrieval",
    },
    {
        "query":         "bm25 vs dense which is better",
        "relevant_text": "BM25 is a sparse retrieval algorithm based on term frequency",
    },
    {
        "query":         "what metrics to use for rag eval",
        "relevant_text": "Evaluation metrics for retrieval include recall@k, precision@k, MRR",
    },
]

print("=" * 70)
print(f"{'Query':<35} {'Baseline r@1':>12} {'Rewritten r@1':>14}")
print("=" * 70)

baseline    = VectorRetriever(embedder=embedder, index=index)
reformulate = QueryRewritingRetriever(embedder=embedder, index=index,
                                      mode="reformulate")
hyde        = QueryRewritingRetriever(embedder=embedder, index=index,
                                      mode="hyde")

for case in TEST_CASES:
    q  = case["query"]
    rt = case["relevant_text"]

    b_res  = baseline.retrieve(q, top_k=3)
    r_res  = reformulate.retrieve(q, top_k=3)
    h_res  = hyde.retrieve(q, top_k=3)

    b_r1   = recall_at_k(b_res,  rt, 1)
    r_r1   = recall_at_k(r_res,  rt, 1)
    h_r1   = recall_at_k(h_res,  rt, 1)

    rewritten = r_res[0].metadata.get("rewritten_query", "—") if r_res else "—"
    print(f"\nQ: {q}")
    print(f"  Rewritten:     {rewritten[:70]}")
    print(f"  Baseline r@1:  {b_r1:.0f}  |  Reformulate r@1: {r_r1:.0f}  "
          f"|  HyDE r@1: {h_r1:.0f}")