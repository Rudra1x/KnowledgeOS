# scripts/test_faithfulness_nli.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from retrievers.vector_retriever      import VectorRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from generation.faithfulness_checker  import FaithfulnessChecker
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

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = VectorRetriever(embedder=embedder, index=index)
reranker  = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

# NLI strategy — no Ollama, dedicated model
checker = FaithfulnessChecker(
    strategy   = "nli",
    model_name = "cross-encoder/nli-MiniLM2-L6-H768",
    threshold  = 0.25,
)

query      = "What is BM25?"
candidates = retriever.retrieve(query, top_k=5)
reranked   = reranker.rerank(query, copy.deepcopy(candidates), top_k=3)

TESTS = [
    {
        "label":  "Faithful",
        "answer": ("BM25 is a sparse retrieval algorithm based on term frequency "
                   "and inverse document frequency. It scores documents by how many "
                   "query terms appear, weighted by rarity."),
    },
    {
        "label":  "Hallucinated",
        "answer": ("BM25 is a sparse retrieval algorithm. "
                   "It was invented by Stephen Robertson in 1994 at Cambridge University. "
                   "BM25 is the default algorithm in Elasticsearch and Apache Solr."),
    },
]

for test in TESTS:
    result = checker.check(test["answer"], reranked)
    print(f"\n{'='*55}")
    print(f"Test: {test['label']}")
    print(f"Faithful:  {result['faithful']}")
    print(f"Score:     {result['score']:.3f}")
    print(f"Latency:   {checker.last_check_ms:.0f}ms")
    for d in result["claim_details"]:
        mark = "✓" if d["supported"] else "✗"
        print(f"  {mark} [{d['score']:.2f}] {d['claim'][:65]}")