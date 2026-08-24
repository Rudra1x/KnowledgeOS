# scripts/test_generation.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader               import TXTLoader
from chunkers.recursive_chunker       import RecursiveChunker
from embedders.bge_embedder           import BGEEmbedder
from indexes.faiss_index              import FaissFlatIndex
from retrievers.vector_retriever      import VectorRetriever
from rerankers.cross_encoder_reranker import CrossEncoderReranker
from generation.local_generator       import LocalLLMGenerator
from generation.prompt_builder        import build_prompt, extract_citations
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
generator = LocalLLMGenerator(max_tokens=300, temperature=0.0)

QUERIES = [
    "What is BM25 and how does it score documents?",
    "How does hybrid retrieval work?",
    "What metrics evaluate retrieval quality?",
]

for query in QUERIES:
    print("=" * 65)
    print(f"Q: {query}\n")

    # Retrieve + rerank
    candidates = retriever.retrieve(query, top_k=5)
    reranked   = reranker.rerank(query, copy.deepcopy(candidates), top_k=3)

    # Generate
    answer = generator.generate(query, reranked)
    cited  = extract_citations(answer, reranked)

    print(f"A: {answer}\n")
    print(f"Citations used: {list(cited.keys())}")
    for num, chunk in cited.items():
        print(f"  [{num}] {chunk.content[:60].strip()}...")
    print()