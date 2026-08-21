# scripts/test_agentic.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from indexes.bm25_index          import BM25Index
from retrievers.agentic_retriever import AgenticRetriever
from generation.local_generator  import LocalLLMGenerator
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)
generator  = LocalLLMGenerator(max_tokens=150, temperature=0.0)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

dense_idx  = FaissFlatIndex(dimension=384)
sparse_idx = BM25Index()
dense_idx.add(copy.deepcopy(chunks))
sparse_idx.add(copy.deepcopy(chunks))

agent = AgenticRetriever(
    embedder     = embedder,
    dense_index  = dense_idx,
    sparse_index = sparse_idx,
    generator    = generator,
    max_steps    = 3,
    fetch_k      = 2,
)

QUERIES = [
    "What is BM25 and how does it score documents?",
    "Compare dense retrieval and sparse retrieval methods.",
    "What evaluation metrics exist for RAG systems?",
]

for query in QUERIES:
    print("=" * 65)
    print(f"QUERY: {query}\n")
        # Patch to show trace
    original_call = generator.call_raw
    step_n = [0]
    def traced_call(prompt):
        step_n[0] += 1
        resp = original_call(prompt)
        print(f"  [LLM step {step_n[0]}] -> {resp[:80].strip()}")
        return resp
    generator.call_raw = traced_call
    results = agent.retrieve(query, top_k=3)
    generator.call_raw = original_call
    step_n[0] = 0
    results = agent.retrieve(query, top_k=3)
    steps_used = {r.metadata.get("agentic_step") for r in results}
    actions    = {r.metadata.get("agentic_action") for r in results}
    print(f"Steps used: {sorted(steps_used)}")
    print(f"Tools used: {actions}")
    print(f"Chunks returned: {len(results)}")
    for r in results:
        step   = r.metadata.get("agentic_step", "?")
        action = r.metadata.get("agentic_action", "?")
        score  = r.metadata.get("score", 0)
        print(f"  [step={step} tool={action}] "
              f"score={score:.4f} {r.content[:55].strip()}...")
    print()