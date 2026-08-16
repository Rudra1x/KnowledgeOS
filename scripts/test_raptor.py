# scripts/test_raptor.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.raptor_index        import RAPTORIndex
from generation.generator        import OpenRouterGenerator
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
generator  = OpenRouterGenerator(
    model       = cfg.get("generator", "openrouter", "model"),
    max_tokens  = 200,
    temperature = 0.0,
    reasoning   = False,   # faster for summarization
)
chunker    = RecursiveChunker(chunk_size=512, chunk_overlap=50)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

print(f"Leaf chunks: {len(chunks)}\n")

# Build RAPTOR tree
raptor = RAPTORIndex(
    embedder         = embedder,
    generator        = generator,
    n_clusters       = 3,
    max_levels       = 2,
    min_cluster_size = 2,
)
raptor.add(chunks)

print(f"\nTree stats: {raptor.tree_stats()}")

# Query at different granularities
QUERIES = [
    ("specific",  "What is BM25 and when does it work well?"),
    ("specific",  "What is FAISS used for?"),
    ("thematic",  "How does retrieval work in RAG systems?"),
    ("thematic",  "What are the main components of a RAG pipeline?"),
]

print("\n" + "=" * 70)
print(f"{'Query type':<12} {'Level':>6} {'Score':>7}  Result")
print("=" * 70)

qvec_cache = {}
for qtype, query in QUERIES:
    if query not in qvec_cache:
        qvec_cache[query] = embedder.embed_query(query)
    results = raptor.search(qvec_cache[query], top_k=2, tenant_id="default")
    for r in results[:1]:
        level    = r.metadata.get("raptor_level", 0)
        node_type = r.metadata.get("raptor_node_type", "leaf")
        score    = r.metadata.get("score", 0)
        content  = r.content[:60].strip()
        print(f"{qtype:<12} L{level} ({node_type:<7}) {score:>6.4f}  {content}...")