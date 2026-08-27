# scripts/audit_gold_set.py

import sys, copy
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader          import TXTLoader
from chunkers.recursive_chunker  import RecursiveChunker
from embedders.bge_embedder      import BGEEmbedder
from indexes.faiss_index         import FaissFlatIndex
from retrievers.vector_retriever import VectorRetriever
from eval.gold_set_v2            import GOLD_SET_V2
from eval.metrics                import is_relevant
from core                        import NormalizationPipeline, load_config

cfg        = load_config()
normalizer = NormalizationPipeline()
embedder   = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
chunker    = RecursiveChunker(chunk_size=300, chunk_overlap=0)

docs   = normalizer.apply_many(TXTLoader().load("scripts/corpus.txt"))
chunks = chunker.chunk(docs[0])
vecs   = embedder.embed([c.content for c in chunks])
for c, v in zip(chunks, vecs):
    c.embedding = v

index     = FaissFlatIndex(dimension=384)
index.add(chunks)
retriever = VectorRetriever(embedder=embedder, index=index)

print("GOLD SET ANNOTATION AUDIT")
print("=" * 65)

issues = []
for item in GOLD_SET_V2:
    q  = item["query"]
    rt = item["relevant_text"]

    if not rt:
        print(f"[NEG] {q[:55]}")
        continue

    results  = retriever.retrieve(q, top_k=5)
    hit_rank = None
    for i, c in enumerate(results, 1):
        if is_relevant(c, rt):
            hit_rank = i
            break

    if hit_rank == 1:
        mark = "OK "
    elif hit_rank:
        mark = "LOW"
    else:
        mark = "BAD"

    rank_str = f"rank={hit_rank}" if hit_rank else "MISS"
    print(f"[{mark}] r@{rank_str:<10} {q[:50]}")

    if mark != "OK ":
        issues.append(item)
        print(f"       relevant_text : {rt!r}")
        if results:
            print(f"       top-1 content : {results[0].content[:80].strip()!r}")
        # Check if relevant_text appears anywhere in corpus
        found_in = [i for i, c in enumerate(chunks)
                    if rt.lower() in c.content.lower()]
        print(f"       found in chunks: {found_in}")
        print()

print()
if issues:
    print(f"Issues found: {len(issues)} queries need gold set fixes")
else:
    print("All gold set annotations verified correct")