# scripts/make_corpus.py

CORPUS = """Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation. RAG retrieves relevant documents from a knowledge base and provides them as context to the language model, grounding the generated answer in retrieved facts.

Chunking strategy has a large impact on retrieval quality. Fixed chunking splits text at regular character intervals regardless of content structure. Recursive chunking splits on paragraph, sentence, and character boundaries in order, preserving natural text structure. Semantic chunking uses embedding similarity to detect topic shifts, producing variable-length chunks around coherent concepts.

BM25 is a sparse retrieval algorithm based on term frequency and inverse document frequency. It scores documents by how many query terms appear, weighted by rarity. BM25 excels at exact keyword matches and short queries with specific named entities.

Dense retrieval uses neural embeddings to capture semantic meaning beyond keyword overlap. Dense retrieval can match paraphrases and synonyms that share no surface form with the query. It performs better on conceptual queries but requires GPU for fast inference at scale.

Hybrid retrieval combines sparse and dense methods. Reciprocal Rank Fusion (RRF) merges ranked lists from BM25 and vector search by summing reciprocal ranks. Hybrid retrieval consistently beats either method alone on diverse query types.

Rerankers refine an initial retrieval result set. Cross-encoders score query-document pairs jointly and produce much higher precision than bi-encoders, at the cost of latency. Rerankers are typically applied to the top 50-100 candidates from the retriever.

Evaluation metrics for retrieval include recall@k, precision@k, MRR, and nDCG. Recall@k measures whether the correct document appears in the top-k results. MRR measures how high the first relevant result ranks on average. nDCG rewards finding relevant documents at higher ranks.

Faithfulness measures whether generated answers are supported by the retrieved context. A high-faithfulness system only asserts claims that appear in the retrieved passages. Faithfulness is evaluated by checking each claim in the answer against the source chunks using NLI models.

FAISS is a library for efficient similarity search on dense vectors developed by Facebook AI Research. FAISS supports both exact search with IndexFlatIP and approximate search with IVF and HNSW indexes. It compares answer spans against retrieved chunks using inner product or L2 distance.

Chunking overlap controls how much content is shared between adjacent chunks. A 50-token overlap ensures that sentences at chunk boundaries appear in both adjacent chunks. Excessive overlap increases index size and can confuse retrieval by duplicating content."""

if __name__ == "__main__":
    import os
    os.makedirs("scripts", exist_ok=True)
    with open("scripts/corpus.txt", "w", encoding="utf-8") as f:
        f.write(CORPUS)
    print(f"Corpus written: {len(CORPUS)} chars, "
          f"{len(CORPUS.splitlines())} lines, "
          f"{len([p for p in CORPUS.split(chr(10)*2) if p.strip()])} paragraphs")