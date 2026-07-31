# eval/gold_set.py

"""
Hand-labeled evaluation set: queries + which chunks should be retrieved.
Chunk match is done by substring — the relevant_text must appear in the retrieved chunk.
"""

GOLD_SET = [
    {
        "query":         "What is RAG?",
        "relevant_text": "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation",
    },
    {
        "query":         "How does chunking affect retrieval quality?",
        "relevant_text": "Chunking strategy has a large impact on retrieval quality",
    },
    {
        "query":         "What is BM25 and when does it work well?",
        "relevant_text": "BM25 is a sparse retrieval algorithm based on term frequency",
    },
    {
        "query":         "How is dense retrieval different from BM25?",
        "relevant_text": "Dense retrieval uses neural embeddings to capture semantic meaning",
    },
    {
        "query":         "What is hybrid retrieval?",
        "relevant_text": "Hybrid retrieval combines sparse and dense methods",
    },
    {
        "query":         "How do cross-encoder rerankers work?",
        "relevant_text": "Cross-encoders score query-document pairs jointly",
    },
    {
        "query":         "Which metrics evaluate retrieval?",
        "relevant_text": "Evaluation metrics for retrieval include recall@k, precision@k, MRR, and nDCG",
    },
    {
        "query":         "What is faithfulness in RAG?",
        "relevant_text": "Faithfulness measures whether generated answers are supported by the retrieved context",
    },
    {
        "query":         "What is FAISS used for?",
        "relevant_text": "FAISS is a library for efficient similarity search on dense vectors",
    },
    {
        "query":         "What does Reciprocal Rank Fusion do?",
        "relevant_text": "Reciprocal Rank Fusion (RRF) merges ranked lists from BM25 and vector search",
    },
]