# eval/multiformat_gold_set.py

"""
Multi-format gold set. Each entry ties a query to a distinctive substring
that should appear in the correctly-retrieved chunk. The correct chunks
live in different source files (TXT, MD, PDF, CSV, HTML, email, YouTube).
"""

MULTIFORMAT_GOLD_SET = [
    # From scripts/corpus.txt (M0 baseline corpus)
    {
        "query":         "What is RAG?",
        "relevant_text": "Retrieval-Augmented Generation (RAG) combines information retrieval",
        "expected_source": "corpus.txt",
    },
    {
        "query":         "What is BM25 and when does it work well?",
        "relevant_text": "BM25 is a sparse retrieval algorithm based on term frequency",
        "expected_source": "corpus.txt",
    },

    # From scripts/sample.md (KnowledgeOS markdown)
    {
        "query":         "What is the KnowledgeOS architecture?",
        "relevant_text": "The system uses a plugin-based design",
        "expected_source": "sample.md",
    },
    {
        "query":         "What was the baseline recall from M0?",
        "relevant_text": "recall@1 of 0.8, recall@3 of 1.0",
        "expected_source": "sample.md",
    },

    # From scripts/faq.csv (row-per-doc)
    {
        "query":         "How do I measure RAG quality?",
        "relevant_text": "recall@k and MRR for retrieval",
        "expected_source": "faq.csv",
    },
    {
        "query":         "What is a cross-encoder reranker?",
        "relevant_text": "cross-encoder scores query-document pairs jointly",
        "expected_source": "faq.csv",
    },

    # From scripts/sample.eml (email content)
    {
        "query":         "What are the Q4 planning action items?",
        "relevant_text": "finalize the retrieval benchmark results by Friday",
        "expected_source": "sample.eml",
    },
]