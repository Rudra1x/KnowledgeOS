# eval/gold_set_v2.py
"""
Expanded gold set with typed queries, full ground truth, and difficulty levels.

Each entry:
  query:          str   the user's question
  query_type:     str   'factoid' | 'comparison' | 'thematic' | 'negative'
  relevant_text:  str   substring of the correct chunk(s)
  expected_chunks: int  how many chunks needed for a complete answer
  difficulty:     str   'easy' | 'medium' | 'hard'
  notes:          str   what makes this query interesting or tricky
"""

GOLD_SET_V2 = [
    # --- Factoid (single chunk, clear answer) ---
    {
        "query":          "What is RAG?",
        "query_type":     "factoid",
        "relevant_text":  "Retrieval-Augmented Generation (RAG) combines information retrieval",
        "expected_chunks": 1,
        "difficulty":     "easy",
        "notes":          "Baseline. Every retriever should get this.",
    },
    {
        "query":          "What is BM25 and when does it work well?",
        "query_type":     "factoid",
        "relevant_text":  "BM25 is a sparse retrieval algorithm based on term frequency",
        "expected_chunks": 1,
        "difficulty":     "easy",
        "notes":          "Term frequency exact match. BM25 and dense both retrieve correctly.",
    },
    {
        "query":          "What is FAISS used for?",
        "query_type":     "factoid",
        "relevant_text":  "FAISS is a library for efficient similarity search",
        "expected_chunks": 1,
        "difficulty":     "easy",
        "notes":          "Exact entity match. All retrievers should get this.",
    },
    {
    "query":          "What is faithfulness in RAG evaluation?",
    "query_type":     "factoid",
    "relevant_text":  "Faithfulness measures whether generated answers are supported",   # ← was "Faithfulness measures whether"
    "expected_chunks": 1,
    "difficulty":     "medium",
    "notes":          "Technical concept. Tests semantic retrieval.",
},
{
    "query":          "What makes dense retrieval different from keyword search?",
    "query_type":     "comparison",
    "relevant_text":  "Dense retrieval uses neural embeddings",
    "expected_chunks": 1,
    "difficulty":     "medium",
    "notes":          "Rephrased comparison — 'keyword search' maps cleanly to dense chunk.",
},
    {
        "query":          "What does Reciprocal Rank Fusion do?",
        "query_type":     "factoid",
        "relevant_text":  "Reciprocal Rank Fusion",
        "expected_chunks": 1,
        "difficulty":     "medium",
        "notes":          "Less common term. Tests rare-term retrieval.",
    },

    # --- Comparison (requires multiple chunks or cross-concept reasoning) ---
    {
    "query":          "How is dense retrieval different from BM25?",
    "query_type":     "comparison",
    "relevant_text":  "Dense retrieval uses neural embeddings",
    "expected_chunks": 2,
    "difficulty":     "hard",
    "notes":          "Known hard query — hybrid chunk ranks above dense chunk. recall@3=1.0, recall@1=0. Correct behavior for this corpus structure.",
},
    {
        "query":          "What is hybrid retrieval?",
        "query_type":     "comparison",
        "relevant_text":  "Hybrid retrieval combines sparse and dense methods",
        "expected_chunks": 1,
        "difficulty":     "medium",
        "notes":          "Synthesis of two retrieval paradigms.",
    },
    {
        "query":          "How do cross-encoder rerankers work?",
        "query_type":     "comparison",
        "relevant_text":  "cross-encoders score query-document pairs jointly",
        "expected_chunks": 1,
        "difficulty":     "medium",
        "notes":          "Technical process question.",
    },

    # --- Thematic (requires synthesis across chunks) ---
    {
        "query":          "Which metrics evaluate retrieval quality?",
        "query_type":     "thematic",
        "relevant_text":  "Evaluation metrics for retrieval include recall@k",
        "expected_chunks": 1,
        "difficulty":     "medium",
        "notes":          "List question. Needs the evaluation chunk.",
    },
    {
        "query":          "How does chunking affect retrieval quality?",
        "query_type":     "thematic",
        "relevant_text":  "Chunking strategy has a large impact",
        "expected_chunks": 1,
        "difficulty":     "medium",
        "notes":          "Process question about pipeline design.",
    },

    # --- Negative (answer not in corpus) ---
    {
        "query":          "What is pgvector and how does it compare to FAISS?",
        "query_type":     "negative",
        "relevant_text":  "",   # no relevant text — not in corpus
        "expected_chunks": 0,
        "difficulty":     "hard",
        "notes":          "pgvector is not in our corpus. System should say 'not found'.",
    },
    {
        "query":          "What is the capital of France?",
        "query_type":     "negative",
        "relevant_text":  "",
        "expected_chunks": 0,
        "difficulty":     "easy",
        "notes":          "General knowledge question. Should trigger SelfRAG gate (NO retrieve).",
    },
]

# Convenience accessors
def get_by_type(query_type: str) -> list[dict]:
    return [q for q in GOLD_SET_V2 if q["query_type"] == query_type]

def get_by_difficulty(difficulty: str) -> list[dict]:
    return [q for q in GOLD_SET_V2 if q["difficulty"] == difficulty]

def get_positive_queries() -> list[dict]:
    """Queries where the answer IS in the corpus."""
    return [q for q in GOLD_SET_V2 if q["relevant_text"]]

def get_negative_queries() -> list[dict]:
    """Queries where the answer is NOT in the corpus."""
    return [q for q in GOLD_SET_V2 if not q["relevant_text"]]