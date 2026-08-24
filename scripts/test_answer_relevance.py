# scripts/test_answer_relevance.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from embedders.bge_embedder       import BGEEmbedder
from generation.local_generator   import LocalLLMGenerator
from generation.answer_relevance  import AnswerRelevanceScorer
from core                         import load_config

cfg       = load_config()
embedder  = BGEEmbedder(cfg.get("embedder", "bge_small", "model_name"))
generator = LocalLLMGenerator(max_tokens=150, temperature=0.3)
scorer    = AnswerRelevanceScorer(
    embedder    = embedder,
    generator   = generator,
    n_questions = 3,
)

TEST_CASES = [
    {
        "question": "What is BM25 and how does it score documents?",
        "answer":   ("BM25 is a sparse retrieval algorithm based on term frequency "
                     "and inverse document frequency. It scores documents by how many "
                     "query terms appear, weighted by their rarity in the corpus."),
        "expected": "relevant",
    },
    {
        "question": "What is BM25 and how does it score documents?",
        "answer":   ("Retrieval systems use various methods to rank documents. "
                     "The choice of method depends on the use case and corpus size."),
        "expected": "partially_relevant or irrelevant",
    },
    {
        "question": "What is faithfulness in RAG?",
        "answer":   ("FAISS is a library for efficient similarity search on dense "
                     "vectors developed by Facebook AI Research."),
        "expected": "irrelevant",
    },
]

print("=" * 68)
print(f"{'Question':<35} {'Expected':>18} {'Got':>10} {'Score':>6}")
print("=" * 68)

for case in TEST_CASES:
    result = scorer.score(case["question"], case["answer"])

    print(f"\nQ: {case['question'][:60]}")
    print(f"A: {case['answer'][:60]}...")
    print(f"Generated questions:")
    for q in result["generated_questions"]:
        print(f"  - {q[:65]}")
    print(f"Similarities: {result['similarities']}")
    print(f"Verdict: {result['verdict']}  "
          f"Score={result['relevance_score']:.3f}  "
          f"({scorer.last_score_ms:.0f}ms)")
    print()