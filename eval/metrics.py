# eval/metrics.py

from core import Chunk
import math

# Add these two functions to the bottom of eval/metrics.py

import math as _math

def ndcg_at_k(retrieved: list, relevant_text: str, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not retrieved or not relevant_text:
        return 0.0
    dcg = 0.0
    for i, chunk in enumerate(retrieved[:k], start=1):
        if is_relevant(chunk, relevant_text):
            dcg += 1.0 / _math.log2(i + 1)
            break
    idcg = 1.0   # perfect: relevant doc at rank 1
    return round(dcg / idcg, 4)


def precision_at_k(retrieved: list, relevant_text: str, k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    n_rel = sum(1 for c in top_k if is_relevant(c, relevant_text))
    return round(n_rel / len(top_k), 4)
def is_relevant(chunk: Chunk, relevant_text: str, min_overlap: float = 0.6) -> bool:
    """
    A chunk is relevant if it contains at least `min_overlap` fraction of the
    relevant_text's words in the same order, tolerating chunk boundary cuts.
    """
    gold_words  = relevant_text.lower().split()
    chunk_lower = chunk.content.lower()

    if not gold_words:
        return False

    # Substring shortcut (fast path)
    if relevant_text.lower() in chunk_lower:
        return True

    # Sliding window: longest run of consecutive gold_words present in chunk
    max_run = 0
    current = 0
    chunk_tokens = chunk_lower.split()
    chunk_text   = " ".join(chunk_tokens)

    # Check every contiguous sub-sequence of gold_words
    for start in range(len(gold_words)):
        for end in range(start + 1, len(gold_words) + 1):
            span = " ".join(gold_words[start:end])
            if span in chunk_text and (end - start) > max_run:
                max_run = end - start

    return (max_run / len(gold_words)) >= min_overlap


def recall_at_k(retrieved: list[Chunk], relevant_text: str, k: int) -> float:
    top_k = retrieved[:k]
    return 1.0 if any(is_relevant(c, relevant_text) for c in top_k) else 0.0


def mean_reciprocal_rank(retrieved: list[Chunk], relevant_text: str) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if is_relevant(chunk, relevant_text):
            return 1.0 / rank
    return 0.0


def faithfulness_check(answer: str, retrieved: list[Chunk]) -> float:
    context = " ".join(c.content.lower() for c in retrieved)
    words   = answer.lower().split()

    if len(words) < 4:
        return 1.0

    ngrams = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 1.0

    grounded = sum(1 for ng in ngrams if ng in context)
    return grounded / len(ngrams)

def ndcg_at_k(retrieved: list, relevant_text: str, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    DCG@k  = sum(rel_i / log2(i+1)) for i in 1..k
    IDCG@k = DCG of perfect ranking (relevant doc at rank 1)
    nDCG@k = DCG@k / IDCG@k

    For binary relevance (0 or 1):
    - IDCG@k = 1/log2(2) = 1.0 (perfect: relevant at rank 1)
    - DCG@k  = 1/log2(rank+1) if relevant doc found, else 0
    """
    if not retrieved or not relevant_text:
        return 0.0

    dcg = 0.0
    for i, chunk in enumerate(retrieved[:k], start=1):
        if is_relevant(chunk, relevant_text):
            dcg += 1.0 / math.log2(i + 1)
            break   # binary relevance: only first relevant counts

    idcg = 1.0   # perfect: relevant doc at rank 1 → 1/log2(2) = 1.0
    return round(dcg / idcg, 4)


def precision_at_k(retrieved: list, relevant_text: str, k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    if not retrieved:
        return 0.0
    top_k   = retrieved[:k]
    n_rel   = sum(1 for c in top_k if is_relevant(c, relevant_text))
    return round(n_rel / len(top_k), 4)