# scripts/test_gold_set_v2.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from eval.gold_set_v2 import (
    GOLD_SET_V2, get_by_type, get_by_difficulty,
    get_positive_queries, get_negative_queries,
)
from collections import Counter

print(f"Total queries: {len(GOLD_SET_V2)}")
print(f"By type:       {dict(Counter(q['query_type'] for q in GOLD_SET_V2))}")
print(f"By difficulty: {dict(Counter(q['difficulty'] for q in GOLD_SET_V2))}")
print(f"Positive:      {len(get_positive_queries())}")
print(f"Negative:      {len(get_negative_queries())}")

print("\nQuery breakdown:")
for q in GOLD_SET_V2:
    neg = " [NEGATIVE]" if not q["relevant_text"] else ""
    print(f"  [{q['difficulty']:6s}] [{q['query_type']:10s}] {q['query'][:50]}{neg}")