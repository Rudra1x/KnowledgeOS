# eval/__init__.py

from .runner   import evaluate, print_report
from .gold_set import GOLD_SET
from .metrics  import recall_at_k, mean_reciprocal_rank, faithfulness_check

__all__ = [
    "evaluate", "print_report", "GOLD_SET",
    "recall_at_k", "mean_reciprocal_rank", "faithfulness_check",
]