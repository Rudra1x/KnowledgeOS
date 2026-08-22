# rerankers/llm_reranker.py

import re
import time
from core import Chunk, Reranker
from generation.local_generator import LocalLLMGenerator


class LLMReranker(Reranker):
    """
    LLM-based reranker using Qwen2.5-3B via Ollama.

    Asks the LLM to score each chunk's relevance to the query
    on a 1-10 scale. No specialized cross-encoder model required.

    Advantages over cross-encoder:
    - Customizable ranking criteria via prompt
    - Interpretable: model can explain its score
    - Already available (same LocalLLMGenerator as retrievers)

    Disadvantages:
    - ~2-3s per chunk vs ~20ms for cross-encoder
    - Score calibration varies by model and prompt

    Two scoring modes:
    - 'score':  ask for 1-10 numeric score per chunk
    - 'compare': ask LLM to rank all chunks at once (faster)

    Parameters
    ----------
    generator  : LocalLLMGenerator
    mode       : 'score' | 'compare'
    criteria   : str   additional ranking criteria appended to prompt
    """

    NAME = "llm_reranker"

    SCORE_PROMPT = """Rate how relevant the following passage is for \
answering the question.

Question: {query}

Passage: {passage}

{criteria}
Rate relevance from 1 (completely irrelevant) to 10 (perfectly relevant).
Respond with ONLY a single integer between 1 and 10:"""

    COMPARE_PROMPT = """Rank the following passages from most to least \
relevant for answering the question.

Question: {query}

Passages:
{passages}

{criteria}
Respond with ONLY the passage numbers in order from most to least relevant.
Example format: 3, 1, 4, 2, 5
Your ranking:"""

    def __init__(
        self,
        generator: LocalLLMGenerator | None = None,
        mode:      str = "score",
        criteria:  str = "",
    ):
        if mode not in {"score", "compare"}:
            raise ValueError(f"mode must be 'score' or 'compare', got {mode!r}")
        self.generator         = generator or LocalLLMGenerator(max_tokens=10)
        self.mode              = mode
        self.criteria          = criteria
        self.last_rerank_ms: float = 0.0

    def rerank(
        self,
        query:  str,
        chunks: list[Chunk],
        top_k:  int | None = None,
    ) -> list[Chunk]:
        if not chunks:
            return []

        t0 = time.perf_counter()

        if self.mode == "score":
            ranked = self._score_mode(query, chunks)
        else:
            ranked = self._compare_mode(query, chunks)

        self.last_rerank_ms = (time.perf_counter() - t0) * 1000
        return ranked[:top_k] if top_k else ranked

    # ------------------------------------------------------------------

    def _score_mode(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Score each chunk independently with the LLM."""
        for i, chunk in enumerate(chunks):
            prompt = self.SCORE_PROMPT.format(
                query    = query,
                passage  = chunk.content[:400],
                criteria = self.criteria,
            )
            response = self.generator.call_raw(prompt).strip()
            score    = self._parse_score(response)

            chunk.metadata["rerank_score"]  = score
            chunk.metadata["rerank_model"]  = "llm_score"
            chunk.metadata["original_rank"] = i + 1
            chunk.metadata["llm_response"]  = response

        return sorted(chunks, key=lambda c: c.metadata["rerank_score"],
                      reverse=True)

    def _compare_mode(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Ask LLM to rank all chunks at once."""
        passages = "\n\n".join(
            f"[{i+1}] {c.content[:300]}" for i, c in enumerate(chunks)
        )
        prompt = self.COMPARE_PROMPT.format(
            query    = query,
            passages = passages,
            criteria = self.criteria,
        )
        response = self.generator.call_raw(prompt).strip()
        order    = self._parse_ranking(response, len(chunks))

        # Tag original ranks
        for i, chunk in enumerate(chunks):
            chunk.metadata["original_rank"] = i + 1
            chunk.metadata["rerank_model"]  = "llm_compare"

        # Reorder by LLM-specified order
        ranked = []
        seen   = set()
        for idx in order:
            if 0 <= idx < len(chunks) and idx not in seen:
                chunks[idx].metadata["rerank_score"] = len(chunks) - len(ranked)
                ranked.append(chunks[idx])
                seen.add(idx)

        # Append any unranked chunks at the end
        for i, chunk in enumerate(chunks):
            if i not in seen:
                chunk.metadata["rerank_score"] = 0
                ranked.append(chunk)

        return ranked

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract integer score 1-10 from LLM response."""
        match = re.search(r"\b([1-9]|10)\b", response)
        if match:
            return float(match.group(1))
        return 5.0   # neutral default

    @staticmethod
    def _parse_ranking(response: str, n: int) -> list[int]:
        """Parse '3, 1, 4, 2' into zero-indexed list [2, 0, 3, 1]."""
        numbers = re.findall(r"\b(\d+)\b", response)
        order   = []
        for num in numbers:
            idx = int(num) - 1   # convert to 0-indexed
            if 0 <= idx < n and idx not in order:
                order.append(idx)
        return order