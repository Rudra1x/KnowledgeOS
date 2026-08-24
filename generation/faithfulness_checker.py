# generation/faithfulness_checker.py

import re
import time
from core import Chunk
from generation.local_generator import LocalLLMGenerator


class FaithfulnessChecker:
    """
    Checks whether a generated answer is supported by retrieved context.

    Faithfulness: every claim in the answer is entailed by at least
    one retrieved chunk. Claims not supported by any chunk are flagged
    as potential hallucinations.

    Two strategies:
    - 'nli':  cross-encoder NLI model (fast, calibrated, no Ollama needed)
    - 'llm':  Qwen2.5-3B via Ollama (flexible, uses existing infrastructure)

    Parameters
    ----------
    strategy   : 'nli' | 'llm'
    model_name : str   NLI model (for strategy='nli')
    generator  : LocalLLMGenerator (for strategy='llm')
    threshold  : float entailment probability threshold (nli only)
    """

    NAME = "faithfulness_checker"

    NLI_ENTAILMENT_LABEL = "entailment"

    LLM_PROMPT = """Is the following claim directly supported by the passage?
Answer YES if the passage explicitly states or clearly implies the claim.
Answer NO if the passage does not support the claim or contradicts it.

Passage: {passage}

Claim: {claim}

Answer YES or NO:"""

    def __init__(
        self,
        strategy:   str                      = "llm",
        model_name: str                      = "cross-encoder/nli-deberta-v3-small",
        generator:  LocalLLMGenerator | None = None,
        threshold:  float                    = 0.5,
    ):
        if strategy not in {"nli", "llm"}:
            raise ValueError(f"strategy must be 'nli' or 'llm'")
        self.strategy   = strategy
        self.threshold  = threshold
        self.last_check_ms: float = 0.0

        if strategy == "nli":
            from sentence_transformers import CrossEncoder
            print(f"  [FaithfulnessChecker] Loading NLI model {model_name}...")
            self.model = CrossEncoder(model_name)
            print(f"  [FaithfulnessChecker] Ready.")
        else:
            self.generator = generator or LocalLLMGenerator(max_tokens=5)

    def check(
        self,
        answer: str,
        chunks: list[Chunk],
    ) -> dict:
        """
        Check faithfulness of an answer against retrieved chunks.

        Returns:
        {
            "faithful":        bool  — True if all claims are supported
            "score":           float — fraction of supported claims
            "claims":          list[str]  — all extracted claims
            "supported":       list[str]  — claims with support
            "unsupported":     list[str]  — potential hallucinations
            "claim_details":   list[dict] — per-claim evidence
        }
        """
        if not answer or not chunks:
            return self._empty_result()

        t0     = time.perf_counter()
        claims = self._extract_claims(answer)

        if not claims:
            self.last_check_ms = (time.perf_counter() - t0) * 1000
            return self._empty_result()

        supported   = []
        unsupported = []
        details     = []

        for claim in claims:
            best_score    = 0.0
            best_chunk_id = None

            for chunk in chunks:
                if self.strategy == "nli":
                    score = self._nli_score(chunk.content, claim)
                else:
                    score = self._llm_score(chunk.content, claim)

                if score > best_score:
                    best_score    = score
                    best_chunk_id = chunk.chunk_id

            is_supported = best_score >= self.threshold
            if is_supported:
                supported.append(claim)
            else:
                unsupported.append(claim)

            details.append({
                "claim":        claim,
                "supported":    is_supported,
                "score":        round(best_score, 3),
                "best_chunk":   best_chunk_id[:8] if best_chunk_id else None,
            })

        n_claims = len(claims)
        score    = len(supported) / n_claims if n_claims else 1.0

        self.last_check_ms = (time.perf_counter() - t0) * 1000

        return {
            "faithful":      len(unsupported) == 0,
            "score":         round(score, 3),
            "claims":        claims,
            "supported":     supported,
            "unsupported":   unsupported,
            "claim_details": details,
        }

    # ------------------------------------------------------------------

    def _extract_claims(self, answer: str) -> list[str]:
        """Split answer into individual claims (sentences)."""
        # Strip citation markers before splitting
        clean = re.sub(r"\[\d+\]", "", answer).strip()
        sents = re.split(r"(?<=[.!?])\s+", clean)
        return [s.strip() for s in sents
                if len(s.strip()) > 15 and not s.strip().startswith("[")]

    def _nli_score(self, passage: str, claim: str) -> float:
        """Score entailment probability using NLI cross-encoder."""
        result = self.model.predict(
            [(passage[:512], claim)],
            apply_softmax=True,
        )
        # Labels order varies by model — find entailment label
        labels = self.model.config.id2label
        for idx, label in labels.items():
            if label.lower() == self.NLI_ENTAILMENT_LABEL:
                return float(result[0][idx])
        return float(result[0][0])

    def _llm_score(self, passage: str, claim: str) -> float:
        """Score support using LLM (YES=1.0, NO=0.0)."""
        prompt   = self.LLM_PROMPT.format(
            passage = passage[:400],
            claim   = claim[:200],
        )
        response = self.generator.call_raw(prompt).strip().upper()
        if response.startswith("YES"):
            return 1.0
        if response.startswith("NO"):
            return 0.0
        return 0.5   # uncertain