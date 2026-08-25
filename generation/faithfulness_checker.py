# generation/faithfulness_checker.py

import re
import time
from core import Chunk
from generation.local_generator import LocalLLMGenerator


class FaithfulnessChecker:
    """
    Checks whether a generated answer is supported by retrieved context.

    Two strategies:
    - 'nli': cross-encoder NLI model (fast, ~500ms, no Ollama needed)
    - 'llm': Qwen2.5-3B via Ollama (flexible, ~30s)

    NLI is the production default. LLM is for complex multi-sentence
    claims that NLI handles poorly.

    Parameters
    ----------
    strategy   : 'nli' | 'llm'
    model_name : str   NLI model (for strategy='nli')
    generator  : LocalLLMGenerator (for strategy='llm')
    threshold  : float entailment threshold (nli=0.25, llm=0.5)
    """

    NAME = "faithfulness_checker"

    LLM_PROMPT = """Is the following claim directly supported by the passage?
Answer YES if the passage explicitly states or clearly implies the claim.
Answer NO if the passage does not support the claim or contradicts it.

Passage: {passage}

Claim: {claim}

Answer YES or NO:"""

    def __init__(
        self,
        strategy:   str                      = "nli",
        model_name: str                      = "cross-encoder/nli-deberta-v3-small",
        generator:  LocalLLMGenerator | None = None,
        threshold:  float                    = 0.25,
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
            "faithful":      bool
            "score":         float  fraction of supported claims
            "claims":        list[str]
            "supported":     list[str]
            "unsupported":   list[str]
            "claim_details": list[dict]
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
                "claim":      claim,
                "supported":  is_supported,
                "score":      round(best_score, 3),
                "best_chunk": best_chunk_id[:8] if best_chunk_id else None,
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
        """Split answer into individual claims (sentences), strip citations."""
        clean = re.sub(r"\[\d+\]", "", answer).strip()
        sents = re.split(r"(?<=[.!?])\s+", clean)
        return [s.strip() for s in sents
                if len(s.strip()) > 15 and not s.strip().startswith("[")]

    def _nli_score(self, passage: str, claim: str) -> float:
        """
        Score entailment probability using NLI cross-encoder.
        Falls back to string overlap for near-verbatim claims
        (NLI models underperform when hypothesis ≈ premise).
        """
        clean_passage = self._clean_passage(passage, claim)

        # String overlap fallback — NLI underscores near-verbatim claims
        overlap = self._word_overlap(clean_passage, claim)
        if overlap >= 0.85:
            return 1.0   # near-verbatim → automatically supported

        result = self.model.predict(
            [(clean_passage[:512], claim)],
            apply_softmax=True,
        )
        scores = result[0]
        for idx, label in self.model.config.id2label.items():
            if "entail" in str(label).lower():
                return float(scores[int(idx)])
        return float(scores[1])

    @staticmethod
    def _word_overlap(passage: str, claim: str) -> float:
        """Fraction of claim words that appear in passage."""
        stopwords    = {"is", "a", "an", "the", "and", "or", "it", "by",
                        "how", "what", "does", "in", "on", "at", "to", "of",
                        "based", "that", "this", "was", "were", "has", "have"}
        claim_words  = {w.lower() for w in re.findall(r"\b\w+\b", claim)
                        if w.lower() not in stopwords}
        if not claim_words:
            return 0.0
        passage_words = {w.lower() for w in re.findall(r"\b\w+\b", passage)}
        return len(claim_words & passage_words) / len(claim_words)

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
        return 0.5

    @staticmethod
    def _clean_passage(passage: str, claim: str) -> str:
        """
        Remove leading sentences that share no content words with the claim.
        Handles chunk boundary artifacts that confuse NLI models.
        """
        sentences = re.split(r"(?<=[.!?])\s+", passage.strip())
        if len(sentences) <= 1:
            return passage

        stopwords   = {"is", "a", "an", "the", "and", "or", "it", "by",
                       "how", "what", "does", "in", "on", "at", "to", "of"}
        claim_words = {w.lower() for w in re.findall(r"\b\w+\b", claim)
                       if w.lower() not in stopwords}

        kept = []
        for sent in sentences:
            sent_words = {w.lower() for w in re.findall(r"\b\w+\b", sent)}
            if kept or (claim_words & sent_words):
                kept.append(sent)

        return " ".join(kept) if kept else passage

    def _empty_result(self) -> dict:
        return {
            "faithful":      True,
            "score":         1.0,
            "claims":        [],
            "supported":     [],
            "unsupported":   [],
            "claim_details": [],
        }