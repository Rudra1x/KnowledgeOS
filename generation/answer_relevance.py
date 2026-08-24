# generation/answer_relevance.py

import re
import time
import numpy as np
from core import Chunk, Embedder
from generation.local_generator import LocalLLMGenerator


class AnswerRelevanceScorer:
    """
    Scores how well an answer addresses the original question.

    Algorithm (RAGAS-inspired):
    1. Generate N questions from the answer using an LLM
    2. Embed the original question and each generated question
    3. Score = mean cosine similarity between original and generated questions
    4. High score: generated questions resemble the original → answer is on-topic
    5. Low score: generated questions diverge → answer is evasive or off-topic

    Intuition: if an answer truly addresses a question, you should be able
    to reconstruct a similar question from the answer alone.

    Parameters
    ----------
    embedder      : Embedder   for question similarity scoring
    generator     : LocalLLMGenerator   for reverse question generation
    n_questions   : int   number of questions to generate from answer
    """

    NAME = "answer_relevance"

    REVERSE_PROMPT = """Given the following answer, generate {n} different \
questions that this answer could be responding to.
Each question should be specific and directly answerable by the given answer.

Answer: {answer}

Generate exactly {n} questions, one per line, numbered 1. 2. etc:"""

    def __init__(
        self,
        embedder:    Embedder,
        generator:   LocalLLMGenerator | None = None,
        n_questions: int = 3,
    ):
        self.embedder    = embedder
        self.generator   = generator or LocalLLMGenerator(max_tokens=150)
        self.n_questions = n_questions
        self.last_score_ms: float = 0.0

    def score(
        self,
        question: str,
        answer:   str,
    ) -> dict:
        """
        Score the relevance of an answer to a question.

        Returns:
        {
            "relevance_score":    float  0-1, higher = more relevant
            "generated_questions": list[str]
            "similarities":        list[float]
            "verdict":            str   'relevant' | 'partially_relevant' | 'irrelevant'
        }
        """
        if not answer.strip():
            return self._empty_result()

        t0 = time.perf_counter()

        # Generate reverse questions from the answer
        gen_questions = self._generate_questions(answer)

        if not gen_questions:
            self.last_score_ms = (time.perf_counter() - t0) * 1000
            return self._empty_result()

        # Embed original question and generated questions
        orig_vec  = np.array(self.embedder.embed_query(question), dtype="float32")
        gen_vecs  = np.array(
            self.embedder.embed(gen_questions), dtype="float32"
        )

        # Normalize
        orig_norm = orig_vec / (np.linalg.norm(orig_vec) + 1e-8)
        gen_norms = gen_vecs / (
            np.linalg.norm(gen_vecs, axis=1, keepdims=True) + 1e-8
        )

        # Cosine similarities
        similarities = (gen_norms @ orig_norm).tolist()
        mean_sim     = float(np.mean(similarities))

        # Verdict
        if mean_sim >= 0.85:
            verdict = "relevant"
        elif mean_sim >= 0.65:
            verdict = "partially_relevant"
        else:
            verdict = "irrelevant"

        self.last_score_ms = (time.perf_counter() - t0) * 1000

        return {
            "relevance_score":     round(mean_sim, 3),
            "generated_questions": gen_questions,
            "similarities":        [round(s, 3) for s in similarities],
            "verdict":             verdict,
        }

    def _generate_questions(self, answer: str) -> list[str]:
        """Generate N questions that the answer could be responding to."""
        prompt   = self.REVERSE_PROMPT.format(
            n      = self.n_questions,
            answer = answer[:500],
        )
        raw      = self.generator.call_raw(prompt)
        if not raw:
            return []

        lines    = raw.strip().split("\n")
        questions = []
        for line in lines:
            line    = line.strip()
            cleaned = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 10 and "?" in cleaned:
                questions.append(cleaned)

        return questions[:self.n_questions]

    def _empty_result(self) -> dict:
        return {
            "relevance_score":     0.0,
            "generated_questions": [],
            "similarities":        [],
            "verdict":             "irrelevant",
        }