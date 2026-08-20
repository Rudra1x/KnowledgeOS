# retrievers/crag_retriever.py

from core import Chunk, Retriever, Embedder, Index
from generation.local_generator import LocalLLMGenerator


class CRAGRetriever(Retriever):
    """
    Corrective RAG: evaluate retrieved docs, correct if poor quality.

    Algorithm:
    1. Retrieve top_k candidates
    2. LLM + keyword heuristic evaluates quality: CORRECT / AMBIGUOUS / INCORRECT
    3. CORRECT   → return as-is
    4. AMBIGUOUS → filter to relevant chunks, re-retrieve to supplement
    5. INCORRECT → reformulate query, re-retrieve entirely

    Two-signal evaluation:
    - LLM evaluation (primary — semantic judgment)
    - Keyword overlap heuristic (backup — catches LLM false positives)

    Parameters
    ----------
    embedder         : Embedder
    index            : Index
    generator        : LocalLLMGenerator
    max_corrections  : int  max re-retrieval attempts
    """

    NAME = "crag"

    EVAL_PROMPT = """You are strictly evaluating whether retrieved passages \
can answer a question. Be critical.

Question: {query}

Retrieved passages:
{passages}

Rules:
- CORRECT: passages directly contain information to answer the question
- INCORRECT: passages are about completely different topics than the question
- AMBIGUOUS: passages are partially related

If the question asks about topics NOT present in the passages (like weather, \
current events, personal info), respond INCORRECT.

Respond with ONLY one word (CORRECT, AMBIGUOUS, or INCORRECT):"""

    CHUNK_RELEVANCE_PROMPT = """Is this passage relevant to answering the question?

Question: {query}
Passage: {passage}

Answer YES or NO:"""

    REFORMULATE_PROMPT = """The following search query returned irrelevant results. \
Rewrite it to be more specific and likely to find relevant information.

Original query: {query}
Why it failed: retrieved passages were not relevant

Improved query (one sentence only):"""

    STOPWORDS = {"is", "the", "a", "an", "how", "what", "does", "do",
                 "and", "or", "of", "in", "to", "for", "it", "with",
                 "where", "when", "who", "which", "are", "was", "were"}

    def __init__(
        self,
        embedder:        Embedder,
        index:           Index,
        generator:       LocalLLMGenerator | None = None,
        max_corrections: int = 2,
    ):
        self.embedder        = embedder
        self.index           = index
        self.generator       = generator or LocalLLMGenerator(max_tokens=50)
        self.max_corrections = max_corrections

    def retrieve(
        self,
        query:     str,
        top_k:     int = 5,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        current_query = query
        final_chunks: list[Chunk] = []

        for attempt in range(self.max_corrections + 1):
            # Retrieve
            qvec       = self.embedder.embed_query(current_query)
            candidates = self.index.search(
                qvec, top_k=top_k * 2, tenant_id=tenant_id
            )

            if not candidates:
                break

            # Evaluate quality
            evaluation = self._evaluate(current_query, candidates[:top_k])

            if evaluation == "CORRECT":
                final_chunks = candidates[:top_k]
                for c in final_chunks:
                    c.metadata["crag_action"]  = "correct"
                    c.metadata["crag_attempt"] = attempt + 1
                break

            elif evaluation == "AMBIGUOUS":
                relevant = [c for c in candidates
                            if self._is_relevant(current_query, c)]
                if len(relevant) >= top_k // 2:
                    final_chunks = relevant[:top_k]
                    for c in final_chunks:
                        c.metadata["crag_action"]  = "ambiguous_filtered"
                        c.metadata["crag_attempt"] = attempt + 1
                    break
                evaluation = "INCORRECT"

            if evaluation == "INCORRECT" and attempt < self.max_corrections:
                current_query = self._reformulate(current_query)
                continue

            # Last attempt — return what we have
            final_chunks = candidates[:top_k]
            for c in final_chunks:
                c.metadata["crag_action"]  = "fallback"
                c.metadata["crag_attempt"] = attempt + 1
            break

        return final_chunks

    # ------------------------------------------------------------------

    def _evaluate(self, query: str, chunks: list[Chunk]) -> str:
        """
        Two-signal evaluation:
        1. LLM judgment (semantic)
        2. Keyword overlap heuristic (catches LLM false positives on 3B models)
        """
        passages = "\n\n".join(
            f"[{i+1}] {c.content[:200]}" for i, c in enumerate(chunks)
        )
        prompt   = self.EVAL_PROMPT.format(query=query, passages=passages)
        response = self.generator.call_raw(prompt).strip().upper()

        llm_eval = "AMBIGUOUS"  # safe default
        for label in ("CORRECT", "AMBIGUOUS", "INCORRECT"):
            if label in response:
                llm_eval = label
                break

        # Keyword overlap heuristic — catches LLM false positives
        query_terms  = set(query.lower().split()) - self.STOPWORDS
        if query_terms:
            all_content  = " ".join(c.content.lower() for c in chunks)
            overlap_rate = sum(1 for t in query_terms if t in all_content) / len(query_terms)
            # LLM says CORRECT but <30% of meaningful query terms appear → override
            if llm_eval == "CORRECT" and overlap_rate < 0.3:
                return "INCORRECT"

        return llm_eval

    def _is_relevant(self, query: str, chunk: Chunk) -> bool:
        prompt   = self.CHUNK_RELEVANCE_PROMPT.format(
            query   = query,
            passage = chunk.content[:300],
        )
        response = self.generator.call_raw(prompt).strip().upper()
        return response.startswith("YES")

    def _reformulate(self, query: str) -> str:
        prompt   = self.REFORMULATE_PROMPT.format(query=query)
        response = self.generator.call_raw(prompt).strip()
        lines    = [l.strip() for l in response.split("\n") if l.strip()]
        return lines[0] if lines else query