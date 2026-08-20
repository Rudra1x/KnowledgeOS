# retrievers/self_rag_retriever.py

from core import Chunk, Retriever, Embedder, Index
from generation.local_generator import LocalLLMGenerator


class SelfRAGRetriever(Retriever):
    """
    Self-RAG: adaptive retrieval with LLM-gated decisions.

    Two decision points:
    1. RETRIEVE GATE: should we retrieve at all for this query?
       - Parametric questions (facts the LLM knows) → skip retrieval
       - Knowledge-intensive questions → retrieve
    2. RELEVANCE FILTER: for each retrieved chunk, is it relevant?
       - Irrelevant chunks are dropped before returning
       - Only genuinely relevant chunks reach the generator

    Why this helps:
    - Avoids irrelevant context poisoning the generator
    - Saves retrieval cost for queries that don't need it
    - Improves faithfulness by filtering noise before generation

    Parameters
    ----------
    embedder          : Embedder
    index             : Index
    generator         : LocalLLMGenerator
    retrieve_threshold: float   confidence threshold for retrieve decision
    relevance_threshold: float  confidence threshold for relevance decision
    always_retrieve   : bool    skip the retrieve gate (always retrieve)
    """

    NAME = "self_rag"

    RETRIEVE_GATE_PROMPT = """Does answering the following question require \
looking up specific facts from an external knowledge base, or can it be \
answered from general knowledge?

Question: {query}

Answer with ONLY one word: YES (needs retrieval) or NO (general knowledge):"""

    RELEVANCE_PROMPT = """Is the following passage relevant and useful for \
answering this question?

Question: {query}

Passage: {passage}

Answer with ONLY one word: YES (relevant) or NO (not relevant):"""

    def __init__(
        self,
        embedder:            Embedder,
        index:               Index,
        generator:           LocalLLMGenerator | None = None,
        always_retrieve:     bool  = False,
        min_relevant_chunks: int   = 1,
    ):
        self.embedder            = embedder
        self.index               = index
        self.generator           = generator or LocalLLMGenerator(max_tokens=10)
        self.always_retrieve     = always_retrieve
        self.min_relevant_chunks = min_relevant_chunks

    def retrieve(
        self,
        query:     str,
        top_k:     int = 5,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        # Step 1: Should we retrieve?
        if not self.always_retrieve:
            needs_retrieval = self._should_retrieve(query)
            if not needs_retrieval:
                # Return empty — caller should use parametric generation
                return []

        # Step 2: Retrieve candidates
        qvec       = self.embedder.embed_query(query)
        candidates = self.index.search(
            qvec, top_k=top_k * 2, tenant_id=tenant_id
        )

        # Step 3: Filter by relevance
        relevant = []
        for chunk in candidates:
            if self._is_relevant(query, chunk):
                chunk.metadata["self_rag_relevant"] = True
                relevant.append(chunk)
            else:
                chunk.metadata["self_rag_relevant"] = False

            if len(relevant) >= top_k:
                break

        # If relevance filter removed everything, fall back to top candidates
        if len(relevant) < self.min_relevant_chunks and candidates:
            relevant = candidates[:self.min_relevant_chunks]
            for c in relevant:
                c.metadata["self_rag_fallback"] = True

        return relevant[:top_k]

    # ------------------------------------------------------------------

    def _should_retrieve(self, query: str) -> bool:
        """Gate: does this query need external retrieval?"""
        prompt   = self.RETRIEVE_GATE_PROMPT.format(query=query)
        response = self.generator.call_raw(prompt).strip().upper()
        # Accept YES variants: "YES", "Yes.", "YES."
        return response.startswith("YES")

    def _is_relevant(self, query: str, chunk: Chunk) -> bool:
        """Filter: is this chunk relevant to the query?"""
        prompt   = self.RELEVANCE_PROMPT.format(
            query   = query,
            passage = chunk.content[:400],  # truncate for prompt budget
        )
        response = self.generator.call_raw(prompt).strip().upper()
        return response.startswith("YES")