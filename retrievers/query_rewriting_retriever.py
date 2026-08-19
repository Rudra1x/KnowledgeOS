# retrievers/query_rewriting_retriever.py

from core import Chunk, Retriever, Embedder, Index
from generation.local_generator import LocalLLMGenerator


class QueryRewritingRetriever(Retriever):
    """
    Retrieves after rewriting the query with an LLM.

    Two modes:
    - 'reformulate': LLM rephrases query to match document language
    - 'hyde':        LLM generates a hypothetical answer document,
                     then embeds that document instead of the query

    Uses LocalLLMGenerator (Ollama first, OpenRouter fallback).

    Parameters
    ----------
    embedder   : Embedder
    index      : Index
    mode       : 'reformulate' | 'hyde'
    generator  : LocalLLMGenerator | None  (creates default if None)
    """

    NAME = "query_rewriting"

    REFORMULATE_PROMPT = """Rewrite the following search query to be more specific, \
complete, and aligned with how technical documentation is written. \
Expand abbreviations, add relevant technical terms, and use formal language. \
Return ONLY the rewritten query, nothing else.

Original query: {query}

Rewritten query:"""

    HYDE_PROMPT = """Write a short, factual paragraph (3-5 sentences) that directly \
answers the following question. Write it as if it were extracted from technical \
documentation. Be specific and include key technical terms.

Question: {query}

Hypothetical answer paragraph:"""

    def __init__(
        self,
        embedder:  Embedder,
        index:     Index,
        mode:      str                   = "reformulate",
        generator: LocalLLMGenerator | None = None,
    ):
        if mode not in {"reformulate", "hyde"}:
            raise ValueError(f"mode must be 'reformulate' or 'hyde', got {mode!r}")
        self.embedder  = embedder
        self.index     = index
        self.mode      = mode
        self.generator = generator or LocalLLMGenerator(max_tokens=150)

    def retrieve(
        self,
        query:     str,
        top_k:     int = 5,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        rewritten = self._rewrite(query)
        qvec      = self.embedder.embed_query(rewritten)
        results   = self.index.search(qvec, top_k=top_k, tenant_id=tenant_id)

        for chunk in results:
            chunk.metadata["rewritten_query"] = rewritten
            chunk.metadata["original_query"]  = query
            chunk.metadata["rewrite_mode"]     = self.mode

        return results

    def _rewrite(self, query: str) -> str:
        prompt = (self.REFORMULATE_PROMPT if self.mode == "reformulate"
                  else self.HYDE_PROMPT).format(query=query)
        result = self.generator.call_raw(prompt)
        return result.strip() if result else query