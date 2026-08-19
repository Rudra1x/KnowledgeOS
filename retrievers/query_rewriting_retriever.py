# retrievers/query_rewriting_retriever.py

import os
import json
import requests
from dotenv import load_dotenv
from core import Chunk, Retriever, Embedder, Index

load_dotenv(override=False)


class QueryRewritingRetriever(Retriever):
    """
    Retrieves after rewriting the query with an LLM.

    Two modes:
    - 'reformulate': LLM rephrases the query to match document language
    - 'hyde':        LLM generates a hypothetical document that would
                     answer the query, then embeds that document instead

    Why this helps:
    - Users write informal, abbreviated, spoken-style queries
    - Documents are written in formal, technical, complete sentences
    - Dense embeddings match the *style* of text, not just the meaning
    - Reformulating the query to match document style closes the style gap

    HyDE specifically:
    - Embeds a hypothetical answer document (not the query)
    - A fake document embeds closer to real documents than a query does
    - Particularly effective for domain-specific corpora

    Parameters
    ----------
    embedder   : Embedder
    index      : Index
    mode       : 'reformulate' | 'hyde'
    model      : OpenRouter model string
    max_tokens : int
    """

    NAME = "query_rewriting"

    REFORMULATE_PROMPT = """Rewrite the following search query to be more specific, 
complete, and aligned with how technical documentation is written.
Expand abbreviations, add relevant technical terms, and use formal language.
Return ONLY the rewritten query, nothing else.

Original query: {query}

Rewritten query:"""

    HYDE_PROMPT = """Write a short, factual paragraph (3-5 sentences) that directly 
answers the following question. Write it as if it were extracted from technical 
documentation. Be specific and include key technical terms.

Question: {query}

Hypothetical answer paragraph:"""

    def __init__(
        self,
        embedder:   Embedder,
        index:      Index,
        mode:       str = "reformulate",
        model:      str = "openrouter/free",
        max_tokens: int = 150,
    ):
        if mode not in {"reformulate", "hyde"}:
            raise ValueError(f"mode must be 'reformulate' or 'hyde', got {mode!r}")
        self.embedder   = embedder
        self.index      = index
        self.mode       = mode
        self.model      = model
        self.max_tokens = max_tokens

        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")

    def retrieve(
        self,
        query:     str,
        top_k:     int = 5,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        # Rewrite the query
        rewritten = self._rewrite(query)

        # Embed the rewritten query (or hypothetical document for HyDE)
        qvec = self.embedder.embed_query(rewritten)

        # Retrieve using the rewritten embedding
        results = self.index.search(qvec, top_k=top_k, tenant_id=tenant_id)

        # Tag each result with the rewritten query for transparency
        for chunk in results:
            chunk.metadata["rewritten_query"] = rewritten
            chunk.metadata["original_query"]  = query
            chunk.metadata["rewrite_mode"]     = self.mode

        return results

    def _rewrite(self, query: str) -> str:
        """Call LLM to rewrite or expand the query."""
        if not self.api_key:
            return query  # graceful degradation — return original if no key

        prompt = (self.REFORMULATE_PROMPT if self.mode == "reformulate"
                  else self.HYDE_PROMPT).format(query=query)

        try:
            resp = requests.post(
                url     = "https://openrouter.ai/api/v1/chat/completions",
                headers = {"Authorization": f"Bearer {self.api_key}",
                           "Content-Type": "application/json"},
                data    = json.dumps({
                    "model":       self.model,
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  self.max_tokens,
                    "temperature": 0.0,
                }),
                timeout = 30,
            )
            resp.raise_for_status()
            data    = resp.json()
            content = (data.get("choices", [{}])[0]
                          .get("message", {})
                          .get("content", ""))
            return content.strip() if content else query
        except Exception as e:
            print(f"  [QueryRewriting] rewrite failed: {e} — using original")
            return query