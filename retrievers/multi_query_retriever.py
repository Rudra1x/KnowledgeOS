# retrievers/multi_query_retriever.py

from collections import defaultdict
from core import Chunk, Retriever, Embedder, Index
from generation.local_generator import LocalLLMGenerator


class MultiQueryRetriever(Retriever):
    """
    Generates K query variants using an LLM, retrieves for each,
    unions the results, and re-ranks by aggregated score.

    Algorithm:
    1. Generate n_variants phrasings of the original query (LLM)
    2. Retrieve top_k candidates for each variant
    3. Union all candidates (deduplicate by chunk_id)
    4. Score each chunk: sum of retrieval scores across all variants
       (chunks retrieved by multiple variants score higher)
    5. Return final top_k by aggregated score

    Why this works:
    - Different phrasings embed in different regions of the vector space
    - Each region may be adjacent to different relevant chunks
    - The union covers more of the relevant space than any single query

    Parameters
    ----------
    embedder    : Embedder
    index       : Index
    generator   : LocalLLMGenerator
    n_variants  : int   number of query variants to generate (default 3)
    fetch_k     : int   candidates per variant before union
    """

    NAME = "multi_query"

    VARIANT_PROMPT = """Generate {n} different search queries that ask about the same \
topic as the original query but use different wording, perspective, or terminology. \
Each query should be a complete sentence and retrievable from technical documentation.

Original query: {query}

Return ONLY the {n} queries, one per line, numbered 1. 2. 3. etc. No explanations."""

    def __init__(
        self,
        embedder:   Embedder,
        index:      Index,
        generator:  LocalLLMGenerator | None = None,
        n_variants: int = 3,
        fetch_k:    int = 5,
    ):
        self.embedder   = embedder
        self.index      = index
        self.generator  = generator or LocalLLMGenerator(max_tokens=200)
        self.n_variants = n_variants
        self.fetch_k    = fetch_k

    def retrieve(
        self,
        query:     str,
        top_k:     int = 5,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        # Step 1: Generate variants
        variants = self._generate_variants(query)
        all_queries = [query] + variants   # always include original

        # Step 2: Retrieve for each variant
        scores:    dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Chunk] = {}

        for q in all_queries:
            qvec     = self.embedder.embed_query(q)
            results  = self.index.search(qvec, top_k=self.fetch_k,
                                         tenant_id=tenant_id)
            for rank, chunk in enumerate(results, start=1):
                cid = chunk.chunk_id
                # Accumulate scores — reciprocal rank within each variant
                scores[cid]    += 1.0 / rank
                chunk_map[cid]  = chunk

        # Step 3: Re-rank by aggregated score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_id, agg_score in ranked[:top_k]:
            chunk = chunk_map[chunk_id]
            chunk.metadata["score"]          = agg_score
            chunk.metadata["score_type"]     = "multi_query_rr"
            chunk.metadata["query_variants"] = all_queries
            results.append(chunk)

        return results

    def _generate_variants(self, query: str) -> list[str]:
        """Ask the LLM for n_variants alternative phrasings."""
        prompt = self.VARIANT_PROMPT.format(
            n=self.n_variants,
            query=query,
        )
        raw = self.generator.call_raw(prompt)
        if not raw:
            return []

        # Parse numbered list: "1. ..." or "1) ..."
        lines   = raw.strip().split("\n")
        variants = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip leading "1." or "1)" or "- "
            import re
            cleaned = re.sub(r"^[\d]+[.)]\s*", "", line).strip()
            cleaned = re.sub(r"^[-*]\s*", "", cleaned).strip()
            if cleaned and len(cleaned) > 10:
                variants.append(cleaned)

        return variants[:self.n_variants]