# retrievers/multi_hop_retriever.py

from collections import defaultdict
from core import Chunk, Retriever, Embedder, Index
from generation.local_generator import LocalLLMGenerator


class MultiHopRetriever(Retriever):
    """
    Iterative retrieval: each hop's results seed the next query.

    Algorithm per hop:
    1. Retrieve top_k chunks for current query
    2. Summarise retrieved context in a few sentences
    3. Ask LLM: "Given this context, what additional information
       do we need to fully answer the original question?"
    4. If LLM returns a follow-up query → next hop
    5. If LLM says "enough" or empty → stop

    Scoring across hops:
    - Hop 1 chunks scored at weight 1.0 (most relevant to original)
    - Hop 2 chunks scored at weight 0.7
    - Hop N chunks scored at weight 0.7^(N-1)
    - Deduplication by chunk_id, keep highest score

    Parameters
    ----------
    embedder   : Embedder
    index      : Index
    generator  : LocalLLMGenerator
    max_hops   : int   maximum number of retrieval hops (default 3)
    fetch_k    : int   chunks per hop
    hop_decay  : float score weight decay per hop (default 0.7)
    """

    NAME = "multi_hop"

    FOLLOWUP_PROMPT = """You are helping answer this question: "{original_query}"

We retrieved this context so far:
{context}

Based on what we have, what specific additional information do we still need
to fully answer the question?

If we have enough information already, respond with exactly: STOP
If we need more, respond with a single specific search query (one sentence, no explanation):"""

    def __init__(
        self,
        embedder:  Embedder,
        index:     Index,
        generator: LocalLLMGenerator | None = None,
        max_hops:  int   = 3,
        fetch_k:   int   = 3,
        hop_decay: float = 0.7,
    ):
        self.embedder  = embedder
        self.index     = index
        self.generator = generator or LocalLLMGenerator(max_tokens=100)
        self.max_hops  = max_hops
        self.fetch_k   = fetch_k
        self.hop_decay = hop_decay

    def retrieve(
        self,
        query:     str,
        top_k:     int = 5,
        tenant_id: str = "default",
    ) -> list[Chunk]:
        all_chunks: dict[str, tuple[Chunk, float]] = {}  # chunk_id → (chunk, score)
        current_query  = query
        hop_context    = []

        for hop in range(1, self.max_hops + 1):
            weight = self.hop_decay ** (hop - 1)

            # Retrieve for current query
            qvec    = self.embedder.embed_query(current_query)
            results = self.index.search(qvec, top_k=self.fetch_k,
                                        tenant_id=tenant_id)

            # Accumulate — keep highest score per chunk
            for chunk in results:
                score = chunk.metadata.get("score", 0.0) * weight
                cid   = chunk.chunk_id
                if cid not in all_chunks or all_chunks[cid][1] < score:
                    chunk.metadata["hop"]         = hop
                    chunk.metadata["hop_query"]   = current_query
                    chunk.metadata["hop_weight"]  = weight
                    all_chunks[cid] = (chunk, score)

            hop_context.extend(r.content[:200] for r in results)

            # Ask LLM for follow-up query
            if hop < self.max_hops:
                context_text   = "\n---\n".join(hop_context[-3:])  # last 3 snippets
                follow_up      = self._get_followup(query, context_text)

                if not follow_up or follow_up.upper().startswith("STOP"):
                    break
                current_query = follow_up

        # Sort by score descending
        ranked = sorted(all_chunks.values(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk, score in ranked[:top_k]:
            chunk.metadata["score"]      = score
            chunk.metadata["score_type"] = "multi_hop"
            results.append(chunk)

        return results

    def _get_followup(self, original_query: str, context: str) -> str:
        prompt   = self.FOLLOWUP_PROMPT.format(
            original_query = original_query,
            context        = context,
        )
        response = self.generator.call_raw(prompt)
        if not response:
            return "STOP"
        # Strip any preamble the model might add
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return lines[0] if lines else "STOP"