# generation/context_compressor.py

import re
import time
import numpy as np
from core import Chunk, Embedder
from generation.local_generator import LocalLLMGenerator


class ContextCompressor:
    """
    Compress retrieved chunks before sending to the generator.

    Reduces token cost and model distraction by extracting only
    the sentences from each chunk that are relevant to the query.

    Three strategies:
    - 'similarity': sentence-level cosine similarity (fast, no LLM)
    - 'llm':        LLM extracts relevant sentences (accurate, slower)
    - 'budget':     keep sentences until token_budget exhausted (cost-controlled)

    Parameters
    ----------
    strategy      : str     'similarity' | 'llm' | 'budget'
    embedder      : Embedder  for similarity strategy
    generator     : LocalLLMGenerator  for llm strategy
    top_sentences : int     max sentences to keep per chunk (similarity)
    token_budget  : int     max tokens per chunk (budget)
    min_sentences : int     always keep at least this many sentences
    """

    EXTRACT_PROMPT = """From the passage below, extract ONLY the sentences \
that are directly relevant to answering the question.
Copy the relevant sentences verbatim. If no sentences are relevant, \
write: [NO RELEVANT SENTENCES]

Question: {query}

Passage:
{passage}

Relevant sentences:"""

    def __init__(
        self,
        strategy:      str                       = "similarity",
        embedder:      Embedder | None           = None,
        generator:     LocalLLMGenerator | None  = None,
        top_sentences: int                       = 3,
        token_budget:  int                       = 150,
        min_sentences: int                       = 1,
    ):
        if strategy not in {"similarity", "llm", "budget"}:
            raise ValueError(f"strategy must be 'similarity', 'llm', or 'budget'")
        if strategy == "similarity" and embedder is None:
            raise ValueError("similarity strategy requires an embedder")
        if strategy == "llm" and generator is None:
            raise ValueError("llm strategy requires a generator")

        self.strategy      = strategy
        self.embedder      = embedder
        self.generator     = generator
        self.top_sentences = top_sentences
        self.token_budget  = token_budget
        self.min_sentences = min_sentences
        self.last_compress_ms: float = 0.0

    def compress(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Compress chunks in-place (modifies .content to compressed version).
        Returns the same list with compressed content.
        Attaches metadata: original_length, compressed_length, compression_ratio.
        """
        if not chunks:
            return chunks

        t0 = time.perf_counter()

        for chunk in chunks:
            original = chunk.content
            if self.strategy == "similarity":
                compressed = self._similarity_compress(query, original)
            elif self.strategy == "llm":
                compressed = self._llm_compress(query, original)
            else:
                compressed = self._budget_compress(original)

            if not compressed.strip():
                compressed = original   # fallback: keep original if empty

            orig_len = len(original.split())
            comp_len = len(compressed.split())

            chunk.metadata["original_content"]    = original
            chunk.metadata["compressed_content"]  = compressed
            chunk.metadata["original_length"]     = orig_len
            chunk.metadata["compressed_length"]   = comp_len
            chunk.metadata["compression_ratio"]   = round(comp_len / max(orig_len, 1), 3)
            chunk.content = compressed

        self.last_compress_ms = (time.perf_counter() - t0) * 1000
        return chunks

    # ------------------------------------------------------------------

    def _similarity_compress(self, query: str, text: str) -> str:
        """Score each sentence by cosine similarity to query, keep top_k."""
        sentences = self._split_sentences(text)
        if len(sentences) <= self.min_sentences:
            return text

        query_vec = np.array(self.embedder.embed_query(query), dtype="float32")
        sent_vecs = np.array(
            self.embedder.embed([s for s in sentences]),
            dtype="float32"
        )

        # Cosine similarity (vectors may already be normalized)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        scores     = sent_vecs @ query_norm

        # Keep top_sentences by score, preserve original order
        n_keep  = max(self.min_sentences, min(self.top_sentences, len(sentences)))
        top_idx = set(np.argsort(scores)[::-1][:n_keep])
        kept    = [s for i, s in enumerate(sentences) if i in top_idx]
        return " ".join(kept)

    def _llm_compress(self, query: str, text: str) -> str:
        """Ask LLM to extract only relevant sentences."""
        prompt   = self.EXTRACT_PROMPT.format(
            query   = query,
            passage = text[:800],   # truncate very long chunks
        )
        response = self.generator.call_raw(prompt).strip()
        if "[NO RELEVANT SENTENCES]" in response or not response:
            return ""
        return response

    def _budget_compress(self, text: str) -> str:
        """Keep sentences until token_budget (approx word count) exhausted."""
        sentences = self._split_sentences(text)
        kept      = []
        count     = 0
        for sent in sentences:
            words = len(sent.split())
            if count + words > self.token_budget and len(kept) >= self.min_sentences:
                break
            kept.append(sent)
            count += words
        return " ".join(kept)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences, filter empty."""
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sents if len(s.strip()) > 10]