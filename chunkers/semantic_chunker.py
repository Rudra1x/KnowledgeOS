# chunkers/semantic_chunker.py

import re
import uuid
import numpy as np
from core import Chunk, Chunker, Document, Embedder


class SemanticChunker(Chunker):
    """
    Content-aware chunker: splits where adjacent sentences are semantically dissimilar.

    Algorithm:
    1. Split document into sentences
    2. Embed each sentence
    3. Compute cosine similarity between every adjacent sentence pair
    4. Any pair whose similarity is below the (100 - breakpoint_percentile)th
       percentile is treated as a topic-shift breakpoint
    5. Group sentences between breakpoints into chunks
    6. Post-process to enforce min/max size (subdivide huge chunks, merge tiny ones)

    Parameters
    ----------
    embedder : Embedder
        Same embedder as the pipeline (BGE, etc.)
    breakpoint_percentile : float
        95 = split at the 5% most-different transitions. Higher = fewer splits.
    min_chunk_size : int
        Chunks smaller than this get merged with a neighbor.
    max_chunk_size : int
        Chunks larger than this get subdivided.

    Trade-offs vs Recursive:
    - Better semantic coherence (topic-aware)
    - Slower — extra embedding pass at ingestion
    - Non-deterministic if the embedder is (BGE is deterministic; API embedders may not be)
    """

    NAME = "semantic"

    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self,
        embedder:              Embedder,
        breakpoint_percentile: float = 95.0,
        min_chunk_size:        int   = 100,
        max_chunk_size:        int   = 1000,
    ):
        if not (0 < breakpoint_percentile < 100):
            raise ValueError(f"breakpoint_percentile must be in (0, 100), got {breakpoint_percentile}")
        if min_chunk_size >= max_chunk_size:
            raise ValueError(f"min_chunk_size ({min_chunk_size}) must be < max_chunk_size ({max_chunk_size})")
        self.embedder              = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chunk_size        = min_chunk_size
        self.max_chunk_size        = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)

        # Edge case: <=1 sentence → return whole doc as one chunk
        if len(sentences) <= 1:
            return self._to_chunks([document.content.strip()], document) if document.content.strip() else []

        # Embed all sentences at once (batched)
        vectors     = np.array(self.embedder.embed(sentences), dtype="float32")

        # Cosine similarity between adjacent sentences (vectors are already normalized by BGE)
        sims        = (vectors[:-1] * vectors[1:]).sum(axis=1)   # dot product = cosine on normalized

        # Percentile-based threshold: the lowest (100 - percentile)% of similarities are breakpoints
        threshold   = float(np.percentile(sims, 100 - self.breakpoint_percentile))

        # Sentences after breakpoints start new groups
        breakpoints = [i + 1 for i, s in enumerate(sims) if s < threshold]

        # Build initial groups
        groups      = self._group_sentences(sentences, breakpoints)

        # Enforce size constraints
        groups      = self._enforce_size(groups)

        return self._to_chunks(groups, document)

    # ------------------------------------------------------------------
    def _split_sentences(self, text: str) -> list[str]:
        """Regex-based sentence splitter. Not perfect (abbreviations trick it),
        but adequate for chunking. For higher quality, plug in spaCy or nltk."""
        sentences = self.SENTENCE_SPLIT_RE.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _group_sentences(sentences: list[str], breakpoints: list[int]) -> list[str]:
        """Group sentences into chunks at breakpoint indices."""
        groups     = []
        prev       = 0
        for bp in breakpoints:
            groups.append(" ".join(sentences[prev:bp]))
            prev = bp
        groups.append(" ".join(sentences[prev:]))
        return [g for g in groups if g]

    def _enforce_size(self, groups: list[str]) -> list[str]:
        """
        Subdivide any group exceeding max_chunk_size (character-level fallback).
        Merge any group below min_chunk_size into a neighbor.
        """
        # 1. Subdivide oversized groups
        subdivided = []
        for g in groups:
            if len(g) <= self.max_chunk_size:
                subdivided.append(g)
                continue
            # Fallback: character slicing at max_chunk_size
            for i in range(0, len(g), self.max_chunk_size):
                subdivided.append(g[i:i + self.max_chunk_size])

        # 2. Merge undersized groups with the next one (or previous if last)
        merged = []
        i = 0
        while i < len(subdivided):
            current = subdivided[i]
            while i + 1 < len(subdivided) and len(current) < self.min_chunk_size:
                current = current + " " + subdivided[i + 1]
                i += 1
            merged.append(current)
            i += 1

        return merged

    def _to_chunks(self, groups: list[str], document: Document) -> list[Chunk]:
        chunks = []
        cursor = 0
        for i, g in enumerate(groups):
            start = document.content.find(g, cursor)
            if start < 0:
                # Sentence join may not exactly match original (whitespace normalization);
                # fall back to cursor position.
                start = cursor
            end    = start + len(g)
            cursor = end

            chunks.append(Chunk(
                chunk_id    = str(uuid.uuid4()),
                doc_id      = document.doc_id,
                content     = g,
                tenant_id   = document.tenant_id,
                start_index = start,
                end_index   = end,
                metadata    = {
                    **document.metadata,
                    "chunker_name":     self.NAME,
                    "chunk_index":      i,
                    "chunk_size_chars": len(g),
                },
            ))
        return chunks