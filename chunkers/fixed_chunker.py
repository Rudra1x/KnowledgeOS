# chunkers/fixed_chunker.py

import uuid
from core import Chunk, Chunker, Document


class OverlappingChunker(Chunker):
    """
    Fixed-size character chunking with configurable overlap.

    Deterministic, fast, meaning-agnostic. This is the baseline every other
    chunker gets measured against.

    Parameters
    ----------
    chunk_size : int
        Target chunk length in characters.
    chunk_overlap : int
        Characters shared between adjacent chunks (should be < chunk_size).

    Metadata added per chunk:
    - chunker_name    : 'overlapping'
    - chunk_index     : position in the document
    - chunk_size_chars: actual size (may be < chunk_size for the last chunk)
    """

    NAME = "overlapping"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            )
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text   = document.content
        step   = self.chunk_size - self.chunk_overlap
        chunks = []

        for i, start in enumerate(range(0, len(text), step)):
            end     = start + self.chunk_size
            snippet = text[start:end]

            if not snippet.strip():
                continue

            chunks.append(Chunk(
                chunk_id    = str(uuid.uuid4()),
                doc_id      = document.doc_id,
                content     = snippet,
                tenant_id   = document.tenant_id,
                start_index = start,
                end_index   = min(end, len(text)),
                metadata    = {
                    **document.metadata,
                    "chunker_name":     self.NAME,
                    "chunk_index":      i,
                    "chunk_size_chars": len(snippet),
                },
            ))

            # Stop once we've covered the whole document
            if end >= len(text):
                break

        return chunks


# Backward-compatible alias so existing scripts keep working
FixedChunker = OverlappingChunker