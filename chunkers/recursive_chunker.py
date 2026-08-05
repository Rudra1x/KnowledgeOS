# chunkers/recursive_chunker.py

import uuid
from core import Chunk, Chunker, Document


class RecursiveChunker(Chunker):
    """
    Splits text on natural boundaries in priority order, then greedily merges
    small pieces to approach chunk_size.

    Separator priority (default):
    1. '\\n\\n' — paragraph
    2. '\\n'   — line
    3. '. '    — sentence (period + space)
    4. ' '     — word
    5. ''      — character-level fallback

    Algorithm:
    - Try highest-priority separator first
    - Any resulting piece still > chunk_size is recursively split with next separator
    - After all splits, greedily merge adjacent pieces until each hits chunk_size
    - Add overlap between adjacent final chunks

    Trade-offs:
    - Respects author's document structure (paragraphs, sentences)
    - Slower than OverlappingChunker (multiple passes) but still O(N)
    - Overlap logic is coarser than OverlappingChunker's exact-character shift
    """

    NAME = "recursive"

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size:    int = 512,
        chunk_overlap: int = 50,
        separators:    list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators    = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._recursive_split(document.content, self.separators)
        merged = self._merge_with_overlap(pieces)

        chunks = []
        cursor = 0
        for i, piece in enumerate(merged):
            # Locate the piece in the original text for start/end offsets.
            # find() from `cursor` handles the common case; overlap chunks
            # legitimately re-scan an earlier region.
            start = document.content.find(piece, max(0, cursor - self.chunk_overlap))
            if start < 0:
                start = cursor
            end = start + len(piece)
            cursor = end - self.chunk_overlap   # allow next find() to backtrack for overlap

            chunks.append(Chunk(
                chunk_id    = str(uuid.uuid4()),
                doc_id      = document.doc_id,
                content     = piece,
                tenant_id   = document.tenant_id,
                start_index = start,
                end_index   = end,
                metadata    = {
                    **document.metadata,
                    "chunker_name":     self.NAME,
                    "chunk_index":      i,
                    "chunk_size_chars": len(piece),
                },
            ))

        return chunks

    # ------------------------------------------------------------------
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Split text hierarchically until every piece is <= chunk_size."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # No separators left — hard-slice at chunk_size boundaries
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        sep, *rest = separators

        if sep == "":
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        parts = text.split(sep)
        # Reattach the separator to each part except the last, so splitting
        # is round-trippable (joining parts reconstructs the original text).
        parts = [p + sep for p in parts[:-1]] + [parts[-1]]

        result = []
        for part in parts:
            if len(part) <= self.chunk_size:
                if part.strip():
                    result.append(part)
            else:
                result.extend(self._recursive_split(part, rest))

        return result

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """
        Greedily concatenate small pieces up to chunk_size, then start a new
        chunk that begins with a chunk_overlap-sized tail of the previous chunk.
        """
        if not pieces:
            return []

        merged  = []
        current = ""

        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                merged.append(current)
                # Start new chunk with overlap from the tail of the previous one
                tail    = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current = tail + piece

        if current.strip():
            merged.append(current)

        return merged