# chunkers/metadata_aware_chunker.py

import uuid
from core import Chunk, Chunker, Document


class MetadataAwareChunker(Chunker):
    """
    Structure-respecting chunker. Uses Document metadata set by loaders to:

    1. Treat 'table' content_type chunks as atomic (never split, never merge
       with adjacent text chunks).
    2. Scale chunk size by heading level (H1 → larger; H3 → smaller).
    3. Treat 'ocr_needed' and 'attachment_needs_processing' as pass-through
       (emit as-is, no splitting).
    4. Fall back to recursive splitting for plain text with no metadata.

    Metadata used:
    - content_type     : 'text' | 'table' | 'ocr_needed' | 'email' | ...
    - heading_level    : int 0-9 (from DOCX/MD loaders)
    - section_title    : str (from DOCX/MD loaders)

    Parameters
    ----------
    base_chunk_size : int   target size for H2 (mid-level) text sections
    overlap_chars   : int   overlap between adjacent text chunks
    heading_size_map: dict  maps heading_level → chunk_size override
    """

    NAME = "metadata_aware"

    SEPARATORS         = ["\n\n", "\n", ". ", " ", ""]
    PASSTHROUGH_TYPES  = {"ocr_needed", "attachment_needs_processing"}
    ATOMIC_TYPES       = {"table"}

    def __init__(
        self,
        base_chunk_size:  int        = 600,
        overlap_chars:    int        = 60,
        heading_size_map: dict | None = None,
    ):
        self.base_chunk_size  = base_chunk_size
        self.overlap_chars    = overlap_chars
        # Default: H1 gets more context, H3 gets less
        self.heading_size_map = heading_size_map or {
            0: base_chunk_size * 2,   # preamble / no heading
            1: base_chunk_size * 2,   # H1 section — widest context
            2: base_chunk_size,       # H2 section — base
            3: base_chunk_size // 2,  # H3 subsection — tighter
            4: base_chunk_size // 2,
            5: base_chunk_size // 2,
        }

    def chunk(self, document: Document) -> list[Chunk]:
        content_type  = document.metadata.get("content_type", "text")
        heading_level = int(document.metadata.get("heading_level", 2))

        # --- Pass-through: emit as-is without splitting ---
        if content_type in self.PASSTHROUGH_TYPES:
            return self._single_chunk(document, reason="passthrough")

        # --- Atomic: table → one chunk, always ---
        if content_type in self.ATOMIC_TYPES:
            return self._single_chunk(document, reason="atomic_table")

        # --- Text: split with heading-level-appropriate size ---
        target_size = self._resolve_target(heading_level)
        pieces      = self._recursive_split(document.content, target_size, self.SEPARATORS)
        pieces      = self._merge_toward(pieces, target_size)
        pieces      = self._apply_overlap(pieces, self.overlap_chars)

        return self._to_chunks(pieces, document, heading_level)

    # ------------------------------------------------------------------
    def _resolve_target(self, heading_level: int) -> int:
        return self.heading_size_map.get(
            heading_level,
            self.base_chunk_size      # fallback for unmapped levels
        )

    def _single_chunk(self, document: Document, reason: str) -> list[Chunk]:
        if not document.content.strip():
            return []
        return [Chunk(
            chunk_id    = str(uuid.uuid4()),
            doc_id      = document.doc_id,
            content     = document.content.strip(),
            tenant_id   = document.tenant_id,
            start_index = 0,
            end_index   = len(document.content),
            metadata    = {
                **document.metadata,
                "chunker_name":     self.NAME,
                "chunk_index":      0,
                "chunk_size_chars": len(document.content.strip()),
                "split_reason":     reason,
            },
        )]

    def _to_chunks(self, pieces: list[str], document: Document, heading_level: int) -> list[Chunk]:
        chunks, cursor = [], 0
        for i, piece in enumerate(pieces):
            start = document.content.find(piece, max(0, cursor - self.overlap_chars))
            if start < 0:
                start = cursor
            end    = start + len(piece)
            cursor = end - self.overlap_chars

            chunks.append(Chunk(
                chunk_id    = str(uuid.uuid4()),
                doc_id      = document.doc_id,
                content     = piece,
                tenant_id   = document.tenant_id,
                start_index = start,
                end_index   = end,
                metadata    = {
                    **document.metadata,
                    "chunker_name":      self.NAME,
                    "chunk_index":       i,
                    "chunk_size_chars":  len(piece),
                    "heading_level_used": heading_level,
                    "target_size_used":  self._resolve_target(heading_level),
                },
            ))
        return chunks

    # ------------------------------------------------------------------
    # Shared split / merge / overlap (same as Recursive / Adaptive)
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, target: int, separators: list[str]) -> list[str]:
        if len(text) <= target:
            return [text] if text.strip() else []
        if not separators:
            return [text[i:i + target] for i in range(0, len(text), target)]
        sep, *rest = separators
        if sep == "":
            return [text[i:i + target] for i in range(0, len(text), target)]
        parts  = text.split(sep)
        parts  = [p + sep for p in parts[:-1]] + [parts[-1]]
        result = []
        for part in parts:
            if len(part) <= target:
                if part.strip():
                    result.append(part)
            else:
                result.extend(self._recursive_split(part, target, rest))
        return result

    @staticmethod
    def _merge_toward(pieces: list[str], target: int) -> list[str]:
        if not pieces:
            return []
        merged, current = [], ""
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + len(piece) <= target:
                current += piece
            else:
                merged.append(current)
                current = piece
        if current.strip():
            merged.append(current)
        return merged

    @staticmethod
    def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
        if overlap <= 0 or len(pieces) < 2:
            return pieces
        result = [pieces[0]]
        for i in range(1, len(pieces)):
            tail = pieces[i - 1][-overlap:]
            result.append(tail + pieces[i])
        return result