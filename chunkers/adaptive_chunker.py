# chunkers/adaptive_chunker.py

import re
import uuid
from core import Chunk, Chunker, Document


class AdaptiveChunker(Chunker):
    """
    Variable-size chunking driven by local content density.

    Dense paragraphs (short words, many sentences, many special chars) get
    smaller target chunk sizes. Sparse prose gets larger.

    Density signals:
    - avg_word_length  : short words → denser (code, abbreviations)
    - sentence_density : sentences-per-100-words (more sentences → denser)
    - special_char_ratio: colon/equals/bracket density (code / formulas / bullets)

    Parameters
    ----------
    min_chunk_size : int   smallest allowed chunk (for very dense content)
    max_chunk_size : int   largest allowed chunk (for very sparse content)
    base_chunk_size: int   starting point; density score scales around this
    overlap_chars  : int   chars of overlap between adjacent chunks
    """

    NAME = "adaptive"

    SEPARATORS    = ["\n\n", "\n", ". ", " ", ""]
    SPECIAL_CHARS = set(":=()[]{}<>/\\|@#$%^&*")

    def __init__(
        self,
        min_chunk_size:  int = 200,
        max_chunk_size:  int = 1000,
        base_chunk_size: int = 512,
        overlap_chars:   int = 50,
    ):
        if min_chunk_size >= max_chunk_size:
            raise ValueError(f"min_chunk_size ({min_chunk_size}) must be < max_chunk_size ({max_chunk_size})")
        if not (min_chunk_size <= base_chunk_size <= max_chunk_size):
            raise ValueError(f"base_chunk_size ({base_chunk_size}) must be between "
                             f"min ({min_chunk_size}) and max ({max_chunk_size})")
        self.min_chunk_size  = min_chunk_size
        self.max_chunk_size  = max_chunk_size
        self.base_chunk_size = base_chunk_size
        self.overlap_chars   = overlap_chars

    def chunk(self, document: Document) -> list[Chunk]:
        # Work paragraph-by-paragraph — density is a local property
        paragraphs = [p.strip() for p in document.content.split("\n\n") if p.strip()]

        pieces = []
        for para in paragraphs:
            target  = self._target_size(para)
            splits  = self._recursive_split(para, target, self.SEPARATORS)
            merged  = self._merge_toward(splits, target)
            pieces.extend(merged)

        # Apply overlap between adjacent final pieces
        pieces = self._apply_overlap(pieces, self.overlap_chars)

        return self._to_chunks(pieces, document)

    # ------------------------------------------------------------------
    # Density scoring
    # ------------------------------------------------------------------

    def _target_size(self, text: str) -> int:
        """
        Compute a density score → interpolate between min and max chunk size.
        Dense text (score → 1.0) → min_chunk_size
        Sparse text (score → 0.0) → max_chunk_size
        """
        density = self._density_score(text)
        # Linear interpolation: high density → small chunks
        target  = int(self.max_chunk_size - density * (self.max_chunk_size - self.min_chunk_size))
        return max(self.min_chunk_size, min(self.max_chunk_size, target))

    def _density_score(self, text: str) -> float:
        """Returns a score in [0.0, 1.0] where 1.0 is maximally dense."""
        words    = text.split()
        if not words:
            return 0.5

        # Signal 1: average word length (shorter = denser)
        avg_word_len       = sum(len(w) for w in words) / len(words)
        # normalize: 2 = very short (dense), 10 = very long (sparse)
        word_len_score     = max(0.0, min(1.0, (10 - avg_word_len) / 8))

        # Signal 2: sentence density (sentences per 100 words)
        sentence_count     = max(1, len(re.findall(r"[.!?]", text)))
        sent_density       = (sentence_count / max(1, len(words))) * 100
        # normalize: 10+ sentences/100 words = max dense, 1 = sparse
        sent_density_score = max(0.0, min(1.0, sent_density / 10))

        # Signal 3: special character ratio
        special_count      = sum(1 for ch in text if ch in self.SPECIAL_CHARS)
        special_ratio      = special_count / max(1, len(text))
        # normalize: 0.15+ = dense (code/formulas), 0 = sparse
        special_score      = max(0.0, min(1.0, special_ratio / 0.15))

        # Weighted average (word length is the primary signal)
        return 0.5 * word_len_score + 0.3 * sent_density_score + 0.2 * special_score

    # ------------------------------------------------------------------
    # Split / merge / overlap (same mechanics as RecursiveChunker)
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

    def _to_chunks(self, pieces: list[str], document: Document) -> list[Chunk]:
        chunks, cursor = [], 0
        for i, piece in enumerate(pieces):
            start = document.content.find(piece, max(0, cursor - self.overlap_chars))
            if start < 0:
                start = cursor
            end    = start + len(piece)
            cursor = end - self.overlap_chars

            density = self._density_score(piece)
            target  = self._target_size(piece)

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
                    "density_score":    round(density, 3),
                    "target_size":      target,
                },
            ))
        return chunks