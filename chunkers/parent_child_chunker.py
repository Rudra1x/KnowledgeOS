# chunkers/parent_child_chunker.py

import uuid
from core import Chunk, Chunker, Document


class ParentChildChunker(Chunker):
    """
    Two-level chunking: large parent chunks split into small child chunks.

    Emits CHILDREN (small) tagged with parent_id and a serialized parent_content
    field in metadata. This gives retrieval-strategy code (M5) everything it
    needs to substitute parent context for generation.

    Algorithm:
    1. First pass: split the document into 'parent' chunks (recursive splitting
       on \\n\\n → \\n → sentence, targeting parent_size).
    2. Second pass: split each parent into 'child' chunks (recursive again,
       targeting child_size, with child_overlap).
    3. Emit children, each carrying parent_id and parent_content in metadata.

    Parameters
    ----------
    parent_size : int
        Target size of parent chunks (large — for context).
    child_size : int
        Target size of child chunks (small — for retrieval).
    child_overlap : int
        Overlap between children within the same parent.

    Metadata added per child chunk:
    - chunker_name    : 'parent_child'
    - chunk_index     : child position across whole document
    - chunk_size_chars: length of this child
    - parent_id       : uuid of the parent chunk it belongs to
    - parent_content  : full content of the parent (used by retrieval at query time)
    - parent_index    : parent's position in the document
    - child_index_in_parent : this child's position within its parent
    """

    NAME = "parent_child"

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        parent_size:   int = 1500,
        child_size:    int = 300,
        child_overlap: int = 40,
    ):
        if child_size >= parent_size:
            raise ValueError(f"child_size ({child_size}) must be < parent_size ({parent_size})")
        if child_overlap >= child_size:
            raise ValueError(f"child_overlap ({child_overlap}) must be < child_size ({child_size})")
        self.parent_size   = parent_size
        self.child_size    = child_size
        self.child_overlap = child_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        # --- Step 1: split into parents ---
        parents = self._recursive_split(document.content, self.parent_size, self.SEPARATORS)
        parents = self._merge_toward(parents, self.parent_size)

        all_children  = []
        global_child_idx = 0

        for parent_idx, parent_text in enumerate(parents):
            parent_id = str(uuid.uuid4())

            # --- Step 2: split parent into children ---
            children = self._recursive_split(parent_text, self.child_size, self.SEPARATORS)
            children = self._merge_toward(children, self.child_size)
            children = self._apply_overlap(children, self.child_overlap)

            for child_idx, child_text in enumerate(children):
                # locate the child inside the original document for offsets
                start = document.content.find(child_text)
                if start < 0:
                    start = 0
                end = start + len(child_text)

                all_children.append(Chunk(
                    chunk_id    = str(uuid.uuid4()),
                    doc_id      = document.doc_id,
                    content     = child_text,
                    tenant_id   = document.tenant_id,
                    start_index = start,
                    end_index   = end,
                    metadata    = {
                        **document.metadata,
                        "chunker_name":          self.NAME,
                        "chunk_index":           global_child_idx,
                        "chunk_size_chars":      len(child_text),
                        "parent_id":             parent_id,
                        "parent_content":        parent_text,
                        "parent_index":          parent_idx,
                        "child_index_in_parent": child_idx,
                    },
                ))
                global_child_idx += 1

        return all_children

    # ------------------------------------------------------------------
    def _recursive_split(self, text: str, target: int, separators: list[str]) -> list[str]:
        """Same recursive split logic as RecursiveChunker but parameterized by target."""
        if len(text) <= target:
            return [text] if text.strip() else []

        if not separators:
            return [text[i:i + target] for i in range(0, len(text), target)]

        sep, *rest = separators
        if sep == "":
            return [text[i:i + target] for i in range(0, len(text), target)]

        parts = text.split(sep)
        parts = [p + sep for p in parts[:-1]] + [parts[-1]]

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
        """Greedy pack up to target."""
        if not pieces:
            return []
        merged  = []
        current = ""
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
        """Prepend the tail of piece N to piece N+1."""
        if overlap <= 0 or len(pieces) < 2:
            return pieces
        overlapped = [pieces[0]]
        for i in range(1, len(pieces)):
            tail = pieces[i - 1][-overlap:]
            overlapped.append(tail + pieces[i])
        return overlapped