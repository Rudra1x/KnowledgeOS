# chunkers/fixed_chunker.py

import uuid
from core import Chunk, Chunker, Document


class FixedChunker(Chunker):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
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
                metadata    = {**document.metadata, "chunk_index": i},
            ))

        return chunks