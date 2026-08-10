# scripts/test_metadata_aware.py

import sys, uuid
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.md_loader           import MarkdownLoader
from loaders.docx_loader         import DOCXLoader
from chunkers.metadata_aware_chunker import MetadataAwareChunker
from core.models                 import Document


chunker = MetadataAwareChunker(base_chunk_size=400, overlap_chars=40)

# --- Test 1: Markdown (heading levels drive chunk size) ---
print("=" * 60)
print("MARKDOWN (heading-level aware)")
print("=" * 60)
md_docs = MarkdownLoader().load("scripts/sample.md")
print(f"Loaded {len(md_docs)} sections from MD\n")

for doc in md_docs:
    chunks = chunker.chunk(doc)
    lvl    = doc.metadata.get("heading_level", "?")
    title  = doc.metadata.get("section_title", "?")
    target = chunker._resolve_target(int(lvl))
    print(f"[H{lvl}] '{title}' → target={target}  "
          f"{len(chunks)} chunk(s)  "
          f"sizes={[c.metadata['chunk_size_chars'] for c in chunks]}")

# --- Test 2: Synthetic mixed content_type documents ---
print("\n" + "=" * 60)
print("MIXED content_type documents")
print("=" * 60)

DOCS = [
    Document(
        doc_id   = str(uuid.uuid4()),
        content  = "This is a long prose section about retrieval. " * 20,
        source   = "synthetic",
        metadata = {"content_type": "text", "heading_level": 2},
    ),
    Document(
        doc_id   = str(uuid.uuid4()),
        content  = "| model | recall@1 | latency |\n| --- | --- | --- |\n| BGE-small | 0.82 | 12ms |\n| E5-base | 0.85 | 18ms |",
        source   = "synthetic",
        metadata = {"content_type": "table"},
    ),
    Document(
        doc_id   = str(uuid.uuid4()),
        content  = "[Image-only page — OCR required for content extraction]",
        source   = "synthetic",
        metadata = {"content_type": "ocr_needed"},
    ),
    Document(
        doc_id   = str(uuid.uuid4()),
        content  = "A brief H3 note about limitations.",
        source   = "synthetic",
        metadata = {"content_type": "text", "heading_level": 3},
    ),
]

for doc in DOCS:
    ct     = doc.metadata.get("content_type", "text")
    hl     = doc.metadata.get("heading_level", "-")
    chunks = chunker.chunk(doc)
    sizes  = [c.metadata["chunk_size_chars"] for c in chunks]
    reason = chunks[0].metadata.get("split_reason", "split") if chunks else "—"
    print(f"  [{ct:30s}  H{hl}]  {len(chunks)} chunk(s)  sizes={sizes}  reason={reason}")