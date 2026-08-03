# scripts/test_docx_md.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from collections import Counter
from loaders.md_loader   import MarkdownLoader
from loaders.docx_loader import DOCXLoader

# --- Markdown ---
print("=" * 60)
print("MARKDOWN LOADER")
print("=" * 60)
md_loader = MarkdownLoader()
md_docs   = md_loader.load("scripts/sample.md")
print(f"Sections: {len(md_docs)}\n")

for d in md_docs:
    lvl   = d.metadata["heading_level"]
    title = d.metadata["section_title"]
    print(f"[H{lvl}] {title}")
    print(f"      {d.content[:100].strip()}...")
    print()

# --- DOCX ---
print("=" * 60)
print("DOCX LOADER")
print("=" * 60)
try:
    docx_loader = DOCXLoader()
    docx_docs   = docx_loader.load("scripts/sample.docx")

    print(f"Documents: {len(docx_docs)}")
    types = Counter(d.metadata.get("content_type") for d in docx_docs)
    print(f"Types: {dict(types)}\n")

    for d in docx_docs[:5]:
        print(f"[{d.metadata.get('content_type')}] {d.metadata.get('section_title', d.metadata.get('table_index'))}")
        print(f"      {d.content[:100].strip()}...")
        print()
except FileNotFoundError:
    print("(Skipping DOCX test — put a sample .docx at scripts/sample.docx to test)")