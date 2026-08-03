# scripts/test_pdf_loader.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from collections import Counter
from loaders.pdf_loader import PDFLoader

loader = PDFLoader(min_chars=20, extract_tables=True)
docs   = loader.load("scripts/sample.pdf")

# --- overview ---
print(f"Loaded {len(docs)} Documents from PDF\n")

types = Counter(d.metadata.get("content_type", "unknown") for d in docs)
print("Content-type breakdown:")
for ct, n in types.items():
    print(f"  {ct:15s} : {n}")
print()

# --- show a preview of each type ---
seen = set()
for d in docs:
    ct = d.metadata.get("content_type")
    if ct in seen:
        continue
    seen.add(ct)
    print(f"--- Example: content_type='{ct}', page {d.metadata['page_number']} ---")
    print(d.content[:400])
    print()