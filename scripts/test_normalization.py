# scripts/test_normalization.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.txt_loader   import TXTLoader
from loaders.pdf_loader   import PDFLoader
from loaders.html_loader  import HTMLLoader
from loaders.md_loader    import MarkdownLoader
from core                 import NormalizationPipeline, Document


# --- Case 1: A deliberately messy Document — see the cleaner in action ---
messy = Document(
    doc_id   = "test-1",
    content  = "Café\u200b   is   nice.\n\n\n\n\nCafé is nice.\x00\x0eHello\r\nworld.",
    source   = "synthetic",
    metadata = {},
    language = "en",
)

print("=" * 60)
print("CASE 1: Synthetic messy document")
print("=" * 60)
print(f"BEFORE (repr): {messy.content!r}")

pipeline = NormalizationPipeline()
cleaned  = pipeline.apply(messy)

print(f"AFTER  (repr): {cleaned.content!r}")
print(f"Metadata: {cleaned.metadata}")
print()

# --- Case 2: Language detection on a French Document ---
french = Document(
    doc_id="test-fr",
    content="La retrieval augmented generation est une technique qui combine la recherche d'information avec la generation de texte par un grand modele de langage.",
    source="synthetic",
    metadata={},
    language="en",     # deliberately wrong to see if detector overrides
)
print("=" * 60)
print("CASE 2: Language detection")
print("=" * 60)
print(f"language BEFORE: {french.language}")
detected = pipeline.apply(french)
print(f"language AFTER:  {detected.language}")
print()

# --- Case 3: Full pipeline on real loaded content ---
print("=" * 60)
print("CASE 3: Real loaders → normalization")
print("=" * 60)

for LoaderCls, path in [
    (TXTLoader,      "scripts/sample.txt"),
    (MarkdownLoader, "scripts/sample.md"),
    (PDFLoader,      "scripts/sample.pdf"),
]:
    try:
        raw  = LoaderCls().load(path)
        norm = pipeline.apply_many(raw)
        d    = norm[0]
        print(f"{LoaderCls.__name__:15s} → {len(norm)} docs | "
              f"lang={d.language} | chars={d.metadata['char_count']} | "
              f"words={d.metadata['word_count']}")
    except FileNotFoundError:
        print(f"{LoaderCls.__name__:15s} → skipped (no sample file)")