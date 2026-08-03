# scripts/debug_bs4.py
import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.html_loader import HTMLLoader

loader = HTMLLoader(strategy="bs4", min_chars=1)   # lower threshold to see what came back
docs   = loader.load("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

if not docs:
    print("bs4 returned nothing at all.")
else:
    d = docs[0]
    print(f"bs4 extracted {len(d.content)} chars")
    print(d.content[:500])