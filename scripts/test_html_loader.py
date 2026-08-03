# scripts/test_html_loader.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.html_loader import HTMLLoader

# A content-heavy Wikipedia page — good real-world test
url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

print("=" * 60)
print("STRATEGY: trafilatura (production default)")
print("=" * 60)
loader = HTMLLoader(strategy="trafilatura")
docs   = loader.load(url)
for d in docs:
    print(f"Title:  {d.metadata.get('title')}")
    print(f"Lang:   {d.language}")
    print(f"Chars:  {len(d.content)}")
    print(f"Method: {d.metadata['extraction_method']}")
    print(f"Preview:\n{d.content[:400]}...")
    print()

print("=" * 60)
print("STRATEGY: both (A/B compare)")
print("=" * 60)
loader2 = HTMLLoader(strategy="both")
docs2   = loader2.load(url)
for d in docs2:
    print(f"[{d.metadata['extraction_method']:12s}]  chars: {len(d.content):>6}")

# Show the delta — how much boilerplate did trafilatura strip that bs4 didn't?
if len(docs2) == 2:
    t_len = next(d for d in docs2 if d.metadata["extraction_method"] == "trafilatura").content
    b_len = next(d for d in docs2 if d.metadata["extraction_method"] == "bs4").content
    diff  = len(b_len) - len(t_len)
    pct   = (diff / max(len(b_len), 1)) * 100
    print(f"\nBoilerplate delta: bs4 kept {diff} more chars ({pct:.1f}% noise removed by trafilatura)")