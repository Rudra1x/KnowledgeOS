# scripts/test_youtube_loader.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.youtube_loader import YouTubeLoader

# Andrej Karpathy - "Intro to Large Language Models" - a widely captioned, English talk
# Fallback options are also popular AI/ML talks with reliable captions
url = "https://www.youtube.com/watch?v=zjkBMFhNj_g"

# --- Segment mode: one Document per timestamp ---
print("=" * 60)
print("SEGMENT MODE (timestamped)")
print("=" * 60)
loader = YouTubeLoader(mode="segment")
docs   = loader.load(url)
print(f"Segments extracted: {len(docs)}\n")

for d in docs[:5]:
    print(f"  [{d.metadata['timestamp']}] {d.content[:80]}...")

print("\n--- Sample metadata (segment 0) ---")
for k, v in docs[0].metadata.items():
    print(f"  {k:20s}: {v}")

# --- Full mode: whole transcript as one Document ---
print()
print("=" * 60)
print("FULL MODE (whole transcript)")
print("=" * 60)
loader_full = YouTubeLoader(mode="full")
docs_full   = loader_full.load(url)
if docs_full:
    d = docs_full[0]
    print(f"Segments merged:   {d.metadata['segment_count']}")
    print(f"Duration (s):      {d.metadata['duration_seconds']:.1f}")
    print(f"Total chars:       {len(d.content)}")
    print(f"Preview:           {d.content[:200]}...")