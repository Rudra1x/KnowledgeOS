# scripts/test_parent_child.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from collections                 import defaultdict
from loaders.txt_loader          import TXTLoader
from chunkers.parent_child_chunker import ParentChildChunker


loader = TXTLoader()
docs   = loader.load("scripts/corpus.txt")
doc    = docs[0]

print(f"Document length: {len(doc.content)} chars\n")

pc = ParentChildChunker(parent_size=1200, child_size=300, child_overlap=40)
children = pc.chunk(doc)

print("=" * 60)
print(f"CHILDREN: {len(children)}")
print("=" * 60)
for c in children:
    p_id   = c.metadata["parent_id"][:8]
    p_idx  = c.metadata["parent_index"]
    c_idx  = c.metadata["child_index_in_parent"]
    size   = c.metadata["chunk_size_chars"]
    print(f"  [P{p_idx}.C{c_idx}]  size={size}  parent={p_id}...  "
          f"{c.content[:60].strip()}...")

# --- Group children by parent to visualize the hierarchy ---
by_parent = defaultdict(list)
for c in children:
    by_parent[c.metadata["parent_id"]].append(c)

print("\n" + "=" * 60)
print("HIERARCHY")
print("=" * 60)
for parent_id, kids in by_parent.items():
    parent_content = kids[0].metadata["parent_content"]
    print(f"\nParent {parent_id[:8]}...  ({len(parent_content)} chars, {len(kids)} children)")
    print(f"  Content preview: {parent_content[:100].strip()}...")
    for kid in kids:
        print(f"    ↳ child ({kid.metadata['chunk_size_chars']} chars): {kid.content[:60].strip()}...")

# --- Ratio check ---
avg_child  = sum(c.metadata['chunk_size_chars'] for c in children) / len(children)
avg_parent = sum(len(kids[0].metadata['parent_content']) for kids in by_parent.values()) / len(by_parent)
print(f"\nAvg child size:  {avg_child:.0f} chars")
print(f"Avg parent size: {avg_parent:.0f} chars")
print(f"Ratio (parent/child): {avg_parent/avg_child:.1f}x")