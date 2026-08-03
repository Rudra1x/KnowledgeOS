# scripts/test_email_loader.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.email_loader import EmailLoader

loader = EmailLoader()
docs   = loader.load("scripts/sample.eml")

print(f"Loaded {len(docs)} email(s)\n")

d = docs[0]
print("=" * 60)
print("METADATA")
print("=" * 60)
for k, v in d.metadata.items():
    if k == "attachments":
        continue   # already shown via attachment_names
    print(f"  {k:20s}: {v}")

print()
print("=" * 60)
print("CONTENT")
print("=" * 60)
print(d.content)