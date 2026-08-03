# scripts/test_csv_loader.py

import sys
sys.path.insert(0, "D:\\Rudraksh\\KnowledgeOS")

from loaders.csv_loader import CSVLoader

# --- Case 1: FAQ, row-per-entity ---
print("=" * 60)
print("CASE 1: FAQ (row-per-entity strategy)")
print("=" * 60)
loader = CSVLoader(strategy="row")
docs   = loader.load("scripts/faq.csv")
print(f"Documents: {len(docs)}\n")
for d in docs[:3]:
    print(f"Row {d.metadata['row_index']}:  {d.content}")
    print()

# --- Case 2: Sales data — WRONG strategy (row) ---
print("=" * 60)
print("CASE 2: Sales — WRONG strategy (row-per-doc)")
print("=" * 60)
docs_wrong = CSVLoader(strategy="row").load("scripts/sales.csv")
print(f"Documents: {len(docs_wrong)}")
print("Example: ", docs_wrong[0].content)
print("\n(Each row is retrievable but nothing captures cross-row patterns.)")
print()

# --- Case 3: Sales data — CORRECT strategy (file) ---
print("=" * 60)
print("CASE 3: Sales — CORRECT strategy (whole-file)")
print("=" * 60)
docs_right = CSVLoader(strategy="file").load("scripts/sales.csv")
print(f"Documents: {len(docs_right)}")
print(docs_right[0].content)
print()
print("(A query like 'total Widget A sales trend' now has one retrievable Document containing all rows.)")