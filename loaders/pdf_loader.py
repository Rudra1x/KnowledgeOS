# loaders/pdf_loader.py

import uuid
import pdfplumber
from core import Document, Loader


class PDFLoader(Loader):
    """
    PDF loader with layout awareness and table extraction.

    Emits multiple Documents per PDF:
    - One Document per page for the prose content (content_type='text')
    - One Document per table (content_type='table', with table_index)

    Detects and skips image-only / empty pages, logging them.

    Limitations (upgrade path):
    - Scanned/image PDFs → need OCR (pytesseract). Marked with content_type='ocr_needed'.
    - Figures/diagrams → not extracted (multimodal, M12).
    - Complex nested tables → best-effort; camelot is better for those.
    """

    def __init__(self, min_chars: int = 20, extract_tables: bool = True):
        self.min_chars      = min_chars
        self.extract_tables = extract_tables

    def load(self, source: str) -> list[Document]:
        docs = []

        with pdfplumber.open(source) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):

                # --- 1. Prose text ---
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    print(f"[PDFLoader] page {page_num} text extraction failed: {e}")
                    text = ""

                text = text.strip()

                if len(text) < self.min_chars:
                    # Likely image-only or empty page → flag for OCR
                    if self._page_has_images(page):
                        docs.append(Document(
                            doc_id   = str(uuid.uuid4()),
                            content  = "[Image-only page — OCR required for content extraction]",
                            source   = source,
                            metadata = {
                                "file_type":    "pdf",
                                "page_number":  page_num,
                                "total_pages":  total_pages,
                                "content_type": "ocr_needed",
                            },
                        ))
                    # else: truly empty page, skip silently
                else:
                    docs.append(Document(
                        doc_id   = str(uuid.uuid4()),
                        content  = text,
                        source   = source,
                        metadata = {
                            "file_type":    "pdf",
                            "page_number":  page_num,
                            "total_pages":  total_pages,
                            "content_type": "text",
                        },
                    ))

                # --- 2. Tables (as separate Documents) ---
                if self.extract_tables:
                    try:
                        tables = page.extract_tables()
                    except Exception as e:
                        print(f"[PDFLoader] page {page_num} table extraction failed: {e}")
                        tables = []

                    for table_idx, table in enumerate(tables):
                        # Filter noise: require at least 2 rows AND 2 columns of real content
                        if not self._is_meaningful_table(table):
                            continue

                        table_md = self._table_to_markdown(table)
                        if not table_md:
                            continue

                        docs.append(Document(
                            doc_id   = str(uuid.uuid4()),
                            content  = table_md,
                            source   = source,
                            metadata = {
                                "file_type":    "pdf",
                                "page_number":  page_num,
                                "total_pages":  total_pages,
                                "content_type": "table",
                                "table_index":  table_idx,
                            },
                        ))

        return docs

    @staticmethod
    def _page_has_images(page) -> bool:
        try:
            return len(page.images) > 0
        except Exception:
            return False

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        """Convert a pdfplumber table (list of rows) to a Markdown table string."""
        if not table or not table[0]:
            return ""

        # Clean cells: replace None with empty string, strip whitespace
        rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table]

        header = rows[0]
        body   = rows[1:]

        md  = "| " + " | ".join(header) + " |\n"
        md += "| " + " | ".join(["---"] * len(header)) + " |\n"
        for row in body:
            # Pad short rows to header width
            while len(row) < len(header):
                row.append("")
            md += "| " + " | ".join(row[:len(header)]) + " |\n"

        return md.strip()
    @staticmethod
    def _is_meaningful_table(table: list[list]) -> bool:
        """A real table has >=2 rows and >=2 columns of non-empty cells."""
        if not table or len(table) < 2:
            return False

        # Count non-empty columns in the header row
        header       = table[0] or []
        non_empty_cols = sum(1 for cell in header if cell and str(cell).strip())
        if non_empty_cols < 2:
            return False

        return True