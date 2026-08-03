# loaders/csv_loader.py

import uuid
import pandas as pd
from core import Document, Loader


class CSVLoader(Loader):
    """
    CSV loader with two strategies:
    - 'row':  one Document per row  (row = one entity: products, FAQs, records)
    - 'file': one Document per file (rows = one dataset: time series, aggregates)

    Row mode uses field-value templating: "col1: val1 | col2: val2 | ..."
    This preserves column semantics — critical for retrieval quality.

    Metadata always captures:
    - file_type='csv'
    - row_count, column_count
    - columns (list of column names)
    - row_index (row mode only)
    """

    def __init__(
        self,
        strategy:      str        = "row",   # 'row' or 'file'
        min_chars:     int        = 5,
        encoding:      str        = "utf-8",
        text_columns:  list[str] | None = None,   # if set, only include these cols in content
        max_rows:      int | None = None,    # cap for large files (safety valve)
    ):
        if strategy not in {"row", "file"}:
            raise ValueError(f"strategy must be 'row' or 'file', got {strategy!r}")
        self.strategy     = strategy
        self.min_chars    = min_chars
        self.encoding     = encoding
        self.text_columns = text_columns
        self.max_rows     = max_rows

    def load(self, source: str) -> list[Document]:
        try:
            df = pd.read_csv(source, encoding=self.encoding, keep_default_na=False)
        except UnicodeDecodeError:
            # Fallback for Windows CSVs saved from Excel
            df = pd.read_csv(source, encoding="latin-1", keep_default_na=False)

        if self.max_rows is not None:
            df = df.head(self.max_rows)

        if df.empty:
            return []

        # Filter columns if the user specified a subset
        if self.text_columns:
            missing = [c for c in self.text_columns if c not in df.columns]
            if missing:
                raise ValueError(f"text_columns not in CSV: {missing}. Available: {list(df.columns)}")
            content_df = df[self.text_columns]
        else:
            content_df = df

        common_meta = {
            "file_type":    "csv",
            "content_type": "table",
            "row_count":    int(len(df)),
            "column_count": int(len(df.columns)),
            "columns":      list(df.columns),
        }

        if self.strategy == "file":
            return self._load_as_file(source, content_df, common_meta)
        else:
            return self._load_as_rows(source, content_df, common_meta)

    # ------------------------------------------------------------------
    def _load_as_rows(self, source: str, df: pd.DataFrame, base_meta: dict) -> list[Document]:
        docs = []
        for row_idx, row in df.iterrows():
            content = self._format_row(row)
            if len(content) < self.min_chars:
                continue
            docs.append(Document(
                doc_id   = str(uuid.uuid4()),
                content  = content,
                source   = source,
                metadata = {**base_meta, "row_index": int(row_idx)},
            ))
        return docs

    def _load_as_file(self, source: str, df: pd.DataFrame, base_meta: dict) -> list[Document]:
        # Render as a Markdown-style table — LLMs read this well
        lines = ["| " + " | ".join(str(c) for c in df.columns) + " |"]
        lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.values) + " |")

        content = "\n".join(lines).strip()
        if len(content) < self.min_chars:
            return []

        return [Document(
            doc_id   = str(uuid.uuid4()),
            content  = content,
            source   = source,
            metadata = base_meta,
        )]

    # ------------------------------------------------------------------
    @staticmethod
    def _format_row(row: pd.Series) -> str:
        """Field-value templating: 'col1: val1 | col2: val2 | ...'."""
        parts = []
        for col, val in row.items():
            val_str = str(val).strip()
            if val_str and val_str.lower() != "nan":
                parts.append(f"{col}: {val_str}")
        return " | ".join(parts)