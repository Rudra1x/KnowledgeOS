# loaders/router.py

import os
from urllib.parse import urlparse
from core import Loader, Document
from .txt_loader     import TXTLoader
from .pdf_loader     import PDFLoader
from .docx_loader    import DOCXLoader
from .md_loader      import MarkdownLoader
from .html_loader    import HTMLLoader
from .csv_loader     import CSVLoader
from .email_loader   import EmailLoader
from .youtube_loader import YouTubeLoader


class LoaderRouter(Loader):
    """
    Auto-dispatches to the right loader based on source type.

    Routing:
    - http(s)://youtube.com | youtu.be   → YouTubeLoader
    - other http(s)://                   → HTMLLoader
    - .txt                               → TXTLoader
    - .md / .markdown                    → MarkdownLoader
    - .html / .htm                       → HTMLLoader
    - .pdf                               → PDFLoader
    - .docx / .doc                       → DOCXLoader
    - .csv                               → CSVLoader
    - .eml                               → EmailLoader

    Unknown extension → raises ValueError with a clear list of supported types.

    The router itself is a Loader — same interface. This means downstream code
    doesn't need to know the routing exists.
    """

    EXTENSION_MAP = {
        ".txt":      TXTLoader,
        ".md":       MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".html":     HTMLLoader,
        ".htm":      HTMLLoader,
        ".pdf":      PDFLoader,
        ".docx":     DOCXLoader,
        ".doc":      DOCXLoader,
        ".csv":      CSVLoader,
        ".eml":      EmailLoader,
    }

    def __init__(self, loader_kwargs: dict | None = None):
        # Optional per-loader kwargs, e.g. {"CSVLoader": {"strategy": "row"}}
        self.loader_kwargs = loader_kwargs or {}

    def load(self, source: str) -> list[Document]:
        loader_cls = self._resolve(source)
        kwargs     = self.loader_kwargs.get(loader_cls.__name__, {})
        return loader_cls(**kwargs).load(source)

    def _resolve(self, source: str) -> type[Loader]:
        parsed = urlparse(source)

        # URL routing
        if parsed.scheme in {"http", "https"}:
            host = parsed.netloc.lower()
            if "youtube.com" in host or "youtu.be" in host:
                return YouTubeLoader
            return HTMLLoader

        # File-extension routing
        ext = os.path.splitext(source)[1].lower()
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]

        raise ValueError(
            f"No loader for source: {source!r}. "
            f"Supported extensions: {sorted(self.EXTENSION_MAP.keys())} "
            f"+ HTTP(S) URLs (routes to HTMLLoader or YouTubeLoader)."
        )