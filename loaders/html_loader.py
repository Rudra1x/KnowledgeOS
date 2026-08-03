# loaders/html_loader.py

import uuid
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from core import Document, Loader


class HTMLLoader(Loader):
    """
    Loads HTML from a URL or a local .html file.

    Extraction strategies:
    - 'trafilatura' (default): ML-heuristic boilerplate remover, best for arbitrary web pages
    - 'bs4':                   BeautifulSoup fallback — grabs <main>/<article>/<body>, strips known chrome
    - 'both':                  runs both, returns TWO Documents (for A/B comparison in eval)

    Metadata captured:
    - source URL or file path
    - extraction_method (which extractor produced this text)
    - title (from <title> or trafilatura's metadata)
    - language (trafilatura only, when available)
    """

    STRIP_TAGS = {"script", "style", "nav", "footer", "header",
                  "aside", "form", "noscript", "svg", "iframe"}

    # Kept minimal — aggressive CSS selectors over-strip on CMS-heavy sites (Wikipedia, docs sites).
    # For site-specific tuning, subclass HTMLLoader and override this.
    STRIP_SELECTORS: list[str] = []

    def __init__(self, strategy: str = "trafilatura", min_chars: int = 100, timeout: int = 20):
        if strategy not in {"trafilatura", "bs4", "both"}:
            raise ValueError(f"strategy must be trafilatura|bs4|both, got {strategy}")
        self.strategy  = strategy
        self.min_chars = min_chars
        self.timeout   = timeout

    def load(self, source: str) -> list[Document]:
        html, resolved_url = self._fetch(source)

        docs = []
        if self.strategy in {"trafilatura", "both"}:
            doc = self._extract_trafilatura(html, resolved_url)
            if doc:
                docs.append(doc)

        if self.strategy in {"bs4", "both"}:
            doc = self._extract_bs4(html, resolved_url)
            if doc:
                docs.append(doc)

        return docs

    # ------------------------------------------------------------------
    def _fetch(self, source: str) -> tuple[str, str]:
        """Returns (html_string, resolved_url_or_path)."""
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            headers = {"User-Agent": "KnowledgeOS/0.1 (+learning-project)"}
            resp    = requests.get(source, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text, source

        # Local file
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), source

    # ------------------------------------------------------------------
    def _extract_trafilatura(self, html: str, source: str) -> Document | None:
        text = trafilatura.extract(
            html,
            include_comments = False,
            include_tables   = True,
            favor_precision  = True,
        )
        if not text or len(text.strip()) < self.min_chars:
            return None

        # Metadata (title, language, author, date...)
        meta = trafilatura.extract_metadata(html)

        return Document(
            doc_id   = str(uuid.uuid4()),
            content  = text.strip(),
            source   = source,
            metadata = {
                "file_type":         "html",
                "content_type":      "text",
                "extraction_method": "trafilatura",
                "title":             meta.title    if meta else None,
                "author":            meta.author   if meta else None,
                "date":              meta.date     if meta else None,
            },
            language = (meta.language if meta and meta.language else "en"),
        )

    # ------------------------------------------------------------------
    def _extract_bs4(self, html: str, source: str) -> Document | None:
        soup = BeautifulSoup(html, "lxml")

        # Remove chrome tags entirely
        for tag_name in self.STRIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Remove elements matched by known-chrome selectors
        for selector in self.STRIP_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()

        # Prefer semantic containers if present
        main = soup.find("main") or soup.find("article") or soup.body or soup

        text = main.get_text(separator="\n", strip=True)
        text = "\n".join(line for line in text.splitlines() if line.strip())

        if len(text) < self.min_chars:
            return None

        title = soup.title.string.strip() if soup.title and soup.title.string else None

        return Document(
            doc_id   = str(uuid.uuid4()),
            content  = text,
            source   = source,
            metadata = {
                "file_type":         "html",
                "content_type":      "text",
                "extraction_method": "bs4",
                "title":             title,
            },
        )