# loaders/md_loader.py

import re
import uuid
from core import Document, Loader


class MarkdownLoader(Loader):
    """
    Loads a Markdown file, using headings (# / ## / ###) as section boundaries.

    Emits ONE Document per section. Preserves heading level in metadata.

    Why not just strip Markdown and treat as plain text?
    - Headings ARE the semantic structure — throwing them away loses signal
    - Downstream retrieval can boost matches whose section title matches the query
    - Citations become more informative ("under '## Installation'")
    """

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

    def __init__(self, min_chars: int = 20):
        self.min_chars = min_chars

    def load(self, source: str) -> list[Document]:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()

        # Find all heading positions
        headings = [(m.start(), len(m.group(1)), m.group(2).strip())
                    for m in self.HEADING_RE.finditer(text)]

        docs = []

        # Handle text before the first heading as "Preamble"
        if not headings or headings[0][0] > 0:
            preamble = text[:headings[0][0]] if headings else text
            preamble = preamble.strip()
            if len(preamble) >= self.min_chars:
                docs.append(Document(
                    doc_id   = str(uuid.uuid4()),
                    content  = preamble,
                    source   = source,
                    metadata = {
                        "file_type":     "markdown",
                        "content_type":  "text",
                        "section_title": "Preamble",
                        "heading_level": 0,
                    },
                ))

        # Handle each heading + its content up to the next heading
        for i, (pos, level, title) in enumerate(headings):
            end = headings[i + 1][0] if i + 1 < len(headings) else len(text)

            # Skip the heading line itself, start after the newline
            section_start = text.find("\n", pos) + 1
            section_text  = text[section_start:end].strip()

            if len(section_text) < self.min_chars:
                continue

            docs.append(Document(
                doc_id   = str(uuid.uuid4()),
                content  = section_text,
                source   = source,
                metadata = {
                    "file_type":     "markdown",
                    "content_type":  "text",
                    "section_title": title,
                    "heading_level": level,
                },
            ))

        return docs