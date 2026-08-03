# loaders/email_loader.py

import uuid
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime, getaddresses
from bs4 import BeautifulSoup
from core import Document, Loader


class EmailLoader(Loader):
    """
    Loads a .eml file into a Document.

    Body extraction:
    - Prefers text/plain; falls back to HTML (stripped via BeautifulSoup)
    - Never uses both — avoids duplicate content

    Content assembly:
    - Subject line included at top of content (highly retrieval-relevant)
    - Body follows

    Metadata:
    - from, to, cc, date (parsed to ISO), subject, message_id
    - attachment_count, attachment_names
    - body_source (which MIME part was used: 'plain' | 'html')

    Attachments:
    - Names and content types captured
    - Content of attachments NOT extracted here (recursive routing = out of M1 scope)
      A production system would route PDFs to PDFLoader, images to VLM, etc.
    """

    def __init__(self, min_chars: int = 20):
        self.min_chars = min_chars

    def load(self, source: str) -> list[Document]:
        with open(source, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        # --- Headers as metadata ---
        subject = (msg.get("Subject") or "").strip()
        sender  = self._addresses(msg, "From")
        to      = self._addresses(msg, "To")
        cc      = self._addresses(msg, "Cc")
        date    = self._parse_date(msg.get("Date"))
        msg_id  = (msg.get("Message-ID") or "").strip()

        # --- Body extraction ---
        body_text, body_source = self._extract_body(msg)

        # --- Attachments (names only) ---
        attachments = self._list_attachments(msg)

        # --- Assemble content: Subject first, then body ---
        content_parts = []
        if subject:
            content_parts.append(f"Subject: {subject}")
        if body_text:
            content_parts.append(body_text)
        content = "\n\n".join(content_parts).strip()

        if len(content) < self.min_chars:
            return []

        return [Document(
            doc_id   = str(uuid.uuid4()),
            content  = content,
            source   = source,
            metadata = {
                "file_type":         "email",
                "content_type":      "email",
                "subject":           subject,
                "from":              sender,
                "to":                to,
                "cc":                cc,
                "date":              date,
                "message_id":        msg_id,
                "body_source":       body_source,
                "attachment_count":  len(attachments),
                "attachment_names":  [a["name"] for a in attachments],
                "attachments":       attachments,   # full info for downstream routing
            },
        )]

    # ------------------------------------------------------------------
    @staticmethod
    def _addresses(msg, header: str) -> list[str]:
        raw = msg.get_all(header) or []
        return [addr for _, addr in getaddresses(raw) if addr]

    @staticmethod
    def _parse_date(raw: str | None) -> str | None:
        if not raw:
            return None
        try:
            return parsedate_to_datetime(raw).isoformat()
        except (TypeError, ValueError):
            return raw   # fallback: keep original string

    # ------------------------------------------------------------------
    def _extract_body(self, msg) -> tuple[str, str]:
        """
        Returns (body_text, source). source is 'plain' | 'html' | 'none'.
        Prefers text/plain over text/html to avoid duplicate content.
        """
        plain_body = None
        html_body  = None

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_disposition() == "attachment":
                    continue
                ctype = part.get_content_type()
                if ctype == "text/plain" and plain_body is None:
                    plain_body = self._decode_part(part)
                elif ctype == "text/html" and html_body is None:
                    html_body = self._decode_part(part)
        else:
            ctype = msg.get_content_type()
            body  = self._decode_part(msg)
            if ctype == "text/plain":
                plain_body = body
            elif ctype == "text/html":
                html_body = body

        if plain_body and plain_body.strip():
            return plain_body.strip(), "plain"
        if html_body:
            stripped = BeautifulSoup(html_body, "lxml").get_text(separator="\n", strip=True)
            if stripped.strip():
                return stripped.strip(), "html"
        return "", "none"

    @staticmethod
    def _decode_part(part) -> str:
        try:
            payload = part.get_content()
            if isinstance(payload, bytes):
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
            return payload or ""
        except Exception:
            return ""

    @staticmethod
    def _list_attachments(msg) -> list[dict]:
        atts = []
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                atts.append({
                    "name":         part.get_filename() or "<unnamed>",
                    "content_type": part.get_content_type(),
                    "size_bytes":   len(part.get_payload(decode=True) or b""),
                })
        return atts