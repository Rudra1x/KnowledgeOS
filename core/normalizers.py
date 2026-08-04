# core/normalizers.py

import re
import unicodedata
from abc import ABC, abstractmethod
from langdetect import detect, LangDetectException, DetectorFactory
from core import Document


# Deterministic language detection (langdetect is nondeterministic by default)
DetectorFactory.seed = 0


class Normalizer(ABC):
    """Base class for all normalization stages."""
    @abstractmethod
    def apply(self, doc: Document) -> Document: ...


class TextCleaner(Normalizer):
    """
    Cleans document content:
    - Unicode normalization to NFC (canonical composition)
    - Removes control chars (0x00-0x1F except tab/newline) and zero-width chars
    - Collapses runs of 3+ newlines to 2 (preserves paragraph breaks, kills excess whitespace)
    - Collapses runs of 2+ spaces to 1
    - Strips leading/trailing whitespace on each line

    Never modifies metadata — pure content cleaning.
    """

    # Zero-width and other invisible/formatting chars we don't want
    INVISIBLE_CHARS = re.compile(r"[\u200b-\u200f\u2028-\u202f\ufeff]")

    # Control chars EXCEPT tab (\t = 0x09) and newline (\n = 0x0A)
    CONTROL_CHARS   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    MULTI_NEWLINE   = re.compile(r"\n{3,}")
    MULTI_SPACE     = re.compile(r"[ \t]{2,}")

    def apply(self, doc: Document) -> Document:
        text = doc.content

        # 1. Unicode NFC normalization
        # 'café' can be encoded two ways in Unicode. NFC picks the canonical form.
        # Without this, "café" and "café" are different strings to the embedder.
        text = unicodedata.normalize("NFC", text)

        # 2. Strip invisible / control chars
        text = self.INVISIBLE_CHARS.sub("", text)
        text = self.CONTROL_CHARS.sub("", text)

        # 3. Collapse whitespace
        text = self.MULTI_NEWLINE.sub("\n\n", text)
        # Strip trailing whitespace per line, then collapse multi-spaces on each line
        lines = [self.MULTI_SPACE.sub(" ", line.rstrip()) for line in text.split("\n")]
        text  = "\n".join(lines).strip()

        doc.content = text
        return doc


class LanguageDetector(Normalizer):
    """
    Detects the primary language of the document if not already set.

    Strategy:
    - Skip very short docs (< min_chars) → fallback
    - For longer docs, use langdetect's probability-based detect_langs()
    - Only accept the top detection if confidence >= min_confidence
    - Otherwise → fallback

    Confidence-based rejection avoids the classic short-text failure mode:
    langdetect will happily label 'NestOpt-Q Rudraksh' as Indonesian if
    forced to pick one. With a confidence gate, uncertain guesses fall back
    to the safe default.
    """

    def __init__(self, min_chars: int = 50, min_confidence: float = 0.85, fallback: str = "en"):
        self.min_chars      = min_chars
        self.min_confidence = min_confidence
        self.fallback       = fallback

    def apply(self, doc: Document) -> Document:
        if doc.language and doc.language not in {"en", "unknown", ""}:
            return doc

        if len(doc.content) < self.min_chars:
            doc.language = self.fallback
            return doc

        try:
            from langdetect import detect_langs
            candidates = detect_langs(doc.content)
            if candidates and candidates[0].prob >= self.min_confidence:
                doc.language = candidates[0].lang
            else:
                doc.language = self.fallback
        except Exception:
            doc.language = self.fallback

        return doc

class MetadataEnricher(Normalizer):
    """
    Adds derived-metadata fields every downstream stage benefits from:
    - char_count
    - word_count
    - line_count

    Cheap to compute now, avoids recomputation in chunkers, retrievers, and eval.
    """

    def apply(self, doc: Document) -> Document:
        content = doc.content
        doc.metadata["char_count"] = len(content)
        doc.metadata["word_count"] = len(content.split())
        doc.metadata["line_count"] = content.count("\n") + 1
        return doc


class NormalizationPipeline:
    """
    Runs a sequence of Normalizers on each Document.

    Default pipeline: clean text → detect language → enrich metadata.
    Order matters: cleaning must precede language detection (garbage in, garbage out).
    """

    def __init__(self, normalizers: list[Normalizer] | None = None):
        self.normalizers = normalizers or [
            TextCleaner(),
            LanguageDetector(),
            MetadataEnricher(),
        ]

    def apply(self, doc: Document) -> Document:
        for n in self.normalizers:
            doc = n.apply(doc)
        return doc

    def apply_many(self, docs: list[Document]) -> list[Document]:
        return [self.apply(d) for d in docs]