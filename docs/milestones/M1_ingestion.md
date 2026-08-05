# Milestone 1 — Ingestion Breadth

**Status:** ✅ Complete
**Duration:** 8 checkpoints
**Deliverable:** A production-shaped ingestion layer that handles 8 formats through one uniform interface, with per-format quality measured through the eval harness.

---

## 1. Milestone summary

### Goal

Widen the front door. Every major document format — TXT, PDF, DOCX, Markdown, HTML, CSV, Email, YouTube — flows through the same normalized `Document` shape, so downstream chunking/embedding/retrieval doesn't care what the source was. Ingestion quality is measured per format, not assumed.

### Why this milestone matters

Ingestion is where most RAG systems quietly fail. A wrongly-parsed PDF, a garbled table extraction, a missed language detection, a dropped chunk — none of these produce error messages. They produce *bad answers,* and the team spends weeks blaming the embedder. This milestone puts a foundation under the whole pipeline: **every source type is a first-class citizen with measured quality.**

### What "done" looks like

- 8 loaders — TXT, PDF, DOCX, Markdown, HTML, CSV, Email, YouTube
- 1 auto-routing dispatcher (extension + URL-scheme based routing)
- 1 unified normalization pipeline (text cleaning, language detection, metadata enrichment)
- Multi-format gold set + per-format eval
- Recorded multi-format baseline: recall@1 = 0.714, recall@3 = 0.857, MRR = 0.821
- All committed with per-checkpoint git tags
- Every failure mode observed in real testing is documented

---

## 2. Architecture recap

### The ingestion flow

```
                     ┌─────────────────┐
Source URL/file ────▶│  LoaderRouter   │
                     └─────────────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
       .txt, .md         .pdf, .docx     http(s)
         TXT/MD           PDF/DOCX      HTML/YouTube
         Loader           Loader          Loader
             │               │               │
             └───────────────┼───────────────┘
                             ▼
                     list[Document]
                             │
                             ▼
                ┌────────────────────────────┐
                │  NormalizationPipeline     │
                │  ├─ TextCleaner            │
                │  ├─ LanguageDetector       │
                │  └─ MetadataEnricher       │
                └────────────────────────────┘
                             │
                             ▼
                list[Document]  (canonical form)
                             │
                             ▼
                   → Chunker (M2) → Embedder → Index → Retriever ...
```

### Loader inventory

| Loader             | Handles        | Key features                                                                    |
| ------------------ | -------------- | ------------------------------------------------------------------------------- |
| `TXTLoader`      | .txt           | Simplest baseline                                                               |
| `PDFLoader`      | .pdf           | pdfplumber, per-page docs, table extraction,`content_type` metadata, OCR flag |
| `DOCXLoader`     | .doc, .docx    | Heading-aware sections, LibreOffice conversion for .doc                         |
| `MarkdownLoader` | .md, .markdown | Regex heading detection, section-per-Document                                   |
| `HTMLLoader`     | http(s), .html | trafilatura + BS4 strategies, metadata extraction                               |
| `CSVLoader`      | .csv           | Row/file strategies, field-value templating                                     |
| `EmailLoader`    | .eml           | MIME-aware, plain-preferred, header metadata, attachment inventory              |
| `YouTubeLoader`  | youtube URLs   | Segment/full modes, timestamped citations, deep-link URLs                       |
| `LoaderRouter`   | *any*        | Composite loader, dispatches by extension or URL scheme                         |

### Normalization stages

| Stage                | Responsibility                                                    |
| -------------------- | ----------------------------------------------------------------- |
| `TextCleaner`      | Unicode NFC + strip invisible/control chars + collapse whitespace |
| `LanguageDetector` | Confidence-gated language detection,`en` fallback               |
| `MetadataEnricher` | Add char/word/line counts                                         |

Order matters: cleaning must precede detection (documented invariant).

---

## 3. Technical deep dive

### 3.1 The "ingestion silently kills quality" pattern

Retrieval failures traced to ingestion typically look like:

- User asks about a table on page 4 → answer says "I don't have context" → table was garbled during extraction
- User asks about content in the sidebar → RAG returns cookie policies → boilerplate wasn't stripped
- Semantically identical strings don't match → Unicode normalization was skipped
- Language-specific content returns nothing → language wasn't detected, routing failed

None of these produce error messages. They produce degraded quality — the failure mode is subtle and diagnosable only with per-format metrics.

### 3.2 The 5 categories of PDF content

| Category         | Example                     | Default extractor                      |
| ---------------- | --------------------------- | -------------------------------------- |
| Plain paragraphs | Reports, articles           | pypdf/pdfplumber both work             |
| Multi-column     | Academic papers, magazines  | pdfplumber with layout detection       |
| Tables           | Financial reports, invoices | pdfplumber.extract_tables() or camelot |
| Images/diagrams  | Presentations               | Extract → VLM caption (M12)           |
| Scanned PDFs     | Legal docs, old archives    | OCR (pytesseract/easyocr, M12)         |

Production RAG needs a **PDF router** — inspect the PDF first, choose the extractor. One-size-fits-all fails.

### 3.3 Multi-format retrieval dynamics

When the corpus spans formats, retrieval develops emergent behaviors:

- **Better content wins.** If two sources contain relevant text, the semantically-closer one is retrieved. This can conflict with an old gold set.
- **Structural formats outperform prose.** CSV rows with field-value templating, emails with subject prefixes, and heading-aware DOCX sections hit higher recall than plain TXT.
- **Chunk-boundary artifacts vary by loader.** PDF pages create natural boundaries; TXT files depend entirely on chunker settings.

### 3.4 Extraction library landscape

| Format  | Options                                            | Winner (2026)                                                          |
| ------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| PDF     | pypdf, pdfplumber, pymupdf, unstructured, camelot  | **pdfplumber** for text+tables, **pymupdf** if GPL is OK   |
| DOCX    | python-docx, docx2txt, mammoth                     | **python-docx** (structure), **mammoth** (HTML conversion) |
| HTML    | trafilatura, readability, boilerpipe, unstructured | **trafilatura** (research standard)                              |
| Email   | stdlib email, mailparser                           | **stdlib** (20 years of production use)                          |
| YouTube | youtube-transcript-api, yt-dlp                     | **youtube-transcript-api** (transcripts only)                    |
| CSV     | pandas, stdlib csv, polars                         | **pandas** (edge cases handled)                                  |

### 3.5 Boilerplate removal quantified

On the Wikipedia RAG article:

- Naive `get_text()`: 22,884 chars — includes nav, citations as line breaks, footer
- Trafilatura: 15,057 chars — clean paragraph flow
- **Delta: 7,827 chars = 34% noise**

That 34% would otherwise sit in your index, get embedded, and compete for retrieval slots against real content.

### 3.6 The confidence-gate pattern

For any "should I trust this model's output" decision, prefer the model's own confidence to a proxy:

- Not "was the text long enough?" but "was the top prediction confident enough?"
- langdetect's `detect_langs()` returns probabilities; use them
- Same principle applies to NER, classification, LLM-as-judge (M8), reranker scores (M6)

This is a general engineering pattern that recurs throughout RAG.

---

## 4. Design decisions and trade-offs

### 4.1 Why per-page/section Documents instead of per-file

- Provenance preserved for citations
- Retrieval granularity matches semantic unit
- Chunker can still further split large sections
- Trade-off: more Documents to manage, slightly higher index cost

### 4.2 Why templated field-value CSV rows

- Column names carry semantic meaning the embedder uses
- Trivial code change (`"col: val | ..."`)
- Trade-off: rows with many empty columns bloat; skip logic mitigates

### 4.3 Why trafilatura as HTML default

- ML-heuristic boilerplate remover, purpose-built for research corpora
- Consistently ranks #1 on extraction benchmarks
- Trade-off: less control than hand-crafted BS4 selectors; strategy='both' preserves the option

### 4.4 Why plain-preferred over HTML in emails

- Plain body is already clean; HTML has boilerplate to strip
- Never embed both — same content twice poisons retrieval
- Trade-off: HTML-only emails need BS4 stripping fallback

### 4.5 Why LibreOffice for .doc → .docx

- Correct, cross-platform, free
- Alternative (writing a binary parser) is months of work
- Trade-off: requires LibreOffice installed; error message points to install page

### 4.6 Why NormalizationPipeline as separate stage

- Enforces canonical form across all loaders
- Composable, testable, extensible (add PII redaction, translation as future stages)
- Trade-off: extra pass over content; cost is negligible

### 4.7 Why segment-mode default for YouTube

- Timestamps are the unique value proposition of video RAG
- Deep-link URLs are citation gold
- Trade-off: many small Documents; chunker downstream normalizes size

---

## 5. Common pitfalls

1. **Naive PDF extraction** — pypdf on multi-column layouts reads across columns, mangling sentence flow
2. **Missing Unicode NFC** — "café" (composed) and "café" (decomposed) treated as different strings, silent retrieval failures
3. **Embedding both plain and HTML email body** — duplicate content poisons embeddings
4. **Aggressive HTML CSS selectors** — over-strip on CMS-heavy sites (Wikipedia's TOC is inside `<nav>`)
5. **Wrong CSV strategy** — treating a sales time-series as row-per-doc means aggregate queries can never be answered
6. **Missing attachment routing** — attachments captured but PDFs inside emails never extracted
7. **Hard-coded language thresholds** — either false positives (short text detected as random) or false negatives (real short text falls back)
8. **Non-deterministic language detection** — langdetect without `DetectorFactory.seed = 0` produces different outputs across runs
9. **Silent OCR skips** — image-only PDF pages disappearing without any log or metadata trace
10. **Gold set drift** — corpus expansion invalidates gold entries; retrieval "regressions" are actually gold-set bugs
11. **Not tagging chunks with source_file** — impossible to compute per-format metrics after the fact
12. **Aggressive whitespace collapse** — collapsing all whitespace to single spaces destroys paragraph boundaries the chunker relies on

---

## 6. Benchmarks and results

### Multi-format baseline (M1)

**Pipeline:** LoaderRouter → NormalizationPipeline → FixedChunker → BGE-small → FAISS flat → VectorRetriever

**Corpus:** 32 chunks across 5 file types (TXT, MD, CSV, EML, PDF)

| Metric   | Score |
| -------- | ----- |
| recall@1 | 0.714 |
| recall@3 | 0.857 |
| MRR      | 0.821 |

### Per-format recall@1

| Format     | recall@1 | Note                              |
| ---------- | -------- | --------------------------------- |
| faq.csv    | 1.000    | Field-value templating wins       |
| sample.eml | 1.000    | Subject-in-content pays off       |
| corpus.txt | 0.500    | Duplicate content elsewhere       |
| sample.md  | 0.500    | recall@3 = 1.0 — reranker signal |

### Interpretation

- **Structured formats retrieve better than prose.** CSV rows and emails with prefixed subjects both hit 100% on their queries.
- **Two "failures" are gold-set staleness**, not retrieval bugs. The CSV FAQ row answered "when does BM25 work well?" better than the original TXT paragraph. Retriever right; gold set stale.
- **recall@1 vs recall@3 gap persists across formats** — same reranker signal from M0, now confirmed with source diversity.

### Comparison with M0

|          | M0 (TXT only) | M1 (multi-format) |
| -------- | ------------- | ----------------- |
| recall@1 | 0.800         | 0.714             |
| recall@3 | 1.000         | 0.857             |
| MRR      | 0.900         | 0.821             |

The drop is largely gold-set staleness, not real regression. Absolute numbers matter less than the diagnostic clarity per-format metrics provide.

---

## 7. Interview mock exam

### Section A — Fundamentals (10 questions)

1. What is the purpose of an ingestion layer in a RAG system?
2. What does a `Loader` return in your architecture?
3. Name three PDF extraction libraries and one distinguishing feature of each.
4. What is Unicode NFC and why does it matter?
5. What is the `content_type` metadata field for?
6. What does trafilatura do that BeautifulSoup doesn't?
7. Why does BGE-small treat "Café" and "Café" differently before NFC?
8. What are the two CSV loading strategies and when do you use each?
9. What is a MIME multi-part email?
10. What is the deep-link URL feature of the YouTube loader?

### Section B — Applied Understanding (15 questions)

11. Your PDF loader extracts tables as separate Documents. What if the "table" is a false positive?
12. Why per-page Documents in the PDF loader instead of per-file?
13. Why is HTML the loudest signal-to-noise problem in ingestion?
14. Your bs4 fallback returned nothing on Wikipedia. What was the root cause?
15. Why prefer text/plain over text/html when extracting email body?
16. How does field-value templating in CSV loading help retrieval?
17. Your CSV file has 500K rows. What in your loader design protects against OOM?
18. Why do you seed langdetect with `DetectorFactory.seed = 0`?
19. Your first `min_chars=50` for language detection produced false positives. Your second attempt with `min_chars=200` produced false negatives. What was the actual fix?
20. Order matters in the normalization pipeline. What breaks if you swap cleaning and detection?
21. How does `LoaderRouter` remain a `Loader` while dispatching to other Loaders?
22. Two gold-set queries "failed" but the retriever returned better answers than expected. Is that a regression?
23. Which of your formats hit 100% recall@1 in the multi-format eval and why?
24. Your normalization pipeline mutates Documents in place. What are the risks?
25. What is the ISO 8601 date format and why does the email loader normalize dates to it?

### Section C — Design and Trade-offs (10 questions)

26. Design an ingestion pipeline for 10 million mixed-format documents. What breaks in your current architecture?
27. A client says "our RAG can't find things in our PDFs." Walk through the diagnostic steps.
28. Your gold set is now stale. Design a maintenance process that scales.
29. You have a corpus with 40% PDF, 30% HTML, 20% DOCX, 10% CSV. Where would you invest engineering time first, and why?
30. Design a system that routes email attachments through the appropriate loader recursively.
31. Compare trafilatura's "content extraction" with LLM-based extraction (e.g., "extract the main article from this HTML"). Trade-offs?
32. Your video RAG is retrieving mostly ASR-mistranscribed segments. What's your fix?
33. Multi-tenancy meets multi-format: a tenant uploads a PDF that references another tenant's data. How do you prevent leakage?
34. Design an ingestion system where "delete this document" actually removes all traces (chunks, embeddings, cached tokens) — GDPR-compliant.
35. How would you eval a boilerplate-removal system? What is the ground truth?

### Section D — Whiteboard Coding (5 questions)

36. Implement a `LoaderRouter._resolve(source: str)` that returns the right Loader class by extension or URL scheme.
37. Implement Unicode NFC normalization + control-char stripping in ≤10 lines.
38. Implement field-value templating for a CSV row: `dict → "col1: val1 | col2: val2"`.
39. Implement a MIME body extractor that prefers text/plain, falls back to HTML with tag-stripping.
40. Implement a per-format recall@1 breakdown given a list of `(query, expected_source, retrieved_chunks)` tuples.

---

## 8. Project walkthrough scripts

### 8.1 The 30-second pitch

> "M1 added seven new loaders to my modular RAG platform — PDF, DOCX, HTML, Markdown, CSV, Email, YouTube — plus a router that auto-dispatches by extension or URL. Every loader emits a normalized `Document` shape, and a normalization pipeline enforces Unicode NFC, language detection, and metadata enrichment before anything hits the chunker. I measure retrieval quality per format through the eval harness — structured formats like CSV and email hit 100% recall@1 while prose formats sit lower, which is the kind of diagnostic clarity that lets you target ingestion work where it matters."

### 8.2 The 2-minute technical walkthrough

> "The ingestion layer has three concerns: reading arbitrary formats, canonicalizing them, and dispatching automatically.
>
> **Loaders.** Each format has its own loader — PDF via pdfplumber with per-page docs and separate table extraction, DOCX with heading-aware sections and LibreOffice fallback for legacy .doc, HTML via trafilatura for automatic boilerplate removal (34% noise reduction on Wikipedia), CSV with row-per-doc and whole-file strategies plus field-value templating, email with MIME-aware body extraction and header metadata, YouTube transcripts with timestamped segments and deep-link URLs. Every loader subclasses the same `Loader` ABC and emits a `Document` with rich metadata.
>
> **Normalization.** Every Document from every loader passes through a normalization pipeline before it reaches the chunker. Unicode NFC canonicalization prevents visually-identical strings from being byte-different (a subtle failure mode that breaks retrieval silently). Confidence-gated language detection replaces a hard length threshold that had false-positive and false-negative failure modes. Metadata enrichment adds char/word/line counts everywhere downstream stages need them.
>
> **Routing.** `LoaderRouter` implements the same `Loader` interface but dispatches by URL scheme or file extension. Downstream code doesn't know or care about routing.
>
> **Measurement.** The eval harness now supports per-format `recall@1` breakdowns. Structured formats (CSV, email) hit 100%. Prose formats sit at 50%, and two of those 'failures' turn out to be gold-set staleness — the retriever returned a better answer from a different source than the gold set specified. That's a production dynamic worth naming: corpus expansion regularly invalidates gold entries."

### 8.3 The 5-minute deep walkthrough

> "Let me start with the architecture, then walk through a specific format's decisions, then show you the measurement.
>
> **Architecture.** The ingestion layer is three stages: `LoaderRouter` → `NormalizationPipeline` → (downstream). The router is a Loader that dispatches to other Loaders — the Composite pattern. Every specific Loader — TXT, PDF, DOCX, Markdown, HTML, CSV, Email, YouTube — subclasses the `Loader` ABC and emits `Document` objects. The normalization pipeline is a sequence of small transformers, each doing one thing: text cleaning, language detection, metadata enrichment.
>
> **A specific loader's design decisions.** Take the PDF loader. First, I use `pdfplumber` over `pypdf` because it's layout-aware — multi-column and tables get handled sensibly. Second, per-page Documents preserve citation provenance. Third, tables are extracted as separate Documents with `content_type: 'table'` — this is the extensibility hook that lets downstream stages filter, boost, or route by type without any reprocessing. Fourth, image-only pages are flagged with `content_type: 'ocr_needed'` instead of silently dropped — a user asking 'why isn't page 7 searchable' gets an answer. Fifth, there's a noise-table filter: real tables have at least 2 rows and 2 meaningful columns, which cuts pdfplumber's high-recall/low-precision false positives.
>
> **A specific normalization decision.** The language detector. My first implementation had `min_chars=50` — short text produced false positives (78 chars of English names detected as Indonesian). I raised to 200 — French text with 158 chars fell back to English incorrectly. Neither threshold works because the problem isn't length, it's confidence. The right fix was langdetect's `detect_langs()` which returns probabilities: try detection on anything ≥50 chars, but reject if confidence is below 0.85. This general pattern — prefer model confidence to input-length proxies — recurs throughout RAG.
>
> **Measurement.** I have a multi-format gold set where each entry has `query`, `relevant_text`, and `expected_source`. The eval script tags every chunk with `source_file` at ingestion, runs the gold set, and reports both aggregate recall/MRR and per-format recall@1. Results: CSV 1.00, email 1.00, TXT 0.50, MD 0.50. And two of those 'failures' were the retriever returning better content from a different source than the gold set specified — a gold-set staleness signal, not a retrieval regression. Multi-format eval isn't just 'does the pipeline still work' — it's 'where should I invest to lift the whole thing.' That diagnostic clarity is the whole point of M1."

---

## 9. Further reading

### Papers / references

- **Unstructured** — [github.com/Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured). A production-oriented document-loader library across ~30 formats; useful as a reference architecture.
- **Trafilatura paper** — Barbaresi 2021, "Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction." ACL demo track.
- **RFC 5322** — Email format spec. Skim for MIME multi-part structure.
- **Unicode UAX #15** — The Unicode normalization annex. Section on NFC vs NFD.
- **Anthropic Contextual Retrieval** (2024) — prepending contextual chunk summaries lifts recall 35%; relevant as an M2 preview but the ingestion implications are useful here too.

### Engineering references

- **pdfplumber docs** — [github.com/jsvine/pdfplumber](https://github.com/jsvine/pdfplumber). Table extraction section especially.
- **LlamaIndex Data Connectors** — good taxonomy of production loader patterns.
- **LangChain Document Loaders** — 100+ loaders; useful for spotting formats you haven't considered.

---

## Milestone status

- [X] 1.1 — PDF loader
- [X] 1.2 — DOCX + Markdown loaders
- [X] 1.3 — HTML/Web loader
- [X] 1.4 — CSV loader
- [X] 1.5 — Email loader
- [X] 1.6 — YouTube transcript loader
- [X] 1.7 — Normalization pipeline
- [X] 1.8 — Multi-format router + eval

**Resume line unlocked (updated with M1):**

> *Built KnowledgeOS, a modular, plugin-based Retrieval-Augmented Generation platform in Python. Implemented an eight-format ingestion layer (TXT, PDF, DOCX, Markdown, HTML, CSV, Email, YouTube) with auto-routing dispatcher, structured metadata extraction (page numbers, section headings, MIME headers, video timestamps), and a Unicode-aware normalization pipeline with confidence-gated language detection. Extended the eval harness to measure per-format retrieval quality, exposing structural strengths (CSV field-value templating and email subject-prepending hit 100% recall@1) and gold-set staleness dynamics. Designed the whole stack behind seven ABCs from M0 so new formats plug in without touching downstream code.*
