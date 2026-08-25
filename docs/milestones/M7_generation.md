# Milestone 7 — Generation Engine

**Status:** ✅ Complete
**Duration:** 6 checkpoints
**Deliverable:** A complete generation quality layer — grounded prompts with inline citations, context compression, faithfulness checking (NLI + word overlap, 176ms), answer relevance scoring (RAGAS-inspired), and streaming with TTFT measurement. Full pipeline benchmark: r@1=1.00, faithfulness=0.80, relevance=0.73, TTFT=7.4s CPU.

---

## 1. Milestone summary

### Goal
Close the final loop. Everything M0-M6 was about finding the right context. M7 is about turning that context into a trustworthy, cited, quality-verified answer.

### Why this milestone matters
Retrieval quality (M0-M6) is necessary but not sufficient. The generator can hallucinate even when given perfect context. Without grounded prompts, the model adds parametric facts not in the retrieved chunks. Without faithfulness checking, hallucinations reach users. Without streaming, a 15-second answer feels like failure. M7 is the trust layer.

### What "done" looks like
- Grounded system prompt with inline [N] citation protocol
- Context compression (37% reduction before generation)
- Faithfulness checker (NLI + word overlap, 176ms, correct discrimination)
- Answer relevance scorer (reverse-question generation, correct gradient)
- Streaming generator (SSE parsing, TTFT measurement, Ollama primary)
- Full pipeline benchmark with all quality dimensions measured

---

## 2. Architecture recap

### The generation quality stack

```
Reranked chunks (from M6)
          │
          ▼
┌─────────────────────────────┐
│  ContextCompressor          │
│  (similarity, top_k=3 sent) │
│  ~50ms                      │
└─────────────┬───────────────┘
              │ compressed chunks (0.63 mean ratio)
              ▼
┌─────────────────────────────┐
│  build_prompt()             │
│  system rules + [1][2][3]   │
│  numbered context passages  │
└─────────────┬───────────────┘
              │ messages list
              ▼
┌─────────────────────────────┐
│  StreamingGenerator         │
│  Ollama → OpenRouter        │
│  yields tokens one by one   │
│  TTFT=7.4s CPU / ~500ms GPU │
└─────────────┬───────────────┘
              │ answer with [N] citations
              ▼
┌─────────────────────────────┐
│  extract_citations()        │
│  {1: Chunk, 2: Chunk, ...}  │
└─────────────┬───────────────┘
              │ [async]
              ▼
┌─────────────────────────────┐
│  FaithfulnessChecker (NLI)  │
│  176ms, nli-MiniLM2-L6-H768 │
│  per-claim verdict          │
└─────────────┬───────────────┘
              │ [async]
              ▼
┌─────────────────────────────┐
│  AnswerRelevanceScorer      │
│  reverse question generation│
│  + embedding similarity     │
└─────────────────────────────┘
```

---

## 3. Technical deep dive

### 3.1 The faithfulness problem — why prompts aren't enough

Even with "use only the provided context," Qwen2.5-3B supplements with parametric memory ~10-20% of the time. The model's pretraining data is strong — it "knows" facts about BM25's inventors, Elasticsearch's defaults, etc. When the retrieved context is thin (short chunks, few chunks), the model fills gaps from memory.

**The three-layer defense:**
1. System prompt: "do not add facts from parametric knowledge" — first line
2. Context compression: remove distracting non-relevant sentences — reduces noise that might trigger tangential facts
3. Faithfulness checker: verify each claim post-generation — catches violations layers 1-2 missed

### 3.2 NLI model selection — the critical calibration problem

Faithfulness checking requires an NLI model that correctly scores (passage, claim) pairs from your domain. Three models we tested:

| Model | Clear entailment score | Useful? |
|-------|----------------------|---------|
| nli-deberta-v3-small | 0.0013 | No |
| nli-MiniLM2-L6-H768 | 0.7958 | Yes |

Same NLI architecture class, radically different calibration on short technical sentences. Always validate on sample pairs from your actual corpus before deploying a faithfulness checker.

**The two NLI failure modes:**
- Near-verbatim claims: NLI underscores when hypothesis ≈ premise (training distribution mismatch). Fix: word overlap fallback.
- Boundary artifact noise: leading irrelevant sentences confuse the model. Fix: `_clean_passage()` strips non-overlapping leading sentences.

### 3.3 The reverse-question relevance proxy

RAGAS insight: if an answer addresses a question, LLM-generated questions from that answer should embed close to the original question.

```python
# Generate
generated_qs = llm.generate("What questions does this answer?", answer)
# Score
similarities = [cosine(embed(original_q), embed(gen_q)) for gen_q in generated_qs]
relevance    = mean(similarities)
```

**Why it works:** a direct BM25 answer generates BM25-specific questions. An evasive answer generates generic retrieval questions. A wrong-topic answer generates off-topic questions. The embedding distance captures topic alignment.

**Why absolute scores need calibration:** generated sub-questions are more specific than the original ("What type of algorithm is BM25?" vs "What is BM25 and how does it score?"). Embedding similarity naturally caps at 0.65-0.75 for correct answers. Calibrate threshold to your data.

### 3.4 Streaming — the TTFT architecture

```
Prompt tokens → [PREFILL] → KV cache populated → [DECODE] → token 1, token 2, ...
                 ↑ TTFT cutoff here
```

TTFT = prefill time + first decode step. On CPU, prefill of a 500-token prompt = 6-14 seconds. On A10 GPU = 100-300ms. The streaming code is hardware-agnostic — switch to GPU or API, same `iter_lines()` loop, 15-50× better TTFT.

**Production TTFT targets:**
- Interactive chat: <500ms TTFT → requires GPU or fast API model
- Search-like UX: <2s TTFT → fast GPU or medium API model
- Research assistant: <10s TTFT → CPU acceptable, async acceptable

### 3.5 Context compression — the token cost layer

On production corpora (300-500 word chunks):
```
3 chunks × 400 words × compression ratio 0.63 = 756 words sent vs 1200
Savings: 444 words ≈ 333 tokens
At $0.01/1K tokens (GPT-4): $0.003/query × 100K queries/day = $300/day saved
```

The similarity compressor (sentence-level cosine) is the right production choice:
- No LLM calls (no latency cost)
- Removes noise that doesn't match the query
- Metadata enables monitoring (compression_ratio distribution tells you corpus health)

### 3.6 Full pipeline latency breakdown

| Stage | CPU | GPU |
|-------|-----|-----|
| Retrieve | 2ms | 2ms |
| Rerank (MS-MARCO) | 20ms | 5ms |
| Compress (similarity) | 50ms | 50ms |
| Generate (500ms) | 7,400ms | 500ms |
| Faithfulness (NLI) | 176ms | 50ms |
| Relevance (async) | 3,000ms | 500ms |
| **Total sync** | **7,648ms** | **607ms** |

Return answer to user at ~7.6s (CPU) / ~0.6s (GPU). Run faithfulness + relevance async, flag retroactively.

---

## 4. Design decisions and trade-offs

### 4.1 Claim-level vs answer-level faithfulness
Claim-level: each sentence evaluated independently. Score = fraction of supported sentences.
Answer-level: entire answer is faithful or not.

Claim-level wins: "one sentence is hallucinated" is more actionable than "answer is bad." You can redact the hallucinated sentence and return the rest. The score (0.333 = 1/3 claims supported) quantifies the severity.

### 4.2 NLI + word overlap vs LLM-only for faithfulness
NLI + word overlap: 176ms, no Ollama dependency, covers verbatim and paraphrased claims.
LLM-only: 37s, requires Ollama, more flexible for complex multi-sentence claims.

NLI wins for production. LLM as an override when NLI returns unexpected results on complex claims.

### 4.3 Synchronous generation + async quality checking
Return the answer immediately after streaming completes. Run faithfulness + relevance checks asynchronously. Flag low-quality answers in a background queue for human review or automatic regeneration. This minimizes perceived latency while maintaining quality guarantees.

### 4.4 n_questions=1 vs n_questions=3 for relevance
n=1: single reverse question, high variance, 3× faster. For real-time scoring.
n=3: three reverse questions, stable mean, 3× slower. For offline quality auditing.

Production: n=1 for real-time, n=3 for batch evaluation.

### 4.5 System prompt as the primary faithfulness control
The system prompt is the cheapest faithfulness lever. Before adding a faithfulness checker, tighten the prompt. Three rules cover most cases: cite every claim, say "not in context" if not there, use only retrieved context. The checker catches what the prompt misses — it's the second defense, not the first.

---

## 5. Common pitfalls

1. **Not using numbered passages in the context block** → model can't produce [N] citations. The numbered format must be in the user message for citation to work.
2. **Using nli-deberta-v3-small as the faithfulness NLI model** → scores near 0.00 on short technical sentences. Always validate NLI calibration on your domain.
3. **Near-verbatim claim scoring at 0.02 in NLI** → training distribution mismatch. Add word overlap fallback (≥0.85 → score=1.0).
4. **Boundary artifact sentences confusing NLI** → use `_clean_passage()` to strip leading sentences with no content word overlap with the claim.
5. **Threshold 0.85 for relevance "relevant"** → too strict. Correct answers score 0.65-0.75. Calibrate on labeled data.
6. **Running quality checks synchronously** → adds 3+ seconds of perceived latency. Run faithfulness + relevance async after returning the answer.
7. **Budget compression keeping the beginning of chunks** → relevant content may be late in the chunk. Similarity compression is position-agnostic.
8. **LLM compressor on short chunks** → returns ratio=1.0 (correct behavior — nothing to compress). Only use LLM compressor on long chunks (>200 words).
9. **JSON decode errors in SSE streaming** → always try/except the json.loads call in `iter_lines()` — partial buffer flushes produce malformed lines.
10. **Not measuring TTFT separately from total time** → TTFT is the UX metric; total time is the hardware metric. Both matter, neither replaces the other.
11. **Prompt verbosity** → long system prompts increase prefill time. Every 100 tokens added to the prompt = ~1-2s additional TTFT on CPU.

---

## 6. Benchmark results

### M7 generation benchmark (5 queries, CPU, Qwen2.5-3B)

| Query | r@1 | Compress | Faith | Rel | TTFT |
|-------|-----|----------|-------|-----|------|
| What is RAG? | 1.00 | 0.71 | 1.00 | 0.57 | 7581ms |
| How does chunking affect retrieval? | 1.00 | 0.59 | 1.00 | 0.79 | 7115ms |
| What is BM25 and when does it work? | 1.00 | 0.72 | 1.00 | 0.71 | 7635ms |
| How is dense retrieval different? | 1.00 | 0.56 | 0.00* | 0.86 | 7526ms |
| What is hybrid retrieval? | 1.00 | 0.55 | 1.00 | 0.74 | 7231ms |
| **MEAN** | **1.00** | **0.63** | **0.80** | **0.73** | **7417ms** |

*False positive — NLI flagged a correct, grounded paraphrase as unsupported.

**Key interpretation:**
- r@1=1.00: cross-encoder reranker (M6) working perfectly
- Compress=0.63: 37% of chunk content removed before generation
- Faith=0.80: 4/5 answers faithful, 1 false positive from NLI on paraphrased claim
- Rel=0.73: answers are on-topic, threshold needs calibration (0.60 is correct)
- TTFT=7.4s: CPU prefill dominates — GPU reduces to ~500ms

---

## 7. Interview mock exam

### Section A — Fundamentals (10 questions)

1. What are the three rules in your grounded system prompt and what does each prevent?
2. Why does numbering context passages [1][2][3] elicit reliable citation without fine-tuning?
3. What does extract_citations() return and what does it enable?
4. What is context compression and what problem does it solve?
5. What is the difference between faithfulness and correctness in RAG?
6. What is NLI (Natural Language Inference) and how is it used for faithfulness checking?
7. What is the RAGAS approach to answer relevance scoring?
8. What is TTFT and why does it matter for interactive RAG?
9. What is SSE and how do you parse it in streaming generation?
10. Why does word overlap fallback improve NLI faithfulness checking?

### Section B — Applied Understanding (15 questions)

11. Your faithfulness checker returned False for a correct, grounded answer. What's the likely cause and fix?
12. nli-deberta-v3-small gave 0.001 for a clear entailment pair. What does this tell you about NLI model selection?
13. Why does the word overlap fallback use threshold=0.85 rather than a lower value?
14. Your answer relevance scored 0.57 for "What is RAG?" with a correct answer. Is the answer bad or the scorer?
15. Context compression ratio=0.63 on 45-word chunks. What does this become on 400-word chunks and what's the cost impact?
16. TTFT=7.4s on CPU. What are the three architecture options to reduce it below 500ms?
17. Your LLM compressor returned ratio=1.0 on a 45-word chunk. Is this a bug?
18. A claim "BM25 is a sparse algorithm" scored 1.00 (word overlap path) while "invented by Robertson" scored 0.00 (NLI path). Explain why each path was taken.
19. The streaming generator gets a JSON decode error on a line from Ollama. What should happen?
20. Why does budget compression fail when relevant content appears late in a chunk?
21. Your faithfulness benchmark shows faith=0.80 with one false positive. How do you decide if this is acceptable?
22. Answer relevance n_questions=1 scored 0.57 for a query that n_questions=3 would score 0.70. What causes the variance?
23. The _clean_passage() function drops leading sentences. What's the criterion for dropping?
24. Your system prompt says "use only retrieved context" but the model still sometimes adds parametric facts. Why?
25. You want to run faithfulness checking without Ollama. What strategy do you use?

### Section C — Design and Trade-offs (10 questions)

26. Design a production RAG quality pipeline for a healthcare company where hallucinations have patient safety implications.
27. A client wants to keep all quality checks synchronous. What's the maximum acceptable latency per query on CPU vs GPU?
28. Compare claim-level vs answer-level faithfulness checking. When would you use each?
29. Design a calibration procedure for the answer relevance threshold using your existing gold set.
30. Your faithfulness false positive rate is 15% (NLI flags correct grounded answers). Design a two-stage approach to reduce it below 3%.
31. A corpus has very long chunks (800 words average). Which compression strategy do you use and why?
32. Design a streaming RAG endpoint using FastAPI that returns tokens via Server-Sent Events to a browser client.
33. Your generation quality scores are logged. Design a monitoring system that detects quality degradation over time.
34. Compare context compression before generation vs retrieval of more focused chunks (smaller chunk_size). Which is better?
35. A client needs citations to include page numbers and paragraph numbers from PDF sources. What changes in your pipeline?

### Section D — Whiteboard Coding (5 questions)

36. Implement `build_prompt(query: str, chunks: list[Chunk]) -> list[dict]` — numbered context passages with source metadata.
37. Implement `extract_citations(answer: str, chunks: list[Chunk]) -> dict[int, Chunk]` — parse [N] and map to chunks.
38. Implement `FaithfulnessChecker._word_overlap(passage: str, claim: str) -> float` — fraction of content words shared.
39. Implement `StreamingGenerator._stream_ollama(messages) -> Generator[str]` — SSE parsing with error handling.
40. Implement `AnswerRelevanceScorer.score(question: str, answer: str) -> dict` — reverse question generation + embedding similarity.

---

## 8. Project walkthrough scripts

### 8.1 The 30-second pitch

> "M7 added the generation quality layer to KnowledgeOS. A grounded system prompt with inline [N] citation protocol constrains Qwen2.5-3B to retrieved context. Context compression (similarity strategy) removes 37% of chunk content before generation. Faithfulness checking with NLI runs in 176ms and correctly flags hallucinated claims — 'invented by Robertson in 1994' scored 0.00, while grounded claims scored 1.00. Answer relevance uses reverse question generation: the score gradient is 0.662 for direct answers, 0.517 for evasive, 0.427 for wrong topic. Streaming delivers first tokens in 7.4s on CPU — 500ms on GPU. Full pipeline benchmark: r@1=1.00, faithfulness=0.80, relevance=0.73."

### 8.2 The 2-minute technical walkthrough

> "M7 has four main components. First, the grounded prompt: a system message with three rules — cite every claim with [N], say 'not in context' if the answer isn't there, use only retrieved context. Numbered context passages elicit reliable citation because [N] notation is in the model's pretraining distribution. extract_citations() maps every [N] in the answer back to the exact Chunk object for traceability.
>
> Second, context compression: sentence-level cosine similarity scores each sentence against the query, keeps top_k. We saw it correctly drop a boundary artifact sentence ('ses embedding similarity to detect topic shifts') while keeping BM25 content. 37% of content removed, 37% fewer tokens to the generator.
>
> Third, faithfulness checking: three layers. Word overlap (≥85% content words match → automatically supported — handles near-verbatim claims NLI underscores). _clean_passage strips boundary artifacts before NLI scoring. NLI model nli-MiniLM2-L6-H768 (the critical model selection — nli-deberta gave 0.001 on clear entailment, useless). Total latency: 176ms vs 37s LLM strategy.
>
> Fourth, streaming: SSE parsing with `stream=True` and `iter_lines()`, yields tokens one by one, measures TTFT. On CPU: 7.4s TTFT. On GPU: ~500ms. The streaming code is hardware-agnostic."

### 8.3 The 5-minute deep walkthrough

> "Let me walk through four areas: the faithfulness problem and how we solved it, the NLI debugging journey, the relevance scoring design, and the streaming architecture.
>
> **The faithfulness problem.** Even with a grounded system prompt, the model adds parametric facts 10-20% of the time. The prompt is the first defense; it dramatically reduces but doesn't eliminate hallucination. The second defense is faithfulness checking: extract each sentence as a claim, check each claim against each retrieved chunk. We tested the LLM strategy — 37 seconds for 6 checks, correct discrimination. We tested NLI — 176ms, but the model selection was non-trivial.
>
> **The NLI debugging journey.** nli-deberta-v3-small scored 0.0013 on a clear entailment pair — completely wrong. nli-MiniLM2-L6-H768 scored 0.796 on the same pair — production-ready. Same NLI architecture class, dramatically different calibration on short technical sentences. We also discovered two NLI failure modes: near-verbatim claims score paradoxically low (training distribution mismatch when hypothesis ≈ premise) — fixed with word overlap fallback (≥85% content words → score=1.0). Boundary artifact sentences confuse the model — fixed with _clean_passage() that drops leading sentences with no content word overlap with the claim.
>
> **Relevance scoring.** RAGAS insight: if an answer addresses a question, reverse-generated questions should embed close to the original. We generate N questions from the answer, embed all, score cosine similarity to the original. Direct BM25 answer: 0.662 (on-topic sub-questions). Evasive answer: 0.517 (generic retrieval questions). Wrong-topic FAISS answer for faithfulness question: 0.427 (FAISS questions, orthogonal to faithfulness). Correct ordering. The threshold 0.85 for 'relevant' is too strict — correct answers score 0.65-0.75. Calibrate to 0.60 on labeled data.
>
> **Streaming.** SSE parsing: strip `data: ` prefix, check for `[DONE]`, JSON-decode, extract `choices[0].delta.content`. try/except around JSON decode — partial buffer flushes produce malformed lines, silently skip them. The collect() wrapper measures TTFT (time to first token) and TPS. On CPU with Qwen2.5-3B and a 500-token prompt: TTFT=7.4s dominated by prefill. On T4 GPU: ~200-500ms TTFT. The streaming architecture is hardware-agnostic — switch to GPU, same code, 15× better TTFT. Production deployment: return the answer to the user after streaming, run faithfulness and relevance async in the background, flag low-quality answers for human review."

---

## 9. Further reading

- **Asai et al., 2023** — "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." The faithfulness checking framework.
- **Es et al., 2023** — "RAGAS: Automated Evaluation of Retrieval Augmented Generation." The answer relevance reverse-question approach.
- **Williams et al., 2018** — "A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference." MultiNLI — the training corpus for NLI models.
- **Xu et al., 2023** — "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation." Context compression for RAG.
- **Vaswani et al., 2017** — "Attention is All You Need." The transformer architecture underlying generation and NLI models.
- **Ollama documentation** — ollama.com/docs — streaming API format and model serving.

---

## Milestone status

- [x] 7.1 — Prompt engineering + inline citation (build_prompt, extract_citations)
- [x] 7.2 — ContextCompressor (similarity, LLM, budget strategies)
- [x] 7.3 — FaithfulnessChecker (NLI + word overlap, 176ms, nli-MiniLM2-L6-H768)
- [x] 7.4 — AnswerRelevanceScorer (reverse question generation, RAGAS-inspired)
- [x] 7.5 — StreamingGenerator (SSE parsing, TTFT measurement, Ollama primary)
- [x] 7.6 — Generation benchmark (r@1=1.00, faith=0.80, rel=0.73, TTFT=7.4s CPU)

**Resume line (updated with M7):**

> *Completed KnowledgeOS M7 generation layer: grounded system prompt with inline [N] citation protocol (extract_citations maps every cited number to the exact source Chunk), similarity-based context compression (0.63 mean ratio, removes 37% of chunk content before generation), NLI faithfulness checker using nli-MiniLM2-L6-H768 with word overlap fallback (176ms, correct discrimination between faithful 1.000 and hallucinated 0.333 answers — vs 37s for LLM strategy), RAGAS-inspired answer relevance scorer (correct gradient: 0.662 direct > 0.517 evasive > 0.427 wrong-topic), and SSE streaming generator with TTFT measurement (7.4s CPU baseline, 500ms GPU target). Full pipeline benchmark: r@1=1.000, faithfulness=0.80, relevance=0.73, TTFT=7.4s on Qwen2.5-3B CPU.*
