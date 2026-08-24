# generation/prompt_builder.py

from core import Chunk


SYSTEM_PROMPT = """You are a precise, citation-driven assistant. \
Answer questions using ONLY the provided context passages.

Rules:
1. Cite every factual claim using [N] where N is the passage number.
2. If the answer is not in the context, say: "The provided context does not contain information about this."
3. Do not add facts from your own knowledge — only use the context.
4. Be concise and direct. One paragraph unless the question requires more."""


def build_prompt(query: str, chunks: list[Chunk]) -> list[dict]:
    """
    Build a chat-format prompt for RAG generation.

    Returns a list of message dicts compatible with the OpenAI
    chat completions API (and Ollama's compatible endpoint).

    Context passages are numbered [1], [2], ... for inline citation.
    chunk_id is included as metadata for downstream citation resolution.

    Parameters
    ----------
    query  : str         the user's question
    chunks : list[Chunk] retrieved and reranked context passages
    """
    if not chunks:
        context_block = "No context passages were retrieved."
    else:
        passages = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", chunk.doc_id)
            passages.append(
                f"[{i}] (source: {source}, chunk_id: {chunk.chunk_id[:8]})\n"
                f"{chunk.content.strip()}"
            )
        context_block = "\n\n".join(passages)

    user_message = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer (cite passages with [N]):"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]


def extract_citations(answer: str, chunks: list[Chunk]) -> dict[int, Chunk]:
    """
    Parse [N] citations from an answer and map to source chunks.

    Returns {citation_number: Chunk} for every [N] that appears
    in the answer and has a corresponding chunk.

    Example:
        answer = "BM25 is sparse [1]. Dense uses embeddings [2]."
        → {1: chunk_0, 2: chunk_1}
    """
    import re
    cited    = {}
    numbers  = re.findall(r"\[(\d+)\]", answer)
    for num_str in numbers:
        n = int(num_str)
        if 1 <= n <= len(chunks) and n not in cited:
            cited[n] = chunks[n - 1]
    return cited