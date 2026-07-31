# generation/prompt_builder.py

from core import Chunk


SYSTEM_PROMPT = """You are a precise question-answering assistant.
Answer the user's question using ONLY the provided context chunks.
At the end of every factual claim, cite the chunk number like [1] or [2].
If the context does not contain enough information, say "I don't have enough context to answer this."
Do NOT use any knowledge outside the provided context."""


def build_prompt(query: str, chunks: list[Chunk]) -> list[dict]:
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        context_lines.append(f"[{i}] {chunk.content.strip()}")

    context_block = "\n\n".join(context_lines)

    user_message = f"""Context:
{context_block}

Question: {query}"""

    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]