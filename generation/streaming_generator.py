# generation/streaming_generator.py

import json
import time
import requests
from typing import Generator as GenType
from dotenv import load_dotenv
from core import Chunk
from generation.prompt_builder import build_prompt

load_dotenv(override=False)


class StreamingGenerator:
    """
    Token-by-token streaming generator.

    Streams from Ollama (primary) or OpenRouter (fallback).
    Yields tokens as they arrive — caller iterates the generator.

    Usage:
        for token in streamer.stream(query, chunks):
            print(token, end="", flush=True)

    Also supports collect() for non-streaming use with timing:
        result = streamer.collect(query, chunks)
        print(result["answer"])
        print(f"TTFT: {result['ttft_ms']:.0f}ms")

    Parameters
    ----------
    local_model    : str
    fallback_model : str
    max_tokens     : int
    temperature    : float
    """

    NAME = "streaming_generator"

    OLLAMA_URL     = "http://localhost:11434/v1/chat/completions"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        local_model:    str   = "qwen2.5:3b-instruct",
        fallback_model: str   = "openrouter/free",
        max_tokens:     int   = 512,
        temperature:    float = 0.0,
    ):
        self.local_model    = local_model
        self.fallback_model = fallback_model
        self.max_tokens     = max_tokens
        self.temperature    = temperature

        import os
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

        self.last_ttft_ms:  float = 0.0   # time to first token
        self.last_total_ms: float = 0.0   # total generation time
        self.last_n_tokens: int   = 0     # tokens generated

    def stream(
        self,
        query:  str,
        chunks: list[Chunk],
    ) -> GenType[str, None, None]:
        """
        Yield tokens one by one as they arrive from the LLM.

        Usage:
            for token in generator.stream(query, chunks):
                print(token, end="", flush=True)
        """
        messages = build_prompt(query, chunks)

        try:
            yield from self._stream_ollama(messages)
        except requests.exceptions.ConnectionError:
            print("\n  [Streaming] Ollama not reachable — falling back to OpenRouter")
            yield from self._stream_openrouter(messages)
        except Exception as e:
            print(f"\n  [Streaming] Ollama failed ({e}) — falling back to OpenRouter")
            yield from self._stream_openrouter(messages)

    def collect(
        self,
        query:  str,
        chunks: list[Chunk],
    ) -> dict:
        """
        Stream internally and collect full answer + timing metrics.

        Returns:
        {
            "answer":    str    full generated answer
            "ttft_ms":   float  time to first token (ms)
            "total_ms":  float  total generation time (ms)
            "n_tokens":  int    approximate token count
            "tps":       float  tokens per second
        }
        """
        t_start  = time.perf_counter()
        tokens   = []
        ttft_ms  = None

        for token in self.stream(query, chunks):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t_start) * 1000
            tokens.append(token)

        total_ms = (time.perf_counter() - t_start) * 1000
        answer   = "".join(tokens)
        n_tokens = len(answer.split())   # approximate

        self.last_ttft_ms  = ttft_ms or 0.0
        self.last_total_ms = total_ms
        self.last_n_tokens = n_tokens

        return {
            "answer":   answer,
            "ttft_ms":  round(ttft_ms or 0.0, 1),
            "total_ms": round(total_ms, 1),
            "n_tokens": n_tokens,
            "tps":      round(n_tokens / (total_ms / 1000), 1) if total_ms > 0 else 0,
        }

    # ------------------------------------------------------------------

    def _stream_ollama(self, messages: list[dict]) -> GenType[str, None, None]:
        """Stream tokens from Ollama."""
        resp = requests.post(
            url     = self.OLLAMA_URL,
            headers = {"Content-Type": "application/json"},
            data    = json.dumps({
                "model":       self.local_model,
                "messages":    messages,
                "max_tokens":  self.max_tokens,
                "temperature": self.temperature,
                "stream":      True,
            }),
            stream  = True,
            timeout = 120,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                data    = json.loads(line)
                delta   = (data.get("choices", [{}])[0]
                               .get("delta", {})
                               .get("content", ""))
                if delta:
                    yield delta
            except json.JSONDecodeError:
                continue

    def _stream_openrouter(self, messages: list[dict]) -> GenType[str, None, None]:
        """Stream tokens from OpenRouter."""
        if not self.openrouter_key:
            yield "I cannot generate a response — no LLM available."
            return

        resp = requests.post(
            url     = self.OPENROUTER_URL,
            headers = {"Authorization": f"Bearer {self.openrouter_key}",
                       "Content-Type": "application/json"},
            data    = json.dumps({
                "model":       self.fallback_model,
                "messages":    messages,
                "max_tokens":  self.max_tokens,
                "temperature": self.temperature,
                "stream":      True,
            }),
            stream  = True,
            timeout = 60,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() == "[DONE]":
                break
            try:
                data  = json.loads(line)
                delta = (data.get("choices", [{}])[0]
                             .get("delta", {})
                             .get("content", ""))
                if delta:
                    yield delta
            except json.JSONDecodeError:
                continue