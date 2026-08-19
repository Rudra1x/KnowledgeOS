# generation/local_generator.py

import json
import requests
from core import Chunk, Generator
from .prompt_builder import build_prompt


class LocalLLMGenerator(Generator):
    """
    Generator that calls a local Ollama server first,
    falls back to OpenRouter if Ollama is unavailable.

    Ollama exposes an OpenAI-compatible endpoint at:
        http://localhost:11434/v1/chat/completions

    No API key required for Ollama.
    Falls back to OpenRouter automatically on connection error.

    Parameters
    ----------
    local_model    : str   Ollama model name (e.g. 'qwen2.5:3b-instruct')
    fallback_model : str   OpenRouter model string
    local_url      : str   Ollama base URL
    max_tokens     : int
    temperature    : float
    """

    NAME = "local_llm"

    OLLAMA_URL    = "http://localhost:11434/v1/chat/completions"
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

        # Load OpenRouter key for fallback (optional)
        import os
        from dotenv import load_dotenv
        load_dotenv(override=False)
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        messages = build_prompt(query, chunks)

        # Try local Ollama first
        try:
            return self._call_ollama(messages)
        except requests.exceptions.ConnectionError:
            print("  [LocalLLM] Ollama not reachable — falling back to OpenRouter")
        except Exception as e:
            print(f"  [LocalLLM] Ollama failed ({e}) — falling back to OpenRouter")

        # Fallback to OpenRouter
        return self._call_openrouter(messages)

    def call_raw(self, prompt: str) -> str:
        """
        Direct string prompt — bypasses RAG prompt builder.
        Used by query rewriting, multi-query, Self-RAG, CRAG.
        Returns raw LLM output string.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._call_ollama(messages)
        except Exception:
            return self._call_openrouter_raw(messages)

    # ------------------------------------------------------------------

    def _call_ollama(self, messages: list[dict]) -> str:
        resp = requests.post(
            url     = self.OLLAMA_URL,
            headers = {"Content-Type": "application/json"},
            data    = json.dumps({
                "model":       self.local_model,
                "messages":    messages,
                "max_tokens":  self.max_tokens,
                "temperature": self.temperature,
                "stream":      False,
            }),
            timeout = 60,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = (data.get("choices", [{}])[0]
                      .get("message", {})
                      .get("content", ""))
        return content.strip()

    def _call_openrouter(self, messages: list[dict]) -> str:
        if not self.openrouter_key:
            return "I cannot generate a response — no LLM available."
        resp = requests.post(
            url     = self.OPENROUTER_URL,
            headers = {"Authorization": f"Bearer {self.openrouter_key}",
                       "Content-Type": "application/json"},
            data    = json.dumps({
                "model":       self.fallback_model,
                "messages":    messages,
                "max_tokens":  self.max_tokens,
                "temperature": self.temperature,
            }),
            timeout = 60,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = (data.get("choices", [{}])[0]
                      .get("message", {})
                      .get("content", ""))
        return content.strip()

    def _call_openrouter_raw(self, messages: list[dict]) -> str:
        if not self.openrouter_key:
            return ""
        try:
            return self._call_openrouter(messages)
        except Exception:
            return ""