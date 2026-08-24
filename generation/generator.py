# generation/generator.py

import os
import json
import requests
from dotenv import load_dotenv
from core import Chunk, Generator
from .prompt_builder import build_prompt

load_dotenv(override=False)


class OpenRouterGenerator(Generator):
    """
    Generator using OpenRouter API (Claude, GPT-4, Llama, etc.)

    Uses the updated prompt_builder with system prompt + citation instructions.

    Parameters
    ----------
    model       : str   OpenRouter model string
    max_tokens  : int
    temperature : float
    reasoning   : bool  enable extended thinking (for supported models)
    """

    NAME = "openrouter"

    def __init__(
        self,
        model:       str   = "openrouter/free",
        max_tokens:  int   = 512,
        temperature: float = 0.0,
        reasoning:   bool  = False,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.reasoning   = reasoning
        self.api_key     = os.environ.get("OPENROUTER_API_KEY", "")

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if not self.api_key:
            return "[OpenRouter] No API key — set OPENROUTER_API_KEY in .env"

        messages = build_prompt(query, chunks)
        payload  = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(
                url     = "https://openrouter.ai/api/v1/chat/completions",
                headers = {"Authorization": f"Bearer {self.api_key}",
                           "Content-Type": "application/json"},
                data    = json.dumps(payload),
                timeout = 60,
            )
            resp.raise_for_status()
            data    = resp.json()
            content = (data.get("choices", [{}])[0]
                          .get("message", {})
                          .get("content", ""))
            return content.strip()
        except Exception as e:
            return f"[OpenRouter] Generation failed: {e}"