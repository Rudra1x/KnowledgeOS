# generation/generator.py

import os
import json
import requests
from dotenv import load_dotenv
from core import Chunk, Generator
from .prompt_builder import build_prompt


# Load .env from repo root on first import.
# override=False means real env vars win — safe for CI/production.
load_dotenv(override=False)


class OpenRouterGenerator(Generator):
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model:       str   = "openrouter/free",
        max_tokens:  int   = 512,
        temperature: float = 0.0,
        reasoning:   bool  = True,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.reasoning   = reasoning

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found. "
                "Add it to a .env file at repo root, or export it as an env var."
            )
        self.api_key = api_key

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        messages = build_prompt(query, chunks)

        payload = {
            "model":       self.model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }
        if self.reasoning:
            payload["reasoning"] = {"enabled": True}

        response = requests.post(
            url     = self.URL,
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
            data    = json.dumps(payload),
            timeout = 120,
        )
        response.raise_for_status()
        data = response.json()

        if "choices" not in data:
            raise RuntimeError(f"OpenRouter error: {data}")

        return data["choices"][0]["message"]["content"].strip()