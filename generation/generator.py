# generation/generator.py

import os
import json
import requests
from core import Chunk, Generator
from .prompt_builder import build_prompt


class OpenRouterGenerator(Generator):
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model:       str   = "nvidia/nemotron-3-ultra-550b-a55b:free",
        max_tokens:  int   = 512,
        temperature: float = 0.0,
        reasoning:   bool  = True,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.reasoning   = reasoning
        self.api_key     = os.environ["OPENROUTER_API_KEY"]

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

        # Defensive: OpenRouter sometimes returns errors as 200-status JSON
        if "choices" not in data:
            raise RuntimeError(f"OpenRouter error: {data}")

        return data["choices"][0]["message"]["content"].strip()