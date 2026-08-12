# embedders/api_embedder.py

import os
import time
import json
import requests
import numpy as np
from dotenv import load_dotenv
from core import Embedder

load_dotenv(override=False)


class APIEmbedder(Embedder):
    """
    API-based embedder using the OpenAI-compatible /v1/embeddings endpoint.

    Works with:
    - OpenAI:      base_url="https://api.openai.com/v1"
    - OpenRouter:  base_url="https://openrouter.ai/api/v1"
    - Cohere:      base_url="https://api.cohere.ai/v2"  (slightly different schema)
    - Voyage:      base_url="https://api.voyageai.com/v1"
    - Local vLLM:  base_url="http://localhost:8000/v1"

    Key difference from local models:
    - No GPU, no model in memory, no warm-up
    - Stateless HTTP calls — perfect for serverless
    - Expensive at scale (pay per token)
    - Rate-limited — we batch carefully and respect limits

    Dimension is inferred from the first API call (different models differ).
    """

    NAME = "api"

    def __init__(
        self,
        model_name:  str        = "text-embedding-3-small",
        base_url:    str        = "https://api.openai.com/v1",
        api_key_env: str        = "OPENAI_API_KEY",
        batch_size:  int        = 100,      # OpenAI allows up to 2048 per call
        timeout:     int        = 60,
        dimensions:  int | None = None,     # Some models support Matryoshka (truncatable dims)
    ):
        self.model_name  = model_name
        self.base_url    = base_url.rstrip("/")
        self.batch_size  = batch_size
        self.timeout     = timeout
        self.dimensions  = dimensions
        self.last_embed_ms: float = 0.0

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"API key not found. Set {api_key_env} in your .env file."
            )
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

        # Infer dimension from a single test call
        self.dimension = self._infer_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0   = time.perf_counter()
        vecs = self._batch_embed(texts)
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vecs.tolist()

    def embed_query(self, query: str) -> list[float]:
        t0   = time.perf_counter()
        vecs = self._batch_embed([query])
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vecs[0].tolist()

    def embed_numpy(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        t0   = time.perf_counter()
        vecs = self._batch_embed(texts)
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vecs

    # ------------------------------------------------------------------

    def _batch_embed(self, texts: list[str]) -> np.ndarray:
        """Chunk texts into batches, embed each, stack and normalize."""
        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            vecs  = self._call_api(batch)
            all_vecs.append(vecs)
        result = np.vstack(all_vecs).astype("float32")
        # Normalize (most embedding APIs return normalized vectors, but verify)
        norms  = np.linalg.norm(result, axis=1, keepdims=True)
        norms  = np.where(norms == 0, 1.0, norms)
        return result / norms

    def _call_api(self, texts: list[str]) -> np.ndarray:
        payload: dict = {"model": self.model_name, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        response = requests.post(
            url     = f"{self.base_url}/embeddings",
            headers = self.headers,
            data    = json.dumps(payload),
            timeout = self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        if "data" not in data:
            raise RuntimeError(f"Unexpected API response: {data}")

        # API returns results in arbitrary order — sort by index
        items = sorted(data["data"], key=lambda x: x["index"])
        return np.array([item["embedding"] for item in items], dtype="float32")

    def _infer_dimension(self) -> int:
        vecs = self._call_api(["dimension probe"])
        return vecs.shape[1]