# embedders/cache.py

import os
import json
import sqlite3
import hashlib
import time
import numpy as np
from core import Embedder


class CachedEmbedder(Embedder):
    """
    Caching wrapper around any Embedder.

    Architecture:
    - L1: in-memory dict (instant, process-scoped)
    - L2: SQLite on disk (persistent, survives restart)

    Cache key: SHA-256(model_name + '::' + text)
    Values: JSON-serialized float list

    On embed(texts):
    1. Check L1 for each text → hits returned immediately
    2. Check L2 for remaining misses
    3. Embed only the true misses using the wrapped embedder
    4. Write new vectors to L1 + L2
    5. Assemble full result in original order

    Metadata tracked per entry:
    - model_name:  which embedder produced this vector
    - created_at:  unix timestamp
    - hit_count:   how many times this entry was served from cache

    Parameters
    ----------
    embedder   : any Embedder — wrapped transparently
    cache_path : path to SQLite file (created if missing)
    max_memory : max entries in L1 before eviction (LRU)
    """

    NAME = property(lambda self: f"cached_{self.embedder.NAME}")

    def __init__(
        self,
        embedder:   Embedder,
        cache_path: str = "cache/embeddings.db",
        max_memory: int = 10_000,
    ):
        self.embedder   = embedder
        self.cache_path = cache_path
        self.max_memory = max_memory
        self.dimension  = embedder.dimension
        self.last_embed_ms: float = 0.0

        # L1: in-memory {cache_key: np.ndarray}
        self._mem: dict[str, np.ndarray] = {}
        self._mem_order: list[str]       = []  # for LRU eviction

        # L2: SQLite
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        t0   = time.perf_counter()
        vecs = self._cached_embed(texts)
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vecs.tolist()

    def embed_query(self, query: str) -> list[float]:
        t0   = time.perf_counter()
        vecs = self._cached_embed([query])
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vecs[0].tolist()

    def embed_numpy(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        t0   = time.perf_counter()
        vecs = self._cached_embed(texts)
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vecs

    # ------------------------------------------------------------------
    # Cache logic
    # ------------------------------------------------------------------

    def _cached_embed(self, texts: list[str]) -> np.ndarray:
        keys    = [self._make_key(t) for t in texts]
        result  = [None] * len(texts)
        misses  = []

        for i, (key, text) in enumerate(zip(keys, texts)):
            vec = self._l1_get(key)
            if vec is None:
                vec = self._l2_get(key)
                if vec is not None:
                    self._l1_set(key, vec)   # ← populate L1 from L2 hit
            if vec is not None:
                result[i] = vec
            else:
                misses.append((i, key, text))

        if misses:
            miss_texts = [m[2] for m in misses]
            new_vecs   = np.array(
                self.embedder.embed(miss_texts), dtype="float32"
            )
            for (i, key, _), vec in zip(misses, new_vecs):
                self._l1_set(key, vec)
                self._l2_set(key, vec)
                result[i] = vec

        return np.vstack(result).astype("float32")
    def _make_key(self, text: str) -> str:
        raw = f"{self.embedder.model_name}::{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # L1 (memory)
    # ------------------------------------------------------------------

    def _l1_get(self, key: str) -> np.ndarray | None:
        if key in self._mem:
            # Move to end (LRU)
            self._mem_order.remove(key)
            self._mem_order.append(key)
            return self._mem[key]
        return None

    def _l1_set(self, key: str, vec: np.ndarray) -> None:
        if key not in self._mem and len(self._mem) >= self.max_memory:
            # Evict oldest
            evict = self._mem_order.pop(0)
            del self._mem[evict]
        self._mem[key] = vec
        if key not in self._mem_order:
            self._mem_order.append(key)

    # ------------------------------------------------------------------
    # L2 (SQLite)
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key  TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                vector     TEXT NOT NULL,
                created_at REAL NOT NULL,
                hit_count  INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model_name)"
        )
        self._db.commit()

    def _l2_get(self, key: str) -> np.ndarray | None:
        row = self._db.execute(
            "SELECT vector FROM embeddings WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        self._db.execute(
            "UPDATE embeddings SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (key,)
        )
        self._db.commit()
        return np.array(json.loads(row[0]), dtype="float32")

    def _l2_set(self, key: str, vec: np.ndarray) -> None:
        self._db.execute("""
            INSERT OR REPLACE INTO embeddings
                (cache_key, model_name, vector, created_at, hit_count)
            VALUES (?, ?, ?, ?, 0)
        """, (key, self.embedder.model_name, json.dumps(vec.tolist()),
              time.time()))
        self._db.commit()

    # ------------------------------------------------------------------
    # Cache stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        row = self._db.execute(
            "SELECT COUNT(*), SUM(hit_count) FROM embeddings WHERE model_name = ?",
            (self.embedder.model_name,)
        ).fetchone()
        return {
            "model":       self.embedder.model_name,
            "db_entries":  row[0] or 0,
            "total_hits":  row[1] or 0,
            "l1_entries":  len(self._mem),
            "cache_path":  self.cache_path,
        }

    def clear(self) -> None:
        """Remove all cached entries for this model."""
        self._db.execute(
            "DELETE FROM embeddings WHERE model_name = ?",
            (self.embedder.model_name,)
        )
        self._db.commit()
        self._mem.clear()
        self._mem_order.clear()

    def close(self) -> None:
        self._db.close()
