# embedders/batch_processor.py

import time
import numpy as np
from core import Chunk, Embedder


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    L2-normalize a float32 array of shape (N, D) in-place.

    Returns the same array, normalized. Handles zero vectors safely
    (leaves them as zero rather than producing NaN).

    Why verify normalization instead of assuming it:
    - Raw HuggingFace models don't normalize automatically
    - Float32/float64 conversion can shift norms slightly off 1.0
    - Cached vectors from different code paths may have been stored un-normalized
    - FAISS IndexFlatIP silently returns wrong cosine scores for un-normalized vectors
    """
    vectors = np.asarray(vectors, dtype="float32")
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms   = np.where(norms < eps, 1.0, norms)   # avoid div-by-zero on zero vectors
    return vectors / norms


def verify_normalization(vectors: np.ndarray, tol: float = 1e-4) -> dict:
    """
    Check whether vectors are already normalized.

    Returns:
        {
            "normalized":  bool  — True if all norms are within tol of 1.0
            "max_dev":     float — max deviation from 1.0 across all vectors
            "n_outliers":  int   — how many vectors have norm deviation > tol
        }

    Use before indexing to catch un-normalized embeddings before they
    silently corrupt retrieval scores.
    """
    norms   = np.linalg.norm(vectors, axis=1)
    devs    = np.abs(norms - 1.0)
    return {
        "normalized": bool(np.all(devs <= tol)),
        "max_dev":    float(np.max(devs)),
        "n_outliers": int(np.sum(devs > tol)),
    }


class BatchEmbedder:
    """
    Production-grade batch embedding processor.

    Wraps any Embedder and embeds a list of Chunks in memory-efficient
    mini-batches, with progress reporting and per-batch error handling.

    Parameters
    ----------
    embedder      : any Embedder (raw or CachedEmbedder)
    batch_size    : chunks per embedding call
    show_progress : print a simple progress line every N batches
    normalize     : verify + correct normalization after each batch
    skip_on_error : if True, log and skip bad batches; if False, re-raise

    Usage
    -----
    processor = BatchEmbedder(embedder, batch_size=64)
    chunks    = processor.embed_chunks(chunks)
    # chunks now have .embedding populated
    """

    def __init__(
        self,
        embedder:      Embedder,
        batch_size:    int  = 64,
        show_progress: bool = True,
        normalize:     bool = True,
        skip_on_error: bool = False,
    ):
        self.embedder      = embedder
        self.batch_size    = batch_size
        self.show_progress = show_progress
        self.normalize     = normalize
        self.skip_on_error = skip_on_error

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed all chunks in-place (sets chunk.embedding).
        Returns the same list with embeddings populated.

        Skips chunks whose content is empty.
        Reports progress and timing.
        """
        if not chunks:
            return chunks

        total      = len(chunks)
        embedded   = 0
        skipped    = 0
        t_start    = time.perf_counter()

        for batch_start in range(0, total, self.batch_size):
            batch = chunks[batch_start:batch_start + self.batch_size]

            # Filter empty content
            valid   = [(i, c) for i, c in enumerate(batch) if c.content.strip()]
            if not valid:
                skipped += len(batch)
                continue

            idxs, valid_chunks = zip(*valid)
            texts  = [c.content for c in valid_chunks]

            try:
                t0      = time.perf_counter()
                vectors = np.array(self.embedder.embed(texts), dtype="float32")
                t_batch = (time.perf_counter() - t0) * 1000

                if self.normalize:
                    check = verify_normalization(vectors)
                    if not check["normalized"]:
                        vectors = normalize_vectors(vectors)

                for local_idx, vec in zip(idxs, vectors):
                    batch[local_idx].embedding = vec.tolist()

                embedded += len(valid_chunks)
                skipped  += len(batch) - len(valid_chunks)

                if self.show_progress:
                    done_pct = (batch_start + len(batch)) / total * 100
                    print(
                        f"  Embedded {batch_start + len(batch):>6}/{total}  "
                        f"({done_pct:5.1f}%)  "
                        f"batch={t_batch:.0f}ms  "
                        f"norm_ok={check['normalized'] if self.normalize else 'skip'}",
                        end="\r",
                    )

            except Exception as e:
                if self.skip_on_error:
                    print(f"\n  [BatchEmbedder] batch {batch_start} failed: {e} — skipping")
                    skipped += len(valid)
                else:
                    raise

        t_total = (time.perf_counter() - t_start) * 1000
        if self.show_progress:
            print(
                f"\n  Done: {embedded} embedded, {skipped} skipped  "
                f"| total={t_total:.0f}ms  "
                f"| throughput={embedded/(t_total/1000):.0f} chunks/sec"
            )

        return chunks

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed raw strings in batches. Returns float32 numpy array (N, D).
        Useful when you don't have Chunk objects yet.
        """
        if not texts:
            return np.empty((0, self.embedder.dimension), dtype="float32")

        all_vecs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            vecs  = np.array(self.embedder.embed(batch), dtype="float32")
            if self.normalize:
                vecs = normalize_vectors(vecs)
            all_vecs.append(vecs)

        return np.vstack(all_vecs)