# embedders/instructor_embedder.py

import time
import numpy as np
from sentence_transformers import SentenceTransformer
from core import Embedder


class InstructionEmbedder(Embedder):
    """
    Instruction-tuned embedding via sentence-transformers.

    Replicates the core value proposition of HKUST Instructor
    without the broken InstructorEmbedding library dependency:
    prepend a task instruction to every text before encoding.

    Works with any instruction-following sentence-transformers model:
    - hkunlp/instructor-base or instructor-large (if installed separately)
    - BAAI/bge-base-en-v1.5 (BGE is instruction-aware)
    - intfloat/e5-base-v2 (with appropriate prefix)

    Default uses BGE-base for a quality step up from BGE-small.

    Example instructions:
    - "Represent the technical documentation paragraph for retrieval: "
    - "Represent the question for retrieving supporting documents: "
    - "Represent the financial report section for retrieval: "

    Why instructions help:
    The model has seen task-prefixed text during training. The prefix shifts
    the embedding toward a task-specific subspace, improving recall on
    out-of-domain corpora by 2-5% vs generic embeddings.
    """

    NAME = "instruction"

    DEFAULT_QUERY_INSTRUCTION = "Represent the question for retrieving relevant documents: "
    DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "

    def __init__(
        self,
        model_name:        str = "BAAI/bge-base-en-v1.5",
        batch_size:        int = 16,
        query_instruction: str | None = None,
        embed_instruction: str | None = None,
    ):
        self.model_name        = model_name
        self.batch_size        = batch_size
        self.query_instruction = query_instruction or self.DEFAULT_QUERY_INSTRUCTION
        self.embed_instruction = embed_instruction or self.DEFAULT_EMBED_INSTRUCTION

        self.model             = SentenceTransformer(model_name)
        self.dimension         = self.model.get_embedding_dimension()
        self.last_embed_ms: float = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prefixed = [f"{self.embed_instruction}{t}" for t in texts]
        t0       = time.perf_counter()
        vectors  = self.model.encode(
            prefixed,
            batch_size           = self.batch_size,
            normalize_embeddings = True,
            show_progress_bar    = False,
        )
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        prefixed = f"{self.query_instruction}{query}"
        t0       = time.perf_counter()
        vector   = self.model.encode(
            [prefixed],
            normalize_embeddings = True,
            show_progress_bar    = False,
        )[0]
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vector.tolist()

    def embed_numpy(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        prefixed = [f"{self.embed_instruction}{t}" for t in texts]
        t0       = time.perf_counter()
        vectors  = self.model.encode(
            prefixed,
            batch_size           = self.batch_size,
            normalize_embeddings = True,
            show_progress_bar    = False,
        )
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        return vectors.astype("float32")


# Alias for clarity in benchmark tables
InstructorEmbedder = InstructionEmbedder