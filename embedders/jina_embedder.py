# embedders/jina_embedder.py

"""
Jina Embedder — STUB

jinaai/jina-embeddings-v2 and v3 both have compatibility issues with
current transformers versions (v2: missing transformers.onnx;
v3: XLMRobertaLoRA missing all_tied_weights_keys attribute).

Architectural lesson preserved in docs/checkpoints/3.3_jina_embedder.md:
- 8192-token context window vs BGE/E5's 512 tokens
- Silent truncation at 512 tokens is the key failure mode
- Jina is the right choice when chunk avg size > 400 chars

Wire this when Jina publishes a compatible release.
Track: https://github.com/jina-ai/jina-embeddings

To use when fixed:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jinaai/jina-embeddings-v3",
                                trust_remote_code=True)
"""


class JinaEmbedder:
    """Stub — see module docstring for status."""
    NAME      = "jina"
    dimension = 1024

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JinaEmbedder is currently incompatible with the installed "
            "transformers version. See embedders/jina_embedder.py for details."
        )