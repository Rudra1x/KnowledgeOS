# generation/__init__.py

from .generator          import OpenRouterGenerator
from .local_generator    import LocalLLMGenerator
from .prompt_builder     import build_prompt, extract_citations
from .context_compressor import ContextCompressor

__all__ = [
    "OpenRouterGenerator", "LocalLLMGenerator",
    "build_prompt", "extract_citations", "ContextCompressor",
]