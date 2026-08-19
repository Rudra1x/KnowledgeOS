# generation/__init__.py

from .generator        import OpenRouterGenerator
from .local_generator  import LocalLLMGenerator
from .prompt_builder   import build_prompt

__all__ = ["OpenRouterGenerator", "LocalLLMGenerator", "build_prompt"]