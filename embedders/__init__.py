# embedders/__init__.py

from .bge_embedder        import BGEEmbedder
from .e5_embedder         import E5Embedder
from .instructor_embedder import InstructionEmbedder, InstructorEmbedder
from .jina_embedder       import JinaEmbedder   # stub — see module docstring

__all__ = ["BGEEmbedder", "E5Embedder", "InstructionEmbedder",
           "InstructorEmbedder", "JinaEmbedder"]