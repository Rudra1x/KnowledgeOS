# core/config.py

import yaml
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineConfig:
    loader:    str
    chunker:   str
    embedder:  str
    index:     str
    retriever: str
    reranker:  str | None
    generator: str


@dataclass
class KnowledgeOSConfig:
    pipeline: PipelineConfig
    settings: dict[str, Any] = field(default_factory=dict)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Dot-path lookup: config.get('chunker', 'fixed', 'chunk_size')"""
        node = self.settings
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node


def load_config(path: str = "configs/default.yaml") -> KnowledgeOSConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    pipeline = PipelineConfig(**raw["pipeline"])

    settings = {k: v for k, v in raw.items() if k != "pipeline"}

    return KnowledgeOSConfig(pipeline=pipeline, settings=settings)