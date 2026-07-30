# core/models.py

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id:    str
    content:   str
    source:    str
    metadata:  dict[str, Any] = field(default_factory=dict)
    tenant_id: str            = "default"
    language:  str            = "en"


@dataclass
class Chunk:
    chunk_id:    str
    doc_id:      str
    content:     str
    embedding:   list[float]  = field(default_factory=list)
    metadata:    dict[str, Any] = field(default_factory=dict)
    tenant_id:   str          = "default"
    start_index: int          = 0
    end_index:   int          = 0