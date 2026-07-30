# loaders/txt_loader.py

import uuid
from core import Document, Loader


class TXTLoader(Loader):
    def load(self, source: str) -> list[Document]:
        with open(source, "r", encoding="utf-8") as f:
            content = f.read().strip()

        return [Document(
            doc_id   = str(uuid.uuid4()),
            content  = content,
            source   = source,
            metadata = {"file_type": "txt"},
        )]