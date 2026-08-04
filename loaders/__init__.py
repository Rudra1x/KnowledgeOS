# loaders/__init__.py

from .txt_loader     import TXTLoader
from .pdf_loader     import PDFLoader
from .docx_loader    import DOCXLoader
from .md_loader      import MarkdownLoader
from .html_loader    import HTMLLoader
from .csv_loader     import CSVLoader
from .email_loader   import EmailLoader
from .youtube_loader import YouTubeLoader
from .router         import LoaderRouter

__all__ = [
    "TXTLoader", "PDFLoader", "DOCXLoader", "MarkdownLoader",
    "HTMLLoader", "CSVLoader", "EmailLoader", "YouTubeLoader",
    "LoaderRouter",
]