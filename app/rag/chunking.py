"""Deterministic fixed-character chunks; sanitize whole documents first."""
from dataclasses import dataclass, field
import re

SAFE_FILENAME = re.compile(r"(?:claim|policy|customer)_[0-9]+\.txt|procedures_[a-z_]+\.txt|malicious_[a-z_]+\.txt")

@dataclass(frozen=True)
class Chunk:
    document_id: str
    filename: str
    chunk_id: str
    text: str = field(repr=False)

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{32}", self.document_id):
            raise ValueError("Invalid document ID")
        if not SAFE_FILENAME.fullmatch(self.filename):
            raise ValueError("Unsafe source filename")
        if not re.fullmatch(self.document_id + r"-[0-9]+", self.chunk_id):
            raise ValueError("Invalid chunk ID")
        if not isinstance(self.text, str) or len(self.text) > 16000:
            raise ValueError("Invalid chunk text")


def chunk_text(text: str, document_id: str, filename: str, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    if not 1 <= size <= 16000 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk configuration")
    if not text.strip():
        return []
    chunks = []
    for start in range(0, len(text), size - overlap):
        chunks.append(Chunk(document_id, filename, f"{document_id}-{len(chunks)}", text[start:start + size]))
        if start + size >= len(text):
            break
    return chunks
