"""Confined local text ingestion. Sanitize before chunking or embedding."""
import hashlib
import os
import stat
from pathlib import Path
from app.rag.chunking import SAFE_FILENAME, chunk_text
from app.core.exceptions import SecureAgentError, ErrorCode
from app.security.documents import protect_document

class IngestionError(SecureAgentError):
    error_code = ErrorCode.INGESTION_FAILED

class DocumentIngestor:
    def __init__(self, directory, gateway, embeddings, store, size=1000, overlap=150):
        self.directory = Path(directory).resolve()
        self.gateway, self.embeddings, self.store = gateway, embeddings, store
        self.size, self.overlap = size, overlap

    def load(self, filename):
        try:
            if not SAFE_FILENAME.fullmatch(filename):
                raise ValueError("Unsupported document")
            # Open relative to an anchored directory descriptor; reject symlinks.
            root = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    os.close(fd)
                    raise ValueError("Unsupported file object")
                with os.fdopen(fd, 'r', encoding='utf-8') as handle:
                    text = handle.read(1_000_001)
                if len(text) > 1_000_000:
                    raise ValueError("Document too large")
            finally:
                os.close(root)
            doc_id = hashlib.sha256(filename.encode()).hexdigest()[:32]
            sanitized, _ = protect_document(text, self.gateway, doc_id)
            return chunk_text(sanitized, doc_id, filename, self.size, self.overlap)
        except SecureAgentError:
            raise
        except Exception:
            raise IngestionError("Local document ingestion failed") from None

    def rebuild(self):
        try:
            chunks = []
            for path in sorted(self.directory.iterdir()):
                if path.name == ".gitkeep":
                    continue
                chunks.extend(self.load(path.name))
            vectors = []
            for start in range(0,len(chunks),16):
                vectors.extend(self.embeddings.embed([c.text for c in chunks[start:start+16]]))
            self.store.replace(chunks,vectors)
            return len(chunks)
        except SecureAgentError:
            raise
        except Exception:
            raise IngestionError("Local document ingestion failed") from None
