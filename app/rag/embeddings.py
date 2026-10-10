"""Local embedding-only abstraction. Inputs must already be sanitized."""
from typing import Protocol
import math
import httpx
from app.core.config import _parse_local_url, _validate_timeout
from app.core.exceptions import SecureAgentError, ErrorCode

class EmbeddingError(SecureAgentError):
    error_code = ErrorCode.EMBEDDING_FAILED

class EmbeddingClient(Protocol):
    model_name: str
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def validate_vectors(vectors, count: int, dimension: int | None = None) -> int:
    if not isinstance(vectors, list) or len(vectors) != count or not vectors:
        raise ValueError("Invalid embedding count")
    dim = dimension or len(vectors[0])
    if not 1 <= dim <= 16384:
        raise ValueError("Invalid embedding dimension")
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dim:
            raise ValueError("Embedding dimension mismatch")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
            raise ValueError("Invalid embedding values")
        if not any(vector):
            raise ValueError("Zero embedding")
    return dim


class OllamaEmbeddingClient:
    def __init__(self, base_url="http://localhost:11434", model_name="nomic-embed-text", timeout=120.0, transport=None):
        _validate_timeout('Embedding timeout', timeout)
        self.model_name = model_name
        self._client = httpx.Client(base_url=_parse_local_url(base_url), timeout=timeout,
                                    trust_env=False, follow_redirects=False, transport=transport)

    def close(self):
        self._client.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            if any(not isinstance(t, str) or not t.strip() for t in texts):
                raise ValueError("Invalid embedding input")
            tags = self._client.get("/api/tags")
            tags.raise_for_status()
            installed = {m["name"] for m in tags.json()["models"]}
            if self.model_name not in installed and not (":" not in self.model_name and self.model_name + ":latest" in installed):
                raise ValueError("Embedding model missing")
            result = self._client.post("/api/embed", json={"model": self.model_name, "input": texts, "truncate": False})
            result.raise_for_status()
            vectors = result.json()["embeddings"]
            validate_vectors(vectors, len(texts))
            return vectors
        except Exception:
            raise EmbeddingError("Local embedding failed") from None
