"""Retrieval receives only sanitized user queries and has no generation client."""
from app.core.exceptions import SecureAgentError, ErrorCode

class RetrievalError(SecureAgentError):
    error_code = ErrorCode.RETRIEVAL_FAILED

class Retriever:
    def __init__(self, embeddings, store):
        self.embeddings, self.store = embeddings, store

    @property
    def embedding_model(self):
        return self.embeddings.model_name

    def retrieve(self, sanitized_query, top_k=3):
        try:
            vectors = self.embeddings.embed([sanitized_query])
            return self.store.search(vectors[0], top_k)
        except SecureAgentError:
            raise
        except Exception:
            raise RetrievalError("Local retrieval failed") from None

    def close(self):
        close = getattr(self.embeddings,"close",None)
        if callable(close): close()
