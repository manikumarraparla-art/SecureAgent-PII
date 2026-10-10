"""Run: python -m app.rag (atomic collection rebuild)."""
from app.core.config import get_settings
from app.security.pii_detector import PIIDetector
from app.security.gateway import PIISecurityGateway
from app.rag.embeddings import OllamaEmbeddingClient
from app.rag.vector_store import SQLiteVectorStore
from app.rag.ingestion import DocumentIngestor
from app.core.exceptions import SecureAgentError


def main():
    settings = get_settings()
    embeddings = OllamaEmbeddingClient(settings.ollama_base_url, settings.ollama_embed_model, settings.ollama_request_timeout)
    try:
        count = DocumentIngestor(settings.rag_documents_dir, PIISecurityGateway(PIIDetector.from_settings(settings)), embeddings,
            SQLiteVectorStore(settings.rag_index_path, settings.rag_collection, settings.ollama_embed_model),
            settings.rag_chunk_size, settings.rag_chunk_overlap).rebuild()
        print(f"Index rebuilt: {count} sanitized chunks; embedding model={settings.ollama_embed_model}")
    except SecureAgentError as exc:
        print(f"Index rebuild failed: {exc.error_code.value}")
        raise SystemExit(1) from None
    finally:
        embeddings.close()

if __name__ == '__main__': main()
