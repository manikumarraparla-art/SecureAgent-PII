"""Reuse Phase 3 retrieval and document protection; no generation client."""
from app.rag.chunking import Chunk
from app.security.documents import protect_retrieved
from app.security.tokenizer import TokenVault
from app.rag.retriever import RetrievalError

def search_knowledge(arguments, session, retriever, gateway, top_k):
    if retriever is None: raise RetrievalError("Knowledge index unavailable")
    hits = retriever.retrieve(arguments['query'],top_k)
    if len(hits)>top_k: raise RetrievalError("Invalid retrieval count")
    sources=[]
    for hit in hits:
        chunk=hit.chunk
        Chunk(chunk.document_id,chunk.filename,chunk.chunk_id,chunk.text)
        vault=session.document_vaults.setdefault(chunk.document_id,TokenVault())
        safe,_=protect_retrieved(chunk.text,gateway,chunk.document_id,session.request_id,vault)
        reference={"document_id":chunk.document_id,"filename":chunk.filename,
                   "chunk_id":chunk.chunk_id,"score":hit.score}
        if reference not in session.sources: session.sources.append(reference)
        sources.append({"document_id":chunk.document_id,"filename":chunk.filename,
                        "chunk_id":chunk.chunk_id,"score":hit.score,"text":safe})
    return {"sources":sources}
