"""SQLite persistence and exact cosine search for small local collections."""
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from dataclasses import dataclass
from app.rag.chunking import Chunk
from app.rag.embeddings import validate_vectors
from app.core.exceptions import SecureAgentError, ErrorCode

class VectorStoreError(SecureAgentError):
    error_code = ErrorCode.VECTOR_STORE_FAILED

@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float

class SQLiteVectorStore:
    def __init__(self, path="data/vectorstore/index.sqlite3", collection="synthetic", model="nomic-embed-text"):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", collection):
            raise ValueError("Invalid collection")
        self.path, self.collection, self.model = Path(path), collection, model

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute("CREATE TABLE IF NOT EXISTS collections (name TEXT PRIMARY KEY, model TEXT, dimension INTEGER)")
        connection.execute("CREATE TABLE IF NOT EXISTS chunks (collection TEXT, id TEXT, document TEXT, filename TEXT, text TEXT, vector TEXT, PRIMARY KEY(collection,id))")
        return connection

    def replace(self, chunks: list[Chunk], vectors: list[list[float]]):
        try:
            if len(chunks) != len(vectors):
                raise ValueError("Chunk/vector mismatch")
            dimension = validate_vectors(vectors, len(chunks)) if chunks else 0
            with self._connect() as db:
                db.execute("DELETE FROM chunks WHERE collection=?", (self.collection,))
                db.execute("INSERT OR REPLACE INTO collections VALUES (?,?,?)", (self.collection, self.model, dimension))
                db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", [(self.collection, c.chunk_id, c.document_id, c.filename, c.text, json.dumps(v)) for c,v in zip(chunks,vectors)])
        except Exception:
            raise VectorStoreError("Local index write failed") from None
        finally:
            if 'db' in locals(): db.close()

    def search(self, vector: list[float], top_k: int = 3) -> list[SearchHit]:
        try:
            if type(top_k) is not int or not 1 <= top_k <= 10:
                raise ValueError("Invalid top_k")
            with self._connect() as db:
                meta = db.execute("SELECT model,dimension FROM collections WHERE name=?", (self.collection,)).fetchone()
                if not meta or meta[0] != self.model or meta[1] == 0:
                    raise ValueError("Index missing or incompatible; rebuild")
                validate_vectors([vector], 1, meta[1])
                rows = db.execute("SELECT document,filename,id,text,vector FROM chunks WHERE collection=?", (self.collection,)).fetchall()
            hits = []
            for doc, filename, cid, text, encoded in rows:
                other = json.loads(encoded)
                validate_vectors([other], 1, meta[1])
                score = sum(a*b for a,b in zip(vector,other)) / (math.sqrt(sum(a*a for a in vector))*math.sqrt(sum(b*b for b in other)))
                hits.append(SearchHit(Chunk(doc,filename,cid,text), max(-1.0,min(1.0,score))))
            return sorted(hits, key=lambda h: (-h.score,h.chunk.chunk_id))[:top_k]
        except Exception:
            raise VectorStoreError("Local index search failed") from None
        finally:
            if 'db' in locals(): db.close()
