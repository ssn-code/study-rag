"""Persistent ChromaDB storage for Study RAG chunks."""

from collections.abc import Sequence

import chromadb

from src.config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR


class VectorStoreError(RuntimeError):
    """Raised when ChromaDB cannot read or write the knowledge base."""


class StudyVectorStore:
    """Store chunk text, metadata, and externally generated embeddings in ChromaDB."""

    def __init__(self) -> None:
        try:
            CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
            self.collection = client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise VectorStoreError("Could not initialize the local ChromaDB database.") from error

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        try:
            result = self.collection.get(ids=list(ids), include=[])
            return set(result["ids"])
        except Exception as error:
            raise VectorStoreError("Could not check existing ChromaDB chunks.") from error

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, str | int]],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if not ids:
            return
        try:
            self.collection.upsert(
                ids=list(ids),
                documents=list(documents),
                metadatas=list(metadatas),
                embeddings=[list(vector) for vector in embeddings],
            )
        except Exception as error:
            raise VectorStoreError("Could not store chunks in ChromaDB.") from error

    def remove_stale_source_chunks(self, source: str, current_ids: set[str]) -> int:
        """Remove chunks from an updated source that no longer exist in its current text."""
        try:
            result = self.collection.get(where={"source": source}, include=[])
            stale_ids = [chunk_id for chunk_id in result["ids"] if chunk_id not in current_ids]
            if stale_ids:
                self.collection.delete(ids=stale_ids)
            return len(stale_ids)
        except Exception as error:
            raise VectorStoreError("Could not reconcile existing chunks for this source.") from error

    def inspect(self, limit: int = 3) -> dict[str, object]:
        """Return a small non-search inspection sample for verification."""
        try:
            return self.collection.get(
                limit=limit,
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception as error:
            raise VectorStoreError("Could not inspect the ChromaDB collection.") from error
