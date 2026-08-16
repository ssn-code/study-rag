"""SQLite document registry for tracking Phase 5 Knowledge Base Synchronization state."""

from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

from src.config import REGISTRY_DB_PATH


class RegistryError(RuntimeError):
    """Raised when registry database operations fail."""


class DocumentRegistry:
    """Track the synchronization state, content hash, metadata, and chunk counts of study files."""

    def __init__(self, db_path: Path = REGISTRY_DB_PATH) -> None:
        self.db_path = db_path
        self._initialize()

    def _initialize(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS registry (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        relative_path TEXT NOT NULL UNIQUE,
                        file_type TEXT NOT NULL,
                        file_hash TEXT NOT NULL,
                        last_modified REAL NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        sync_status TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as error:
            raise RegistryError(f"Could not initialize registry: {error}") from error

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Return a list of all registered documents."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM registry")
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as error:
            raise RegistryError(f"Could not retrieve documents: {error}") from error

    def get_document(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document's registration metadata by its relative path."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM registry WHERE relative_path = ?", (relative_path,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as error:
            raise RegistryError(f"Could not retrieve document: {error}") from error

    def add_or_update_document(
        self,
        document_id: str,
        filename: str,
        relative_path: str,
        file_type: str,
        file_hash: str,
        last_modified: float,
        chunk_count: int,
        sync_status: str,
    ) -> None:
        """Add a new document entry or update an existing one based on document_id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO registry (
                        document_id, filename, relative_path, file_type, file_hash, last_modified, chunk_count, sync_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id) DO UPDATE SET
                        filename=excluded.filename,
                        relative_path=excluded.relative_path,
                        file_type=excluded.file_type,
                        file_hash=excluded.file_hash,
                        last_modified=excluded.last_modified,
                        chunk_count=excluded.chunk_count,
                        sync_status=excluded.sync_status
                    """,
                    (
                        document_id,
                        filename,
                        relative_path,
                        file_type,
                        file_hash,
                        last_modified,
                        chunk_count,
                        sync_status,
                    ),
                )
        except sqlite3.Error as error:
            raise RegistryError(f"Could not insert or update document: {error}") from error

    def delete_document(self, relative_path: str) -> None:
        """Delete a document entry from the registry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM registry WHERE relative_path = ?", (relative_path,)
                )
        except sqlite3.Error as error:
            raise RegistryError(f"Could not delete document from registry: {error}") from error
