import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src import sync


class SyncTests(unittest.TestCase):
    def test_failed_document_with_same_hash_requires_retry(self) -> None:
        registered_doc = {"file_hash": "current-hash", "sync_status": "failed"}

        self.assertFalse(sync.is_document_synchronized(registered_doc, "current-hash"))

    def test_successfully_synchronized_document_with_same_hash_is_skipped(self) -> None:
        registered_doc = {"file_hash": "current-hash", "sync_status": "synchronized"}

        self.assertTrue(sync.is_document_synchronized(registered_doc, "current-hash"))

    def test_discover_documents_returns_only_pdfs_recursively(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "a.PDF"
            nested_pdf = root / "nested" / "b.pdf"
            pdf.touch()
            nested_pdf.parent.mkdir()
            nested_pdf.touch()
            (root / "slides.pptx").touch()
            (root / "notes.docx").touch()

            self.assertEqual(sync.discover_documents(root), [pdf, nested_pdf])

    def test_dry_run_does_not_create_missing_registry(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "offline.pdf").touch()
            registry_path = root / "chroma_db" / "registry.db"

            with (
                patch.object(sync, "DATA_DIR", data_dir),
                patch.object(sync, "REGISTRY_DB_PATH", registry_path),
                patch.object(sys, "argv", ["src.sync", "--dry-run"]),
                redirect_stdout(io.StringIO()),
            ):
                sync.main()

            self.assertFalse(registry_path.exists())


if __name__ == "__main__":
    unittest.main()