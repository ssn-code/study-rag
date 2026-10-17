from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from src import ingest
from src.ingest import Chunk


class IngestTests(unittest.TestCase):
    def test_ingest_reconciles_chunks_by_relative_path(self) -> None:
        with TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "shared.pdf"
            chunk = Chunk(
                id="chunk-id",
                text="study text",
                metadata={
                    "source": "shared.pdf",
                    "relative_path": "data/course-a/shared.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "document_type": "pdf",
                },
            )
            embedder = Mock()
            embedder.embed_texts.return_value = [[0.1]]
            store = Mock()
            store.existing_ids.return_value = set()

            with (
                patch.object(ingest, "extract_pdf_pages", return_value=["study text"]),
                patch.object(ingest, "create_chunks", return_value=[chunk]),
            ):
                ingest.ingest_pdf(pdf_path, embedder, store)

            store.remove_stale_source_chunks.assert_called_once_with(
                "shared.pdf",
                {"chunk-id"},
                relative_path="data/course-a/shared.pdf",
            )


if __name__ == "__main__":
    unittest.main()