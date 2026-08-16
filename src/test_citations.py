"""Unit and integration tests for Phase 6 Citations and Source Grounding."""

import sys
from pathlib import Path
import unittest

from src.rag import SourceReference, RAGResult, get_chunk_source_info, build_context, extract_citations
from src.retriever import RetrievalResult


class TestCitations(unittest.TestCase):
    """Verify citations formatting, deduplication, validation, and metadata extraction."""

    def test_metadata_extraction_and_keys(self) -> None:
        """Test metadata key parsing and extraction for PDF, PPTX, and DOCX."""
        # PDF Chunk
        pdf_meta = {
            "source": "networks.pdf",
            "document_type": "pdf",
            "page_number": 42,
            "chunk_index": 5,
        }
        key, info = get_chunk_source_info(pdf_meta)
        self.assertEqual(key, ("networks.pdf", "page", 42))
        self.assertEqual(info["page"], 42)
        self.assertEqual(info["type"], "pdf")

        # PPTX Chunk
        pptx_meta = {
            "source": "intro.pptx",
            "document_type": "pptx",
            "slide_number": 18,
            "chunk_index": 2,
        }
        key, info = get_chunk_source_info(pptx_meta)
        self.assertEqual(key, ("intro.pptx", "slide", 18))
        self.assertEqual(info["slide"], 18)
        self.assertEqual(info["type"], "pptx")

        # DOCX Chunk
        docx_meta = {
            "source": "notes.docx",
            "document_type": "docx",
            "section": "Abstract",
            "chunk_index": 1,
        }
        key, info = get_chunk_source_info(docx_meta)
        self.assertEqual(key, ("notes.docx", "section", "Abstract"))
        self.assertEqual(info["section"], "Abstract")
        self.assertEqual(info["type"], "docx")

    def test_context_builder_and_deduplication(self) -> None:
        """Test build_context outputs correctly matched SOURCE [ID] headers and mapping IDs."""
        # Setup mock retrieval results
        results = [
            RetrievalResult(
                id="chunk_1",
                text="Text 1",
                metadata={"source": "networks.pdf", "document_type": "pdf", "page_number": 42},
                distance=0.1,
            ),
            RetrievalResult(
                id="chunk_2",
                text="Text 2",
                metadata={"source": "networks.pdf", "document_type": "pdf", "page_number": 42},
                distance=0.2,
            ),
            RetrievalResult(
                id="chunk_3",
                text="Text 3",
                metadata={"source": "intro.pptx", "document_type": "pptx", "slide_number": 18},
                distance=0.3,
            ),
            RetrievalResult(
                id="chunk_4",
                text="Text 4",
                metadata={"source": "notes.docx", "document_type": "docx", "section": "Abstract"},
                distance=0.4,
            ),
        ]

        # Map unique sources manually to get indices
        unique_sources_map = {}
        mapping = []
        for result in results:
            key, _ = get_chunk_source_info(result.metadata)
            if key not in unique_sources_map:
                unique_sources_map[key] = len(unique_sources_map) + 1
            mapping.append(unique_sources_map[key])

        # Verify mapping deduplication
        self.assertEqual(mapping, [1, 1, 2, 3])

        # Build context using mapped indices
        context = build_context(results, mapping)

        # Assert context contains correct SOURCE [ID] labels and structure
        self.assertIn("SOURCE [1]\nDocument: networks.pdf\nPage: 42\n\nText 1", context)
        self.assertIn("SOURCE [1]\nDocument: networks.pdf\nPage: 42\n\nText 2", context)
        self.assertIn("SOURCE [2]\nDocument: intro.pptx\nSlide: 18\n\nText 3", context)
        self.assertIn("SOURCE [3]\nDocument: notes.docx\nSection: Abstract\n\nText 4", context)

    def test_citation_validation(self) -> None:
        """Test extraction and boundary validation of citation keys from LLM outputs."""
        answer = "TCP connection is established using a handshake [1][2]. Extra claim [7]."
        cited_ids = extract_citations(answer, max_valid_id=3)

        # Verify citation validation boundary checking (1 and 2 are kept, out-of-bounds 7 is skipped)
        self.assertIn(1, cited_ids)
        self.assertIn(2, cited_ids)
        self.assertNotIn(7, cited_ids)
        self.assertEqual(cited_ids, {1, 2})

    def test_multiple_citations_in_same_bracket(self) -> None:
        """Test parsing of multi-citations like [1, 2] or spaces [1] [2]."""
        answer_comma = "TCP uses flow and congestion control [1, 2]."
        cited_ids_comma = extract_citations(answer_comma, max_valid_id=3)
        self.assertEqual(cited_ids_comma, {1, 2})

        answer_spaces = "TCP uses handshake [1] [ 2 ]."
        cited_ids_spaces = extract_citations(answer_spaces, max_valid_id=3)
        self.assertEqual(cited_ids_spaces, {1, 2})


if __name__ == "__main__":
    unittest.main()
