from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sys

from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, PROJECT_ROOT
from src.embeddings import EmbeddingError, NvidiaEmbedder
from src.vector_store import StudyVectorStore, VectorStoreError


# This is the folder where you will place study PDFs.
DOCUMENTS_DIR = DATA_DIR


@dataclass(frozen=True)
class Chunk:
    """A page-aware text chunk ready for embedding and storage."""

    id: str
    text: str
    metadata: dict[str, str | int]


def extract_pdf_text(pdf_path: Path) -> str:
    """Read a PDF and return the text from all of its pages."""
    reader = PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")

    # Some PDF pages contain no selectable text, so use an empty string as a
    # safe fallback before joining all page text into one document string.
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """Read each page separately so stored chunks retain their page number."""
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]


def create_chunks(pdf_path: Path, pages: list[str], document_hash: str | None = None) -> list[Chunk]:
    """Split page text into small overlapping chunks for the embedding API."""
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_number, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        start = 0
        while start < len(text):
            chunk_text = text[start : start + CHUNK_SIZE].strip()
            if chunk_text:
                if document_hash:
                    chunk_id = f"{document_hash}_{chunk_index}"
                else:
                    stable_value = f"{pdf_path.name}|{page_number}|{chunk_index}|{chunk_text}"
                    chunk_id = sha256(stable_value.encode("utf-8")).hexdigest()

                try:
                    rel_path = str(pdf_path.relative_to(PROJECT_ROOT).as_posix())
                except ValueError:
                    rel_path = str(pdf_path.as_posix())

                metadata: dict[str, str | int] = {
                    "source": pdf_path.name,
                    "relative_path": rel_path,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "document_type": pdf_path.suffix[1:].lower() or "pdf",
                }
                if document_hash:
                    metadata["document_hash"] = document_hash

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )
                chunk_index += 1
            end = start + CHUNK_SIZE
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP

    return chunks


def find_pdf_paths(filename: str | None) -> list[Path]:
    """Return one requested PDF or every top-level PDF in the data directory."""
    if filename:
        pdf_path = DOCUMENTS_DIR / Path(filename).name
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {pdf_path.name}")
        return [pdf_path]

    return sorted(path for path in DOCUMENTS_DIR.glob("*.pdf") if path.is_file())


def ingest_pdf(pdf_path: Path, embedder: NvidiaEmbedder, store: StudyVectorStore) -> dict[str, int]:
    """Extract, embed, and persist a single PDF without duplicating known chunks."""
    pages = extract_pdf_pages(pdf_path)
    chunks = create_chunks(pdf_path, pages)
    if not chunks:
        return {"pages": len(pages), "chunks": 0, "embeddings": 0, "stored": 0, "skipped": 0}

    existing_ids = store.existing_ids([chunk.id for chunk in chunks])
    new_chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]

    if new_chunks:
        embeddings = embedder.embed_texts([chunk.text for chunk in new_chunks])
        store.upsert(
            ids=[chunk.id for chunk in new_chunks],
            documents=[chunk.text for chunk in new_chunks],
            metadatas=[chunk.metadata for chunk in new_chunks],
            embeddings=embeddings,
        )

    store.remove_stale_source_chunks(pdf_path.name, {chunk.id for chunk in chunks})
    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "embeddings": len(new_chunks),
        "stored": len(new_chunks),
        "skipped": len(existing_ids),
    }


def main() -> None:
    if len(sys.argv) > 2:
        print("Usage: python -m src.ingest [pdf-filename]")
        print("Example: python -m src.ingest notes.pdf")
        sys.exit(1)

    try:
        pdf_paths = find_pdf_paths(sys.argv[1] if len(sys.argv) == 2 else None)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    if not pdf_paths:
        print(f"No PDF files found in: {DOCUMENTS_DIR}")
        return

    try:
        embedder = NvidiaEmbedder()
        store = StudyVectorStore()
    except (EmbeddingError, VectorStoreError) as error:
        print(f"Error: {error}")
        sys.exit(1)

    totals = {"documents": 0, "pages": 0, "chunks": 0, "embeddings": 0, "stored": 0, "skipped": 0, "errors": 0}
    for pdf_path in pdf_paths:
        try:
            result = ingest_pdf(pdf_path, embedder, store)
            totals["documents"] += 1
            for name in ("pages", "chunks", "embeddings", "stored", "skipped"):
                totals[name] += result[name]
            print(
                f"Processed {pdf_path.name}: {result['pages']} pages, "
                f"{result['chunks']} chunks, {result['stored']} stored, "
                f"{result['skipped']} skipped."
            )
        except (EmbeddingError, VectorStoreError) as error:
            totals["errors"] += 1
            print(f"Error processing {pdf_path.name}: {error}")
        except Exception:
            totals["errors"] += 1
            print(f"Error processing {pdf_path.name}: could not read or chunk the PDF.")

    print("\nIngestion summary")
    print(f"Documents processed: {totals['documents']}")
    print(f"Pages processed: {totals['pages']}")
    print(f"Chunks created: {totals['chunks']}")
    print(f"Embeddings generated: {totals['embeddings']}")
    print(f"Chunks stored: {totals['stored']}")
    print(f"Skipped/duplicate chunks: {totals['skipped']}")
    print(f"Errors: {totals['errors']}")
    if totals["errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
