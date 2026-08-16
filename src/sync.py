"""Orchestrates the Phase 5 incremental vector database updates and synchronization."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import DATA_DIR, PROJECT_ROOT
from src.embeddings import EmbeddingError, NvidiaEmbedder
from src.ingest import create_chunks, extract_pdf_pages
from src.registry import DocumentRegistry, RegistryError
from src.vector_store import StudyVectorStore, VectorStoreError


def calculate_file_hash(filepath: Path) -> str:
    """Calculate the SHA-256 hash of a file's contents."""
    hasher = sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as error:
        raise OSError(f"Could not calculate hash for {filepath.name}: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize study documents with ChromaDB.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and display proposed changes without writing to database or calling APIs.",
    )
    args = parser.parse_args()

    print("STUDY RAG KNOWLEDGE BASE SYNC\n")

    if not DATA_DIR.exists():
        print(f"Error: Data directory does not exist: {DATA_DIR}")
        sys.exit(1)

    print("Scanning documents...")

    try:
        registry = DocumentRegistry()
    except RegistryError as error:
        print(f"Error: Registry initialization failed: {error}")
        sys.exit(1)

    # Scan the data directory for documents (primarily .pdf)
    current_files = sorted(
        path for path in DATA_DIR.glob("**/*")
        if path.is_file() and path.suffix.lower() in (".pdf", ".pptx", ".docx")
    )

    new_docs = []
    modified_docs = []
    unchanged_docs = []
    deleted_docs = []

    # Map scanned files by relative path
    scanned_rel_paths = set()

    for file_path in current_files:
        try:
            rel_path = str(file_path.relative_to(PROJECT_ROOT).as_posix())
        except ValueError:
            rel_path = str(file_path.as_posix())

        scanned_rel_paths.add(rel_path)

        try:
            file_hash = calculate_file_hash(file_path)
            last_modified = file_path.stat().st_mtime
        except Exception as error:
            print(f"Error accessing file {file_path.name}: {error}")
            continue

        registered_doc = registry.get_document(rel_path)

        doc_info = {
            "path": file_path,
            "relative_path": rel_path,
            "filename": file_path.name,
            "file_type": file_path.suffix[1:].lower(),
            "file_hash": file_hash,
            "last_modified": last_modified,
        }

        if not registered_doc:
            new_docs.append(doc_info)
        elif registered_doc["file_hash"] != file_hash:
            doc_info["old_hash"] = registered_doc["file_hash"]
            modified_docs.append(doc_info)
        else:
            unchanged_docs.append(doc_info)

    # Check for DELETED documents
    all_registered = registry.get_all_documents()
    for reg_doc in all_registered:
        if reg_doc["relative_path"] not in scanned_rel_paths:
            deleted_docs.append(reg_doc)

    # Print scan report
    print("\nNEW:")
    if new_docs:
        for doc in new_docs:
            print(f"- {doc['filename']}")
    else:
        print("(None)")

    print("\nMODIFIED:")
    if modified_docs:
        for doc in modified_docs:
            print(f"- {doc['filename']}")
    else:
        print("(None)")

    print("\nUNCHANGED:")
    if unchanged_docs:
        for doc in unchanged_docs:
            print(f"- {doc['filename']}")
    else:
        print("(None)")

    print("\nDELETED:")
    if deleted_docs:
        for doc in deleted_docs:
            print(f"- {doc['filename']}")
    else:
        print("(None)")

    print("\nSummary:")
    print(f"New documents: {len(new_docs)}")
    print(f"Modified documents: {len(modified_docs)}")
    print(f"Unchanged documents: {len(unchanged_docs)}")
    print(f"Deleted documents: {len(deleted_docs)}")

    if args.dry_run:
        print("\nDry run mode enabled. No changes were made.")
        return

    # If no updates are needed, exit early
    if not (new_docs or modified_docs or deleted_docs):
        print("\nSynchronization complete. No updates required.")
        return

    # Setup embeddings and vector store clients
    try:
        store = StudyVectorStore()
    except VectorStoreError as error:
        print(f"\nError: Vector database initialization failed: {error}")
        sys.exit(1)

    embedder = None
    if new_docs or modified_docs:
        try:
            embedder = NvidiaEmbedder()
        except EmbeddingError as error:
            print(f"\nError: Nvidia embedding client initialization failed: {error}")
            sys.exit(1)

    chunks_added = 0
    chunks_deleted = 0
    api_calls = 0
    sync_errors = 0

    # 1. Process DELETED documents first
    for doc in deleted_docs:
        try:
            print(f"\nRemoving deleted document: {doc['filename']}")
            # Delete old chunks
            deleted_count = store.delete_chunks_by_hash(doc["file_hash"])
            # Fallback delete by source in case of old ID format
            deleted_count += store.delete_source_chunks(doc["filename"])
            chunks_deleted += deleted_count
            # Remove from registry
            registry.delete_document(doc["relative_path"])
        except (VectorStoreError, RegistryError) as error:
            print(f"Error removing {doc['filename']}: {error}")
            sync_errors += 1

    # Helper function for ingesting a document (used for both NEW and MODIFIED)
    def process_document(doc: dict, is_modified: bool = False, old_hash: str = None) -> bool:
        nonlocal chunks_added, chunks_deleted, api_calls
        doc_path = doc["path"]
        doc_hash = doc["file_hash"]

        # Limit parser to PDF for now
        if doc["file_type"] != "pdf":
            print(f"Error processing {doc['filename']}: Multi-format PPTX/DOCX parsing is not configured in requirements.txt.")
            return False

        try:
            pages = extract_pdf_pages(doc_path)
            chunks = create_chunks(doc_path, pages, doc_hash)
            if not chunks:
                print(f"No content extracted from {doc['filename']}.")
                return False

            # Embed chunks
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_texts(texts)
            api_calls += len(texts)

            # Store new chunks in ChromaDB
            store.upsert(
                ids=[c.id for c in chunks],
                documents=texts,
                metadatas=[c.metadata for c in chunks],
                embeddings=embeddings,
            )
            chunks_added += len(chunks)

            # Remove the old chunks *after* confirming new chunks are successfully stored
            old_deleted = 0
            if old_hash:
                old_deleted += store.delete_chunks_by_hash(old_hash)
            # Remove any chunks matching this source filename that are not in the new set
            old_deleted += store.remove_stale_source_chunks(doc["filename"], {c.id for c in chunks})
            chunks_deleted += old_deleted

            # Update Document Registry
            doc_id = sha256(doc["relative_path"].encode("utf-8")).hexdigest()[:16]
            registry.add_or_update_document(
                document_id=doc_id,
                filename=doc["filename"],
                relative_path=doc["relative_path"],
                file_type=doc["file_type"],
                file_hash=doc_hash,
                last_modified=doc["last_modified"],
                chunk_count=len(chunks),
                sync_status="synchronized",
            )
            return True

        except Exception as error:
            print(f"Error processing document {doc['filename']}: {error}")
            # Mark document as failed in the registry so it retries next run
            try:
                doc_id = sha256(doc["relative_path"].encode("utf-8")).hexdigest()[:16]
                registry.add_or_update_document(
                    document_id=doc_id,
                    filename=doc["filename"],
                    relative_path=doc["relative_path"],
                    file_type=doc["file_type"],
                    file_hash=doc_hash,
                    last_modified=doc["last_modified"],
                    chunk_count=0,
                    sync_status="failed",
                )
            except RegistryError:
                pass
            return False

    # 2. Process NEW documents
    for doc in new_docs:
        print(f"\nProcessing new document: {doc['filename']}...")
        success = process_document(doc, is_modified=False)
        if not success:
            sync_errors += 1

    # 3. Process MODIFIED documents
    for doc in modified_docs:
        print(f"\nProcessing modified document: {doc['filename']}...")
        success = process_document(doc, is_modified=True, old_hash=doc["old_hash"])
        if not success:
            sync_errors += 1

    # Print final synchronization summary
    print("\nSummary:")
    print(f"Chunks added: {chunks_added}")
    print(f"Chunks updated: {chunks_added if (chunks_added and chunks_deleted) else 0}")
    print(f"Chunks deleted: {chunks_deleted}")
    print(f"Embedding API calls: {api_calls}")

    if sync_errors:
        print(f"\nSynchronization completed with {sync_errors} error(s).")
        sys.exit(1)
    else:
        print("\nSynchronization complete.")


if __name__ == "__main__":
    main()
