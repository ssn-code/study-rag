"""Command-line retrieval over the persisted Study RAG ChromaDB collection."""

import argparse
from dataclasses import dataclass
import sys

from src.config import DEFAULT_TOP_K
from src.embeddings import EmbeddingError, NvidiaEmbedder
from src.vector_store import StudyVectorStore, VectorStoreError


class RetrievalError(RuntimeError):
    """Raised when a retrieval request is invalid."""


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved document chunk and ChromaDB's distance for the query."""

    id: str
    text: str
    metadata: dict[str, str | int]
    distance: float | None


class StudyRetriever:
    """Embed a user query and return its nearest persisted document chunks."""

    def __init__(self) -> None:
        self._embedder = NvidiaEmbedder()
        self._store = StudyVectorStore(create_if_missing=False)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        filters: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty.")
        if top_k < 1:
            raise RetrievalError("top_k must be at least 1.")

        where = self._validate_filters(filters)
        query_embedding = self._embedder.embed_query(query.strip())
        response = self._store.similarity_search(
            query_embedding,
            top_k=top_k,
            where=where or None,
        )

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        return [
            RetrievalResult(
                id=chunk_id,
                text=document,
                metadata=metadata,
                distance=distance,
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    @staticmethod
    def _validate_filters(filters: dict[str, str] | None) -> dict[str, str]:
        if not filters:
            return {}
        allowed = {"source", "document_type"}
        invalid = set(filters) - allowed
        if invalid:
            raise RetrievalError(
                f"Unsupported metadata filter(s): {', '.join(sorted(invalid))}."
            )
        if any(not isinstance(value, str) or not value.strip() for value in filters.values()):
            raise RetrievalError("Metadata filter values must be non-empty strings.")
        return filters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve relevant Study RAG chunks.")
    parser.add_argument("query", help="Question or search text to embed and search for.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Maximum chunks to return.")
    parser.add_argument("--source", help="Only search chunks from this source filename.")
    parser.add_argument("--document-type", help="Only search this document type, such as pdf.")
    return parser.parse_args()


def print_results(query: str, results: list[RetrievalResult]) -> None:
    print(f"Query:\n{query}\n")
    if not results:
        print("No matching chunks were found.")
        return

    print("Results:")
    for position, result in enumerate(results, start=1):
        print(f"\n[{position}]")
        print(f"ID: {result.id}")
        print(f"Source: {result.metadata['source']}")
        print(f"Page: {result.metadata['page_number']}")
        print(f"Chunk: {result.metadata['chunk_index']}")
        if result.distance is not None:
            print(f"Distance: {result.distance:.4f}")
        print(f"Text:\n{result.text}")


def main() -> None:
    args = parse_args()
    filters = {
        name: value
        for name, value in {
            "source": args.source,
            "document_type": args.document_type,
        }.items()
        if value is not None
    }
    try:
        results = StudyRetriever().retrieve(args.query, top_k=args.top_k, filters=filters)
        print_results(args.query, results)
    except (EmbeddingError, VectorStoreError, RetrievalError) as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
