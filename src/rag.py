"""RAG orchestration: retrieve study chunks, build context, and generate an answer."""

import argparse
from dataclasses import dataclass
import sys

from src.config import DEFAULT_TOP_K
from src.embeddings import EmbeddingError
from src.generator import GenerationError, NvidiaGenerator
from src.retriever import RetrievalError, RetrievalResult, StudyRetriever
from src.vector_store import VectorStoreError


class RAGError(RuntimeError):
    """Raised when a RAG question cannot be answered safely."""


@dataclass(frozen=True)
class SourceReference:
    """Source metadata retained alongside a grounded generated answer."""

    source: str
    page_number: int
    chunk_index: int


@dataclass(frozen=True)
class RAGResult:
    """Generated answer and the retrieved chunks used as context."""

    answer: str
    sources: list[SourceReference]


def build_context(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks with their original metadata for the LLM prompt."""
    context_sections: list[str] = []
    for position, result in enumerate(results, start=1):
        metadata = result.metadata
        try:
            source = metadata["source"]
            page_number = metadata["page_number"]
            chunk_index = metadata["chunk_index"]
        except KeyError as error:
            raise RAGError("A retrieved chunk is missing required source metadata.") from error
        if not isinstance(source, str) or not isinstance(page_number, int) or not isinstance(chunk_index, int):
            raise RAGError("A retrieved chunk contains invalid source metadata.")
        context_sections.append(
            f"SOURCE {position}\n"
            f"Document: {source}\n"
            f"Page: {page_number}\n"
            f"Chunk: {chunk_index}\n\n"
            f"{result.text}"
        )
    return "\n\n---\n\n".join(context_sections)


def answer_query(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, str] | None = None,
) -> RAGResult:
    """Retrieve context and return a generated answer plus its source metadata."""
    if not query or not query.strip():
        raise RAGError("Query cannot be empty.")

    results = StudyRetriever().retrieve(query, top_k=top_k, filters=filters)
    if not results:
        return RAGResult("No study chunks were retrieved for this question.", [])

    context = build_context(results)
    answer = NvidiaGenerator().generate(query, context)
    sources = [
        SourceReference(
            source=result.metadata["source"],
            page_number=result.metadata["page_number"],
            chunk_index=result.metadata["chunk_index"],
        )
        for result in results
    ]
    return RAGResult(answer=answer, sources=sources)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a Study RAG question.")
    parser.add_argument("query", help="Question to answer from the ingested study material.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Maximum chunks to use.")
    parser.add_argument("--source", help="Only retrieve chunks from this source filename.")
    parser.add_argument("--document-type", help="Only retrieve this document type, such as pdf.")
    return parser.parse_args()


def print_result(query: str, result: RAGResult) -> None:
    print(f"QUESTION:\n{query}\n")
    print(f"ANSWER:\n{result.answer}\n")
    print("SOURCES:")
    if not result.sources:
        print("No retrieved sources.")
        return
    for position, source in enumerate(result.sources, start=1):
        print(f"{position}. {source.source} — Page {source.page_number}, Chunk {source.chunk_index}")


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
        result = answer_query(args.query, top_k=args.top_k, filters=filters)
        print_result(args.query, result)
    except (EmbeddingError, VectorStoreError, RetrievalError, GenerationError, RAGError) as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
