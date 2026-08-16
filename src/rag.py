"""RAG orchestration: retrieve study chunks, build context, and generate an answer."""

import argparse
from dataclasses import dataclass
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from src.config import DEFAULT_TOP_K
from src.embeddings import EmbeddingError
from src.generator import GenerationError, NvidiaGenerator
from src.retriever import RetrievalError, RetrievalResult, StudyRetriever
from src.vector_store import VectorStoreError


class RAGError(RuntimeError):
    """Raised when a RAG question cannot be answered safely."""


@dataclass(frozen=True)
class SourceReference:
    """Verified source metadata citation."""

    id: int
    source: str
    type: str
    page: int | None = None
    slide: int | None = None
    section: str | None = None
    chunk_index: int | None = None


@dataclass(frozen=True)
class RAGResult:
    """Generated answer and the retrieved chunks used as context."""

    answer: str
    sources: list[SourceReference]


def get_chunk_source_info(metadata: dict) -> tuple[tuple, dict]:
    """Extract location parameters and format source info structure."""
    source = metadata.get("source", "Unknown")
    file_type = metadata.get("document_type", "pdf")
    
    page = metadata.get("page_number")
    slide = metadata.get("slide_number")
    section = metadata.get("section")
    chunk_index = metadata.get("chunk_index")
    
    # Construct a unique key for deduplication
    if page is not None:
        key = (source, "page", page)
        info = {"source": source, "type": file_type, "page": page}
    elif slide is not None:
        key = (source, "slide", slide)
        info = {"source": source, "type": file_type, "slide": slide}
    elif section is not None:
        key = (source, "section", section)
        info = {"source": source, "type": file_type, "section": section}
    else:
        key = (source, "chunk", chunk_index)
        info = {"source": source, "type": file_type, "chunk_index": chunk_index}
        
    return key, info


def build_context(results: list[RetrievalResult], mapping: list[int] | None = None) -> str:
    """Format retrieved chunks with their original metadata for the LLM prompt."""
    context_sections: list[str] = []
    for position, result in enumerate(results, start=1):
        metadata = result.metadata
        source = metadata.get("source", "Unknown")
        source_id = mapping[position - 1] if mapping else position
        
        meta_lines = [f"SOURCE [{source_id}]", f"Document: {source}"]
        if "page_number" in metadata:
            meta_lines.append(f"Page: {metadata['page_number']}")
        if "slide_number" in metadata:
            meta_lines.append(f"Slide: {metadata['slide_number']}")
        if "section" in metadata:
            meta_lines.append(f"Section: {metadata['section']}")
        if "chunk_index" in metadata:
            meta_lines.append(f"Chunk: {metadata['chunk_index']}")
            
        meta_header = "\n".join(meta_lines)
        context_sections.append(
            f"{meta_header}\n\n"
            f"{result.text}"
        )
    return "\n\n---\n\n".join(context_sections)


def extract_citations(answer: str, max_valid_id: int) -> set[int]:
    """Parse valid citations in formats like [1], [1, 2], [1] [2] from the answer."""
    bracketed = re.findall(r'\[([^\]]+)\]', answer)
    citations = set()
    for item in bracketed:
        parts = re.split(r'[\s,]+', item)
        for p in parts:
            if p.isdigit():
                val = int(p)
                if 1 <= val <= max_valid_id:
                    citations.add(val)
    return citations


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

    # Map unique sources to sequential IDs (1-based)
    unique_sources_map = {}
    unique_sources_list = []
    mapping = []
    
    for result in results:
        key, info = get_chunk_source_info(result.metadata)
        if key not in unique_sources_map:
            new_id = len(unique_sources_map) + 1
            unique_sources_map[key] = new_id
            info["id"] = new_id
            unique_sources_list.append(info)
        mapping.append(unique_sources_map[key])

    context = build_context(results, mapping)
    answer = NvidiaGenerator().generate(query, context)
    
    # Extract and validate citations cited in the generated answer
    cited_ids = extract_citations(answer, len(unique_sources_list))
    
    # Convert validated unique sources to SourceReference objects
    sources = []
    for info in unique_sources_list:
        if info["id"] in cited_ids:
            sources.append(
                SourceReference(
                    id=info["id"],
                    source=info["source"],
                    type=info["type"],
                    page=info.get("page"),
                    slide=info.get("slide"),
                    section=info.get("section"),
                    chunk_index=info.get("chunk_index"),
                )
            )
            
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
        print("No verified sources.")
        return
    for source in result.sources:
        location = ""
        if source.page is not None:
            location = f" — Page {source.page}"
        elif source.slide is not None:
            location = f" — Slide {source.slide}"
        elif source.section is not None:
            location = f" — Section: {source.section}"
        elif source.chunk_index is not None:
            location = f" — Chunk {source.chunk_index}"
            
        print(f"[{source.id}] {source.source}{location}")


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
