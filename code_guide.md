# Study RAG Codebase Guide

This document provides a detailed walkthrough of the **Study RAG** application codebase, explaining the role, configuration, and step-by-step logic of each file.

---

## 1. System Architecture Overview

The application follows a standard **Retrieval-Augmented Generation (RAG)** architecture using:
- **PDF Extraction**: `pypdf` for reading local documents.
- **Vector Database**: `ChromaDB` (persistent local database) for storing vector representations of text.
- **LLM/Embeddings Services**: NVIDIA API integration via the `openai` Python SDK.

```
       [1. Ingestion Phase]
          PDF Files (data/)
                 ↓
         extract_pdf_pages()
                 ↓
          create_chunks()
                 ↓
     NvidiaEmbedder.embed_texts()
                 ↓
       StudyVectorStore.upsert() -> persisted in chroma_db/
                 

       [2. Retrieval & Generation Phase]
            User Question
                 ↓
     NvidiaEmbedder.embed_query()
                 ↓
       StudyVectorStore.similarity_search()
                 ↓
          build_context()
                 ↓
        NvidiaGenerator.generate() -> NVIDIA LLM
                 ↓
            Grounded Answer
```

---

## 2. File-by-File Breakdown

### 📂 `src/config.py`
**Purpose**: Centralized configuration management.
- **Key Variables**:
  - `DATA_DIR` & `CHROMA_DB_DIR`: Set file paths relative to the project root directory.
  - `CHUNK_SIZE` (1,000) & `CHUNK_OVERLAP` (150): Control the token-like size and window overlap for text chunking.
  - `NVIDIA_EMBEDDING_MODEL` & `NVIDIA_LLM_MODEL`: API target models.
- **Helper Functions**:
  - `get_embedding_api_key()`: Retrieves the credentials for embedding, throwing a clear `ValueError` if missing.
  - `get_llm_api_key()`: Retrieves the credentials for the language model, throwing a `ValueError` if missing.

---

### 📂 `src/embeddings.py`
**Purpose**: Interfaces with the NVIDIA Embeddings API using the `openai` client library.
- **`NvidiaEmbedder` Class**:
  - `__init__()`: Initializes the `OpenAI` client using NVIDIA's base URL (`https://integrate.api.nvidia.com/v1`).
  - `embed_texts(texts)`: Splits input texts into batches of size `EMBEDDING_BATCH_SIZE` (32) and requests embeddings.
  - `embed_query(query)`: Generates an embedding specifically for the search query (flagged as `input_type="query"`).
  - `_embed_batch(texts, input_type)`: Performs the network request. It contains a retry loop up to `EMBEDDING_MAX_RETRIES` (3) with exponential backoff (`2 ** (attempt - 1)`) to handle rate limits and transient connection issues.

---

### 📂 `src/vector_store.py`
**Purpose**: Manages interactions with the local ChromaDB instance.
- **`StudyVectorStore` Class**:
  - `__init__(create_if_missing)`: Connects to a persistent Chroma database (`chroma_db/`) using the cosine similarity space (`hnsw:space: cosine`).
  - `existing_ids(ids)`: Returns the subset of IDs that already exist in the database (used to skip re-indexing duplicate chunks).
  - `upsert(ids, documents, metadatas, embeddings)`: Writes chunks and their vectors to the database.
  - `remove_stale_source_chunks(source, current_ids)`: Detects and deletes chunks that were stored under a source filename in a previous run but are no longer present (helps keep indices clean when files are edited).
  - `similarity_search(query_embedding, top_k, where)`: Queries ChromaDB for the closest `top_k` matches, with optional metadata filters.

---

### 📂 `src/ingest.py`
**Purpose**: The CLI entrypoint for extracting PDF text and seeding the vector database.
- **Helper Functions**:
  - `extract_pdf_pages(pdf_path)`: Uses `PdfReader` to extract clean text from each PDF page.
  - `create_chunks(pdf_path, pages)`: Splits each page's text into chunks of size `CHUNK_SIZE` with a sliding overlap window. It assigns each chunk a stable hash ID:
    ```python
    stable_value = f"{pdf_path.name}|{page_number}|{chunk_index}|{chunk_text}"
    id = sha256(stable_value.encode("utf-8")).hexdigest()
    ```
  - `ingest_pdf(...)`: Orchestrates extraction, filters out already existing chunks, runs embedding generation on new chunks, upserts them to the store, and cleans up stale items.
  - `main()`: Command-line driver that processes single files or scans the entire `data/` directory.

---

### 📂 `src/retriever.py`
**Purpose**: CLI entrypoint and class to embed queries and query the vector database.
- **`StudyRetriever` Class**:
  - `retrieve(query, top_k, filters)`: Encapsulates the process of embedding the query text and running a similarity search in the vector store.
  - `_validate_filters(filters)`: Limits filter keys strictly to `source` and `document_type` and validates that query constraints are string values.
- **`main()`**: CLI script that parses incoming arguments, executes retrieval, and outputs matching details including chunk IDs, source document pages, and the similarity distances.

---

### 📂 `src/generator.py`
**Purpose**: Handles grounded text generation using retrieved knowledge.
- **`NvidiaGenerator` Class**:
  - `generate(question, context)`: Sends a system prompt and a structured instruction prompt containing context sections alongside the user's question.
  - **`SYSTEM_PROMPT`**: Directs the LLM to behave as a careful, student-friendly assistant. It strictly instructs the model:
    1. To answer using *only* the retrieved context.
    2. To explicitly say if context is insufficient.
    3. To omit any thinking process, internal reasoning, or monologue, returning only the direct answer.
  - Retries requests with exponential backoff on connection or rate limits.

---

### 📂 `src/rag.py`
**Purpose**: Orchestrates the full end-to-end RAG pipeline.
- **Helper Functions**:
  - `build_context(results)`: Formats multiple retrieval chunks into a clear, structured template containing the source labels, page numbers, and chunk text.
  - `answer_query(query, top_k, filters)`: Runs the full pipeline:
    1. Retrieves top chunks using `StudyRetriever`.
    2. Builds the structured context.
    3. Runs LLM generation using `NvidiaGenerator`.
    4. Gathers source references (`source`, `page_number`, `chunk_index`) to return alongside the final response.
- **`main()`**: Parses arguments, processes the RAG pipeline, and outputs the final answer and source list to the terminal.

---

## 3. Workflow Control Flow

### Ingestion Flow
1. Run `python -m src.ingest`
2. `ingest.py` scans `data/*.pdf`.
3. For each PDF, text is extracted page by page, and then split into chunks.
4. Active chunk IDs are queried from `vector_store.py` (`existing_ids`).
5. Only new/modified chunks are sent to `embeddings.py` for representation.
6. New chunks and embeddings are stored, while stale chunks from prior ingestion runs are deleted.

### Question-Answering (RAG) Flow
1. Run `python -m src.rag "Question text"`
2. `rag.py` calls `StudyRetriever.retrieve()`.
3. Query text is embedded into a vector representation by `NvidiaEmbedder`.
4. `StudyVectorStore` queries ChromaDB to find closest matches based on cosine distance.
5. `rag.py` formats these matches into a markdown context block.
6. The context block is sent to `NvidiaGenerator.generate()` alongside the system rules.
7. The assistant returns the direct answer, which is printed along with the page/document sources.
