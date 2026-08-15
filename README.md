# Study RAG

A Python study assistant that retrieves material from locally persisted ChromaDB and generates grounded answers with NVIDIA APIs.

## Project layout

- `data/`: input study materials.
- `chroma_db/`: local ChromaDB persistence data.
- `src/`: application source code.
- `.env`: local secrets and configuration; never commit it.

## Phase 2: embeddings and ChromaDB

The pipeline extracts text from PDFs, creates page-aware chunks, generates embeddings with NVIDIA's API, and stores vectors locally in ChromaDB. Retrieval and generation are not implemented.

Set the NVIDIA configuration in `.env`:

```text
NVIDIA_EMBEDDING_API_KEY=your_embedding_api_key
NVIDIA_LLM_API_KEY=your_llm_api_key
NVIDIA_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-1b-v2
NVIDIA_LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
LLM_TEMPERATURE=0.2
LLM_TOP_P=0.7
LLM_MAX_TOKENS=1024
```

`.env` is ignored by Git. The embedding and LLM credentials and model names are independently configurable.

## Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run ingestion

Place PDFs in `data/`, then run from the repository root. With no filename, every top-level PDF is processed:

```powershell
.\.venv\Scripts\python.exe -m src.ingest
```

To process one document:

```powershell
.\.venv\Scripts\python.exe -m src.ingest notes.pdf
```

ChromaDB data is persisted under `chroma_db/` and is excluded from Git. Each record contains a stable chunk ID, the original chunk text, its NVIDIA embedding, and metadata for the source filename, page number, chunk index, and document type. Re-running ingestion skips unchanged chunks rather than duplicating them.

## Phase 3: retrieval

Retrieval embeds only the user's query with the same NVIDIA embedding model used for ingestion. The query vector is then sent to ChromaDB for a top-K similarity search over the persisted document vectors; documents are not re-embedded.

Run a retrieval test:

```powershell
.\.venv\Scripts\python.exe -m src.retriever "What topics are covered in the computer networks course?"
```

`top_k` defaults to 5 and controls the maximum number of nearest chunks returned. Use `--top-k` to change it, and optionally limit the search with stored metadata:

```powershell
.\.venv\Scripts\python.exe -m src.retriever "What is TCP?" --top-k 3 --source Networks-syl.pdf
```

Each result prints its chunk ID, source filename, page number, chunk index, ChromaDB distance, and original text. Retrieval returns the closest chunks even for unrelated questions; no answer generation or similarity threshold is implemented.

## Phase 4: RAG generation

Phase 4 connects the retrieval pipeline to the NVIDIA-hosted LLM `nvidia/nemotron-3.5-lightning-30b-a3b` to generate grounded answers using the retrieved study material.

### RAG Generation Architecture

```text
User Question
      ↓
Existing Phase 3 Retriever (NvidiaEmbedder + ChromaDB)
      ↓
Nearest K Relevant Chunks
      ↓
Context Builder (Structured layout with Document/Page/Chunk metadata)
      ↓
NVIDIA LLM (nvidia/nemotron-3.5-lightning-30b-a3b)
      ↓
Grounded Answer + Preserved Sources
```

### NVIDIA API Configuration

Add the following environment variables to your `.env` file (which is gitignored):

```text
NVIDIA_EMBEDDING_API_KEY=your_embedding_api_key
NVIDIA_LLM_API_KEY=your_llm_api_key

NVIDIA_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-1b-v2
NVIDIA_LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b

LLM_TEMPERATURE=0.2
LLM_TOP_P=0.7
LLM_MAX_TOKENS=1024
```

### Run the RAG CLI

To answer questions from the ingested study material:

```powershell
.\.venv\Scripts\python.exe -m src.rag "Explain the TCP three-way handshake"
```

To limit the retriever to search a specific source file or document type:

```powershell
.\.venv\Scripts\python.exe -m src.rag "Explain connection establishment" --source Networks-syl.pdf
```

### Example Output

```text
QUESTION:
What is the course code and title of the networks course in the syllabus?

ANSWER:
Based on the syllabus document provided, the course details are:

- **Course Code:** CS23502
- **Course Title:** NETWORKS AND DATA COMMUNICATION

This is specified at the beginning of the syllabus document under Unit I.

SOURCES:
1. ComputerNetworking.pdf — Page 18, Chunk 43
2. Networks-syl.pdf — Page 1, Chunk 0
3. Networks-syl.pdf — Page 2, Chunk 3
```

No chat memory, web search, or additional external search strategies are implemented.

