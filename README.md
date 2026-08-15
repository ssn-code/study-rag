# Study RAG

A Python study-material ingestion project using NVIDIA embeddings and locally persisted ChromaDB.

## Project layout

- `data/`: input study materials.
- `chroma_db/`: local ChromaDB persistence data.
- `src/`: application source code.
- `.env`: local secrets and configuration; never commit it.

## Phase 2: embeddings and ChromaDB

The pipeline extracts text from PDFs, creates page-aware chunks, generates embeddings with NVIDIA's API, and stores vectors locally in ChromaDB. Retrieval and generation are not implemented.

Set `NVIDIA_API_KEY` in `.env`:

```text
NVIDIA_API_KEY=your_key_here
```

The embedding model is configured in `src/config.py` as `nvidia/llama-nemotron-embed-1b-v2`.

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
