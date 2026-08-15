"""Central configuration for the Study RAG project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

NVIDIA_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
NVIDIA_GENERATION_MODEL = "meta/llama-3.3-70b-instruct"
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"

CHROMA_COLLECTION_NAME = "study_rag_knowledge_base"
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_RETRIES = 3
DEFAULT_TOP_K = 5
