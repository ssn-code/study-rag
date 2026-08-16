"""Central configuration for the Study RAG project."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
REGISTRY_DB_PATH = CHROMA_DB_DIR / "registry.db"

NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"

CHROMA_COLLECTION_NAME = "study_rag_knowledge_base"
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_RETRIES = 3
DEFAULT_TOP_K = 5
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))


def get_embedding_api_key() -> str:
    """Return the credential reserved for NVIDIA embedding requests."""
    api_key = os.getenv("NVIDIA_EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError(
            "Embedding configuration error: NVIDIA_EMBEDDING_API_KEY is missing."
        )
    return api_key


def get_llm_api_key() -> str:
    """Return the credential reserved for NVIDIA LLM generation requests."""
    api_key = os.getenv("NVIDIA_LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM configuration error: NVIDIA_LLM_API_KEY is missing.")
    return api_key
