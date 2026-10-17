"""Central configuration for the Study RAG project."""

import math
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
REGISTRY_DB_PATH = CHROMA_DB_DIR / "registry.db"


def _read_float_setting(name: str, default: str) -> float:
    value = os.getenv(name, default)
    try:
        parsed = float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number; received {value!r}.") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number; received {value!r}.")
    return parsed


def _read_int_setting(name: str, default: str) -> int:
    value = os.getenv(name, default)
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer; received {value!r}.") from None


NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"

CHROMA_COLLECTION_NAME = "study_rag_knowledge_base"
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_RETRIES = 3
DEFAULT_TOP_K = 5
LLM_TEMPERATURE = _read_float_setting("LLM_TEMPERATURE", "0.2")
LLM_TOP_P = _read_float_setting("LLM_TOP_P", "0.7")
LLM_MAX_TOKENS = _read_int_setting("LLM_MAX_TOKENS", "1024")
LLM_TIMEOUT_SECONDS = _read_float_setting("LLM_TIMEOUT_SECONDS", "60")

if not 0 <= LLM_TEMPERATURE <= 2:
    raise ValueError("LLM_TEMPERATURE must be between 0 and 2.")
if not 0 <= LLM_TOP_P <= 1:
    raise ValueError("LLM_TOP_P must be between 0 and 1.")
if LLM_MAX_TOKENS < 1:
    raise ValueError("LLM_MAX_TOKENS must be at least 1.")
if LLM_TIMEOUT_SECONDS <= 0:
    raise ValueError("LLM_TIMEOUT_SECONDS must be greater than 0.")


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
