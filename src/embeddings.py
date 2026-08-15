"""NVIDIA API embedding client for Study RAG ingestion."""

import os
import time
from collections.abc import Sequence

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, OpenAI, RateLimitError

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    NVIDIA_API_BASE_URL,
    NVIDIA_EMBEDDING_MODEL,
    PROJECT_ROOT,
)


class EmbeddingError(RuntimeError):
    """Raised when the NVIDIA embedding service cannot produce vectors."""


class NvidiaEmbedder:
    """Generate passage embeddings through NVIDIA's OpenAI-compatible API."""

    def __init__(self) -> None:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise EmbeddingError(
                "NVIDIA_API_KEY is missing. Add it to .env before running ingestion."
            )

        self._client = OpenAI(api_key=api_key, base_url=NVIDIA_API_BASE_URL)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one float embedding for each non-empty input text."""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Cannot generate embeddings for empty chunks.")

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
            embeddings.extend(self._embed_batch(batch))
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
            try:
                response = self._client.embeddings.create(
                    model=NVIDIA_EMBEDDING_MODEL,
                    input=texts,
                    encoding_format="float",
                    extra_body={"input_type": "passage", "modality": "text"},
                )
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors = [list(item.embedding) for item in ordered]
                if len(vectors) != len(texts):
                    raise EmbeddingError("NVIDIA returned an incomplete embedding batch.")
                return vectors
            except (RateLimitError, APIConnectionError) as error:
                if attempt == EMBEDDING_MAX_RETRIES:
                    raise EmbeddingError(
                        "NVIDIA embedding request failed after retries. Check network access "
                        "and API rate limits."
                    ) from error
                time.sleep(2 ** (attempt - 1))
            except APIStatusError as error:
                if error.status_code >= 500 and attempt < EMBEDDING_MAX_RETRIES:
                    time.sleep(2 ** (attempt - 1))
                    continue
                raise EmbeddingError(
                    "NVIDIA embedding request was rejected. Check the API key, model access, "
                    "and request size."
                ) from error
            except APIError as error:
                raise EmbeddingError("NVIDIA embedding request failed.") from error

        raise EmbeddingError("NVIDIA embedding request failed.")
