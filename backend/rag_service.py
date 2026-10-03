import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator

import lmstudio as lms
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

load_dotenv()

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", "qdrant_data")).expanduser()
if not QDRANT_PATH.is_absolute():
    QDRANT_PATH = BACKEND_DIR / QDRANT_PATH

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
LM_STUDIO_EMBEDDING_MODEL = os.getenv(
    "LM_STUDIO_EMBEDDING_MODEL",
    "Qwen/Qwen3-Embedding-0.6B-GGUF",
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
DOCUMENT_PREFIX = os.getenv("EMBEDDING_DOCUMENT_PREFIX", "")
QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX",
    "Instruct: Given a question, retrieve relevant passages from the user's books "
    "that answer the question.\nQuery: ",
)
STRUCTURING_MODEL = os.getenv(
    "STRUCTURING_MODEL",
    "lmstudio-community/Qwen3-4B-Instruct-2507-GGUF",
)


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be set to true or false.")


ENABLE_SEMANTIC_CHUNKING = _read_bool_env("ENABLE_SEMANTIC_CHUNKING", True)

STRUCTURING_SYSTEM_PROMPT = """You are a document structuring engine for a Retrieval-Augmented Generation (RAG) system.

Analyze the document and split it into logically coherent retrieval chunks.

Requirements:

- Detect sections and subsections.
- Infer hierarchy when headings are missing.
- Group related information together.
- Preserve original wording.
- Do not summarize away important details.
- Create retrieval-friendly metadata.

Return ONLY valid JSON.

Schema:

[
  {
    "title": "document title",
    "section": "section name",
    "subsection": "subsection name or null",
    "header_path": ["root","section","subsection"],
    "summary": "short description",
    "keywords": ["keyword1","keyword2"],
    "chunk_type": "overview",
    "content": "original text"
  }
]

No markdown.
No explanations.
JSON only."""


class LMStudioModelUnloadError(RuntimeError):
    """Raised when LM Studio cannot release a model after an operation."""


class RAGService:
    def __init__(self, collection_name="rag_collection"):
        self.collection_name = collection_name
        self._model_lock = threading.Lock()

        if EMBEDDING_DIMENSION <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be a positive integer")

        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(QDRANT_PATH))

        self.vector_size = EMBEDDING_DIMENSION

        self._ensure_collection()

    @contextmanager
    def _loaded_model(
        self,
        loader: Callable[[str], Any],
        model_name: str,
    ) -> Generator[Any, None, None]:
        """Load one LM Studio model at a time and always attempt to unload it."""
        with self._model_lock:
            model = loader(model_name)
            try:
                yield model
            finally:
                try:
                    model.unload()
                except Exception as error:
                    logger.exception("Failed to unload LM Studio model '%s'.", model_name)
                    raise LMStudioModelUnloadError(
                        f"LM Studio failed to unload model '{model_name}'. "
                        "Further model operations were stopped to avoid keeping "
                        "multiple models loaded."
                    ) from error

    def structure_document(self, content: str) -> list[dict[str, Any]]:
        """Ask the configured local chat model to create semantic chunks and metadata."""
        with self._loaded_model(lms.llm, STRUCTURING_MODEL):
            response = requests.post(
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                json={
                    "model": STRUCTURING_MODEL,
                    "messages": [
                        {"role": "system", "content": STRUCTURING_SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                    "temperature": 0,
                },
                timeout=300,
            )
            response.raise_for_status()
            response_content = response.json()["choices"][0]["message"]["content"]
        parsed_chunks = json.loads(response_content)
        return self._validate_structured_chunks(parsed_chunks)

    def _validate_structured_chunks(self, chunks: Any) -> list[dict[str, Any]]:
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Structuring model response must be a non-empty JSON array.")

        validated_chunks = []
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Structured chunk {index} must be a JSON object.")

            content = chunk.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Structured chunk {index} has no valid content.")

            header_path = chunk.get("header_path", [])
            keywords = chunk.get("keywords", [])
            if not isinstance(header_path, list) or not all(
                isinstance(item, str) for item in header_path
            ):
                raise ValueError(f"Structured chunk {index} has an invalid header_path.")
            if not isinstance(keywords, list) or not all(
                isinstance(item, str) for item in keywords
            ):
                raise ValueError(f"Structured chunk {index} has invalid keywords.")

            validated_chunks.append(
                {
                    "title": self._optional_text(chunk.get("title")),
                    "section": self._optional_text(chunk.get("section")),
                    "subsection": self._nullable_text(chunk.get("subsection")),
                    "header_path": header_path,
                    "summary": self._optional_text(chunk.get("summary")),
                    "keywords": keywords,
                    "chunk_type": self._optional_text(chunk.get("chunk_type")),
                    "content": content,
                }
            )

        return validated_chunks

    @staticmethod
    def _optional_text(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("Structuring metadata fields must be strings or null.")
        return value

    @staticmethod
    def _nullable_text(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Structuring metadata fields must be strings or null.")
        return value

    def _fallback_chunks(self, content: str, filename: str) -> list[dict[str, Any]]:
        title = Path(filename).stem
        return [
            {
                "title": title,
                "section": "",
                "subsection": None,
                "header_path": [],
                "summary": "",
                "keywords": [],
                "chunk_type": "text",
                "content": chunk,
            }
            for chunk in self._chunk_text(content)
        ]

    def _document_chunks(self, content: str, filename: str) -> list[dict[str, Any]]:
        if not ENABLE_SEMANTIC_CHUNKING:
            logger.info("Semantic chunking is disabled; using fixed-size fallback chunks.")
            return self._fallback_chunks(content, filename)

        try:
            return self.structure_document(content)
        except LMStudioModelUnloadError:
            raise
        except Exception:
            logger.exception(
                "Semantic structuring failed for %s; falling back to fixed-size chunks.",
                filename,
            )
            return self._fallback_chunks(content, filename)

    @staticmethod
    def _build_search_text(chunk: dict[str, Any]) -> str:
        return (
            f"Title: {chunk['title']}\n\n"
            f"Section: {chunk['section']}\n\n"
            f"Subsection: {chunk['subsection']}\n\n"
            f"Summary: {chunk['summary']}\n\n"
            f"Keywords: {','.join(chunk['keywords'])}\n\n"
            f"{chunk['content']}"
        )

    def _ensure_collection(self):
        collections = self.client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if self.collection_name in collection_names:
            collection_info = self.client.get_collection(self.collection_name)
            existing_size = collection_info.config.params.vectors.size
            if existing_size != self.vector_size:
                raise ValueError(
                    f"Qdrant collection '{self.collection_name}' has vector size "
                    f"{existing_size}, but the configured embedding dimension is "
                    f"{self.vector_size}. Set EMBEDDING_DIMENSION to match or use a "
                    "different collection name; the existing collection was not changed."
                )
            print(f"Collection '{self.collection_name}' exists with correct dimensions.")
            return

        print(f"Creating collection '{self.collection_name}' with {self.vector_size} dimensions...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        print("Collection created successfully.")

    def generate_embedding(self, text: str):
        with self._loaded_model(lms.embedding_model, LM_STUDIO_EMBEDDING_MODEL) as model:
            return self._request_embedding(model, f"{DOCUMENT_PREFIX}{text}")

    def loaded_chat_model(self, model_name: str):
        """Load a chat model within the shared, serialized LM Studio model lifecycle."""
        return self._loaded_model(lms.llm, model_name)

    def generate_query_embedding(self, text: str):
        with self._loaded_model(lms.embedding_model, LM_STUDIO_EMBEDDING_MODEL) as model:
            return self._request_embedding(model, f"{QUERY_PREFIX}{text}")

    def _request_embedding(self, model: Any, text: str):
        embedding = model.embed(text)
        if len(embedding) != self.vector_size:
            raise ValueError(
                f"LM Studio returned an embedding with {len(embedding)} dimensions, "
                f"but EMBEDDING_DIMENSION is set to {self.vector_size}."
            )
        return embedding

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50):
        """
        Splits text into overlapping chunks of words.
        Args:
            text: The text to chunk
            chunk_size: Number of words per chunk
            overlap: Number of words to overlap between chunks
        """
        words = text.split()
        if not words:
            return []

        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            if end >= len(words):
                break

            start += chunk_size - overlap

        return chunks

    def ingest_document(self, content: str, filename: str):
        print(f"Ingesting document: {filename}, length: {len(content)}")

        chunks = self._document_chunks(content, filename)
        print(f"Created {len(chunks)} chunks from document.")

        points = []
        if chunks:
            with self._loaded_model(
                lms.embedding_model,
                LM_STUDIO_EMBEDDING_MODEL,
            ) as model:
                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i + 1}/{len(chunks)}")
                    vector = self._request_embedding(
                        model,
                        f"{DOCUMENT_PREFIX}{self._build_search_text(chunk)}",
                    )
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                "filename": filename,
                                **chunk,
                                "chunk_index": i,
                            },
                        )
                    )

        if points:
            print(f"Upserting {len(points)} points to Qdrant...")
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print("Upsert successful.")
        return len(points)

    def search(
        self,
        query: str,
        limit: int = 5,
        filename: str | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self.generate_query_embedding(query)
        query_filter = None
        if filename:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename),
                    )
                ]
            )
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
        )
        defaults = {
            "filename": "unknown",
            "content": "",
            "title": "",
            "section": "",
            "subsection": None,
            "header_path": [],
            "summary": "",
            "keywords": [],
            "chunk_type": "",
            "chunk_index": 0,
        }
        return [
            {
                "score": point.score,
                **{
                    field: point.payload.get(field, default)
                    for field, default in defaults.items()
                },
            }
            for point in search_result.points
        ]

    def document_outline(self, filename: str | None = None) -> list[dict[str, Any]]:
        """Return the unique title/section hierarchy for one document or the collection."""
        query_filter = None
        if filename:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename),
                    )
                ]
            )

        outlines: dict[tuple[str, str, str], dict[str, Any]] = {}
        offset = None
        max_points = 5000
        points_read = 0
        while points_read < max_points:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=min(256, max_points - points_read),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                payload = point.payload or {}
                key = (
                    str(payload.get("filename", "unknown")),
                    str(payload.get("title", "")),
                    " / ".join(payload.get("header_path", [])),
                )
                outlines.setdefault(
                    key,
                    {
                        "filename": key[0],
                        "title": key[1],
                        "header_path": payload.get("header_path", []),
                        "section": payload.get("section", ""),
                        "subsection": payload.get("subsection"),
                    },
                )
            points_read += len(points)
            if next_offset is None:
                break
            offset = next_offset

        return list(outlines.values())
