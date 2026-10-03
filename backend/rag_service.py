import os
import uuid
from pathlib import Path

import lmstudio as lms
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", "qdrant_data")).expanduser()
if not QDRANT_PATH.is_absolute():
    QDRANT_PATH = BACKEND_DIR / QDRANT_PATH

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


class RAGService:
    def __init__(self, collection_name="rag_collection"):
        self.collection_name = collection_name

        if EMBEDDING_DIMENSION <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be a positive integer")

        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(QDRANT_PATH))

        self.embedding_model = lms.embedding_model(LM_STUDIO_EMBEDDING_MODEL)
        self.vector_size = EMBEDDING_DIMENSION

        self._ensure_collection()

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
        return self._request_embedding(f"{DOCUMENT_PREFIX}{text}")

    def generate_query_embedding(self, text: str):
        return self._request_embedding(f"{QUERY_PREFIX}{text}")

    def _request_embedding(self, text: str):
        embedding = self.embedding_model.embed(text)
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

        paragraphs = self._chunk_text(content)
        print(f"Created {len(paragraphs)} chunks from document.")

        points = []
        for i, paragraph in enumerate(paragraphs):
            print(f"Processing chunk {i + 1}/{len(paragraphs)}")
            vector = self.generate_embedding(paragraph)
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "filename": filename,
                        "content": paragraph,
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

    def search(self, query: str, limit: int = 5):
        query_vector = self.generate_query_embedding(query)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        return search_result.points
