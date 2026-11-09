import logging
import os
import uuid
from datetime import datetime
from typing import Sequence

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
from sentence_transformers import SentenceTransformer

from chatmem.schema import Chunk
from chatmem.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_VECTOR_SIZE = 384
DEFAULT_COLLECTION_NAME = "chunks_dense"


def get_qdrant_client(
    url: str | None = None,
    api_key: str | None = None,
    path: str | None = None,
    in_memory: bool = False,
) -> QdrantClient:
    """Connect to Qdrant Cloud, local Docker instance, local disk, or in-memory.

    Order of priority:
    1. in_memory=True -> QdrantClient(":memory:")
    2. path provided -> QdrantClient(path=path)
    3. Explicit or env QDRANT_CLUSTER_ENDPOINT + QDRANT_API_KEY
    4. Explicit or env QDRANT_URL (default: http://localhost:6333)
    5. Fallback -> local disk storage in data/qdrant_storage
    """
    if in_memory:
        logger.info("Initializing in-memory Qdrant client")
        return QdrantClient(":memory:")

    if path:
        logger.info("Initializing local disk Qdrant client at: %s", path)
        return QdrantClient(path=path)

    cloud_endpoint = url or os.getenv("QDRANT_CLUSTER_ENDPOINT")
    cloud_key = api_key or os.getenv("QDRANT_API_KEY")

    if cloud_endpoint and cloud_key:
        logger.info("Connecting to Qdrant Cloud at: %s", cloud_endpoint)
        try:
            return QdrantClient(url=cloud_endpoint, api_key=cloud_key)
        except Exception as e:
            logger.warning("Failed to connect to Qdrant Cloud (%s), falling back to local storage", e)

    local_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=local_url, timeout=2.0)
        # Test connection
        client.get_collections()
        logger.info("Connected to Qdrant instance at %s", local_url)
        return client
    except Exception:
        fallback_path = os.getenv("QDRANT_STORAGE_PATH", "data/qdrant_storage")
        logger.info(
            "Qdrant server not reachable at %s. Falling back to embedded local storage: %s",
            local_url,
            fallback_path,
        )
        return QdrantClient(path=fallback_path)


class DenseIndex:
    """Dense vector index backed by SentenceTransformers and Qdrant."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        client: QdrantClient | None = None,
        in_memory: bool = False,
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.client = client or get_qdrant_client(in_memory=in_memory)
        self._model: SentenceTransformer | None = None
        self._vector_dim: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._vector_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def vector_dim(self) -> int:
        if self._vector_dim is None:
            _ = self.model
        return self._vector_dim or DEFAULT_VECTOR_SIZE

    def init_collection(self, force_recreate: bool = False) -> None:
        """Ensure the target collection exists in Qdrant with cosine distance."""
        exists = self.client.collection_exists(self.collection_name)
        if exists and force_recreate:
            logger.info("Recreating collection: %s", self.collection_name)
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(
                "Creating collection '%s' with dimension %d",
                self.collection_name,
                self.vector_dim,
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    def index_chunks(self, chunks: Sequence[Chunk], batch_size: int = 32) -> int:
        """Embed and upsert a batch of Chunk objects into Qdrant."""
        if not chunks:
            return 0

        self.init_collection(force_recreate=False)

        texts = [c.text for c in chunks]
        logger.info("Computing embeddings for %d chunks (batch_size=%d)...", len(texts), batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        points: list[PointStruct] = []
        for chunk, vec in zip(chunks, embeddings):
            # Deterministic UUID prevents duplicate points on re-indexing
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            time_val = (
                chunk.time.isoformat()
                if isinstance(chunk.time, datetime)
                else str(chunk.time)
            )

            point = PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "conversation_id": chunk.conversation_id,
                    "platform": chunk.platform,
                    "title": chunk.title or "Untitled",
                    "text": chunk.text,
                    "message_ids": chunk.message_ids,
                    "time": time_val,
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info("Indexed %d points into '%s'", len(points), self.collection_name)
        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 10,
        platform_filter: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search the dense index for chunks semantically matching the query.

        Returns:
            List of (Chunk, similarity_score) sorted descending by relevance.
        """
        if not query.strip():
            return []

        query_vector = self.model.encode(query, convert_to_numpy=True).tolist()

        query_filter = None
        if platform_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="platform",
                        match=MatchValue(value=platform_filter),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        results: list[tuple[Chunk, float]] = []
        for scored_point in response.points:
            payload = scored_point.payload or {}
            time_raw = payload.get("time")
            try:
                time_dt = datetime.fromisoformat(time_raw) if time_raw else datetime.now()
            except Exception:
                time_dt = datetime.now()

            chunk = Chunk(
                chunk_id=str(payload.get("chunk_id", scored_point.id)),
                conversation_id=str(payload.get("conversation_id", "")),
                platform=str(payload.get("platform", "unknown")),
                title=str(payload.get("title", "Untitled")),
                text=str(payload.get("text", "")),
                message_ids=list(payload.get("message_ids", [])),
                time=time_dt,
            )
            results.append((chunk, float(scored_point.score)))

        return results


def init_collections(client: QdrantClient, vector_size: int = DEFAULT_VECTOR_SIZE) -> None:
    """Helper to initialize standard collection schemas."""
    collections = ["chunks_pm", "chunks_qa", "chunks_ws"]
    for name in collections:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", name)


def create_embedding(chunks: Sequence[Chunk], model: SentenceTransformer | None = None) -> list[PointStruct]:
    """Helper function to convert chunks to Qdrant PointStructs."""
    if not chunks:
        return []
    if model is None:
        model = SentenceTransformer(DEFAULT_MODEL_NAME)

    texts = [c.text for c in chunks]
    vectors = model.encode(texts, convert_to_numpy=True)

    points = []
    for chunk, vec in zip(chunks, vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
        points.append(
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "conversation_id": chunk.conversation_id,
                    "platform": chunk.platform,
                    "title": chunk.title,
                    "text": chunk.text,
                    "message_ids": chunk.message_ids,
                    "time": chunk.time.isoformat() if isinstance(chunk.time, datetime) else str(chunk.time),
                },
            )
        )
    return points


if __name__ == "__main__":
    from chatmem.chunking.qa_pair import QA_Chunks

    print(f"Sample QA chunks ready for indexing: {len(QA_Chunks)}")
    index = DenseIndex(collection_name="chunks_qa_test", in_memory=True)
    index.index_chunks(QA_Chunks)

    test_query = "How to fix KeyError in pandas groupby?"
    print(f"\nQuerying: '{test_query}'")
    hits = index.search(test_query, top_k=3)
    for i, (chunk, score) in enumerate(hits, 1):
        print(f"\n[{i}] Score: {score:.4f} | Chat: {chunk.title} ({chunk.platform})")
        print(f"Text snippet: {chunk.text[:120]}...")
