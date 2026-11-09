import logging
import os
import uuid
from datetime import datetime
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)
from sentence_transformers import SentenceTransformer

from chatmem.index.store import QdrantStore, get_qdrant_store
from chatmem.schema import Chunk
from chatmem.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_VECTOR_SIZE = 384
DEFAULT_COLLECTION_NAME = "chunks_qa"
DEFAULT_LOCAL_QDRANT_URL = "http://localhost:6333"

_GLOBAL_EMBED_MODEL: SentenceTransformer | None = None


def get_qdrant_client(url: str | None = None) -> QdrantClient:
    """Connect to local Qdrant instance (default: http://localhost:6333)."""
    target_url = url or os.getenv("QDRANT_URL") or DEFAULT_LOCAL_QDRANT_URL
    return QdrantClient(url=target_url)


def _get_embed_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    global _GLOBAL_EMBED_MODEL
    if _GLOBAL_EMBED_MODEL is None:
        logger.info("Loading embedding model '%s' for dense search...", model_name)
        _GLOBAL_EMBED_MODEL = SentenceTransformer(model_name)
    return _GLOBAL_EMBED_MODEL


def search_dense(
    client: QdrantClient,
    collection: str,
    query: str,
    top_k: int = 30,
) -> list[tuple[str, float]]:
    """Search Qdrant dense vector collection using query text.

    Args:
        client: QdrantClient instance.
        collection: Target Qdrant collection name (e.g. 'chunks_qa').
        query: Query string.
        top_k: Max points to return.

    Returns:
        List of (chunk_id, score) tuples sorted best-first.
    """
    if not query or not query.strip():
        return []

    model = _get_embed_model()
    query_vector = model.encode(query, convert_to_numpy=True).tolist()

    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    results: list[tuple[str, float]] = []
    for p in response.points:
        payload = p.payload or {}
        chunk_id = str(payload.get("chunk_id", p.id))
        results.append((chunk_id, float(p.score)))

    return results


class DenseIndex:
    """Dense vector search powered by SentenceTransformers embeddings and QdrantStore."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        store: QdrantStore | None = None,
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.store = store or get_qdrant_store()
        self._model: SentenceTransformer | None = None
        self._vector_dim: int | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the SentenceTransformer embedding model."""
        if self._model is None:
            self._model = _get_embed_model(self.model_name)
            self._vector_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def vector_dim(self) -> int:
        if self._vector_dim is None:
            _ = self.model
        return self._vector_dim or DEFAULT_VECTOR_SIZE

    def init_collection(self, force_recreate: bool = False) -> None:
        """Ensure collection exists in Qdrant store with correct dimensions."""
        self.store.ensure_collection(
            collection_name=self.collection_name,
            vector_size=self.vector_dim,
            distance=Distance.COSINE,
            force_recreate=force_recreate,
        )

    def index_chunks(self, chunks: Sequence[Chunk], batch_size: int = 32) -> int:
        """Embed chunks and persist them into Qdrant store."""
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

        return self.store.upsert_points(self.collection_name, points)

    def search(
        self,
        query: str,
        top_k: int = 10,
        platform_filter: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search Qdrant for chunks semantically matching the query."""
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

        scored_points = self.store.query_nearest(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            query_filter=query_filter,
        )

        results: list[tuple[Chunk, float]] = []
        for sp in scored_points:
            payload = sp.payload or {}
            time_raw = payload.get("time")
            try:
                time_dt = datetime.fromisoformat(time_raw) if time_raw else datetime.now()
            except Exception:
                time_dt = datetime.now()

            chunk = Chunk(
                chunk_id=str(payload.get("chunk_id", sp.id)),
                conversation_id=str(payload.get("conversation_id", "")),
                platform=str(payload.get("platform", "unknown")),
                title=str(payload.get("title", "Untitled")),
                text=str(payload.get("text", "")),
                message_ids=list(payload.get("message_ids", [])),
                time=time_dt,
            )
            results.append((chunk, float(sp.score)))

        return results
