import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

from chatmem.schema import Chunk, Conversation, Message
from chatmem.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chatmem_metadata.db"
DEFAULT_LOCAL_QDRANT_URL = "http://localhost:6333"


# ============================================================================
# 1. SQLite Metadata Store (Conversations, Messages, Chunks)
# ============================================================================

class SQLiteMetadataStore:
    """Manages SQLite storage for raw conversations, messages, and chunk metadata."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Create tables for conversations, messages, and chunks."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    title TEXT NOT NULL,
                    creation_time TEXT NOT NULL,
                    updation_time TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    role TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    message_ids TEXT NOT NULL,
                    time TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv_id ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_messages_platform ON messages(platform);
                CREATE INDEX IF NOT EXISTS idx_chunks_conv_id ON chunks(conversation_id);
            """)
        logger.info("Initialized SQLite metadata store at %s", self.db_path)

    def save_conversations(self, conversations: Sequence[Conversation]) -> int:
        """Save a batch of Conversation objects and all their child messages."""
        if not conversations:
            return 0

        conv_rows = []
        msg_rows = []

        for conv in conversations:
            c_time = conv.creation_time.isoformat() if isinstance(conv.creation_time, datetime) else str(conv.creation_time)
            u_time = conv.updation_time.isoformat() if isinstance(conv.updation_time, datetime) else str(conv.updation_time)
            conv_rows.append((conv.conversation_id, conv.platform, conv.title, c_time, u_time))

            for msg in conv.messages:
                m_time = msg.timestamp.isoformat() if isinstance(msg.timestamp, datetime) else str(msg.timestamp)
                msg_rows.append((
                    msg.message_id,
                    conv.conversation_id,
                    msg.platform,
                    msg.role,
                    msg.title or conv.title or "Untitled",
                    msg.text,
                    m_time,
                ))

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO conversations (conversation_id, platform, title, creation_time, updation_time)
                VALUES (?, ?, ?, ?, ?);
            """, conv_rows)

            conn.executemany("""
                INSERT OR REPLACE INTO messages (message_id, conversation_id, platform, role, title, text, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, msg_rows)

        logger.info("Saved %d conversations and %d messages into SQLite", len(conv_rows), len(msg_rows))
        return len(conv_rows)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Retrieve a Conversation and all its Messages by ID."""
        with self._get_connection() as conn:
            c_row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?;",
                (conversation_id,),
            ).fetchone()

            if not c_row:
                return None

            m_rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC;",
                (conversation_id,),
            ).fetchall()

            messages = [
                Message(
                    message_id=r["message_id"],
                    conversation_id=r["conversation_id"],
                    platform=r["platform"],
                    role=r["role"],
                    title=r["title"],
                    text=r["text"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
                for r in m_rows
            ]

            return Conversation(
                conversation_id=c_row["conversation_id"],
                platform=c_row["platform"],
                title=c_row["title"],
                creation_time=datetime.fromisoformat(c_row["creation_time"]),
                updation_time=datetime.fromisoformat(c_row["updation_time"]),
                messages=messages,
            )

    def get_all_conversations(self, platform: str | None = None) -> list[Conversation]:
        """Fetch all stored conversations."""
        query = "SELECT conversation_id FROM conversations"
        params = ()
        if platform:
            query += " WHERE platform = ?"
            params = (platform,)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        conversations = []
        for r in rows:
            c = self.get_conversation(r["conversation_id"])
            if c:
                conversations.append(c)
        return conversations

    def save_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Store chunk metadata for quick citation hydration."""
        if not chunks:
            return 0

        chunk_rows = []
        for c in chunks:
            t_str = c.time.isoformat() if isinstance(c.time, datetime) else str(c.time)
            chunk_rows.append((
                c.chunk_id,
                c.conversation_id,
                c.platform,
                c.title,
                c.text,
                json.dumps(c.message_ids),
                t_str,
            ))

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks (chunk_id, conversation_id, platform, title, text, message_ids, time)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, chunk_rows)

        logger.info("Saved %d chunks into SQLite", len(chunk_rows))
        return len(chunk_rows)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a Chunk by its chunk_id."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?;", (chunk_id,)).fetchone()
            if not row:
                return None
            return Chunk(
                chunk_id=row["chunk_id"],
                conversation_id=row["conversation_id"],
                platform=row["platform"],
                title=row["title"],
                text=row["text"],
                message_ids=json.loads(row["message_ids"]),
                time=datetime.fromisoformat(row["time"]),
            )


# ============================================================================
# 2. Local Qdrant Store (Vector Database Layer)
# ============================================================================

class QdrantStore:
    """Manages collections, vectors, and vector searches in local Qdrant."""

    def __init__(self, url: str | None = None):
        target_url = url or os.getenv("QDRANT_URL") or DEFAULT_LOCAL_QDRANT_URL
        self.client = QdrantClient(url=target_url)
        self.url = target_url
        logger.info("Initialized QdrantStore connected to %s", self.url)

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int = 384,
        distance: Distance = Distance.COSINE,
        force_recreate: bool = False,
    ) -> None:
        """Create a collection if it doesn't already exist."""
        exists = self.client.collection_exists(collection_name)
        if exists and force_recreate:
            logger.info("Deleting collection %s for recreation", collection_name)
            self.client.delete_collection(collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            logger.info("Created Qdrant collection: %s (size=%d)", collection_name, vector_size)

    def upsert_points(self, collection_name: str, points: list[PointStruct]) -> int:
        """Upsert vector points with payload into Qdrant."""
        if not points:
            return 0
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        logger.info("Upserted %d points to Qdrant collection '%s'", len(points), collection_name)
        return len(points)

    def query_nearest(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 10,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        """Perform vector similarity search against a collection."""
        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return response.points

    def list_collections(self) -> list[str]:
        """Return list of collection names."""
        return [c.name for c in self.client.get_collections().collections]

    def count_points(self, collection_name: str) -> int:    
        """Get the number of indexed points in a collection."""
        if not self.client.collection_exists(collection_name):
            return 0
        return self.client.get_collection(collection_name).points_count or 0


# Default instances for convenient import across modules
def get_metadata_store(db_path: str | Path | None = None) -> SQLiteMetadataStore:
    return SQLiteMetadataStore(db_path=db_path or DEFAULT_DB_PATH)


def get_qdrant_store(url: str | None = None) -> QdrantStore:
    return QdrantStore(url=url)


if __name__ == "__main__":
    from chatmem.chunking.per_message import conversations_list
    from chatmem.chunking.qa_pair import QA_Chunks

    # 1. Test SQLite Metadata Store
    print("Testing SQLite Metadata Store...")
    sql_store = get_metadata_store()
    convs = list(conversations_list.values())
    sql_store.save_conversations(convs)
    sql_store.save_chunks(QA_Chunks)

    loaded = sql_store.get_all_conversations()
    print(f"SQLite successfully saved & verified: {len(loaded)} conversations.")

    # 2. Test Qdrant Store
    print("\nTesting Qdrant Store...")
    q_store = get_qdrant_store()
    print(f"Local Qdrant active collections: {q_store.list_collections()}")
    for name in q_store.list_collections():
        print(f" - {name}: {q_store.count_points(name)} points")

