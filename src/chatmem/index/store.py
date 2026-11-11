import json
import os
import re
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
# 1. Connection & Core Query Helpers
# ============================================================================

def connect(db_path: str | Path) -> sqlite3.Connection:
    """Connect to SQLite database and configure row_factory."""
    path_str = str(db_path)
    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path_str)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def bm25_search(
    conn: sqlite3.Connection,
    strategy: str,
    query: str,
    top_k: int = 30,
) -> list[tuple[str, float]]:
    """Search chunks via BM25 for the given strategy.

    Returns:
        List of (chunk_id, score) tuples sorted best-first.
    """
    if not query or not query.strip():
        return []

    cursor = conn.cursor()

    # 1. Check for FTS5 tables first
    fts_table = None
    for candidate in [f"chunks_{strategy}_fts", "chunks_fts"]:
        row = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (candidate,)
        ).fetchone()
        if row:
            fts_table = candidate
            break

    if fts_table:
        tokens = re.findall(r"\w+", query)
        if tokens:
            fts_query = " OR ".join(tokens)
            try:
                table_cols = [c["name"] for c in cursor.execute(f"PRAGMA table_info({fts_table});").fetchall()]
                if "strategy" in table_cols:
                    rows = cursor.execute(
                        f"SELECT chunk_id, -bm25({fts_table}) AS score FROM {fts_table} "
                        f"WHERE {fts_table} MATCH ? AND (strategy = ? OR strategy IS NULL) "
                        f"ORDER BY score DESC LIMIT ?",
                        (fts_query, strategy, top_k),
                    ).fetchall()
                else:
                    rows = cursor.execute(
                        f"SELECT chunk_id, -bm25({fts_table}) AS score FROM {fts_table} "
                        f"WHERE {fts_table} MATCH ? "
                        f"ORDER BY score DESC LIMIT ?",
                        (fts_query, top_k),
                    ).fetchall()
                return [(str(r["chunk_id"]), float(r["score"])) for r in rows]
            except sqlite3.OperationalError:
                pass

    # 2. Fallback: load chunks from chunks table and score with BM25Index
    from chatmem.index.bm25 import BM25Index

    table_name = "chunks"
    try:
        col_info = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
        has_strategy_col = any(c["name"] == "strategy" for c in col_info)
    except Exception:
        return []

    if has_strategy_col:
        rows = cursor.execute(
            f"SELECT chunk_id, text, conversation_id, platform, title, message_ids, time "
            f"FROM {table_name} WHERE strategy = ? OR strategy IS NULL",
            (strategy,),
        ).fetchall()
    else:
        rows = cursor.execute(
            f"SELECT chunk_id, text, conversation_id, platform, title, message_ids, time FROM {table_name}"
        ).fetchall()

    if not rows:
        return []

    chunks: list[Chunk] = []
    for r in rows:
        try:
            m_ids = json.loads(r["message_ids"]) if isinstance(r["message_ids"], str) else list(r["message_ids"])
        except Exception:
            m_ids = []
        try:
            t = datetime.fromisoformat(r["time"]) if r["time"] else datetime.now()
        except Exception:
            t = datetime.now()

        chunks.append(
            Chunk(
                chunk_id=r["chunk_id"],
                conversation_id=r["conversation_id"],
                platform=r["platform"],
                title=r["title"] or "Untitled",
                text=r["text"],
                message_ids=m_ids,
                time=t,
            )
        )

    bm25_idx = BM25Index()
    bm25_idx.index_chunks(chunks)
    hits = bm25_idx.search(query, top_k=top_k)
    return [(c.chunk_id, score) for c, score in hits]


def get_chunks(
    conn: sqlite3.Connection,
    strategy: str,
    chunk_ids: Sequence[str],
) -> dict[str, dict]:
    """Retrieve full chunk records by chunk_ids.

    Returns:
        dict[chunk_id, record dict with keys:
            chunk_id, conversation_id, platform, title, text, message_ids (list[str]), time (ISO string)
        ]
    """
    if not chunk_ids:
        return {}

    placeholders = ",".join("?" for _ in chunk_ids)
    cursor = conn.cursor()

    query = f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})"
    try:
        rows = cursor.execute(query, list(chunk_ids)).fetchall()
    except sqlite3.OperationalError:
        return {}

    records: dict[str, dict] = {}
    for r in rows:
        cid = str(r["chunk_id"])
        m_ids = r["message_ids"]
        if isinstance(m_ids, str):
            try:
                m_ids = json.loads(m_ids)
            except Exception:
                m_ids = [m_ids]
        elif not isinstance(m_ids, list):
            m_ids = list(m_ids)

        t_val = str(r["time"]) if r["time"] else ""

        records[cid] = {
            "chunk_id": cid,
            "conversation_id": str(r["conversation_id"]),
            "platform": str(r["platform"]),
            "title": str(r["title"]) if r["title"] is not None else None,
            "text": str(r["text"]),
            "message_ids": [str(mid) for mid in m_ids],
            "time": t_val,
        }

    return records


# ============================================================================
# 2. SQLite Metadata Store Class
# ============================================================================

class SQLiteMetadataStore:
    """Manages SQLite storage for raw conversations, messages, and chunk metadata."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return connect(self.db_path)

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
                    strategy TEXT,
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

    def save_chunks(self, chunks: Sequence[Chunk], strategy: str = "qa") -> int:
        """Store chunk metadata for quick citation hydration."""
        if not chunks:
            return 0

        chunk_rows = []
        for c in chunks:
            t_str = c.time.isoformat() if isinstance(c.time, datetime) else str(c.time)
            chunk_rows.append((
                c.chunk_id,
                strategy,
                c.conversation_id,
                c.platform,
                c.title,
                c.text,
                json.dumps(c.message_ids),
                t_str,
            ))

        with self._get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks (chunk_id, strategy, conversation_id, platform, title, text, message_ids, time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, chunk_rows)

        logger.info("Saved %d chunks into SQLite (strategy=%s)", len(chunk_rows), strategy)
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
# 3. Local Qdrant Store (Vector Database Layer)
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


def get_metadata_store(db_path: str | Path | None = None) -> SQLiteMetadataStore:
    return SQLiteMetadataStore(db_path=db_path or DEFAULT_DB_PATH)


def get_qdrant_store(url: str | None = None) -> QdrantStore:
    return QdrantStore(url=url)
