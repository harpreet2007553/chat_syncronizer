"""Service layer for chatmem local API wrapping retrieval, indexing, and storage."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from chatmem.api.models import (
    ApiConfig,
    Citation,
    ContextResponse,
    HealthResponse,
    IngestMessagesRequest,
    SearchRequest,
    SearchResponse,
    SearchResultChunk,
)
from chatmem.chunking.qa_pair import qa_chunks
from chatmem.index.dense import DEFAULT_MODEL_NAME, DenseIndex
from chatmem.index.store import DEFAULT_DB_PATH, SQLiteMetadataStore, connect, get_qdrant_store
from chatmem.retrieval.pipeline import (
    PipelineConfig,
    PipelineResult,
    QdrantConfig,
    Retriever,
)
from chatmem.retrieval.rerank import get_cross_encoder
from chatmem.schema import Conversation, Message
from chatmem.utils.logging import get_logger

logger = get_logger(__name__)


def _format_context_block(
    chunks: list[SearchResultChunk],
    max_chars: int,
) -> tuple[str, int]:
    """Deterministically format retrieved chunks into a bounded plain-text prompt injection block.

    Truncates at message boundaries where possible while strictly enforcing max_chars.
    """
    if not chunks:
        return "", 0

    header = "--- PAST CONVERSATION CONTEXT (chatmem) ---\n\n"
    footer = "\n--- END CONTEXT ---"
    fixed_overhead = len(header) + len(footer)

    if max_chars <= fixed_overhead:
        return "", 0

    available_chars = max_chars - fixed_overhead
    used_chunks = 0
    body_parts: list[str] = []
    current_len = 0

    for idx, c in enumerate(chunks, 1):
        cit = c.citation
        title = cit.title or "Untitled"
        lead = f"[{idx}] From {cit.platform} conversation \"{title}\" ({cit.timestamp}):\n"
        chunk_full = lead + c.text.strip() + "\n\n"

        if current_len + len(chunk_full) <= available_chars:
            body_parts.append(chunk_full)
            current_len += len(chunk_full)
            used_chunks += 1
        else:
            # Need truncation at message/paragraph boundary
            remaining_for_this = available_chars - current_len - len(lead) - 5  # leeway for ellipsis
            if remaining_for_this > 50:
                text_slice = c.text.strip()[:remaining_for_this]
                # Look for last paragraph or sentence boundary
                boundary = text_slice.rfind("\n\n")
                if boundary == -1:
                    boundary = text_slice.rfind(". ")
                if boundary != -1 and boundary > 40:
                    text_slice = text_slice[: boundary + 1]
                else:
                    text_slice = text_slice.rstrip() + "..."
                truncated_chunk = lead + text_slice + "\n\n"
                body_parts.append(truncated_chunk)
                current_len += len(truncated_chunk)
                used_chunks += 1
            break

    if used_chunks == 0:
        return "", 0

    final_text = header + "".join(body_parts).rstrip() + footer
    return final_text, used_chunks


class ChatmemService:
    """Manages index lifecycle, asynchronous retrieval, and concurrency serialization."""

    def __init__(
        self,
        api_config: ApiConfig,
        qdrant_config: QdrantConfig | None = None,
        db_path: str | Path | None = None,
    ):
        self.config = api_config
        self.qdrant_config = qdrant_config or QdrantConfig()
        self.db_path = Path(db_path or DEFAULT_DB_PATH)

        # Persistent storage instances
        self.sqlite_store = SQLiteMetadataStore(db_path=self.db_path)
        self.sqlite_store.init_db()

        self.qdrant_store = get_qdrant_store(
            config=self.qdrant_config,
        )
        self.dense_index = DenseIndex(
            store=self.qdrant_store,
            collection_name="chunks_qa",
        )

        # Process-level serialization lock for concurrent writes
        self.write_lock = asyncio.Lock()

        # Shared retriever client connections
        self._qdrant_client = self.qdrant_store.client
        self._sqlite_conn = connect(self.db_path)

        # Fast retriever instance
        self.fast_config = PipelineConfig(
            name="fast",
            strategy="qa",
            use_bm25=True,
            use_dense=True,
            use_rerank=False,
            dense_gate_threshold=self.config.fast_gate_threshold,
            final_top_k=self.config.max_results,
            qdrant=self.qdrant_config,
        )
        self.fast_retriever = Retriever(
            config=self.fast_config,
            conn=self._sqlite_conn,
            client=self._qdrant_client,
        )

        # Precise retriever (lazy-loaded on first call to precise mode)
        self._precise_retriever: Retriever | None = None

    def get_precise_retriever(self) -> Retriever:
        """Lazily build precise retriever with cross-encoder reranker on first use."""
        if self._precise_retriever is None:
            logger.info("Initializing precise retriever with cross-encoder reranker...")
            precise_config = PipelineConfig(
                name="precise",
                strategy="qa",
                use_bm25=True,
                use_dense=True,
                use_rerank=True,
                rerank_candidates=self.config.precise_rerank_candidates,
                gate_threshold=self.config.precise_gate_threshold,
                dense_gate_threshold=None,
                final_top_k=self.config.max_results,
                qdrant=self.qdrant_config,
            )
            self._precise_retriever = Retriever(
                config=precise_config,
                conn=self._sqlite_conn,
                client=self._qdrant_client,
            )
        return self._precise_retriever

    def warmup(self) -> None:
        """Run a warmup query at startup to eliminate model loading cold-starts."""
        logger.info("Running warmup query on fast retriever...")
        try:
            _ = self.fast_retriever.search("warmup query")
            logger.info("Warmup query completed successfully.")
        except Exception as e:
            logger.warning("Warmup query failed or skipped: %s", e)

    async def search(self, req: SearchRequest) -> SearchResponse:
        """Execute search asynchronously off the main event loop thread."""
        mode = req.mode or self.config.default_mode
        top_k = req.top_k or self.config.max_results

        retriever = self.get_precise_retriever() if mode == "precise" else self.fast_retriever

        # Run blocking model inference in worker threadpool
        res: PipelineResult = await asyncio.to_thread(retriever.search, req.query)

        results: list[SearchResultChunk] = []
        for c in res.chunks:
            # Apply platform filter if requested
            if req.platform and c.platform.lower() != req.platform.lower():
                continue
            # Apply since timestamp filter if requested
            if req.since and c.time < req.since:
                continue

            results.append(
                SearchResultChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    score=round(c.score, 4),
                    citation=Citation(
                        platform=c.platform,
                        conversation_id=c.conversation_id,
                        title=c.title,
                        timestamp=c.time,
                    ),
                )
            )

        if top_k is not None:
            results = results[:top_k]

        return SearchResponse(
            injected=res.injected,
            reason=res.reason,
            gate_score=round(res.top_score, 4) if res.top_score is not None else None,
            score_scale=res.score_scale,
            results=results,
            timings_ms=res.timings_ms,
        )

    async def format_context(
        self,
        query: str,
        mode: str | None = None,
        max_chars: int | None = None,
    ) -> ContextResponse:
        """Retrieve results and format into a bounded context injection string."""
        target_mode = mode or self.config.default_mode
        cap = max_chars or self.config.max_injected_chars

        search_req = SearchRequest(query=query, mode=target_mode, top_k=self.config.max_results)
        search_res = await self.search(search_req)

        if not search_res.injected or not search_res.results:
            return ContextResponse(
                injected=False,
                reason=search_res.reason,
                context="",
                chunks_used=0,
                total_chars=0,
            )

        context_str, chunks_used = _format_context_block(search_res.results, max_chars=cap)
        return ContextResponse(
            injected=True if chunks_used > 0 else False,
            reason="ok" if chunks_used > 0 else "exceeds_max_chars",
            context=context_str,
            chunks_used=chunks_used,
            total_chars=len(context_str),
        )

    async def ingest_messages(self, req: IngestMessagesRequest) -> int:
        """Idempotently ingest or update messages from browser capture."""
        # 1. Validate roles and text length
        now = datetime.now(timezone.utc)
        messages: list[Message] = []
        for idx, m in enumerate(req.messages):
            role_norm = m.role.strip().lower()
            if role_norm not in {"user", "assistant"}:
                raise ValueError(f"Invalid message role '{m.role}'. Role must be 'user' or 'assistant'.")
            text_norm = m.text.strip()
            if not text_norm:
                raise ValueError("Message text cannot be empty or whitespace-only.")
            if len(text_norm) > self.config.max_message_chars:
                raise ValueError(f"Message exceeds maximum character length of {self.config.max_message_chars}.")

            msg_time = m.timestamp or now
            msg_id = m.external_id or f"{req.conversation_id}_{idx}"
            messages.append(
                Message(
                    message_id=msg_id,
                    conversation_id=req.conversation_id,
                    platform=req.platform,
                    role=role_norm,
                    title=req.title or "Untitled",
                    text=text_norm,
                    timestamp=msg_time,
                )
            )

        conv = Conversation(
            conversation_id=req.conversation_id,
            platform=req.platform,
            title=req.title or "Untitled",
            creation_time=messages[0].timestamp,
            updation_time=messages[-1].timestamp,
            messages=messages,
        )

        # 2. Chunk conversation
        chunks = qa_chunks([conv])

        # 3. Synchronize storage under exclusive write lock
        def _sync_write():
            self.sqlite_store.save_conversations([conv])
            self.sqlite_store.delete_chunks_by_conversation(req.conversation_id, strategy="qa")
            if chunks:
                self.sqlite_store.save_chunks(chunks, strategy="qa")

            self.dense_index.delete_conversation(req.conversation_id)
            if chunks:
                self.dense_index.upsert_chunks(chunks)
            return len(chunks)

        async with self.write_lock:
            chunks_count = await asyncio.to_thread(_sync_write)

        return chunks_count

    async def delete_conversation(self, conversation_id: str, platform: str | None = None) -> int:
        """Delete all chunks, messages, and conversation metadata from both stores."""
        def _sync_delete():
            deleted_chunks = self.sqlite_store.delete_chunks_by_conversation(conversation_id)
            self.sqlite_store.delete_conversation(conversation_id)
            self.dense_index.delete_conversation(conversation_id)
            return deleted_chunks

        async with self.write_lock:
            return await asyncio.to_thread(_sync_delete)

    def get_health(self) -> HealthResponse:
        """Retrieve system status and metadata without leaking private content."""
        with self.sqlite_store._get_connection() as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations;").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]

        is_reranker_loaded = False
        try:
            is_reranker_loaded = get_cross_encoder.cache_info().currsize > 0
        except Exception:
            pass

        return HealthResponse(
            status="ok",
            version="0.1.0",
            embedding_model=DEFAULT_MODEL_NAME,
            rerank_model="BAAI/bge-reranker-base",
            reranker_loaded=is_reranker_loaded,
            conversations_indexed=conv_count,
            chunks_indexed=chunk_count,
        )

    def close(self) -> None:
        """Close opened connections."""
        try:
            self._sqlite_conn.close()
        except Exception:
            pass
        try:
            self.fast_retriever.close()
        except Exception:
            pass
        if self._precise_retriever:
            try:
                self._precise_retriever.close()
            except Exception:
                pass
        try:
            self.dense_index.close()
        except Exception:
            pass
