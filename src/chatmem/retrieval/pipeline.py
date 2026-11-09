from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from qdrant_client import QdrantClient

from chatmem.index.dense import get_qdrant_client, search_dense
from chatmem.index.store import bm25_search, connect, get_chunks
from chatmem.retrieval.fusion import reciprocal_rank_fusion
from chatmem.retrieval.gate import apply_gate
from chatmem.retrieval.rerank import rerank

logger = logging.getLogger(__name__)

STRATEGY_COLLECTIONS: dict[str, str] = {
    "pm": "chunks_pm",
    "qa": "chunks_qa",
    "ws": "chunks_ws",
}


# ============================================================================
# 1. Pipeline Configuration
# ============================================================================

class PipelineConfig(BaseModel):
    """Configuration for retrieval experiments and execution pipelines."""

    name: str = "default"
    strategy: Literal["pm", "qa", "ws"] = "qa"
    use_bm25: bool = True
    use_dense: bool = True
    search_top_k: int = Field(default=30, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    bm25_weight: float = Field(default=1.0, ge=0.0)
    dense_weight: float = Field(default=1.0, ge=0.0)
    use_rerank: bool = True
    rerank_candidates: int = Field(default=30, ge=1)
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_normalize: bool = False
    gate_threshold: float | None = None
    min_results: int = Field(default=1, ge=1)
    final_top_k: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_pipeline_settings(self) -> PipelineConfig:
        if not self.use_bm25 and not self.use_dense:
            raise ValueError("At least one of use_bm25 or use_dense must be True.")
        if self.gate_threshold is not None and not self.use_rerank:
            raise ValueError("gate_threshold cannot be set when use_rerank is False (gate requires cross-encoder scores).")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load and parse configuration from a YAML file."""
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


# ============================================================================
# 2. Result Data Models
# ============================================================================

@dataclass
class RetrievedChunk:
    """A retrieved and scored chunk enriched with full metadata and citation details."""

    chunk_id: str
    score: float
    text: str
    conversation_id: str
    platform: str
    title: str | None
    time: str
    message_ids: list[str]
    ranks: dict[str, int]


@dataclass
class PipelineResult:
    """The complete result of a retrieval pipeline search run.

    Note:
        top_score is on a different scale depending on the config
        (RRF score, raw cross-encoder score, or sigmoid score).
        It must not be directly compared across different pipeline configurations.
    """

    chunks: list[RetrievedChunk]
    injected: bool
    reason: str
    top_score: float | None
    timings_ms: dict[str, float]


# ============================================================================
# 3. Retriever Implementation
# ============================================================================

class Retriever:
    """Retrieval orchestrator executing search, fusion, reranking, and gating."""

    def __init__(
        self,
        config: PipelineConfig,
        conn: sqlite3.Connection,
        client: QdrantClient,
    ) -> None:
        self.config = config
        self.conn = conn
        self.client = client

    def close(self) -> None:
        """Close opened database connections and clients."""
        try:
            self.conn.close()
        except Exception:
            pass
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass

    def __enter__(self) -> Retriever:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def search(self, query: str) -> PipelineResult:
        """Execute the configured multi-stage retrieval pipeline.

        Steps:
            a. Query validation (reject empty / whitespace)
            b. First-stage search (BM25 and/or Dense)
            c. RRF Fusion across enabled sources
            d. Full record hydration from SQLite
            e. Early exit if no candidates remain
            f. Optional Cross-Encoder reranking and relevance gating
            g. Truncation and packaging with per-stage timings
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only.")

        pipeline_start = time.perf_counter()
        timings_ms: dict[str, float] = {}

        # --------------------------------------------------------------------
        # Step b: First-stage search
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        results_by_source: dict[str, list[tuple[str, float]]] = {}

        if self.config.use_bm25:
            results_by_source["bm25"] = bm25_search(
                conn=self.conn,
                strategy=self.config.strategy,
                query=query,
                top_k=self.config.search_top_k,
            )

        if self.config.use_dense:
            collection = STRATEGY_COLLECTIONS[self.config.strategy]
            results_by_source["dense"] = search_dense(
                client=self.client,
                collection=collection,
                query=query,
                top_k=self.config.search_top_k,
            )

        timings_ms["search"] = (time.perf_counter() - t0) * 1000.0

        # --------------------------------------------------------------------
        # Step c: Fusion (Always call RRF, even for single sources)
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        weights: dict[str, float] = {}
        if "bm25" in results_by_source:
            weights["bm25"] = self.config.bm25_weight
        if "dense" in results_by_source:
            weights["dense"] = self.config.dense_weight

        fusion_top_n = (
            self.config.rerank_candidates
            if self.config.use_rerank
            else self.config.final_top_k
        )
        fused_hits = reciprocal_rank_fusion(
            results_by_source=results_by_source,
            k=self.config.rrf_k,
            weights=weights,
            top_n=fusion_top_n,
        )
        timings_ms["fusion"] = (time.perf_counter() - t0) * 1000.0

        # --------------------------------------------------------------------
        # Step d: Fetch full records from SQLite
        # --------------------------------------------------------------------
        t0 = time.perf_counter()
        fused_chunk_ids = [h.chunk_id for h in fused_hits]
        records = get_chunks(self.conn, self.config.strategy, fused_chunk_ids)

        missing_count = sum(1 for cid in fused_chunk_ids if cid not in records)
        if missing_count > 0:
            logger.warning("Dropped %d chunk IDs not found in SQLite store", missing_count)

        kept_fused_hits = [h for h in fused_hits if h.chunk_id in records]
        timings_ms["fetch"] = (time.perf_counter() - t0) * 1000.0

        # --------------------------------------------------------------------
        # Step e: If nothing remains, return no_candidates without loading reranker
        # --------------------------------------------------------------------
        if not kept_fused_hits:
            timings_ms["total"] = (time.perf_counter() - pipeline_start) * 1000.0
            rounded_timings = {k: round(v, 2) for k, v in timings_ms.items()}
            logger.info(
                "Retriever [%s]: returned=0, injected=False, reason=no_candidates, total_ms=%.2f",
                self.config.name,
                rounded_timings["total"],
            )
            return PipelineResult(
                chunks=[],
                injected=False,
                reason="no_candidates",
                top_score=None,
                timings_ms=rounded_timings,
            )

        fusion_ranks = {h.chunk_id: h.ranks for h in kept_fused_hits}
        retrieved_chunks: list[RetrievedChunk] = []

        # --------------------------------------------------------------------
        # Step f: Rerank and optional relevance gate
        # --------------------------------------------------------------------
        if self.config.use_rerank:
            t0 = time.perf_counter()
            candidates = [(h.chunk_id, records[h.chunk_id]["text"]) for h in kept_fused_hits]
            reranked_hits = rerank(
                query=query,
                candidates=candidates,
                top_n=None,
                model_name=self.config.rerank_model,
                normalize=self.config.rerank_normalize,
            )
            timings_ms["rerank"] = (time.perf_counter() - t0) * 1000.0

            if self.config.gate_threshold is not None:
                t0 = time.perf_counter()
                decision = apply_gate(
                    hits=reranked_hits,
                    threshold=self.config.gate_threshold,
                    max_results=self.config.final_top_k,
                    min_results=self.config.min_results,
                )
                timings_ms["gate"] = (time.perf_counter() - t0) * 1000.0

                injected = decision.passed
                reason = decision.reason
                top_score = decision.top_score
                final_hits = decision.hits
            else:
                final_hits = reranked_hits[: self.config.final_top_k]
                injected = True
                reason = "no_gate"
                top_score = final_hits[0].score if final_hits else None

            for h in final_hits:
                rec = records[h.chunk_id]
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=h.chunk_id,
                        score=h.score,
                        text=rec["text"],
                        conversation_id=rec["conversation_id"],
                        platform=rec["platform"],
                        title=rec["title"],
                        time=rec["time"],
                        message_ids=rec["message_ids"],
                        ranks=fusion_ranks.get(h.chunk_id, {}),
                    )
                )

        # --------------------------------------------------------------------
        # Step g: No reranker
        # --------------------------------------------------------------------
        else:
            final_fused = kept_fused_hits[: self.config.final_top_k]
            injected = True
            reason = "no_gate"
            top_score = final_fused[0].score if final_fused else None

            for h in final_fused:
                rec = records[h.chunk_id]
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=h.chunk_id,
                        score=h.score,
                        text=rec["text"],
                        conversation_id=rec["conversation_id"],
                        platform=rec["platform"],
                        title=rec["title"],
                        time=rec["time"],
                        message_ids=rec["message_ids"],
                        ranks=h.ranks,
                    )
                )

        # --------------------------------------------------------------------
        # Step i: Finalize timings and log
        # --------------------------------------------------------------------
        timings_ms["total"] = (time.perf_counter() - pipeline_start) * 1000.0
        rounded_timings = {k: round(v, 2) for k, v in timings_ms.items()}

        logger.info(
            "Retriever [%s]: returned=%d, injected=%s, reason=%s, total_ms=%.2f",
            self.config.name,
            len(retrieved_chunks),
            injected,
            reason,
            rounded_timings["total"],
        )

        return PipelineResult(
            chunks=retrieved_chunks,
            injected=injected,
            reason=reason,
            top_score=top_score,
            timings_ms=rounded_timings,
        )


# ============================================================================
# 4. Pipeline Builder
# ============================================================================

def build_retriever(
    config: PipelineConfig | str | Path,
    db_path: str | Path | None = None,
) -> Retriever:
    """Build and initialize a Retriever instance from a config object or YAML path."""
    if isinstance(config, (str, Path)):
        config = PipelineConfig.from_yaml(config)

    if db_path is None:
        db_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "chatmem.db"

    conn = connect(db_path)
    client = get_qdrant_client()

    return Retriever(config=config, conn=conn, client=client)

