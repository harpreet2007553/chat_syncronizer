from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import QdrantClient

from chatmem.index.dense import get_qdrant_client, search_dense
from chatmem.index.store import DEFAULT_DB_PATH, bm25_search, connect, get_chunks
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

    model_config = ConfigDict(extra="forbid")

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
            raise ValueError(
                "gate_threshold cannot be set when use_rerank is False (gate requires cross-encoder scores)."
            )
        if self.min_results > self.final_top_k:
            raise ValueError(
                f"min_results ({self.min_results}) cannot exceed final_top_k ({self.final_top_k})."
            )
        if self.use_rerank and self.rerank_candidates < self.final_top_k:
            raise ValueError(
                f"rerank_candidates ({self.rerank_candidates}) must be >= final_top_k ({self.final_top_k}) when use_rerank is True."
            )
        return self

    @property
    def score_scale(self) -> str:
        """'rrf' if no reranker, else 'sigmoid' or 'raw' per rerank_normalize."""
        if not self.use_rerank:
            return "rrf"
        return "sigmoid" if self.rerank_normalize else "raw"

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
        top_score is on a different scale depending on score_scale ('rrf', 'raw', or 'sigmoid').
        It is only comparable between results with the same score_scale AND the same rerank_model.
    """

    chunks: list[RetrievedChunk]
    injected: bool
    reason: str
    top_score: float | None
    score_scale: str
    timings_ms: dict[str, float]


# ============================================================================
# 3. Retriever Implementation
# ============================================================================

def _to_chunk(
    record: dict[str, Any],
    chunk_id: str,
    score: float,
    ranks: dict[str, int],
) -> RetrievedChunk:
    """Build a RetrievedChunk from a SQLite record plus score and ranks."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        text=record["text"],
        conversation_id=record["conversation_id"],
        platform=record["platform"],
        title=record["title"],
        time=record["time"],
        message_ids=record["message_ids"],
        ranks=ranks,
    )


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
            a. Query validation
            b. First-stage search
            c. Fusion
            d. Record hydration
            e. Early exit if no candidates
            f. Rerank + optional gate (rerank branch) / no-rerank branch
            g. Finalize timings, log, and package the result
        """
        # --------------------------------------------------------------------
        # Step a: Query validation
        # --------------------------------------------------------------------
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
        # Step d: Record hydration (Fetch full records from SQLite)
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
        # Step e: Early exit if no candidates
        # --------------------------------------------------------------------
        if not kept_fused_hits:
            timings_ms["total"] = (time.perf_counter() - pipeline_start) * 1000.0
            rounded_timings = {k: round(v, 2) for k, v in timings_ms.items()}
            logger.info(
                "Retriever [%s]: returned=0, injected=False, reason=no_candidates, score_scale=%s, total_ms=%.2f",
                self.config.name,
                self.config.score_scale,
                rounded_timings["total"],
            )
            return PipelineResult(
                chunks=[],
                injected=False,
                reason="no_candidates",
                top_score=None,
                score_scale=self.config.score_scale,
                timings_ms=rounded_timings,
            )

        fusion_ranks = {h.chunk_id: h.ranks for h in kept_fused_hits}
        retrieved_chunks: list[RetrievedChunk] = []

        # --------------------------------------------------------------------
        # Step f: Rerank + optional gate (rerank branch)
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
                    _to_chunk(
                        record=rec,
                        chunk_id=h.chunk_id,
                        score=h.score,
                        ranks=fusion_ranks.get(h.chunk_id, {}),
                    )
                )

        # --------------------------------------------------------------------
        # Step f (no rerank): No reranker branch
        # --------------------------------------------------------------------
        else:
            final_fused = kept_fused_hits[: self.config.final_top_k]
            injected = True
            reason = "no_gate"
            top_score = final_fused[0].score if final_fused else None

            for h in final_fused:
                rec = records[h.chunk_id]
                retrieved_chunks.append(
                    _to_chunk(
                        record=rec,
                        chunk_id=h.chunk_id,
                        score=h.score,
                        ranks=h.ranks,
                    )
                )

        # --------------------------------------------------------------------
        # Step g: Finalize timings, log, and package the result
        # --------------------------------------------------------------------
        timings_ms["total"] = (time.perf_counter() - pipeline_start) * 1000.0
        rounded_timings = {k: round(v, 2) for k, v in timings_ms.items()}

        logger.info(
            "Retriever [%s]: returned=%d, injected=%s, reason=%s, score_scale=%s, total_ms=%.2f",
            self.config.name,
            len(retrieved_chunks),
            injected,
            reason,
            self.config.score_scale,
            rounded_timings["total"],
        )

        return PipelineResult(
            chunks=retrieved_chunks,
            injected=injected,
            reason=reason,
            top_score=top_score,
            score_scale=self.config.score_scale,
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
        db_path = DEFAULT_DB_PATH

    conn = connect(db_path)
    client = get_qdrant_client()

    return Retriever(config=config, conn=conn, client=client)

