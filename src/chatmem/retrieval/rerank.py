import functools
import logging
import math
import time
from dataclasses import dataclass
from typing import Sequence

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RerankedHit:
    """Represents a candidate chunk scored by a cross-encoder model.

    Attributes:
        chunk_id: Unique identifier of the chunk.
        score: Relevance score from the cross-encoder. Raw logit score by default,
            or sigmoid-normalized in [0, 1] if normalize=True was requested.
        prev_rank: 1-based index position in the input candidate list.
    """

    chunk_id: str
    score: float
    prev_rank: int


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid mapping arbitrary real numbers into (0, 1)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


@functools.lru_cache(maxsize=4)
def get_cross_encoder(
    model_name: str = "BAAI/bge-reranker-base",
    max_length: int = 512,
) -> CrossEncoder:
    """Load and cache a CrossEncoder model instance lazily.

    Auto-detects CUDA if available, falling back to CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug("Loading CrossEncoder model '%s' on %s (max_length=%d)", model_name, device, max_length)
    return CrossEncoder(model_name, max_length=max_length, device=device)


def rerank(
    query: str,
    candidates: Sequence[tuple[str, str]],
    top_n: int | None = None,
    model_name: str = "BAAI/bge-reranker-base",
    batch_size: int = 16,
    max_length: int = 512,
    normalize: bool = False,
) -> list[RerankedHit]:
    """Re-score a list of candidate chunks against a query using a local CrossEncoder.

    This step runs after reciprocal rank fusion and before the relevance gate.
    All cross-encoder inferences run strictly locally. Candidate text and queries
    are never transmitted to external APIs or printed to logs.

    Args:
        query: User search query string.
        candidates: Sequence of (chunk_id, text) tuples in initial ranking order.
        top_n: Optional maximum number of hits to return. Must be >= 1 if provided.
        model_name: Identifier of the cross-encoder model (default 'BAAI/bge-reranker-base').
        batch_size: Inference batch size for CrossEncoder.predict.
        max_length: Maximum token length for chunk+query pairs before truncation.
        normalize: If False (default), returns raw cross-encoder output scores.
            If True, applies a sigmoid transform to map scores strictly into (0, 1).
            Downstream gate thresholds in gate.py should align with this flag.

    Returns:
        List of RerankedHit instances sorted in descending order of cross-encoder score.
        Ties are broken deterministically by earlier prev_rank.

    Raises:
        ValueError: If query is empty or whitespace-only, or if top_n < 1.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty or whitespace-only.")

    if top_n is not None and top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    if not candidates:
        return []

    start_time = time.perf_counter()

    # Deduplicate candidate chunks while preserving their first 1-based rank position
    unique_candidates: list[tuple[str, str, int]] = []
    seen_ids: set[str] = set()

    for rank, (chunk_id, text) in enumerate(candidates, start=1):
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_candidates.append((chunk_id, text, rank))

    # Construct (query, candidate_text) pairs for CrossEncoder scoring
    pairs = [(query, text) for _, text, _ in unique_candidates]

    model = get_cross_encoder(model_name=model_name, max_length=max_length)
    raw_scores = model.predict(pairs, batch_size=batch_size)

    hits: list[RerankedHit] = []
    for (chunk_id, _, prev_rank), score in zip(unique_candidates, raw_scores):
        score_val = float(score)
        if normalize:
            score_val = _sigmoid(score_val)
        hits.append(RerankedHit(chunk_id=chunk_id, score=score_val, prev_rank=prev_rank))

    # Sort descending by score; ties broken deterministically by lower prev_rank
    hits.sort(key=lambda h: (-h.score, h.prev_rank))

    if top_n is not None:
        hits = hits[:top_n]

    elapsed = time.perf_counter() - start_time
    logger.info("Reranked %d candidates in %.3f seconds", len(unique_candidates), elapsed)

    return hits

