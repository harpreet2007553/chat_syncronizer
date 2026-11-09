import logging
import math
from dataclasses import dataclass
from typing import Sequence

from chatmem.retrieval.rerank import RerankedHit

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    """The outcome of the relevance gate.

    Attributes:
        passed: True if any hits are kept and deemed relevant to return.
        hits: The list of kept RerankedHit objects (sorted best-first), or [] if rejected.
        reason: Decision code: 'ok', 'no_candidates', 'below_threshold', or 'too_few_results'.
        top_score: The highest valid score among the candidates, or None if no valid candidates.
    """

    passed: bool
    hits: list[RerankedHit]
    reason: str
    top_score: float | None


def apply_gate(
    hits: Sequence[RerankedHit],
    threshold: float,
    max_results: int | None = None,
    min_results: int = 1,
) -> GateDecision:
    """Decide whether retrieved and reranked hits meet quality requirements for context injection.

    If candidate scores fail to meet the threshold, injecting context is worse than
    injecting nothing. This function filters candidates, verifies that at least min_results
    clear the threshold, and truncates to max_results if specified.

    Args:
        hits: Sequence of RerankedHit instances from the reranker.
        threshold: Minimum relevance score required to keep a hit. Threshold scale
            must match the reranker score format (raw logit or sigmoid-normalized).
        max_results: Optional maximum number of hits to return. Must be >= 1 if provided.
        min_results: Minimum number of hits required to pass. If fewer hits clear
            the threshold, the entire query is rejected with 'too_few_results'.

    Returns:
        GateDecision containing passed status, kept hits, reason code, and top score.

    Raises:
        ValueError: If threshold is NaN, if min_results < 1, or if max_results < 1.
    """
    if math.isnan(threshold):
        raise ValueError("threshold cannot be NaN.")

    if min_results < 1:
        raise ValueError(f"min_results must be >= 1, got {min_results}")

    if max_results is not None and max_results < 1:
        raise ValueError(f"max_results must be >= 1, got {max_results}")

    if not hits:
        logger.debug("Gate decision: no_candidates | candidates=0 | kept=0 | top_score=None")
        return GateDecision(
            passed=False,
            hits=[],
            reason="no_candidates",
            top_score=None,
        )

    # Filter out invalid NaN or infinite scores
    valid_hits: list[RerankedHit] = []
    invalid_count = 0
    for h in hits:
        if math.isnan(h.score) or math.isinf(h.score):
            invalid_count += 1
        else:
            valid_hits.append(h)

    if invalid_count > 0:
        logger.warning("Skipped %d hits with NaN or infinite scores", invalid_count)

    if not valid_hits:
        logger.debug("Gate decision: no_candidates | candidates=0 | kept=0 | top_score=None")
        return GateDecision(
            passed=False,
            hits=[],
            reason="no_candidates",
            top_score=None,
        )

    # Stable sort by score descending (preserves original input order for ties)
    sorted_hits = sorted(valid_hits, key=lambda h: h.score, reverse=True)
    top_score = sorted_hits[0].score

    # Filter candidates meeting or exceeding the threshold
    passed_hits = [h for h in sorted_hits if h.score >= threshold]

    if not passed_hits:
        decision = GateDecision(
            passed=False,
            hits=[],
            reason="below_threshold",
            top_score=top_score,
        )
    elif len(passed_hits) < min_results:
        decision = GateDecision(
            passed=False,
            hits=[],
            reason="too_few_results",
            top_score=top_score,
        )
    else:
        if max_results is not None:
            passed_hits = passed_hits[:max_results]
        decision = GateDecision(
            passed=True,
            hits=passed_hits,
            reason="ok",
            top_score=top_score,
        )

    logger.debug(
        "Gate decision: %s | candidates=%d | kept=%d | top_score=%.4f",
        decision.reason,
        len(valid_hits),
        len(decision.hits),
        top_score,
    )
    return decision

