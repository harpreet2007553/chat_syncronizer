import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FusedHit:
    """Represents a search result after Reciprocal Rank Fusion.

    Attributes:
        chunk_id: Unique identifier of the retrieved chunk.
        score: Fused RRF relevance score.
        ranks: Mapping of source name to the 1-based rank where this chunk appeared.
    """

    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    results_by_source: dict[str, list[tuple[str, float]]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    top_n: int | None = None,
) -> list[FusedHit]:
    """Merge ranked results from multiple sources into a single ranking using RRF.

    Args:
        results_by_source: Mapping of source name to its ranked (chunk_id, score) list,
            sorted best-first. List position (1-based) is used as rank; scores are ignored.
        k: Smoothing constant to control the penalty for lower ranks (default 60).
        weights: Optional multiplier per source (default 1.0 for each source).
        top_n: Optional maximum number of hits to return.

    Returns:
        List of FusedHit objects sorted in descending order of fused score.
        Ties are broken deterministically by best single-source rank, then chunk_id.

    Raises:
        ValueError: If k <= 0 or if any weight is negative.
    """
    if k <= 0:
        raise ValueError(f"k must be strictly positive, got {k}")

    if weights is not None:
        for source, w in weights.items():
            if w < 0:
                raise ValueError(f"Weight for source '{source}' cannot be negative, got {w}")

    if not results_by_source:
        return []

    # Map chunk_id -> {"score": float, "ranks": dict[str, int]}
    fused_data: dict[str, dict] = {}

    for source, results in results_by_source.items():
        if not results:
            continue

        weight = (weights.get(source, 1.0)) if weights is not None else 1.0
        seen_in_source: set[str] = set()

        for rank, (chunk_id, _) in enumerate(results, start=1):
            if chunk_id in seen_in_source:
                # Count only the first (best) occurrence within a single source list
                continue
            seen_in_source.add(chunk_id)

            rrf_score = weight * (1.0 / (k + rank))

            if chunk_id not in fused_data:
                fused_data[chunk_id] = {
                    "score": 0.0,
                    "ranks": {},
                }

            fused_data[chunk_id]["score"] += rrf_score
            fused_data[chunk_id]["ranks"][source] = rank

    if not fused_data:
        return []

    fused_hits: list[FusedHit] = [
        FusedHit(
            chunk_id=cid,
            score=data["score"],
            ranks=data["ranks"],
        )
        for cid, data in fused_data.items()
    ]

    # Sort descending by score.
    # Ties are broken by best (lowest) single-source rank, then alphabetically by chunk_id.
    fused_hits.sort(
        key=lambda h: (-h.score, min(h.ranks.values()), h.chunk_id)
    )

    if top_n is not None:
        if top_n <= 0:
            return []
        fused_hits = fused_hits[:top_n]

    return fused_hits

