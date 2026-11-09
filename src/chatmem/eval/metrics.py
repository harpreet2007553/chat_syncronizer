from __future__ import annotations

import math
from typing import Any, Sequence


def recall_at_k(
    ranked: Sequence[str],
    relevant: set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Recall@k: fraction of relevant items found in the top k.

    Raises:
        ValueError: If relevant is empty.
    """
    rel_set = set(relevant)
    if not rel_set:
        raise ValueError("Answerable queries must have at least 1 relevant ID.")
    if k <= 0:
        return 0.0

    top_k = ranked[:k]
    hits = len(rel_set.intersection(top_k))
    return float(hits / len(rel_set))


def hit_at_k(
    ranked: Sequence[str],
    relevant: set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Hit@k: 1.0 if any relevant item is found in top k, else 0.0."""
    rel_set = set(relevant)
    if not rel_set or k <= 0:
        return 0.0
    return 1.0 if any(item in rel_set for item in ranked[:k]) else 0.0


def reciprocal_rank(
    ranked: Sequence[str],
    relevant: set[str] | Sequence[str],
) -> float:
    """Compute Reciprocal Rank: 1 / rank of first relevant item, or 0.0 if none."""
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    for idx, item in enumerate(ranked, start=1):
        if item in rel_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(
    ranked: Sequence[str],
    relevant: set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k (nDCG@k) with binary gains."""
    rel_set = set(relevant)
    if not rel_set or k <= 0:
        return 0.0

    top_k = ranked[:k]
    dcg = sum(
        1.0 / math.log2(idx + 1)
        for idx, item in enumerate(top_k, start=1)
        if item in rel_set
    )

    num_ideal = min(len(rel_set), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, num_ideal + 1))
    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def dedupe_conversations(chunks: Sequence[Any]) -> list[str]:
    """Extract conversation IDs from retrieved chunks while preserving first-occurrence order."""
    seen: set[str] = set()
    deduped: list[str] = []
    for c in chunks:
        cid = getattr(c, "conversation_id", None)
        if cid is None and isinstance(c, dict):
            cid = c.get("conversation_id")
        elif cid is None and isinstance(c, str):
            cid = c

        if cid is not None:
            cid_str = str(cid)
            if cid_str not in seen:
                seen.add(cid_str)
                deduped.append(cid_str)
    return deduped


def mean(values: Sequence[float]) -> float:
    """Compute arithmetic mean of a sequence of numbers.

    Returns:
        0.0 if the input sequence is empty.
    """
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def gate_sweep(
    rows: list[dict[str, Any]],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Simulate relevance gate thresholds offline over evaluated queries.

    Args:
        rows: list of dicts with:
            - 'answerable': bool
            - 'top_score': float | None
            - 'relevant_in_topk': bool
        thresholds: list of numeric score thresholds to evaluate.

    Returns:
        List of dicts containing rates for each threshold. Zero-denominator fields return None.
    """
    total_ans = sum(1 for r in rows if r.get("answerable", False))
    total_no_ans = sum(1 for r in rows if not r.get("answerable", False))
    total_good = sum(
        1 for r in rows if r.get("answerable", False) and r.get("relevant_in_topk", False)
    )

    results: list[dict[str, Any]] = []

    for t in thresholds:
        ans_injected = sum(
            1
            for r in rows
            if r.get("answerable", False)
            and r.get("top_score") is not None
            and r["top_score"] >= t
        )
        no_ans_injected = sum(
            1
            for r in rows
            if not r.get("answerable", False)
            and r.get("top_score") is not None
            and r["top_score"] >= t
        )
        no_ans_not_injected = total_no_ans - no_ans_injected

        useful = sum(
            1
            for r in rows
            if r.get("answerable", False)
            and r.get("relevant_in_topk", False)
            and r.get("top_score") is not None
            and r["top_score"] >= t
        )
        missed_good = sum(
            1
            for r in rows
            if r.get("answerable", False)
            and r.get("relevant_in_topk", False)
            and not (r.get("top_score") is not None and r["top_score"] >= t)
        )

        inject_rate_ans = (ans_injected / total_ans) if total_ans > 0 else None
        correct_abstain_no_ans = (no_ans_not_injected / total_no_ans) if total_no_ans > 0 else None
        false_inject_rate = (no_ans_injected / total_no_ans) if total_no_ans > 0 else None
        useful_inject_rate = (useful / total_ans) if total_ans > 0 else None
        missed_good_rate = (missed_good / total_good) if total_good > 0 else None

        results.append(
            {
                "threshold": t,
                "inject_rate_answerable": inject_rate_ans,
                "correct_abstain_rate_no_answer": correct_abstain_no_ans,
                "false_injection_rate": false_inject_rate,
                "useful_injection_rate": useful_inject_rate,
                "missed_good_rate": missed_good_rate,
            }
        )

    return results


def suggest_thresholds(scores: Sequence[float | None], n: int = 25) -> list[float]:
    """Generate n evenly spaced threshold candidate values spanning min to max of observed scores."""
    valid_scores = [float(s) for s in scores if s is not None]
    if not valid_scores or n <= 0:
        return []

    min_s = min(valid_scores)
    max_s = max(valid_scores)
    if min_s == max_s or n == 1:
        return [float(min_s)]

    step = (max_s - min_s) / (n - 1)
    return [float(min_s + i * step) for i in range(n)]

