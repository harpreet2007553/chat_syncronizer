import math
import pytest

from chatmem.retrieval.gate import GateDecision, apply_gate
from chatmem.retrieval.rerank import RerankedHit


def test_hits_above_threshold_pass_sorted_best_first():
    """Hits meeting threshold pass and return in descending score order."""
    hits = [
        RerankedHit(chunk_id="c1", score=0.65, prev_rank=1),
        RerankedHit(chunk_id="c2", score=0.85, prev_rank=2),
        RerankedHit(chunk_id="c3", score=0.40, prev_rank=3),
    ]

    decision = apply_gate(hits, threshold=0.50)

    assert decision.passed is True
    assert decision.reason == "ok"
    assert decision.top_score == 0.85
    assert len(decision.hits) == 2
    assert [h.chunk_id for h in decision.hits] == ["c2", "c1"]
    assert [h.score for h in decision.hits] == [0.85, 0.65]


def test_everything_below_threshold_rejects_with_below_threshold():
    """When all hits are below threshold, reject with 'below_threshold' and report top_score."""
    hits = [
        RerankedHit(chunk_id="c1", score=0.35, prev_rank=1),
        RerankedHit(chunk_id="c2", score=0.48, prev_rank=2),
    ]

    decision = apply_gate(hits, threshold=0.50)

    assert decision.passed is False
    assert decision.hits == []
    assert decision.reason == "below_threshold"
    assert decision.top_score == 0.48


def test_score_exactly_equal_to_threshold_passes():
    """Scores exactly equal to threshold must pass."""
    hits = [
        RerankedHit(chunk_id="c1", score=0.50, prev_rank=1),
        RerankedHit(chunk_id="c2", score=0.49, prev_rank=2),
    ]

    decision = apply_gate(hits, threshold=0.50)

    assert decision.passed is True
    assert decision.reason == "ok"
    assert len(decision.hits) == 1
    assert decision.hits[0].chunk_id == "c1"
    assert decision.top_score == 0.50


def test_min_results_rejection():
    """Reject with 'too_few_results' when fewer than min_results clear the threshold."""
    hits = [
        RerankedHit(chunk_id="c1", score=0.90, prev_rank=1),
        RerankedHit(chunk_id="c2", score=0.40, prev_rank=2),
    ]

    decision = apply_gate(hits, threshold=0.50, min_results=2)

    assert decision.passed is False
    assert decision.hits == []
    assert decision.reason == "too_few_results"
    assert decision.top_score == 0.90


def test_max_results_truncation():
    """max_results truncates kept hits to the top N."""
    hits = [
        RerankedHit(chunk_id="c1", score=0.60, prev_rank=1),
        RerankedHit(chunk_id="c2", score=0.95, prev_rank=2),
        RerankedHit(chunk_id="c3", score=0.80, prev_rank=3),
        RerankedHit(chunk_id="c4", score=0.70, prev_rank=4),
    ]

    decision = apply_gate(hits, threshold=0.50, max_results=2)

    assert decision.passed is True
    assert len(decision.hits) == 2
    assert [h.chunk_id for h in decision.hits] == ["c2", "c3"]


def test_unsorted_input_and_stable_tie_breaking():
    """Handles unsorted input correctly, and preserves input order for ties."""
    hits = [
        RerankedHit(chunk_id="first_tie", score=0.75, prev_rank=3),
        RerankedHit(chunk_id="low", score=0.60, prev_rank=1),
        RerankedHit(chunk_id="high", score=0.90, prev_rank=4),
        RerankedHit(chunk_id="second_tie", score=0.75, prev_rank=2),
    ]

    decision = apply_gate(hits, threshold=0.50)

    assert decision.passed is True
    # "high" first (0.90), then "first_tie" (0.75), "second_tie" (0.75), then "low" (0.60)
    assert [h.chunk_id for h in decision.hits] == ["high", "first_tie", "second_tie", "low"]


def test_empty_input_returns_no_candidates():
    """Empty candidate list yields passed=False, reason='no_candidates', top_score=None."""
    decision = apply_gate([], threshold=0.50)

    assert decision.passed is False
    assert decision.hits == []
    assert decision.reason == "no_candidates"
    assert decision.top_score is None


def test_nan_and_inf_scores_skipped():
    """NaN and infinite scores are safely skipped; all-invalid yields 'no_candidates'."""
    hits = [
        RerankedHit(chunk_id="c_nan", score=float("nan"), prev_rank=1),
        RerankedHit(chunk_id="c_inf", score=float("inf"), prev_rank=2),
        RerankedHit(chunk_id="c_neginf", score=float("-inf"), prev_rank=3),
        RerankedHit(chunk_id="c_valid", score=0.70, prev_rank=4),
    ]

    decision = apply_gate(hits, threshold=0.50)
    assert decision.passed is True
    assert len(decision.hits) == 1
    assert decision.hits[0].chunk_id == "c_valid"
    assert decision.top_score == 0.70

    # All-invalid
    all_invalid = [
        RerankedHit(chunk_id="c1", score=float("nan"), prev_rank=1),
        RerankedHit(chunk_id="c2", score=float("inf"), prev_rank=2),
    ]
    invalid_decision = apply_gate(all_invalid, threshold=0.50)
    assert invalid_decision.passed is False
    assert invalid_decision.reason == "no_candidates"
    assert invalid_decision.top_score is None


def test_invalid_arguments_raise_value_error():
    """Invalid parameters raise ValueError."""
    hits = [RerankedHit(chunk_id="c1", score=0.8, prev_rank=1)]

    with pytest.raises(ValueError, match="threshold cannot be NaN"):
        apply_gate(hits, threshold=float("nan"))

    with pytest.raises(ValueError, match="min_results must be >= 1"):
        apply_gate(hits, threshold=0.5, min_results=0)

    with pytest.raises(ValueError, match="max_results must be >= 1"):
        apply_gate(hits, threshold=0.5, max_results=0)

    with pytest.raises(ValueError, match="max_results must be >= 1"):
        apply_gate(hits, threshold=0.5, max_results=-2)


def test_input_list_is_not_mutated():
    """Input list and its elements are not modified in-place."""
    h1 = RerankedHit(chunk_id="c1", score=0.40, prev_rank=1)
    h2 = RerankedHit(chunk_id="c2", score=0.80, prev_rank=2)
    original_hits = [h1, h2]

    apply_gate(original_hits, threshold=0.50)

    # original_hits should still have c1 at index 0 and c2 at index 1
    assert original_hits == [h1, h2]
    assert original_hits[0].chunk_id == "c1"
    assert original_hits[1].chunk_id == "c2"

