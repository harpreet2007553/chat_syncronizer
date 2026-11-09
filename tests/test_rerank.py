import math
import numpy as np
import pytest

from chatmem.retrieval.rerank import RerankedHit, rerank


class FakeCrossEncoder:
    """Mock cross-encoder that returns predetermined scores without loading weights."""

    def __init__(self, scores_map: dict[str, float] | None = None, default_score: float = 0.0):
        self.scores_map = scores_map or {}
        self.default_score = default_score
        self.call_count = 0

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        self.call_count += 1
        # Returns numpy floats to test that rerank converts them to Python floats
        return [np.float32(self.scores_map.get(text, self.default_score)) for _, text in pairs]


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeCrossEncoder()
    monkeypatch.setattr("chatmem.retrieval.rerank.get_cross_encoder", lambda **kwargs: model)
    return model


def test_rerank_sorts_descending_and_tracks_prev_rank(fake_model):
    """Output is sorted by score descending and prev_rank reflects the 1-based input position."""
    fake_model.scores_map = {
        "text_low": -2.5,
        "text_high": 4.2,
        "text_mid": 1.0,
    }

    candidates = [
        ("chunk_low", "text_low"),    # prev_rank 1
        ("chunk_high", "text_high"),  # prev_rank 2
        ("chunk_mid", "text_mid"),    # prev_rank 3
    ]

    hits = rerank("test query", candidates)

    assert len(hits) == 3
    # Best score first
    assert hits[0].chunk_id == "chunk_high"
    assert pytest.approx(hits[0].score, rel=1e-5) == 4.2
    assert hits[0].prev_rank == 2
    assert isinstance(hits[0].score, float)

    # Second best
    assert hits[1].chunk_id == "chunk_mid"
    assert pytest.approx(hits[1].score, rel=1e-5) == 1.0
    assert hits[1].prev_rank == 3

    # Lowest
    assert hits[2].chunk_id == "chunk_low"
    assert pytest.approx(hits[2].score, rel=1e-5) == -2.5
    assert hits[2].prev_rank == 1


def test_top_n_truncation(fake_model):
    """top_n parameter properly truncates the sorted results."""
    fake_model.scores_map = {
        "t1": 10.0,
        "t2": 20.0,
        "t3": 30.0,
    }
    candidates = [("c1", "t1"), ("c2", "t2"), ("c3", "t3")]

    hits = rerank("my query", candidates, top_n=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "c3"
    assert hits[1].chunk_id == "c2"


def test_empty_candidates_returns_empty_without_loading_model(monkeypatch):
    """Empty candidates return [] and the model loader is never invoked."""
    called = False

    def explode(**kwargs):
        nonlocal called
        called = True
        raise RuntimeError("Model should not be loaded for empty candidates!")

    monkeypatch.setattr("chatmem.retrieval.rerank.get_cross_encoder", explode)

    result = rerank("search query", [])
    assert result == []
    assert not called


def test_duplicate_chunk_ids_counted_once(fake_model):
    """If a chunk_id appears multiple times, only the first occurrence is kept."""
    fake_model.scores_map = {
        "first_text": 5.0,
        "other_text": 2.0,
    }
    candidates = [
        ("c1", "first_text"),   # rank 1
        ("c2", "other_text"),   # rank 2
        ("c1", "second_text"),  # duplicate c1 -> ignored
    ]

    hits = rerank("query", candidates)
    assert len(hits) == 2
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert hits[0].prev_rank == 1
    assert hits[1].prev_rank == 2


def test_normalize_applies_sigmoid_and_preserves_order(fake_model):
    """normalize=True maps arbitrary scores strictly into [0, 1] while preserving ranking."""
    fake_model.scores_map = {
        "very_negative": -10.0,
        "zero": 0.0,
        "very_positive": 10.0,
    }
    candidates = [
        ("c_neg", "very_negative"),
        ("c_zero", "zero"),
        ("c_pos", "very_positive"),
    ]

    raw_hits = rerank("query", candidates, normalize=False)
    norm_hits = rerank("query", candidates, normalize=True)

    # Rank ordering remains identical
    assert [h.chunk_id for h in raw_hits] == [h.chunk_id for h in norm_hits] == ["c_pos", "c_zero", "c_neg"]

    # Sigmoid checks: 0.0 -> 0.5, positive -> >0.5, negative -> <0.5
    for h in norm_hits:
        assert 0.0 < h.score < 1.0

    hit_pos = next(h for h in norm_hits if h.chunk_id == "c_pos")
    hit_zero = next(h for h in norm_hits if h.chunk_id == "c_zero")
    hit_neg = next(h for h in norm_hits if h.chunk_id == "c_neg")

    assert pytest.approx(hit_zero.score, rel=1e-5) == 0.5
    assert hit_pos.score > 0.999
    assert hit_neg.score < 0.001


def test_ties_broken_deterministically(fake_model):
    """Equal scores are broken deterministically by earlier prev_rank."""
    fake_model.scores_map = {
        "text_a": 2.5,
        "text_b": 2.5,
        "text_c": 1.0,
    }
    candidates = [
        ("chunk_first", "text_a"),   # prev_rank 1
        ("chunk_second", "text_b"),  # prev_rank 2
        ("chunk_third", "text_c"),   # prev_rank 3
    ]

    hits = rerank("query", candidates)
    assert len(hits) == 3
    assert hits[0].score == hits[1].score
    # chunk_first had prev_rank 1, chunk_second had prev_rank 2
    assert hits[0].chunk_id == "chunk_first"
    assert hits[0].prev_rank == 1
    assert hits[1].chunk_id == "chunk_second"
    assert hits[1].prev_rank == 2


def test_invalid_query_and_top_n_raise_value_error():
    """Empty query or top_n < 1 raises ValueError."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        rerank("", [("c1", "text")])

    with pytest.raises(ValueError, match="Query cannot be empty"):
        rerank("   \n\t  ", [("c1", "text")])

    with pytest.raises(ValueError, match="top_n must be >= 1"):
        rerank("valid query", [("c1", "text")], top_n=0)

    with pytest.raises(ValueError, match="top_n must be >= 1"):
        rerank("valid query", [("c1", "text")], top_n=-3)


def test_import_does_not_load_model(monkeypatch):
    """Importing the module does not invoke model loading."""
    import sys

    called = False

    def explode(**kwargs):
        nonlocal called
        called = True
        raise RuntimeError("Model was loaded on import!")

    # Check cache info of get_cross_encoder
    from chatmem.retrieval.rerank import get_cross_encoder
    assert get_cross_encoder.cache_info().currsize == 0
    assert not called

