import pytest
from chatmem.retrieval.fusion import FusedHit, reciprocal_rank_fusion


def test_worked_example():
    """Chunk A is rank 1 in bm25 and rank 3 in dense (score ~0.03227).

    Chunk B is rank 2 in bm25 only (score ~0.01613).
    A must rank above B with correct scores and ranks dicts.
    """
    results = {
        "bm25": [("chunk_a", 15.0), ("chunk_b", 12.0)],
        "dense": [("chunk_x", 0.95), ("chunk_y", 0.90), ("chunk_a", 0.85)],
    }

    hits = reciprocal_rank_fusion(results, k=60)

    assert len(hits) == 4
    assert hits[0].chunk_id == "chunk_a"

    expected_score_a = (1.0 / (60 + 1)) + (1.0 / (60 + 3))
    expected_score_b = 1.0 / (60 + 2)

    assert pytest.approx(hits[0].score, rel=1e-5) == expected_score_a
    assert hits[0].ranks == {"bm25": 1, "dense": 3}

    hit_b = next(h for h in hits if h.chunk_id == "chunk_b")
    assert pytest.approx(hit_b.score, rel=1e-5) == expected_score_b
    assert hit_b.ranks == {"bm25": 2}


def test_both_sources_beats_single_source():
    """A chunk found by both sources beats a chunk found by only one source

    at a similar rank.
    """
    results = {
        "bm25": [("chunk_single", 20.0), ("chunk_both", 18.0)],
        "dense": [("chunk_other", 0.99), ("chunk_both", 0.95)],
    }
    # chunk_both: rank 2 in bm25, rank 2 in dense -> 1/62 + 1/62 = 2/62 ~ 0.032258
    # chunk_single: rank 1 in bm25 only -> 1/61 ~ 0.016393
    # chunk_other: rank 1 in dense only -> 1/61 ~ 0.016393

    hits = reciprocal_rank_fusion(results, k=60)

    assert hits[0].chunk_id == "chunk_both"
    assert hits[0].score > hits[1].score


def test_weights_change_order():
    """Weights adjust the relative contribution of each source."""
    results = {
        "bm25": [("chunk_bm25", 10.0)],
        "dense": [("chunk_dense", 0.9)],
    }

    # Equal weights -> rank 1 tie, chunk_bm25 wins by chunk_id alphabetical
    default_hits = reciprocal_rank_fusion(results, k=60)
    assert default_hits[0].chunk_id == "chunk_bm25"
    assert default_hits[1].chunk_id == "chunk_dense"

    # Heavily favor dense
    dense_favored = reciprocal_rank_fusion(results, k=60, weights={"bm25": 0.1, "dense": 2.0})
    assert dense_favored[0].chunk_id == "chunk_dense"
    assert dense_favored[1].chunk_id == "chunk_bm25"
    assert dense_favored[0].score > dense_favored[1].score


def test_duplicates_in_one_list_counted_once():
    """If a chunk appears multiple times in a single list, only its first

    (best) occurrence is counted.
    """
    results = {
        "bm25": [
            ("chunk_dup", 10.0),
            ("chunk_dup", 8.0),
            ("chunk_other", 6.0),
            ("chunk_dup", 4.0),
        ]
    }

    hits = reciprocal_rank_fusion(results, k=60)

    assert len(hits) == 2
    assert hits[0].chunk_id == "chunk_dup"
    # Should only count rank 1
    assert pytest.approx(hits[0].score, rel=1e-5) == 1.0 / 61
    assert hits[0].ranks == {"bm25": 1}

    assert hits[1].chunk_id == "chunk_other"
    assert hits[1].ranks == {"bm25": 3}


def test_empty_inputs_and_top_n():
    """Empty inputs return [], and top_n truncates correctly."""
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"bm25": [], "dense": []}) == []

    results = {
        "bm25": [("c1", 1.0), ("c2", 0.9), ("c3", 0.8), ("c4", 0.7)],
    }

    hits_top2 = reciprocal_rank_fusion(results, top_n=2)
    assert len(hits_top2) == 2
    assert [h.chunk_id for h in hits_top2] == ["c1", "c2"]

    assert reciprocal_rank_fusion(results, top_n=0) == []
    assert len(reciprocal_rank_fusion(results, top_n=10)) == 4


def test_deterministic_tie_breaking():
    """Ties are broken first by better single-source rank, then by chunk_id."""
    # Scenario A: Equal RRF scores, but different best single-source ranks
    # chunk_best_rank: rank 1 with weight 0.5 -> 0.5 * (1/61) = 0.5 / 61
    # chunk_worse_rank: rank 61 with weight 1.0 -> 1.0 * (1/121) != equal
    # Let's test equal scores with different ranks using two sources:
    # Source A (weight 1.0): chunk_1 at rank 1 -> 1/61
    # Source B (weight 61/62): chunk_2 at rank 1 -> (61/62) * (1/61) !=
    # Cleaner: exact equal scores from symmetric inputs:
    # Source A has ("chunk_b", 1.0), Source B has ("chunk_a", 1.0)
    # Both chunk_a and chunk_b have rank 1 and score 1/61.
    # Tie broken alphabetically: "chunk_a" before "chunk_b"
    results = {
        "source1": [("chunk_b", 1.0)],
        "source2": [("chunk_a", 1.0)],
    }
    hits = reciprocal_rank_fusion(results, k=60)
    assert len(hits) == 2
    assert hits[0].score == hits[1].score
    assert hits[0].chunk_id == "chunk_a"
    assert hits[1].chunk_id == "chunk_b"

    # Scenario B: Tie broken by better single-source rank
    # Suppose chunk_x has rank 1 in source_a (weight 1.0) -> score = 1/61
    # chunk_y has rank 2 in source_a (weight 61/62) -> score = (61/62)*(1/62)
    # Let's create an exact tie where single-source rank differs:
    # k=2:
    # chunk_high: rank 1 in s1 (1/(2+1) = 1/3 ~ 0.3333333333333333)
    # chunk_low: rank 4 in s1 (1/6) + rank 4 in s2 (1/6) -> 1/6 + 1/6 = 2/6 = 1/3!
    # Both have score = 1/3!
    # chunk_high best rank is 1. chunk_low best rank is 4.
    # chunk_high must win because min(ranks) is 1 < 4!
    results_rank_tie = {
        "s1": [("chunk_high", 1.0), ("dummy1", 0.9), ("dummy2", 0.8), ("chunk_low", 0.7)],
        "s2": [("dummy3", 1.0), ("dummy4", 0.9), ("dummy5", 0.8), ("chunk_low", 0.7)],
    }
    hits_rank_tie = reciprocal_rank_fusion(results_rank_tie, k=2)
    # Both chunk_high and chunk_low have score 1/3
    chunk_high_hit = next(h for h in hits_rank_tie if h.chunk_id == "chunk_high")
    chunk_low_hit = next(h for h in hits_rank_tie if h.chunk_id == "chunk_low")
    assert pytest.approx(chunk_high_hit.score) == pytest.approx(chunk_low_hit.score)

    # chunk_high has best rank 1, chunk_low has best rank 4 -> chunk_high comes first
    high_idx = hits_rank_tie.index(chunk_high_hit)
    low_idx = hits_rank_tie.index(chunk_low_hit)
    assert high_idx < low_idx


def test_invalid_arguments_raise_value_error():
    """Raise ValueError if k <= 0 or any weight is negative."""
    with pytest.raises(ValueError, match="k must be strictly positive"):
        reciprocal_rank_fusion({"bm25": [("c1", 1.0)]}, k=0)

    with pytest.raises(ValueError, match="k must be strictly positive"):
        reciprocal_rank_fusion({"bm25": [("c1", 1.0)]}, k=-5)

    with pytest.raises(ValueError, match="cannot be negative"):
        reciprocal_rank_fusion(
            {"bm25": [("c1", 1.0)]},
            weights={"bm25": -0.5},
        )

