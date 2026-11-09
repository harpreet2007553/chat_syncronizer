import math
import pytest

from chatmem.eval.metrics import (
    dedupe_conversations,
    gate_sweep,
    hit_at_k,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    suggest_thresholds,
)


def test_recall_at_k_hand_computed():
    """1. recall_at_k: relevant {a,b}, ranked [x,a,y,b] -> k=2 gives 0.5, k=4 gives 1.0."""
    relevant = {"a", "b"}
    ranked = ["x", "a", "y", "b"]

    assert recall_at_k(ranked, relevant, k=2) == 0.5
    assert recall_at_k(ranked, relevant, k=4) == 1.0
    assert recall_at_k(ranked, relevant, k=1) == 0.0
    assert recall_at_k(ranked, relevant, k=3) == 0.5


def test_reciprocal_rank_hand_computed():
    """2. reciprocal_rank: first relevant at rank 3 -> 1/3; none -> 0.0."""
    relevant = {"target"}
    ranked = ["c1", "c2", "target", "c4"]
    assert reciprocal_rank(ranked, relevant) == pytest.approx(1.0 / 3.0)

    assert reciprocal_rank(["c1", "c2"], relevant) == 0.0
    assert reciprocal_rank([], relevant) == 0.0


def test_ndcg_at_k_hand_computed():
    """3. ndcg_at_k: ranked [a,x,b], relevant {a,b}, k=3 compute DCG/IDCG by hand.

    DCG = 1 + 0 + 0.5 = 1.5; IDCG = 1 + 1/log2(3) = 1.63092975...
    DCG / IDCG = 1.5 / 1.63092975... = 0.91972...
    Also a perfect ranking gives 1.0.
    """
    relevant = {"a", "b"}
    ranked = ["a", "x", "b"]

    expected_dcg = 1.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4)  # 1.0 + 0.5 = 1.5
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # 1.0 + 0.63092975...
    expected_ndcg = expected_dcg / expected_idcg

    assert ndcg_at_k(ranked, relevant, k=3) == pytest.approx(expected_ndcg, rel=1e-5)

    # Perfect ranking gives 1.0
    perfect_ranked = ["a", "b", "x"]
    assert ndcg_at_k(perfect_ranked, relevant, k=3) == 1.0
    assert ndcg_at_k(perfect_ranked, relevant, k=2) == 1.0


def test_recall_at_k_empty_relevant_raises_value_error():
    """4. recall_at_k with empty relevant raises ValueError."""
    with pytest.raises(ValueError, match="Answerable queries must have at least 1 relevant ID."):
        recall_at_k(["c1", "c2"], set(), k=2)

    with pytest.raises(ValueError):
        recall_at_k(["c1"], [], k=1)


def test_dedupe_conversations_preserves_first_occurrence():
    """5. dedupe_conversations keeps first-occurrence order."""
    chunks = [
        {"conversation_id": "conv_1"},
        {"conversation_id": "conv_2"},
        {"conversation_id": "conv_1"},
        {"conversation_id": "conv_3"},
        {"conversation_id": "conv_2"},
    ]
    assert dedupe_conversations(chunks) == ["conv_1", "conv_2", "conv_3"]

    # Also supports string lists or object lists
    assert dedupe_conversations(["a", "b", "a", "c"]) == ["a", "b", "c"]


def test_gate_sweep_hand_computed_table():
    """6. gate_sweep on a hand-built 6-row table (3 answerable, 3 no_answer) with 2 thresholds.

    Assert every rate by hand. Include a case where no_answer count is 0 -> those fields are None.
    """
    # 3 answerable:
    #   r1: ans=True, top_score=0.8, relevant_in_topk=True
    #   r2: ans=True, top_score=0.5, relevant_in_topk=True
    #   r3: ans=True, top_score=0.3, relevant_in_topk=False
    # 3 no_answer:
    #   r4: ans=False, top_score=0.6, relevant_in_topk=False
    #   r5: ans=False, top_score=0.4, relevant_in_topk=False
    #   r6: ans=False, top_score=0.2, relevant_in_topk=False
    rows = [
        {"answerable": True, "top_score": 0.8, "relevant_in_topk": True},
        {"answerable": True, "top_score": 0.5, "relevant_in_topk": True},
        {"answerable": True, "top_score": 0.3, "relevant_in_topk": False},
        {"answerable": False, "top_score": 0.6, "relevant_in_topk": False},
        {"answerable": False, "top_score": 0.4, "relevant_in_topk": False},
        {"answerable": False, "top_score": 0.2, "relevant_in_topk": False},
    ]

    # Threshold t = 0.5:
    #   Ans injected: r1 (0.8), r2 (0.5) -> 2 / 3 = 0.6667
    #   NoAns injected: r4 (0.6) -> 1 / 3 = 0.3333
    #   NoAns NOT injected: r5 (0.4), r6 (0.2) -> 2 / 3 = 0.6667
    #   Useful injection: ans & injected & relevant_in_topk -> r1, r2 -> 2 / 3 = 0.6667
    #   Missed good: ans & relevant_in_topk & NOT injected -> none -> 0 / 2 = 0.0
    #
    # Threshold t = 0.7:
    #   Ans injected: r1 (0.8) -> 1 / 3 = 0.3333
    #   NoAns injected: none -> 0 / 3 = 0.0
    #   NoAns NOT injected: r4, r5, r6 -> 3 / 3 = 1.0
    #   Useful injection: r1 -> 1 / 3 = 0.3333
    #   Missed good: r2 (0.5 < 0.7) -> 1 / 2 = 0.5
    sweep = gate_sweep(rows, [0.5, 0.7])
    assert len(sweep) == 2

    # Check t = 0.5
    s0 = sweep[0]
    assert s0["threshold"] == 0.5
    assert s0["inject_rate_answerable"] == pytest.approx(2.0 / 3.0)
    assert s0["correct_abstain_rate_no_answer"] == pytest.approx(2.0 / 3.0)
    assert s0["false_injection_rate"] == pytest.approx(1.0 / 3.0)
    assert s0["useful_injection_rate"] == pytest.approx(2.0 / 3.0)
    assert s0["missed_good_rate"] == pytest.approx(0.0)

    # Check t = 0.7
    s1 = sweep[1]
    assert s1["threshold"] == 0.7
    assert s1["inject_rate_answerable"] == pytest.approx(1.0 / 3.0)
    assert s1["correct_abstain_rate_no_answer"] == pytest.approx(1.0)
    assert s1["false_injection_rate"] == pytest.approx(0.0)
    assert s1["useful_injection_rate"] == pytest.approx(1.0 / 3.0)
    assert s1["missed_good_rate"] == pytest.approx(0.5)

    # Case where no_answer count is 0 -> those fields are None
    ans_only_rows = [
        {"answerable": True, "top_score": 0.8, "relevant_in_topk": True},
    ]
    sweep_ans_only = gate_sweep(ans_only_rows, [0.5])
    assert sweep_ans_only[0]["correct_abstain_rate_no_answer"] is None
    assert sweep_ans_only[0]["false_injection_rate"] is None


def test_gate_sweep_none_top_score_not_injected():
    """7. gate_sweep: top_score None counts as not injected."""
    rows = [
        {"answerable": True, "top_score": None, "relevant_in_topk": True},
        {"answerable": False, "top_score": None, "relevant_in_topk": False},
    ]
    sweep = gate_sweep(rows, [0.1])
    assert sweep[0]["inject_rate_answerable"] == 0.0
    assert sweep[0]["false_injection_rate"] == 0.0
    assert sweep[0]["correct_abstain_rate_no_answer"] == 1.0
    assert sweep[0]["missed_good_rate"] == 1.0


def test_suggest_thresholds():
    """8. suggest_thresholds: empty -> []; n values within [min, max]."""
    assert suggest_thresholds([]) == []
    assert suggest_thresholds([None, None]) == []

    thresholds = suggest_thresholds([0.1, 0.5, 0.9], n=5)
    assert len(thresholds) == 5
    assert thresholds[0] == pytest.approx(0.1)
    assert thresholds[-1] == pytest.approx(0.9)
    assert thresholds == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])

