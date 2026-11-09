import pytest
from pydantic import ValidationError

from chatmem.retrieval.pipeline import (
    PipelineConfig,
    RetrievedChunk,
    _to_chunk,
)


def test_unknown_key_raises_validation_error():
    """1. Unknown key raises ValidationError (use rrf_K=40 and gate_treshold=0.3)."""
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig(rrf_K=40)
    assert "extra_forbidden" in str(exc_info.value) or "rrf_K" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig(gate_treshold=0.3)
    assert "extra_forbidden" in str(exc_info.value) or "gate_treshold" in str(exc_info.value)


def test_min_results_exceeds_final_top_k_raises():
    """2. min_results > final_top_k raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig(min_results=10, final_top_k=5)
    assert "min_results (10) cannot exceed final_top_k (5)." in str(exc_info.value)


def test_use_rerank_candidates_less_than_final_top_k_raises():
    """3. use_rerank=True with rerank_candidates < final_top_k raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig(use_rerank=True, rerank_candidates=3, final_top_k=5)
    assert "rerank_candidates (3) must be >= final_top_k (5) when use_rerank is True." in str(exc_info.value)


def test_use_rerank_false_with_rerank_candidates_less_than_final_top_k_succeeds():
    """4. use_rerank=False with rerank_candidates < final_top_k does NOT raise."""
    config = PipelineConfig(use_rerank=False, rerank_candidates=3, final_top_k=5)
    assert config.use_rerank is False
    assert config.rerank_candidates == 3
    assert config.final_top_k == 5


def test_gate_threshold_with_no_rerank_raises():
    """5. gate_threshold set with use_rerank=False still raises (existing rule)."""
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig(gate_threshold=0.5, use_rerank=False)
    assert "gate_threshold cannot be set when use_rerank is False" in str(exc_info.value)


def test_neither_bm25_nor_dense_raises():
    """6. Neither use_bm25 nor use_dense still raises (existing rule)."""
    with pytest.raises(ValidationError) as exc_info:
        PipelineConfig(use_bm25=False, use_dense=False)
    assert "At least one of use_bm25 or use_dense must be True." in str(exc_info.value)


def test_score_scale_returns_expected_scale():
    """7. score_scale returns 'rrf', 'raw', 'sigmoid' for the three cases."""
    # Case a: No reranker -> "rrf"
    c_rrf = PipelineConfig(use_rerank=False)
    assert c_rrf.score_scale == "rrf"

    # Case b: Reranker enabled, normalize=False -> "raw"
    c_raw = PipelineConfig(use_rerank=True, rerank_normalize=False)
    assert c_raw.score_scale == "raw"

    # Case c: Reranker enabled, normalize=True -> "sigmoid"
    c_sigmoid = PipelineConfig(use_rerank=True, rerank_normalize=True)
    assert c_sigmoid.score_scale == "sigmoid"


def test_from_yaml_with_typo_raises_and_valid_yaml_loads(tmp_path):
    """8. from_yaml with a temp YAML file containing a typo'd key raises;

    a valid YAML loads correctly (use pytest's tmp_path).
    """
    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text("name: bad\nrrf_K: 40\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        PipelineConfig.from_yaml(bad_yaml)

    good_yaml = tmp_path / "good_config.yaml"
    good_yaml.write_text("name: good\nrrf_k: 40\nfinal_top_k: 5\n", encoding="utf-8")
    config = PipelineConfig.from_yaml(good_yaml)
    assert config.name == "good"
    assert config.rrf_k == 40
    assert config.final_top_k == 5


def test_to_chunk_maps_every_field_correctly():
    """9. _to_chunk maps every field correctly from a hand-built record dict."""
    record = {
        "text": "sample text",
        "conversation_id": "conv_123",
        "platform": "chatgpt",
        "title": "A Great Conversation",
        "time": "2026-11-09T12:00:00Z",
        "message_ids": ["m1", "m2", "m3"],
    }
    chunk = _to_chunk(
        record=record,
        chunk_id="chunk_abc",
        score=0.88,
        ranks={"bm25": 1, "dense": 3},
    )
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.chunk_id == "chunk_abc"
    assert chunk.score == 0.88
    assert chunk.text == "sample text"
    assert chunk.conversation_id == "conv_123"
    assert chunk.platform == "chatgpt"
    assert chunk.title == "A Great Conversation"
    assert chunk.time == "2026-11-09T12:00:00Z"
    assert chunk.message_ids == ["m1", "m2", "m3"]
    assert chunk.ranks == {"bm25": 1, "dense": 3}

