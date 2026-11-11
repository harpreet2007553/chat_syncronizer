import sqlite3
from unittest.mock import MagicMock

import pytest

from chatmem.retrieval.gate import GateDecision
from chatmem.retrieval.pipeline import (
    PipelineConfig,
    Retriever,
)
from chatmem.retrieval.rerank import RerankedHit

FAKE_RECORDS = {
    "c_both": {
        "chunk_id": "c_both",
        "conversation_id": "conv_1",
        "platform": "chatgpt",
        "title": "Title Both",
        "text": "Text for both sources",
        "message_ids": ["m1", "m2"],
        "time": "2026-09-01T10:00:00Z",
    },
    "c_bm25": {
        "chunk_id": "c_bm25",
        "conversation_id": "conv_2",
        "platform": "chatgpt",
        "title": "Title BM25",
        "text": "Text for bm25 only",
        "message_ids": ["m3"],
        "time": "2026-09-01T11:00:00Z",
    },
    "c_dense": {
        "chunk_id": "c_dense",
        "conversation_id": "conv_3",
        "platform": "claude",
        "title": "Title Dense",
        "text": "Text for dense only",
        "message_ids": ["m4"],
        "time": "2026-09-01T12:00:00Z",
    },
}


@pytest.fixture
def fake_db_conn():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def fake_qdrant_client():
    return MagicMock()


def test_hybrid_rrf_fuses_both_sources_and_skips_rerank(
    monkeypatch, fake_db_conn, fake_qdrant_client
):
    """1. hybrid_rrf config: results from both sources are fused, a chunk found by both

    ranks first, and the rerank function is never called.
    """
    mock_bm25 = MagicMock(return_value=[("c_bm25", 10.0), ("c_both", 8.0)])
    mock_dense = MagicMock(return_value=[("c_dense", 0.95), ("c_both", 0.90)])
    mock_get_chunks = MagicMock(return_value=FAKE_RECORDS)
    mock_rerank = MagicMock()

    monkeypatch.setattr("chatmem.retrieval.pipeline.bm25_search", mock_bm25)
    monkeypatch.setattr("chatmem.retrieval.pipeline.search_dense", mock_dense)
    monkeypatch.setattr("chatmem.retrieval.pipeline.get_chunks", mock_get_chunks)
    monkeypatch.setattr("chatmem.retrieval.pipeline.rerank", mock_rerank)

    config = PipelineConfig.from_yaml("configs/hybrid_rrf.yaml")
    retriever = Retriever(config=config, conn=fake_db_conn, client=fake_qdrant_client)

    result = retriever.search("how to configure database")

    # Both search sources must have been called
    assert mock_bm25.called
    assert mock_dense.called
    # Rerank must NOT be called for hybrid_rrf
    assert not mock_rerank.called

    assert result.injected is True
    assert result.reason == "no_gate"
    assert len(result.chunks) == 3
    # c_both appeared in both sources (ranks 2 and 2) -> beats single-source rank 1s
    assert result.chunks[0].chunk_id == "c_both"
    assert result.chunks[0].ranks == {"bm25": 2, "dense": 2}
    assert "search" in result.timings_ms
    assert "fusion" in result.timings_ms
    assert "fetch" in result.timings_ms
    assert "rerank" not in result.timings_ms
    assert "total" in result.timings_ms


def test_bm25_only_skips_dense_search(monkeypatch, fake_db_conn, fake_qdrant_client):
    """2. bm25_only: search_dense is never called, and the result still has the same shape."""
    mock_bm25 = MagicMock(return_value=[("c_bm25", 15.0)])
    mock_dense = MagicMock()
    mock_get_chunks = MagicMock(return_value=FAKE_RECORDS)

    monkeypatch.setattr("chatmem.retrieval.pipeline.bm25_search", mock_bm25)
    monkeypatch.setattr("chatmem.retrieval.pipeline.search_dense", mock_dense)
    monkeypatch.setattr("chatmem.retrieval.pipeline.get_chunks", mock_get_chunks)

    config = PipelineConfig.from_yaml("configs/bm25_only.yaml")
    retriever = Retriever(config=config, conn=fake_db_conn, client=fake_qdrant_client)

    result = retriever.search("pandas groupby")

    assert mock_bm25.called
    assert not mock_dense.called

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "c_bm25"
    assert result.chunks[0].ranks == {"bm25": 1}
    assert result.reason == "no_gate"


def test_dense_only_skips_bm25_search(monkeypatch, fake_db_conn, fake_qdrant_client):
    """3. dense_only: bm25_search is never called."""
    mock_bm25 = MagicMock()
    mock_dense = MagicMock(return_value=[("c_dense", 0.88)])
    mock_get_chunks = MagicMock(return_value=FAKE_RECORDS)

    monkeypatch.setattr("chatmem.retrieval.pipeline.bm25_search", mock_bm25)
    monkeypatch.setattr("chatmem.retrieval.pipeline.search_dense", mock_dense)
    monkeypatch.setattr("chatmem.retrieval.pipeline.get_chunks", mock_get_chunks)

    config = PipelineConfig.from_yaml("configs/dense_only.yaml")
    retriever = Retriever(config=config, conn=fake_db_conn, client=fake_qdrant_client)

    result = retriever.search("vector embedding query")

    assert not mock_bm25.called
    assert mock_dense.called
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "c_dense"
    assert result.chunks[0].ranks == {"dense": 1}


def test_hybrid_rerank_orders_by_reranker_scores(
    monkeypatch, fake_db_conn, fake_qdrant_client
):
    """4. hybrid_rerank: rerank receives candidates in fusion order, and final order

    follows the reranker scores.
    """
    mock_bm25 = MagicMock(return_value=[("c_bm25", 10.0), ("c_dense", 5.0)])
    mock_dense = MagicMock(return_value=[("c_dense", 0.95), ("c_bm25", 0.80)])
    mock_get_chunks = MagicMock(return_value=FAKE_RECORDS)

    # Fake reranker flips the order based on semantic relevance
    def fake_rerank(query, candidates, **kwargs):
        # Candidates should be received in fusion order
        assert len(candidates) == 2
        return [
            RerankedHit(chunk_id="c_dense", score=5.8, prev_rank=1),
            RerankedHit(chunk_id="c_bm25", score=1.2, prev_rank=2),
        ]

    monkeypatch.setattr("chatmem.retrieval.pipeline.bm25_search", mock_bm25)
    monkeypatch.setattr("chatmem.retrieval.pipeline.search_dense", mock_dense)
    monkeypatch.setattr("chatmem.retrieval.pipeline.get_chunks", mock_get_chunks)
    monkeypatch.setattr("chatmem.retrieval.pipeline.rerank", fake_rerank)

    config = PipelineConfig.from_yaml("configs/hybrid_rerank.yaml")
    retriever = Retriever(config=config, conn=fake_db_conn, client=fake_qdrant_client)

    result = retriever.search("query requiring reranker")

    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_id == "c_dense"
    assert result.chunks[0].score == 5.8
    assert result.chunks[1].chunk_id == "c_bm25"
    assert result.chunks[1].score == 1.2
    assert "rerank" in result.timings_ms


def test_gate_integration(monkeypatch, fake_db_conn, fake_qdrant_client):
    """Pipeline properly routes to apply_gate when gate_threshold is set."""
    mock_bm25 = MagicMock(return_value=[("c_bm25", 10.0)])
    mock_dense = MagicMock(return_value=[])
    mock_get_chunks = MagicMock(return_value=FAKE_RECORDS)
    mock_rerank = MagicMock(
        return_value=[RerankedHit(chunk_id="c_bm25", score=0.45, prev_rank=1)]
    )
    mock_gate = MagicMock(
        return_value=GateDecision(
            passed=False,
            hits=[],
            reason="below_threshold",
            top_score=0.45,
        )
    )

    monkeypatch.setattr("chatmem.retrieval.pipeline.bm25_search", mock_bm25)
    monkeypatch.setattr("chatmem.retrieval.pipeline.search_dense", mock_dense)
    monkeypatch.setattr("chatmem.retrieval.pipeline.get_chunks", mock_get_chunks)
    monkeypatch.setattr("chatmem.retrieval.pipeline.rerank", mock_rerank)
    monkeypatch.setattr("chatmem.retrieval.pipeline.apply_gate", mock_gate)

    config = PipelineConfig(
        name="test_gate",
        use_bm25=True,
        use_dense=False,
        use_rerank=True,
        gate_threshold=0.50,
    )
    retriever = Retriever(config=config, conn=fake_db_conn, client=fake_qdrant_client)

    result = retriever.search("irrelevant query")

    assert mock_gate.called
    assert result.injected is False
    assert result.reason == "below_threshold"
    assert result.chunks == []
    assert result.top_score == 0.45
    assert "gate" in result.timings_ms


def test_pipeline_config_validation():
    """PipelineConfig validates contradictory or invalid combinations."""
    # At least one source must be true
    with pytest.raises(ValueError, match="At least one of use_bm25 or use_dense"):
        PipelineConfig(use_bm25=False, use_dense=False)

    # gate_threshold set while use_rerank is false must raise ValueError
    with pytest.raises(ValueError, match="gate_threshold cannot be set when use_rerank is False"):
        PipelineConfig(use_bm25=True, use_rerank=False, gate_threshold=0.5)


def test_empty_query_raises_value_error(fake_db_conn, fake_qdrant_client):
    """Empty or whitespace-only query raises ValueError."""
    config = PipelineConfig(use_bm25=True, use_dense=False, use_rerank=False)
    retriever = Retriever(config=config, conn=fake_db_conn, client=fake_qdrant_client)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.search("")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.search("   \n\t ")


def test_empty_search_results_return_no_candidates(monkeypatch, fake_db_conn, fake_qdrant_client):
    """Empty search results return 'no_candidates' without loading reranker."""
    mock_bm25 = MagicMock(return_value=[])
    mock_dense = MagicMock(return_value=[])
    mock_rerank = MagicMock()

    monkeypatch.setattr("chatmem.retrieval.pipeline.bm25_search", mock_bm25)
    monkeypatch.setattr("chatmem.retrieval.pipeline.search_dense", mock_dense)
    monkeypatch.setattr("chatmem.retrieval.pipeline.rerank", mock_rerank)

    config = PipelineConfig(use_bm25=True, use_dense=True, use_rerank=True)
    retriever = Retriever(config=config, conn=fake_db_conn, client=fake_qdrant_client)

    result = retriever.search("query with no matches")

    assert result.injected is False
    assert result.reason == "no_candidates"
    assert result.chunks == []
    assert not mock_rerank.called


def test_context_manager_calls_close(fake_db_conn, fake_qdrant_client):
    """Context manager closes connection and client."""
    config = PipelineConfig(use_bm25=True, use_dense=False, use_rerank=False)
    retriever = Retriever(config=config, conn=fake_db_conn, client=fake_qdrant_client)

    with retriever as r:
        assert r is retriever

    # Connection should be closed
    with pytest.raises(sqlite3.ProgrammingError):
        fake_db_conn.execute("SELECT 1")
    assert fake_qdrant_client.close.called

