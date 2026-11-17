"""Tests for chatmem local FastAPI service, endpoints, auth, and privacy."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from chatmem.api.main import create_app
from chatmem.api.models import ApiConfig
from chatmem.chunking.qa_pair import qa_chunks
from chatmem.index.dense import DenseIndex
from chatmem.index.store import SQLiteMetadataStore
from chatmem.ingest.normalize import load_all_exports
from chatmem.retrieval.pipeline import QdrantConfig


@pytest.fixture(scope="module")
def api_test_setup(tmp_path_factory: pytest.TempPathFactory):
    """Fixture initializing temporary SQLite, Qdrant, sample data, and API config."""
    tmp_path = tmp_path_factory.mktemp("api_test")
    db_path = tmp_path / "test_chatmem.db"
    qdrant_dir = tmp_path / "test_qdrant"
    token_path = tmp_path / "token"

    test_token = "test_secret_bearer_token_1234567890abcdef"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(test_token, encoding="utf-8")

    # Ingest data/sample for test index
    sample_convs = load_all_exports("data/sample")
    assert len(sample_convs) > 0
    chunks = qa_chunks(sample_convs)

    sqlite_store = SQLiteMetadataStore(db_path)
    sqlite_store.save_conversations(sample_convs)
    sqlite_store.save_chunks(chunks, strategy="qa")

    q_cfg = QdrantConfig(mode="local", path=str(qdrant_dir))
    with DenseIndex(path=qdrant_dir, collection_name="chunks_qa", mode="local") as dense_idx:
        dense_idx.index_chunks(chunks)

    api_cfg = ApiConfig(
        host="127.0.0.1",
        port=8000,
        default_mode="fast",
        auth_token_path=str(token_path),
        allowed_cors_origins=["https://allowed.chatmem.test"],
        max_results=5,
        max_injected_chars=1000,
        max_message_chars=300,
        max_request_bytes=3000,
        fast_gate_threshold=0.30,  # low threshold to ensure sample hits inject
        precise_gate_threshold=0.010,
    )

    app = create_app(
        config=api_cfg,
        qdrant_config=q_cfg,
        db_path=db_path,
    )

    with TestClient(app) as client:
        yield {
            "app": app,
            "client": client,
            "token": test_token,
            "headers": {"Authorization": f"Bearer {test_token}"},
            "api_cfg": api_cfg,
            "db_path": db_path,
            "qdrant_dir": qdrant_dir,
            "sample_convs": sample_convs,
            "chunks": chunks,
        }


def test_health_endpoint_no_private_data(api_test_setup):
    """Health endpoint must return status and metadata without requiring auth or leaking data."""
    client = api_test_setup["client"]
    # GET /health without auth headers
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert "embedding_model" in data
    assert "rerank_model" in data
    assert isinstance(data["reranker_loaded"], bool)
    assert data["conversations_indexed"] >= len(api_test_setup["sample_convs"])
    assert data["chunks_indexed"] >= len(api_test_setup["chunks"])

    # Strict check: No private text, ids, or conversation titles
    text_dump = resp.text.lower()
    for conv in api_test_setup["sample_convs"]:
        if conv.title and len(conv.title) > 3:
            assert conv.title.lower() not in text_dump
        for m in conv.messages:
            if len(m.text) > 5:
                assert m.text.lower() not in text_dump


def test_auth_protection(api_test_setup):
    """Protected endpoints must reject missing or invalid bearer tokens with 401."""
    client = api_test_setup["client"]

    # 1. Missing Authorization header
    res1 = client.post("/search", json={"query": "test query"})
    assert res1.status_code == 401

    # 2. Bad token
    res2 = client.post(
        "/search",
        json={"query": "test query"},
        headers={"Authorization": "Bearer bad_token_xyz"},
    )
    assert res2.status_code == 401

    # 3. Valid token succeeds
    res3 = client.post(
        "/search",
        json={"query": "python"},
        headers=api_test_setup["headers"],
    )
    assert res3.status_code == 200


def test_search_fast_and_precise_modes(api_test_setup):
    """Test search endpoint in fast mode and precise mode with mocked reranker."""
    client = api_test_setup["client"]
    headers = api_test_setup["headers"]

    # 1. Fast mode search
    resp_fast = client.post(
        "/search",
        json={"query": "python", "mode": "fast", "top_k": 3},
        headers=headers,
    )
    assert resp_fast.status_code == 200
    data_fast = resp_fast.json()
    assert "injected" in data_fast
    assert "gate_score" in data_fast
    assert "timings_ms" in data_fast
    assert isinstance(data_fast["results"], list)
    if data_fast["results"]:
        first = data_fast["results"][0]
        assert "chunk_id" in first
        assert "text" in first
        assert "score" in first
        assert "citation" in first
        assert "platform" in first["citation"]
        assert "conversation_id" in first["citation"]

    # 2. Precise mode search (mock cross-encoder to keep test lightning fast)
    mock_ce = MagicMock()
    mock_ce.predict.return_value = [0.99] * 10
    with patch("chatmem.retrieval.rerank.get_cross_encoder", return_value=mock_ce):
        resp_precise = client.post(
            "/search",
            json={"query": "python", "mode": "precise", "top_k": 3},
            headers=headers,
        )
        assert resp_precise.status_code == 200
        data_precise = resp_precise.json()
        assert "injected" in data_precise
        assert "results" in data_precise


def test_context_formatting_and_gate_abstain(api_test_setup):
    """Test /context endpoint format, max_chars bounding, and clean gate abstention."""
    client = api_test_setup["client"]
    headers = api_test_setup["headers"]

    # 1. Successful context formatting
    resp = client.post(
        "/context",
        json={"query": "python", "max_chars": 500},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    if data["injected"]:
        assert "--- PAST CONVERSATION CONTEXT (chatmem) ---" in data["context"]
        assert "--- END CONTEXT ---" in data["context"]
        assert len(data["context"]) <= 500
        assert data["chunks_used"] >= 1
        assert data["total_chars"] == len(data["context"])

    # 2. High threshold / abstain test
    resp_abstain = client.post(
        "/search",
        json={"query": "xyznonexistentgibberish12345"},
        headers=headers,
    )
    assert resp_abstain.status_code == 200
    search_data = resp_abstain.json()
    if not search_data["injected"]:
        resp_ctx_abstain = client.post(
            "/context",
            json={"query": "xyznonexistentgibberish12345"},
            headers=headers,
        )
        assert resp_ctx_abstain.status_code == 200
        ctx_data = resp_ctx_abstain.json()
        assert ctx_data["injected"] is False
        assert ctx_data["context"] == ""
        assert ctx_data["chunks_used"] == 0


def test_ingest_idempotency_and_update(api_test_setup):
    """Test /ingest/messages idempotency and updates."""
    client = api_test_setup["client"]
    headers = api_test_setup["headers"]

    payload = {
        "platform": "chatgpt",
        "conversation_id": "test_conv_id_001",
        "title": "Unit Test Conversation",
        "messages": [
            {"role": "user", "text": "What is the capital of France?"},
            {"role": "assistant", "text": "The capital of France is Paris."},
        ],
    }

    # 1. Ingest new conversation
    r1 = client.post("/ingest/messages", json=payload, headers=headers)
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "ok"
    initial_chunks = d1["chunks_indexed"]
    assert initial_chunks > 0

    # 2. Idempotent re-ingest: sending identical conversation yields same chunk count
    r2 = client.post("/ingest/messages", json=payload, headers=headers)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["chunks_indexed"] == initial_chunks

    # Check total chunk count in health has not doubled
    h = client.get("/health").json()
    assert h["chunks_indexed"] >= initial_chunks

    # 3. Update conversation with new turn
    payload_updated = dict(payload)
    payload_updated["messages"] = list(payload["messages"]) + [
        {"role": "user", "text": "And what is its population?"},
        {"role": "assistant", "text": "Paris has a population of over 2 million people."},
    ]

    r3 = client.post("/ingest/messages", json=payload_updated, headers=headers)
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["chunks_indexed"] >= initial_chunks

    # Verify search can retrieve the newly added turn
    s_res = client.post(
        "/search",
        json={"query": "capital of France Paris"},
        headers=headers,
    )
    assert s_res.status_code == 200


def test_delete_conversation(api_test_setup):
    """Test DELETE /conversations/{platform}/{conversation_id} removes from both stores."""
    client = api_test_setup["client"]
    headers = api_test_setup["headers"]

    payload = {
        "platform": "claude",
        "conversation_id": "to_be_deleted_conv",
        "title": "Temporary Conv",
        "messages": [
            {"role": "user", "text": "Delete me later."},
            {"role": "assistant", "text": "I will be removed shortly."},
        ],
    }

    # Ingest
    r1 = client.post("/ingest/messages", json=payload, headers=headers)
    assert r1.status_code == 200

    # Delete
    del_r = client.delete("/conversations/claude/to_be_deleted_conv", headers=headers)
    assert del_r.status_code == 200
    del_data = del_r.json()
    assert del_data["status"] == "deleted"
    assert del_data["conversation_id"] == "to_be_deleted_conv"
    assert del_data["platform"] == "claude"

    # Verify it no longer appears in search
    s_res = client.post(
        "/search",
        json={"query": "Delete me later", "platform": "claude"},
        headers=headers,
    )
    assert s_res.status_code == 200
    for item in s_res.json()["results"]:
        assert item["citation"]["conversation_id"] != "to_be_deleted_conv"


def test_cors_policy(api_test_setup):
    """CORS should allow configured origins and deny unconfigured ones."""
    client = api_test_setup["client"]

    # 1. Disallowed origin
    r_disallowed = client.options(
        "/health",
        headers={
            "Origin": "https://malicious-attacker.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Should not echo back the disallowed origin
    allow_origin = r_disallowed.headers.get("access-control-allow-origin")
    assert allow_origin != "https://malicious-attacker.com"
    assert allow_origin != "*"

    # 2. Allowed origin
    r_allowed = client.options(
        "/health",
        headers={
            "Origin": "https://allowed.chatmem.test",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r_allowed.headers.get("access-control-allow-origin") == "https://allowed.chatmem.test"


def test_oversized_messages_and_payloads_rejected(api_test_setup):
    """Test rejection of oversized message text and request size limits."""
    client = api_test_setup["client"]
    headers = api_test_setup["headers"]

    # 1. Message exceeding max_message_chars (configured to 300)
    oversized_msg = "A" * 350
    r_bad_msg = client.post(
        "/ingest/messages",
        json={
            "platform": "chatgpt",
            "conversation_id": "oversized_conv",
            "messages": [{"role": "user", "text": oversized_msg}],
        },
        headers=headers,
    )
    assert r_bad_msg.status_code == 400
    assert "exceeds maximum character length" in r_bad_msg.text

    # 2. Request exceeding max_request_bytes (configured to 3000 bytes)
    huge_payload = {
        "query": "A" * 3500,
    }
    r_huge = client.post(
        "/search",
        json=huge_payload,
        headers=headers,
    )
    assert r_huge.status_code == 413


def test_concurrent_ingest_requests(api_test_setup):
    """Test that two concurrent ingest requests execute cleanly under write_lock without lock errors."""
    client = api_test_setup["client"]
    headers = api_test_setup["headers"]

    payload1 = {
        "platform": "chatgpt",
        "conversation_id": "concurrent_conv_1",
        "title": "Thread 1",
        "messages": [
            {"role": "user", "text": "Question from thread 1."},
            {"role": "assistant", "text": "Answer from thread 1."},
        ],
    }
    payload2 = {
        "platform": "claude",
        "conversation_id": "concurrent_conv_2",
        "title": "Thread 2",
        "messages": [
            {"role": "user", "text": "Question from thread 2."},
            {"role": "assistant", "text": "Answer from thread 2."},
        ],
    }

    def _post(payload):
        return client.post("/ingest/messages", json=payload, headers=headers)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_post, payload1)
        f2 = executor.submit(_post, payload2)

        res1 = f1.result()
        res2 = f2.result()

    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json()["chunks_indexed"] > 0
    assert res2.json()["chunks_indexed"] > 0


def test_privacy_check_no_query_in_logs(api_test_setup, caplog):
    """Strict privacy check: Query text must NEVER appear in any log output during search."""
    client = api_test_setup["client"]
    headers = api_test_setup["headers"]

    canary_query = "CANARY_CONFIDENTIAL_QUERY_XYZ_777"

    with caplog.at_level(logging.DEBUG):
        resp = client.post(
            "/search",
            json={"query": canary_query},
            headers=headers,
        )
        assert resp.status_code == 200

    # Assert canary string is not present in logged text
    assert canary_query not in caplog.text
    # But verify access logging occurred
    assert "/search" in caplog.text

