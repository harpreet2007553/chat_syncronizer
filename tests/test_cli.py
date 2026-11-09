import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from chatmem.main import app
from chatmem.retrieval.pipeline import PipelineResult, RetrievedChunk
from chatmem.schema import Chunk, Conversation, Message

runner = CliRunner()


@pytest.fixture
def mock_conversation():
    msg = Message(
        message_id="m1",
        conversation_id="conv_1",
        platform="chatgpt",
        role="user",
        title="Sample Conv",
        text="Hello world",
        timestamp="2026-11-09T10:00:00",
    )
    return Conversation(
        conversation_id="conv_1",
        platform="chatgpt",
        title="Sample Conv",
        creation_time="2026-11-09T10:00:00",
        updation_time="2026-11-09T10:00:00",
        messages=[msg],
    )


@pytest.fixture
def mock_chunk():
    return Chunk(
        chunk_id="chunk_qa_0",
        conversation_id="conv_1",
        platform="chatgpt",
        title="Sample Conv",
        text="User: Hello world",
        message_ids=["m1"],
        time="2026-11-09T10:00:00",
    )


class FakeRetriever:
    def __init__(self, result=None, raise_on_search=False):
        self.result = result
        self.raise_on_search = raise_on_search
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.closed = True

    def search(self, query: str):
        if self.raise_on_search:
            raise RuntimeError("search failed")
        return self.result


# ----------------------------------------------------------------------------
# Test 1: --help exits 0 and lists all commands
# ----------------------------------------------------------------------------
def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.output
    assert "search" in result.output
    assert "eval" in result.output
    assert "serve" in result.output


# ----------------------------------------------------------------------------
# Test 2: search "" exits 2 and prints error without traceback
# ----------------------------------------------------------------------------
def test_search_empty_query_exits_2():
    result = runner.invoke(app, ["search", ""])
    assert result.exit_code == 2
    assert "Query cannot be empty" in result.output
    assert "Traceback" not in result.output


# ----------------------------------------------------------------------------
# Test 3: search "x" with injected=True prints score, platform, and title
# ----------------------------------------------------------------------------
def test_search_injected_true_prints_details(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("name: test\n", encoding="utf-8")
    db_file = tmp_path / "test.db"
    db_file.touch()

    chunk = RetrievedChunk(
        chunk_id="c1",
        score=0.9521,
        text="Some retrieved text content",
        conversation_id="conv_1",
        platform="chatgpt",
        title="Title ChatGPT",
        time="2026-11-09T10:00:00Z",
        message_ids=["m1"],
        ranks={"dense": 1},
    )
    fake_res = PipelineResult(
        chunks=[chunk],
        injected=True,
        reason="ok",
        top_score=0.9521,
        score_scale="raw",
        timings_ms={"total": 12.3},
    )
    fake_retriever = FakeRetriever(result=fake_res)
    monkeypatch.setattr(
        "chatmem.retrieval.pipeline.build_retriever",
        lambda *args, **kwargs: fake_retriever,
    )

    result = runner.invoke(
        app,
        ["search", "my query", "--config", str(cfg_file), "--db", str(db_file)],
    )
    assert result.exit_code == 0
    assert "0.9521" in result.output
    assert "chatgpt" in result.output
    assert "Title ChatGPT" in result.output
    assert "conv_1" in result.output


# ----------------------------------------------------------------------------
# Test 4: search "x" with injected=False prints NOT INJECTED and chunks
# ----------------------------------------------------------------------------
def test_search_injected_false_prints_not_injected_and_chunks(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("name: test\n", encoding="utf-8")
    db_file = tmp_path / "test.db"
    db_file.touch()

    chunk = RetrievedChunk(
        chunk_id="c2",
        score=0.1234,
        text="Low relevance text",
        conversation_id="conv_2",
        platform="claude",
        title="Gate Rejected Title",
        time="2026-11-09T11:00:00Z",
        message_ids=["m2"],
        ranks={"bm25": 1},
    )
    fake_res = PipelineResult(
        chunks=[chunk],
        injected=False,
        reason="below_threshold",
        top_score=0.1234,
        score_scale="raw",
        timings_ms={"total": 5.0},
    )
    fake_retriever = FakeRetriever(result=fake_res)
    monkeypatch.setattr(
        "chatmem.retrieval.pipeline.build_retriever",
        lambda *args, **kwargs: fake_retriever,
    )

    result = runner.invoke(
        app,
        ["search", "my query", "--config", str(cfg_file), "--db", str(db_file)],
    )
    assert result.exit_code == 0
    assert "NOT INJECTED" in result.output
    assert "Gate Rejected Title" in result.output


# ----------------------------------------------------------------------------
# Test 5: search "x" --json outputs valid JSON
# ----------------------------------------------------------------------------
def test_search_json_flag_outputs_valid_json(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("name: test\n", encoding="utf-8")
    db_file = tmp_path / "test.db"
    db_file.touch()

    fake_res = PipelineResult(
        chunks=[],
        injected=False,
        reason="no_candidates",
        top_score=None,
        score_scale="rrf",
        timings_ms={"total": 1.0},
    )
    fake_retriever = FakeRetriever(result=fake_res)
    monkeypatch.setattr(
        "chatmem.retrieval.pipeline.build_retriever",
        lambda *args, **kwargs: fake_retriever,
    )

    result = runner.invoke(
        app,
        ["search", "my query", "--config", str(cfg_file), "--db", str(db_file), "--json"],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert isinstance(parsed, dict)
    assert parsed["reason"] == "no_candidates"
    assert parsed["chunks"] == []


# ----------------------------------------------------------------------------
# Test 6: retriever is closed after search even when search() raises
# ----------------------------------------------------------------------------
def test_retriever_closed_on_search_exception(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("name: test\n", encoding="utf-8")
    db_file = tmp_path / "test.db"
    db_file.touch()

    fake_retriever = FakeRetriever(raise_on_search=True)
    monkeypatch.setattr(
        "chatmem.retrieval.pipeline.build_retriever",
        lambda *args, **kwargs: fake_retriever,
    )

    with pytest.raises(RuntimeError):
        runner.invoke(
            app,
            ["search", "my query", "--config", str(cfg_file), "--db", str(db_file)],
            catch_exceptions=False,
        )

    assert fake_retriever.closed is True


# ----------------------------------------------------------------------------
# Test 7: index with nonexistent --input path exits non-zero with clear message
# ----------------------------------------------------------------------------
def test_index_nonexistent_input_exits_2():
    result = runner.invoke(app, ["index", "--input", "nonexistent/path/nowhere"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


# ----------------------------------------------------------------------------
# Test 8: index where one conversation raises during chunking finishes and reports skipped=1
# ----------------------------------------------------------------------------
def test_index_skipped_conversation_reported(tmp_path, monkeypatch, mock_conversation, mock_chunk):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    conv_good = mock_conversation
    conv_bad = Conversation(
        conversation_id="conv_bad",
        platform="chatgpt",
        title="Bad Conv",
        creation_time="2026-11-09T10:00:00",
        updation_time="2026-11-09T10:00:00",
        messages=[],
    )

    monkeypatch.setattr(
        "chatmem.ingest.normalize.load_all_exports",
        lambda p: [conv_good, conv_bad],
    )
    mock_store = MagicMock()
    mock_store.save_chunks.return_value = 1
    monkeypatch.setattr("chatmem.index.store.SQLiteMetadataStore", lambda p: mock_store)

    # Chunker raises if conv_bad is processed
    def mock_qa_chunks(convs):
        if any(c.conversation_id == "conv_bad" for c in convs):
            raise ValueError("bad conv chunking failure")
        return [mock_chunk]

    monkeypatch.setattr("chatmem.chunking.qa_pair.qa_chunks", mock_qa_chunks)

    result = runner.invoke(
        app,
        ["index", "--input", str(export_dir), "--strategy", "qa", "--skip-dense"],
    )
    assert result.exit_code == 0
    assert "skipped=1" in result.output


# ----------------------------------------------------------------------------
# Test 9: eval command invokes run_sweep and exits cleanly
# ----------------------------------------------------------------------------
def test_eval_command_runs_sweep(tmp_path, monkeypatch):
    q_file = tmp_path / "queries.jsonl"
    q_file.write_text('{"qid": "q1", "query": "hi", "relevant": ["c1"], "type": "keyword"}\n', encoding="utf-8")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("name: test\n", encoding="utf-8")
    out_dir = tmp_path / "eval_out"
    out_dir.mkdir()
    (out_dir / "summary.md").write_text("# Summary\n", encoding="utf-8")

    mock_run = MagicMock(return_value=[MagicMock(errors=[], warnings=[])])
    monkeypatch.setattr("chatmem.eval.run_eval.run_sweep", mock_run)

    result = runner.invoke(
        app,
        ["eval", "--queries", str(q_file), "--configs", str(cfg_file), "--out", str(out_dir)],
    )
    assert result.exit_code == 0
    assert mock_run.called
    assert "# Summary" in result.output


# ----------------------------------------------------------------------------
# Test 10: importing chatmem.main does not import torch or qdrant_client
# ----------------------------------------------------------------------------
def test_import_main_does_not_import_heavy_modules():
    code = (
        "import sys, chatmem.main; "
        "assert 'torch' not in sys.modules, 'torch was imported!'; "
        "assert 'qdrant_client' not in sys.modules, 'qdrant_client was imported!'"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Import check failed: {res.stderr}"


# ----------------------------------------------------------------------------
# Test 11: index --input <file> without --platform exits 2 with required message
# ----------------------------------------------------------------------------
def test_index_file_without_platform_exits_2(tmp_path):
    f = tmp_path / "export.json"
    f.write_text("[]", encoding="utf-8")

    result = runner.invoke(app, ["index", "--input", str(f)])
    assert result.exit_code == 2
    assert "A single file needs --platform chatgpt|claude." in result.output


# ----------------------------------------------------------------------------
# Test 12: index --input <dir> calls load_all_exports and DenseIndex per strategy
# ----------------------------------------------------------------------------
def test_index_directory_calls_load_all_exports_and_dense_index(
    tmp_path, monkeypatch, mock_conversation, mock_chunk
):
    from chatmem.retrieval.pipeline import STRATEGY_COLLECTIONS

    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    mock_load_all = MagicMock(return_value=[mock_conversation])
    monkeypatch.setattr("chatmem.ingest.normalize.load_all_exports", mock_load_all)

    mock_store = MagicMock()
    mock_store.save_chunks.return_value = 1
    monkeypatch.setattr("chatmem.index.store.SQLiteMetadataStore", lambda p: mock_store)

    monkeypatch.setattr("chatmem.chunking.per_message.chunk_per_message", lambda c: [mock_chunk])
    monkeypatch.setattr("chatmem.chunking.qa_pair.qa_chunks", lambda c: [mock_chunk])
    monkeypatch.setattr("chatmem.chunking.sliding_window.window_chunks", lambda c: [mock_chunk])

    created_collections = []

    class MockDenseIndex:
        def __init__(self, collection_name, **kwargs):
            created_collections.append(collection_name)

        def index_chunks(self, chunks):
            return len(chunks)

    monkeypatch.setattr("chatmem.index.dense.DenseIndex", MockDenseIndex)

    result = runner.invoke(app, ["index", "--input", str(export_dir), "--strategy", "all"])
    assert result.exit_code == 0
    assert mock_load_all.called
    assert mock_store.save_chunks.call_count == 3
    assert created_collections == [
        STRATEGY_COLLECTIONS["pm"],
        STRATEGY_COLLECTIONS["qa"],
        STRATEGY_COLLECTIONS["ws"],
    ]


# ----------------------------------------------------------------------------
# Test 13: index --skip-dense never constructs DenseIndex
# ----------------------------------------------------------------------------
def test_index_skip_dense_never_constructs_dense_index(
    tmp_path, monkeypatch, mock_conversation, mock_chunk
):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    monkeypatch.setattr(
        "chatmem.ingest.normalize.load_all_exports",
        lambda p: [mock_conversation],
    )
    mock_store = MagicMock()
    mock_store.save_chunks.return_value = 1
    monkeypatch.setattr("chatmem.index.store.SQLiteMetadataStore", lambda p: mock_store)

    monkeypatch.setattr("chatmem.chunking.qa_pair.qa_chunks", lambda c: [mock_chunk])

    mock_dense = MagicMock()
    monkeypatch.setattr("chatmem.index.dense.DenseIndex", mock_dense)

    result = runner.invoke(
        app,
        ["index", "--input", str(export_dir), "--strategy", "qa", "--skip-dense"],
    )
    assert result.exit_code == 0
    assert not mock_dense.called


# ----------------------------------------------------------------------------
# Test 14: Qdrant connection error prints one-line message and exits 1
# ----------------------------------------------------------------------------
def test_index_qdrant_connection_error_exits_1_no_traceback(
    tmp_path, monkeypatch, mock_conversation, mock_chunk
):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    monkeypatch.setattr(
        "chatmem.ingest.normalize.load_all_exports",
        lambda p: [mock_conversation],
    )
    mock_store = MagicMock()
    mock_store.save_chunks.return_value = 1
    monkeypatch.setattr("chatmem.index.store.SQLiteMetadataStore", lambda p: mock_store)

    monkeypatch.setattr("chatmem.chunking.qa_pair.qa_chunks", lambda c: [mock_chunk])

    class FailingDenseIndex:
        def __init__(self, *args, **kwargs):
            pass

        def index_chunks(self, chunks):
            raise ConnectionError("Connection refused to Qdrant")

    monkeypatch.setattr("chatmem.index.dense.DenseIndex", FailingDenseIndex)

    result = runner.invoke(
        app,
        ["index", "--input", str(export_dir), "--strategy", "qa"],
    )
    assert result.exit_code == 1
    assert "Qdrant is not reachable at" in result.output
    assert "--skip-dense" in result.output
    assert "Traceback" not in result.output


# ----------------------------------------------------------------------------
# Test 15: One strategy chunker raising does not stop others; exits 1
# ----------------------------------------------------------------------------
def test_index_strategy_failure_continues_others_and_exits_1(
    tmp_path, monkeypatch, mock_conversation, mock_chunk
):
    export_dir = tmp_path / "exports"
    export_dir.mkdir()

    monkeypatch.setattr(
        "chatmem.ingest.normalize.load_all_exports",
        lambda p: [mock_conversation],
    )
    mock_store = MagicMock()
    mock_store.save_chunks.return_value = 1
    monkeypatch.setattr("chatmem.index.store.SQLiteMetadataStore", lambda p: mock_store)

    monkeypatch.setattr("chatmem.chunking.per_message.chunk_per_message", lambda c: [mock_chunk])
    # qa_chunks explodes
    def failing_qa(c):
        raise RuntimeError("qa exploded")

    monkeypatch.setattr("chatmem.chunking.qa_pair.qa_chunks", failing_qa)
    monkeypatch.setattr("chatmem.chunking.sliding_window.window_chunks", lambda c: [mock_chunk])

    result = runner.invoke(
        app,
        ["index", "--input", str(export_dir), "--strategy", "all", "--skip-dense"],
    )
    assert result.exit_code == 1
    assert "'qa': 'failed'" in result.output
    assert "'pm': 1" in result.output
    assert "'ws': 1" in result.output


# ----------------------------------------------------------------------------
# Test 16: search with missing database prints message and exits 1 without build_retriever
# ----------------------------------------------------------------------------
def test_search_missing_db_exits_1(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("name: test\n", encoding="utf-8")
    missing_db = tmp_path / "nonexistent.db"

    mock_build = MagicMock()
    monkeypatch.setattr("chatmem.retrieval.pipeline.build_retriever", mock_build)

    result = runner.invoke(
        app,
        ["search", "hello", "--config", str(cfg_file), "--db", str(missing_db)],
    )
    assert result.exit_code == 1
    assert "No index found" in result.output
    assert not mock_build.called


# ----------------------------------------------------------------------------
# Test 17: build_retriever default db path equals store.DEFAULT_DB_PATH
# ----------------------------------------------------------------------------
def test_build_retriever_default_db_path(monkeypatch):
    from chatmem.index.store import DEFAULT_DB_PATH
    from chatmem.retrieval.pipeline import PipelineConfig, build_retriever

    captured_db_path = []

    def fake_connect(db_path):
        captured_db_path.append(db_path)
        return MagicMock()

    monkeypatch.setattr("chatmem.retrieval.pipeline.connect", fake_connect)
    monkeypatch.setattr("chatmem.retrieval.pipeline.get_qdrant_client", lambda: MagicMock())

    cfg = PipelineConfig(name="test")
    retriever = build_retriever(cfg)
    assert len(captured_db_path) == 1
    assert captured_db_path[0] == DEFAULT_DB_PATH

