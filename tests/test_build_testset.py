"""Tests for chatmem dataset builder, validator, and eval split support."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chatmem.eval.build_testset import (
    assign_splits,
    build_dataset,
    validate_dataset,
)
from chatmem.eval.run_eval import EvalQuery, load_queries, run_sweep
from chatmem.ingest.normalize import load_all_exports
from chatmem.retrieval.pipeline import PipelineConfig


# ---------------------------------------------------------------------------
# Test 1 & 2: Build from tiny batch & relevant is union over facts
# ---------------------------------------------------------------------------

def test_build_tiny_batch_emits_exports_and_queries(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # 3 conversations: c01 (chatgpt), c02 (claude), c03 (chatgpt)
    batch = [
        {
            "id": "c01",
            "platform": "chatgpt",
            "title": "Conv 1",
            "created": 1673773200.0,
            "facts": ["python_env", "shared_fact"],
            "messages": [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi"}],
        },
        {
            "id": "c02",
            "platform": "claude",
            "title": "Conv 2",
            "created": "2023-01-16T10:00:00Z",
            "facts": ["shared_fact"],
            "messages": [{"role": "user", "text": "ping"}, {"role": "assistant", "text": "pong"}],
        },
        {
            "id": "c03",
            "platform": "chatgpt",
            "title": "Conv 3",
            "created": 1673773300.0,
            "facts": ["docker_setup"],
            "messages": [{"role": "user", "text": "docker"}, {"role": "assistant", "text": "qdrant"}],
        },
    ]
    (raw_dir / "batch_01.json").write_text(json.dumps(batch), encoding="utf-8")

    queries = [
        {"qid": "q01", "query": "how to set up python env", "type": "paraphrase", "needs": ["python_env"], "split": "dev"},
        {"qid": "q02", "query": "shared question", "type": "multi_chat", "needs": ["shared_fact"], "split": "dev"},
        {"qid": "q03", "query": "docker container", "type": "keyword", "needs": ["docker_setup"], "split": "test"},
        {"qid": "q04", "query": "offtopic query", "type": "no_answer_offtopic", "needs": [], "split": "test"},
    ]
    with (raw_dir / "queries_authored.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    num_c, num_q = build_dataset(raw_dir=raw_dir, out_dir=out_dir)
    assert num_c == 3
    assert num_q == 4

    cg_file = out_dir / "chatgpt" / "conversations.json"
    cl_file = out_dir / "claude" / "conversations.json"
    queries_out_file = out_dir / "queries.jsonl"
    facts_out_file = out_dir / "facts.json"

    assert cg_file.exists()
    assert cl_file.exists()
    assert queries_out_file.exists()
    assert facts_out_file.exists()

    cg_data = json.loads(cg_file.read_text(encoding="utf-8"))
    assert len(cg_data) == 2
    cl_data = json.loads(cl_file.read_text(encoding="utf-8"))
    assert len(cl_data) == 1

    out_queries = [json.loads(line) for line in queries_out_file.read_text(encoding="utf-8").splitlines() if line]
    q_map = {q["qid"]: q for q in out_queries}

    assert q_map["q01"]["relevant"] == ["c01"]
    assert q_map["q03"]["relevant"] == ["c03"]
    assert q_map["q04"]["relevant"] == []


def test_relevant_is_union_over_facts(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # c01 and c02 share 'shared_fact'
    batch = [
        {
            "id": "c01",
            "platform": "chatgpt",
            "title": "Conv 1",
            "created": 1673773200.0,
            "facts": ["shared_fact"],
            "messages": [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi"}],
        },
        {
            "id": "c02",
            "platform": "claude",
            "title": "Conv 2",
            "created": "2023-01-16T10:00:00Z",
            "facts": ["shared_fact"],
            "messages": [{"role": "user", "text": "ping"}, {"role": "assistant", "text": "pong"}],
        },
    ]
    (raw_dir / "batch_01.json").write_text(json.dumps(batch), encoding="utf-8")
    queries = [
        {"qid": "q_union", "query": "shared question", "type": "multi_chat", "needs": ["shared_fact"], "split": "dev"},
        {"qid": "q_union_test", "query": "shared question 2", "type": "multi_chat", "needs": ["shared_fact"], "split": "test"},
    ]
    with (raw_dir / "queries_authored.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    build_dataset(raw_dir=raw_dir, out_dir=out_dir)
    queries_out = [json.loads(line) for line in (out_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines() if line]
    q_map = {q["qid"]: q for q in queries_out}
    assert set(q_map["q_union"]["relevant"]) == {"c01", "c02"}


# ---------------------------------------------------------------------------
# Test 3: Validate flags errors
# ---------------------------------------------------------------------------

def test_validate_flags_integrity_errors(tmp_path: Path):
    out_dir = tmp_path / "invalid_dataset"
    out_dir.mkdir()
    cg_dir = out_dir / "chatgpt"
    cg_dir.mkdir()
    cl_dir = out_dir / "claude"
    cl_dir.mkdir()

    # 1. Duplicate conversation ID in ChatGPT
    cg_convs = [
        {"id": "dup_cid", "conversation_id": "dup_cid", "title": "A", "create_time": 0, "mapping": {"root": {"id": "root", "message": None, "parent": None, "children": []}}, "current_node": "root"},
        {"id": "dup_cid", "conversation_id": "dup_cid", "title": "B", "create_time": 0, "mapping": {"root": {"id": "root", "message": None, "parent": None, "children": []}}, "current_node": "root"},
    ]
    (cg_dir / "conversations.json").write_text(json.dumps(cg_convs), encoding="utf-8")
    (cl_dir / "conversations.json").write_text(json.dumps([]), encoding="utf-8")
    (out_dir / "facts.json").write_text(json.dumps({"dup_cid": ["fact1"]}), encoding="utf-8")

    queries = [
        # Answerable query with no relevant
        {"qid": "q_no_rel", "query": "where", "type": "paraphrase", "relevant": [], "split": "dev"},
        # No-answer query with non-empty relevant
        {"qid": "q_no_ans_with_rel", "query": "why", "type": "no_answer_offtopic", "relevant": ["dup_cid"], "split": "dev"},
        # Type missing from test split (paraphrase only in dev)
        {"qid": "q_keyword_test", "query": "token", "type": "keyword", "relevant": ["dup_cid"], "split": "test"},
    ]
    with (out_dir / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    stats, errors, warnings = validate_dataset(out_dir)
    assert len(errors) > 0
    err_text = " ".join(errors)
    assert "Duplicate conversation ID" in err_text
    assert "empty relevant list" in err_text
    assert "non-empty relevant set" in err_text
    assert "missing from test split" in err_text or "missing from dev split" in err_text


# ---------------------------------------------------------------------------
# Test 4: Round trip loads through load_all_exports
# ---------------------------------------------------------------------------

def test_round_trip_load_all_exports(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    batch = [
        {
            "id": "c101",
            "platform": "chatgpt",
            "title": "ChatGPT Conv",
            "created": 1673773200.0,
            "facts": ["f1"],
            "messages": [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi"}],
        },
        {
            "id": "c102",
            "platform": "claude",
            "title": "Claude Conv",
            "created": "2023-01-16T10:00:00Z",
            "facts": ["f2"],
            "messages": [{"role": "user", "text": "foo"}, {"role": "assistant", "text": "bar"}],
        },
    ]
    (raw_dir / "batch_01.json").write_text(json.dumps(batch), encoding="utf-8")
    (raw_dir / "queries_authored.jsonl").write_text(
        json.dumps({"qid": "q1", "query": "foo", "type": "keyword", "needs": ["f1"], "split": "dev"}) + "\n"
        + json.dumps({"qid": "q2", "query": "bar", "type": "keyword", "needs": ["f2"], "split": "test"}) + "\n",
        encoding="utf-8",
    )

    build_dataset(raw_dir=raw_dir, out_dir=out_dir)
    loaded = load_all_exports(out_dir)
    assert len(loaded) == 2
    loaded_map = {c.conversation_id: c for c in loaded}
    assert "c101" in loaded_map
    assert "c102" in loaded_map
    assert loaded_map["c101"].platform == "chatgpt"
    assert loaded_map["c102"].platform == "claude"


# ---------------------------------------------------------------------------
# Test 5: Paraphrase overlap warning triggers on deliberately copied query
# ---------------------------------------------------------------------------

def test_paraphrase_overlap_warning_triggers(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Copied message text
    copied_sentence = "why does groupby say KeyError even though the column is there"
    batch = [
        {
            "id": "c201",
            "platform": "chatgpt",
            "title": "Pandas test",
            "created": 1673773200.0,
            "facts": ["groupby_err"],
            "messages": [{"role": "user", "text": copied_sentence}, {"role": "assistant", "text": "check index"}],
        },
    ]
    (raw_dir / "batch_01.json").write_text(json.dumps(batch), encoding="utf-8")

    queries = [
        # Deliberately copied query
        {"qid": "q_copied_dev", "query": copied_sentence, "type": "paraphrase", "needs": ["groupby_err"], "split": "dev"},
        {"qid": "q_copied_test", "query": copied_sentence, "type": "paraphrase", "needs": ["groupby_err"], "split": "test"},
    ]
    with (raw_dir / "queries_authored.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    build_dataset(raw_dir=raw_dir, out_dir=out_dir)
    stats, errors, warnings = validate_dataset(out_dir)

    warn_text = " ".join(warnings)
    assert "high Jaccard overlap (>0.5)" in warn_text


# ---------------------------------------------------------------------------
# Test 6: Split assignment is deterministic for a fixed seed and stratified
# ---------------------------------------------------------------------------

def test_split_assignment_is_deterministic_and_stratified():
    queries = [
        {"qid": f"q_p_{i}", "query": "text", "type": "paraphrase", "needs": ["f"]} for i in range(10)
    ] + [
        {"qid": f"q_k_{i}", "query": "text", "type": "keyword", "needs": ["f"]} for i in range(10)
    ]

    split1 = assign_splits(queries, seed=1234)
    split2 = assign_splits(queries, seed=1234)
    assert split1 == split2

    # Check stratification
    for qtype in ("paraphrase", "keyword"):
        dev_count = sum(1 for q in split1 if q["type"] == qtype and q["split"] == "dev")
        test_count = sum(1 for q in split1 if q["type"] == qtype and q["split"] == "test")
        assert dev_count == 5
        assert test_count == 5


# ---------------------------------------------------------------------------
# Test 7: eval --split dev only evaluates dev queries (fake retriever)
# ---------------------------------------------------------------------------

def test_eval_split_dev_only_evaluates_dev_queries(tmp_path: Path):
    queries_file = tmp_path / "queries.jsonl"
    queries = [
        {"qid": "q_dev1", "query": "dev query 1", "relevant": ["c01"], "type": "keyword", "split": "dev"},
        {"qid": "q_dev2", "query": "dev query 2", "relevant": ["c02"], "type": "keyword", "split": "dev"},
        {"qid": "q_test1", "query": "test query 1", "relevant": ["c03"], "type": "keyword", "split": "test"},
    ]
    with queries_file.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    # Load with split="dev"
    dev_queries = load_queries(queries_file, split="dev")
    assert len(dev_queries) == 2
    assert {q.id for q in dev_queries} == {"q_dev1", "q_dev2"}

    # Mock retriever
    fake_retriever = MagicMock()
    mock_result = MagicMock()
    mock_result.injected = True
    mock_result.chunks = []
    mock_result.top_score = 0.8
    mock_result.score_scale = "raw"
    mock_result.timings_ms = {"total": 5.0}
    mock_result.ranked_conv_ids = ["c01"]
    fake_retriever.search.return_value = mock_result

    # Minimal pipeline config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("strategy: pm\nuse_bm25: true\nuse_dense: false\n", encoding="utf-8")

    out_dir = tmp_path / "eval_out"
    results = run_sweep(
        config_paths=[config_file],
        queries_path=queries_file,
        out_dir=out_dir,
        split="dev",
        retriever_factory=lambda cfg, db_path=None: fake_retriever,
    )
    assert len(results) == 1
    # Evaluated exactly the 2 dev queries
    assert len(results[0].query_results) == 2
    assert {qr["query_id"] for qr in results[0].query_results} == {"q_dev1", "q_dev2"}


# ---------------------------------------------------------------------------
# Test 8: Old queries file without split key still loads
# ---------------------------------------------------------------------------

def test_old_queries_file_without_split_key_loads():
    old_queries_path = Path("data/eval/queries.sample.jsonl")
    assert old_queries_path.exists()

    queries = load_queries(old_queries_path)
    assert len(queries) == 11
    # Check that split defaults to None and qids match
    assert queries[0].id == "q01"
    assert queries[0].split is None
    assert queries[-1].id == "q11"
    assert queries[-1].split is None
    assert not queries[-1].answerable

