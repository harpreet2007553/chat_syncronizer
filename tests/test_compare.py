"""Unit tests for eval/compare.py and chatmem.eval.compare."""
from __future__ import annotations

import json
from pathlib import Path

from chatmem.eval.compare import analyze_run, generate_markdown_report, load_query_metadata


def test_load_query_metadata(tmp_path: Path) -> None:
    queries_file = tmp_path / "queries.jsonl"
    data = [
        {"qid": "q1", "type": "keyword", "split": "dev", "relevant": ["conv1"]},
        {"qid": "q2", "type": "paraphrase", "split": "test", "relevant": ["conv2"]},
    ]
    with queries_file.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    meta = load_query_metadata(queries_file)
    assert len(meta) == 2
    assert meta["q1"]["type"] == "keyword"
    assert meta["q1"]["split"] == "dev"
    assert meta["q1"]["relevant"] == ["conv1"]
    assert meta["q2"]["split"] == "test"


def test_load_query_metadata_nonexistent(tmp_path: Path) -> None:
    meta = load_query_metadata(tmp_path / "nonexistent.jsonl")
    assert meta == {}


def test_analyze_run_and_generate_markdown_report(tmp_path: Path) -> None:
    # Setup mock query metadata
    query_meta = {
        "q1": {"type": "keyword", "split": "dev", "relevant": ["c1"]},
        "q2": {"type": "no_answer", "split": "dev", "relevant": []},
    }

    # Setup mock run directory
    run_dir = tmp_path / "run_hybrid"
    run_dir.mkdir(parents=True)
    results_json = run_dir / "results.json"

    results_data = [
        {
            "config_name": "hybrid_test",
            "score_scale": "rrf",
            "metrics": {"recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0, "ndcg@10": 1.0},
            "latency_mean_ms": 25.0,
            "latency_p95_ms": 30.0,
            "query_results": [
                {
                    "query_id": "q1",
                    "query": "find c1",
                    "answerable": True,
                    "relevant": ["c1"],
                    "ranked_conv_ids": ["c1", "c2"],
                    "injected": True,
                    "timings_ms": {"total": 20.0},
                },
                {
                    "query_id": "q2",
                    "query": "unanswerable",
                    "answerable": False,
                    "relevant": [],
                    "ranked_conv_ids": [],
                    "injected": False,
                    "timings_ms": {"total": 10.0},
                },
            ],
        }
    ]
    results_json.write_text(json.dumps(results_data), encoding="utf-8")

    # Analyze run
    summaries = analyze_run(run_dir, query_meta)
    assert len(summaries) == 1
    s = summaries[0]
    assert s["config_name"] == "hybrid_test"
    assert s["split"] == "dev"
    assert s["total_n"] == 2
    assert s["ans_n"] == 1
    assert s["no_ans_n"] == 1
    assert s["recall@1"] == 1.0
    assert s["gate_precision"] == 1.0

    # Test report generation
    report = generate_markdown_report(results_base=tmp_path, query_meta=query_meta)
    assert "# Retrieval Evaluation Comparison" in report
    assert "hybrid_test" in report
    assert "| `hybrid_test` | **dev** |" in report
