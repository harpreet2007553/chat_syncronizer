import csv
import json
from pathlib import Path

import pytest

from chatmem.eval.run_eval import (
    ConfigResult,
    EvalQuery,
    evaluate_config,
    gate_report,
    load_queries,
    run_sweep,
)
from chatmem.retrieval.pipeline import PipelineConfig, PipelineResult, RetrievedChunk


class ScriptedRetriever:
    """Fake retriever returning pre-scripted PipelineResults for testing."""

    active_instances = 0
    max_concurrent = 0

    def __init__(self, script_map=None, default_chunks=None, raise_on_query=None):
        self.script_map = script_map or {}
        self.default_chunks = default_chunks or []
        self.raise_on_query = raise_on_query or set()
        self.closed = False
        ScriptedRetriever.active_instances += 1
        ScriptedRetriever.max_concurrent = max(
            ScriptedRetriever.max_concurrent, ScriptedRetriever.active_instances
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if not self.closed:
            self.closed = True
            ScriptedRetriever.active_instances -= 1

    def search(self, query: str) -> PipelineResult:
        if query in self.raise_on_query:
            raise RuntimeError(f"Simulated search crash on '{query}'")

        chunks = self.script_map.get(query, self.default_chunks)
        top_score = chunks[0].score if chunks else None
        return PipelineResult(
            chunks=chunks,
            injected=True,
            reason="ok",
            top_score=top_score,
            score_scale="raw",
            timings_ms={"total": 10.0, "search": 5.0},
        )


def make_chunk(cid: str, score: float = 0.9, ranks: dict | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chk_{cid}",
        score=score,
        text=f"Content for {cid}",
        conversation_id=cid,
        platform="chatgpt",
        title=f"Title {cid}",
        time="2026-11-09T10:00:00Z",
        message_ids=["m1"],
        ranks=ranks or {"bm25": 1, "dense": 1},
    )


# ----------------------------------------------------------------------------
# Test 9: load_queries validations
# ----------------------------------------------------------------------------
def test_load_queries_validations(tmp_path):
    """9. load_queries: bad answerable (empty relevant), bad no_answer (non-empty relevant),

    and duplicate ids each raise with the query id in the message.
    """
    # Bad answerable: relevant is empty
    bad_ans_file = tmp_path / "bad_ans.jsonl"
    bad_ans_file.write_text(
        '{"qid": "q_bad_ans", "query": "how to code", "relevant": [], "type": "paraphrase"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_queries(bad_ans_file)
    assert "q_bad_ans" in str(exc.value)

    # Bad no_answer: relevant is non-empty
    bad_no_ans_file = tmp_path / "bad_no_ans.jsonl"
    bad_no_ans_file.write_text(
        '{"qid": "q_bad_no_ans", "query": "random question", "relevant": ["c1"], "type": "no_answer"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_queries(bad_no_ans_file)
    assert "q_bad_no_ans" in str(exc.value)

    # Duplicate query id
    dup_file = tmp_path / "dup.jsonl"
    dup_file.write_text(
        '{"qid": "q_dup", "query": "q1", "relevant": ["c1"], "type": "keyword"}\n'
        '{"qid": "q_dup", "query": "q2", "relevant": ["c2"], "type": "keyword"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as exc:
        load_queries(dup_file)
    assert "q_dup" in str(exc.value)


# ----------------------------------------------------------------------------
# Test 10: evaluate_config computes accurate metrics; excludes no_answer
# ----------------------------------------------------------------------------
def test_evaluate_config_metrics_exclude_no_answer(tmp_path):
    """10. evaluate_config with a fake retriever returning scripted rankings reproduces

    hand-computed Recall@k/MRR; no_answer queries excluded from those means.
    """
    queries = [
        EvalQuery(id="q1", text="q1 text", relevant={"c1", "c2"}, answerable=True),
        EvalQuery(id="q2", text="q2 text", relevant={"c3"}, answerable=True),
        EvalQuery(id="q3", text="q3 no answer", relevant=set(), answerable=False),
    ]

    # q1 ranking: [c9, c1, c8, c2] -> Recall@2 = 1/2 = 0.5, Recall@4 = 2/2 = 1.0, MRR = 1/2 = 0.5
    # q2 ranking: [c8, c7, c3] -> Recall@2 = 0, Recall@4 = 1/1 = 1.0, MRR = 1/3 = 0.3333...
    # q3 ranking: [c5, c6] -> no_answer (excluded from Recall/MRR)
    script_map = {
        "q1 text": [
            make_chunk("c9", 0.9),
            make_chunk("c1", 0.8),
            make_chunk("c8", 0.7),
            make_chunk("c2", 0.6),
        ],
        "q2 text": [
            make_chunk("c8", 0.9),
            make_chunk("c7", 0.8),
            make_chunk("c3", 0.7),
        ],
        "q3 no answer": [
            make_chunk("c5", 0.5),
            make_chunk("c6", 0.4),
        ],
    }

    cfg = PipelineConfig(
        name="test_cfg",
        use_bm25=True,
        use_dense=True,
        use_rerank=False,
        final_top_k=5,
    )

    def factory(c):
        return ScriptedRetriever(script_map=script_map)

    res = evaluate_config(
        config_path=cfg,
        queries=queries,
        ks=(2, 4),
        retriever_factory=factory,
    )

    # Expected:
    # Recall@2: mean(0.5, 0.0) = 0.25
    # Recall@4: mean(1.0, 1.0) = 1.0
    # MRR: mean(0.5, 1/3) = (0.5 + 0.333333) / 2 = 0.416666...
    assert res.metrics["recall@2"] == pytest.approx(0.25)
    assert res.metrics["recall@4"] == pytest.approx(1.0)
    assert res.metrics["mrr"] == pytest.approx((0.5 + 1.0 / 3.0) / 2.0)


# ----------------------------------------------------------------------------
# Test 11: Source coverage warning triggers
# ----------------------------------------------------------------------------
def test_source_coverage_warning_triggers(tmp_path):
    """11. Source coverage warning triggers when use_dense is true and no chunk has a dense rank."""
    queries = [
        EvalQuery(id="q1", text="q1", relevant={"c1"}, answerable=True),
    ]

    # Only bm25 in ranks, dense is missing
    bm25_only_chunks = [make_chunk("c1", ranks={"bm25": 1})]

    cfg = PipelineConfig(
        name="hybrid_test",
        use_bm25=True,
        use_dense=True,
        use_rerank=False,
    )

    def factory(c):
        return ScriptedRetriever(default_chunks=bm25_only_chunks)

    res = evaluate_config(
        config_path=cfg,
        queries=queries,
        retriever_factory=factory,
    )

    assert any("dense enabled but contributed 0 results" in w for w in res.warnings)


# ----------------------------------------------------------------------------
# Test 12: Search crash recorded as error and run continues
# ----------------------------------------------------------------------------
def test_query_search_crash_continues_run():
    """12. A query whose search() raises is counted as an error and the run continues."""
    queries = [
        EvalQuery(id="q1", text="crash query", relevant={"c1"}, answerable=True),
        EvalQuery(id="q2", text="ok query", relevant={"c2"}, answerable=True),
    ]

    cfg = PipelineConfig(name="crash_test")

    def factory(c):
        return ScriptedRetriever(
            default_chunks=[make_chunk("c2")],
            raise_on_query={"crash query"},
        )

    res = evaluate_config(config_path=cfg, queries=queries, retriever_factory=factory)
    assert len(res.errors) == 1
    assert res.errors[0]["query_id"] == "q1"
    assert len(res.query_results) == 2
    assert res.query_results[0]["error"] is not None
    assert res.query_results[1]["error"] is None


# ----------------------------------------------------------------------------
# Test 13: Retrievers closed after each config and never two open at once
# ----------------------------------------------------------------------------
def test_retrievers_closed_and_never_concurrent(tmp_path):
    """13. retrievers are closed after each config and never two open at once."""
    ScriptedRetriever.active_instances = 0
    ScriptedRetriever.max_concurrent = 0

    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text(
        '{"qid": "q1", "query": "hello", "relevant": ["c1"], "type": "keyword"}\n',
        encoding="utf-8",
    )

    cfg1 = PipelineConfig(name="cfg1")
    cfg2 = PipelineConfig(name="cfg2")

    def factory(c):
        return ScriptedRetriever(default_chunks=[make_chunk("c1")])

    run_sweep(
        config_paths=[cfg1, cfg2],
        queries_path=queries_file,
        out_dir=tmp_path / "out",
        retriever_factory=factory,
    )

    assert ScriptedRetriever.max_concurrent <= 1
    assert ScriptedRetriever.active_instances == 0


# ----------------------------------------------------------------------------
# Test 14: All four configs/eval/*.yaml load via PipelineConfig.from_yaml
# ----------------------------------------------------------------------------
def test_eval_configs_load_validly():
    """14. All four configs/eval/*.yaml load via PipelineConfig.from_yaml."""
    eval_dir = Path("configs/eval")
    assert eval_dir.exists()

    expected_files = ["bm25_only.yaml", "dense_only.yaml", "hybrid.yaml", "hybrid_rerank.yaml"]
    for fname in expected_files:
        fpath = eval_dir / fname
        assert fpath.exists(), f"Missing config file: {fpath}"
        cfg = PipelineConfig.from_yaml(fpath)
        assert cfg.strategy == "qa"
        assert cfg.final_top_k == 10
        assert cfg.search_top_k == 30
        assert cfg.gate_threshold is None


# ----------------------------------------------------------------------------
# Test 15: gate_report header and warning check
# ----------------------------------------------------------------------------
def test_gate_report_header_and_small_sample_warning():
    """15. gate_report header contains score_scale, rerank_model, rerank_normalize;

    the small-sample warning appears with <5 no_answer queries.
    """
    cfg = PipelineConfig(
        name="rerank_test",
        use_rerank=True,
        rerank_model="BAAI/bge-reranker-base",
        rerank_normalize=False,
    )
    res = ConfigResult(
        config_name="rerank_test",
        config=cfg,
        score_scale="raw",
        rerank_model="BAAI/bge-reranker-base",
        metrics={},
        latency_mean_ms=10.0,
        latency_p95_ms=12.0,
        warnings=[],
        errors=[],
        query_results=[
            {
                "query_id": "q1",
                "answerable": True,
                "relevant": ["c1"],
                "ranked_conv_ids": ["c1"],
                "top_score": 0.8,
            },
            {
                "query_id": "q2",
                "answerable": False,
                "relevant": [],
                "ranked_conv_ids": ["c2"],
                "top_score": 0.2,
            },
        ],
    )

    report = gate_report(res, thresholds=[0.5])
    assert "Score scale" in report
    assert "raw" in report
    assert "BAAI/bge-reranker-base" in report
    assert "Rerank normalize" in report
    assert "only 1 no_answer queries; threshold is not statistically meaningful" in report


# ----------------------------------------------------------------------------
# Test 16: run_sweep writes all expected output files
# ----------------------------------------------------------------------------
def test_run_sweep_writes_all_files(tmp_path):
    """16. run_sweep writes results.json, summary.md, per_query.csv and, for rerank, scores_<name>.csv."""
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text(
        '{"qid": "q1", "query": "hello", "relevant": ["c1"], "type": "keyword"}\n'
        '{"qid": "q2", "query": "bye", "relevant": [], "type": "no_answer"}\n',
        encoding="utf-8",
    )

    cfg_bm25 = PipelineConfig(name="bm25_test", use_rerank=False)
    cfg_rerank = PipelineConfig(name="rerank_test", use_rerank=True, rerank_candidates=10, final_top_k=5)

    out_dir = tmp_path / "sweep_out"

    def factory(c):
        return ScriptedRetriever(default_chunks=[make_chunk("c1")])

    run_sweep(
        config_paths=[cfg_bm25, cfg_rerank],
        queries_path=queries_file,
        out_dir=out_dir,
        retriever_factory=factory,
    )

    assert (out_dir / "results.json").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "per_query.csv").exists()
    assert (out_dir / "scores_rerank_test.csv").exists()
    assert not (out_dir / "scores_bm25_test.csv").exists()
    assert (out_dir / "gate_report.md").exists()

    # Validate results.json parse
    with (out_dir / "results.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 2

    # Validate per_query.csv rows
    with (out_dir / "per_query.csv").open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert reader[0] == ["query_id", "config", "answerable", "top_score", "rank_first_relevant", "error"]
        assert len(reader) == 5  # 1 header + 2 queries * 2 configs

