from __future__ import annotations

import csv
import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

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
from chatmem.retrieval.pipeline import PipelineConfig, build_retriever

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Query & Result Data Structures
# ============================================================================

@dataclass
class EvalQuery:
    """A labeled benchmark evaluation query."""

    id: str
    text: str
    relevant: set[str]
    answerable: bool
    split: str | None = None


@dataclass
class ConfigResult:
    """Aggregated evaluation results for a single pipeline configuration."""

    config_name: str
    config: PipelineConfig
    score_scale: str
    rerank_model: str
    metrics: dict[str, float]
    latency_mean_ms: float
    latency_p95_ms: float
    warnings: list[str]
    errors: list[dict[str, Any]]
    query_results: list[dict[str, Any]] = field(default_factory=list)


def percentile_95(values: Sequence[float]) -> float:
    """Compute 95th percentile using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * 0.95
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return float(d0 + d1)


# ============================================================================
# 2. Query Loader
# ============================================================================

def load_queries(path: str | Path, split: str | None = None) -> list[EvalQuery]:
    """Load and validate benchmark evaluation queries from a JSON Lines file.

    Raises:
        ValueError: If query data has duplicate IDs, answerable queries without relevant IDs,
                    no_answer queries with relevant IDs, or an invalid split filter.
    """
    valid_splits = {None, "dev", "test", "all"}
    if split is not None and split.lower() not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Expected one of: 'dev', 'test', 'all', or None.")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Queries file not found: {file_path}")

    queries: list[EvalQuery] = []
    seen_ids: set[str] = set()

    with file_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {file_path}: {e}")

            qid = str(data.get("qid") or data.get("id") or "").strip()
            if not qid:
                raise ValueError(f"Missing query ID on line {line_num} of {file_path}")

            if qid in seen_ids:
                raise ValueError(f"Duplicate query ID found: '{qid}'")
            seen_ids.add(qid)

            query_text = str(data.get("query") or data.get("text") or "").strip()
            relevant_raw = data.get("relevant", [])
            relevant = {str(r).strip() for r in relevant_raw if str(r).strip()}
            qtype = str(data.get("type") or "").strip().lower()
            split_raw = data.get("split")
            query_split = str(split_raw).strip().lower() if split_raw is not None else None

            if qtype == "no_answer" or qtype.startswith("no_answer"):
                if relevant:
                    raise ValueError(
                        f"Query '{qid}' is marked {qtype} but has non-empty relevant set: {relevant}"
                    )
                answerable = False
            else:
                if not relevant:
                    raise ValueError(
                        f"Query '{qid}' is answerable but has an empty relevant set."
                    )
                answerable = True

            queries.append(
                EvalQuery(
                    id=qid,
                    text=query_text,
                    relevant=relevant,
                    answerable=answerable,
                    split=query_split,
                )
            )

    has_splits = any(q.split is not None for q in queries)
    if (split is None or split.lower() == "all") and has_splits:
        print("Choose thresholds on --split dev, report on --split test.")

    if split is not None and split.lower() in ("dev", "test"):
        target_split = split.lower()
        queries = [q for q in queries if q.split == target_split]

    return queries


# ============================================================================
# 3. Pipeline Evaluator
# ============================================================================

def evaluate_config(
    config_path: str | Path,
    queries: Sequence[EvalQuery],
    ks: Sequence[int] = (1, 3, 5, 10),
    retriever_factory: Callable[..., Any] = build_retriever,
) -> ConfigResult:
    """Run retrieval pipeline over evaluation queries and compute retrieval metrics."""
    cfg = (
        PipelineConfig.from_yaml(config_path)
        if isinstance(config_path, (str, Path))
        else config_path
    )

    query_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    latencies: list[float] = []

    seen_dense_ranks = False
    seen_bm25_ranks = False

    retriever = retriever_factory(cfg)
    try:
        for q in queries:
            try:
                result = retriever.search(q.text)
                ranked_convs = dedupe_conversations(result.chunks)

                # Track sources contributing to hits
                for c in result.chunks:
                    ranks_dict = getattr(c, "ranks", {}) or {}
                    if "dense" in ranks_dict:
                        seen_dense_ranks = True
                    if "bm25" in ranks_dict:
                        seen_bm25_ranks = True

                first_rank: int | None = None
                if q.answerable:
                    for idx, cid in enumerate(ranked_convs, start=1):
                        if cid in q.relevant:
                            first_rank = idx
                            break

                total_time = result.timings_ms.get("total", 0.0)
                latencies.append(total_time)

                query_results.append(
                    {
                        "query_id": q.id,
                        "query_text": q.text,
                        "answerable": q.answerable,
                        "relevant": sorted(list(q.relevant)),
                        "ranked_conv_ids": ranked_convs,
                        "top_score": result.top_score,
                        "score_scale": result.score_scale,
                        "injected": result.injected,
                        "timings_ms": result.timings_ms,
                        "rank_first_relevant": first_rank,
                        "error": None,
                    }
                )
            except Exception as e:
                err_msg = str(e)
                logger.error("Query %s raised an exception during evaluation: %s", q.id, err_msg)
                errors.append({"query_id": q.id, "error": err_msg})
                query_results.append(
                    {
                        "query_id": q.id,
                        "query_text": q.text,
                        "answerable": q.answerable,
                        "relevant": sorted(list(q.relevant)),
                        "ranked_conv_ids": [],
                        "top_score": None,
                        "score_scale": cfg.score_scale,
                        "injected": False,
                        "timings_ms": {},
                        "rank_first_relevant": None,
                        "error": err_msg,
                    }
                )
    finally:
        if hasattr(retriever, "close"):
            retriever.close()

    # Source coverage check
    warnings: list[str] = []
    if cfg.use_dense and not seen_dense_ranks:
        warnings.append("dense enabled but contributed 0 results (collection empty or search failing)")
    if cfg.use_bm25 and not seen_bm25_ranks:
        warnings.append("bm25 enabled but contributed 0 results (collection empty or search failing)")

    # Compute metrics across answerable queries only
    ans_queries = [q for q in queries if q.answerable]
    ans_results_map = {r["query_id"]: r for r in query_results if r["answerable"]}

    computed_metrics: dict[str, float] = {}
    for k in ks:
        recalls = []
        hits = []
        ndcgs = []
        for q in ans_queries:
            r = ans_results_map.get(q.id)
            ranked = r["ranked_conv_ids"] if r else []
            recalls.append(recall_at_k(ranked, q.relevant, k))
            hits.append(hit_at_k(ranked, q.relevant, k))
            ndcgs.append(ndcg_at_k(ranked, q.relevant, k))

        computed_metrics[f"recall@{k}"] = mean(recalls)
        computed_metrics[f"hit@{k}"] = mean(hits)
        computed_metrics[f"ndcg@{k}"] = mean(ndcgs)

    mrrs = []
    for q in ans_queries:
        r = ans_results_map.get(q.id)
        ranked = r["ranked_conv_ids"] if r else []
        mrrs.append(reciprocal_rank(ranked, q.relevant))
    computed_metrics["mrr"] = mean(mrrs)

    return ConfigResult(
        config_name=cfg.name,
        config=cfg,
        score_scale=cfg.score_scale,
        rerank_model=cfg.rerank_model,
        metrics=computed_metrics,
        latency_mean_ms=mean(latencies),
        latency_p95_ms=percentile_95(latencies),
        warnings=warnings,
        errors=errors,
        query_results=query_results,
    )


# ============================================================================
# 4. Gate Report Generator
# ============================================================================

def gate_report(
    result: ConfigResult,
    thresholds: list[float] | None = None,
    split: str | None = None,
) -> str:
    """Generate threshold sweep evaluation table for cross-encoder rerank configs."""
    if not result.config.use_rerank:
        return ""

    rows: list[dict[str, Any]] = []
    for q in result.query_results:
        relevant_set = set(q.get("relevant", []))
        top_k = q.get("ranked_conv_ids", [])[: result.config.final_top_k]
        rel_in_topk = bool(relevant_set.intersection(top_k))
        rows.append(
            {
                "answerable": q["answerable"],
                "top_score": q["top_score"],
                "relevant_in_topk": rel_in_topk,
            }
        )

    if thresholds is None:
        thresholds = suggest_thresholds([r["top_score"] for r in rows], n=25)

    sweep_results = gate_sweep(rows, thresholds)
    no_ans_count = sum(1 for r in rows if not r["answerable"])

    title_suffix = f" (Split: {split})" if split else ""
    lines: list[str] = [
        f"## Gate Report: {result.config_name}{title_suffix}",
        f"- **Score scale**: `{result.score_scale}`",
        f"- **Rerank model**: `{result.config.rerank_model}`",
        f"- **Rerank normalize**: `{result.config.rerank_normalize}`",
    ]
    if split:
        lines.append(f"- **Split**: `{split}`")

    if no_ans_count < 5:
        lines.append(
            f"\n> [!WARNING]\n> only {no_ans_count} no_answer queries; threshold is not statistically meaningful.\n"
        )

    lines.append(
        "\n| Threshold | Inject Rate (Ans) | Correct Abstain (NoAns) | False Injection (NoAns) | Useful Injection | Missed Good |"
    )
    lines.append(
        "| ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    for item in sweep_results:
        t = item["threshold"]
        inj_ans = f"{item['inject_rate_answerable']:.1%}" if item["inject_rate_answerable"] is not None else "N/A"
        c_abs = f"{item['correct_abstain_rate_no_answer']:.1%}" if item["correct_abstain_rate_no_answer"] is not None else "N/A"
        f_inj = f"{item['false_injection_rate']:.1%}" if item["false_injection_rate"] is not None else "N/A"
        u_inj = f"{item['useful_injection_rate']:.1%}" if item["useful_injection_rate"] is not None else "N/A"
        m_good = f"{item['missed_good_rate']:.1%}" if item["missed_good_rate"] is not None else "N/A"

        lines.append(
            f"| {t:.4f} | {inj_ans} | {c_abs} | {f_inj} | {u_inj} | {m_good} |"
        )

    return "\n".join(lines)


# ============================================================================
# 5. Benchmark Sweep Runner
# ============================================================================

def run_sweep(
    config_paths: Sequence[str | Path],
    queries_path: str | Path,
    ks: Sequence[int] = (1, 3, 5, 10),
    out_dir: str | Path | None = None,
    split: str | None = "all",
    retriever_factory: Callable[..., Any] = build_retriever,
) -> list[ConfigResult]:
    """Execute sequential evaluation across multiple configurations and persist artifacts."""
    queries = load_queries(queries_path, split=split)

    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("eval_results") / timestamp
    else:
        output_dir = Path(out_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ConfigResult] = []
    for cp in config_paths:
        res = evaluate_config(
            config_path=cp,
            queries=queries,
            ks=ks,
            retriever_factory=retriever_factory,
        )
        results.append(res)

    # 1. results.json
    results_data = []
    for res in results:
        results_data.append(
            {
                "config_name": res.config_name,
                "score_scale": res.score_scale,
                "rerank_model": res.rerank_model,
                "metrics": res.metrics,
                "latency_mean_ms": res.latency_mean_ms,
                "latency_p95_ms": res.latency_p95_ms,
                "warnings": res.warnings,
                "errors": res.errors,
                "query_results": res.query_results,
            }
        )

    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # 2. summary.md
    split_str = str(split) if split else "all"
    summary_lines = [
        f"# Retrieval Evaluation Summary (Split: {split_str})",
        "",
        f"- **Split**: `{split_str}`",
        "",
        "| Config | Score Scale | Rerank Model | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | Latency Mean (ms) | Latency P95 (ms) | Warnings |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for res in results:
        m = res.metrics
        r5 = f"{m.get('recall@5', 0.0):.4f}"
        r10 = f"{m.get('recall@10', 0.0):.4f}"
        mrr = f"{m.get('mrr', 0.0):.4f}"
        ndcg10 = f"{m.get('ndcg@10', 0.0):.4f}"
        h1 = f"{m.get('hit@1', 0.0):.4f}"
        l_mean = f"{res.latency_mean_ms:.2f}"
        l_p95 = f"{res.latency_p95_ms:.2f}"
        warn_str = "; ".join(res.warnings) if res.warnings else "none"

        summary_lines.append(
            f"| {res.config_name} | {res.score_scale} | {res.rerank_model} | {r5} | {r10} | {mrr} | {ndcg10} | {h1} | {l_mean} | {l_p95} | {warn_str} |"
        )

    summary_content = "\n".join(summary_lines)
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write(summary_content + "\n")

    # 3. per_query.csv
    with (output_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "config", "answerable", "top_score", "rank_first_relevant", "error"])
        for res in results:
            for qr in res.query_results:
                writer.writerow(
                    [
                        qr["query_id"],
                        res.config_name,
                        qr["answerable"],
                        qr["top_score"],
                        qr["rank_first_relevant"],
                        qr["error"],
                    ]
                )

    # 4. scores_<config>.csv & gate_report.md
    gate_reports: list[str] = []
    for res in results:
        if res.config.use_rerank:
            score_file = output_dir / f"scores_{res.config_name}.csv"
            with score_file.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["query_id", "answerable", "top_score"])
                for qr in res.query_results:
                    writer.writerow([qr["query_id"], qr["answerable"], qr["top_score"]])

            report = gate_report(res, split=split_str)
            if report:
                gate_reports.append(report)

    if gate_reports:
        with (output_dir / "gate_report.md").open("w", encoding="utf-8") as f:
            f.write("\n\n".join(gate_reports) + "\n")

    return results

