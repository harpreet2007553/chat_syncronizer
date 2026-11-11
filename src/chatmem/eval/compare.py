"""Merge evaluation results into comprehensive markdown comparison tables."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from chatmem.eval.metrics import hit_at_k, recall_at_k, reciprocal_rank


def load_query_metadata(queries_path: str | Path = "data/eval_v2/queries.jsonl") -> dict[str, dict]:
    meta = {}
    path = Path(queries_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    meta[item["qid"]] = {
                        "type": item.get("type", "unknown"),
                        "split": item.get("split", "unknown"),
                        "relevant": item.get("relevant", []),
                    }
    return meta


def analyze_run(run_dir: Path, query_meta: dict[str, dict]) -> list[dict[str, Any]]:
    results_file = run_dir / "results.json"
    if not results_file.exists():
        return []

    data = json.loads(results_file.read_text(encoding="utf-8"))
    summaries = []

    for cfg_res in data:
        cfg_name = cfg_res["config_name"]
        _score_scale = cfg_res.get("score_scale", "raw")
        q_results = cfg_res.get("query_results", [])
        total_n = len(q_results)
        if total_n == 0:
            continue

        # Determine split from queries
        splits = {query_meta[qr["query_id"]]["split"] for qr in q_results if qr["query_id"] in query_meta}
        split_str = "/".join(sorted(splits)) if splits else "unknown"

        ans_results = [qr for qr in q_results if qr.get("answerable")]
        no_ans_results = [qr for qr in q_results if not qr.get("answerable")]
        n_ans = len(ans_results)
        n_no_ans = len(no_ans_results)

        # Retrieval metrics on answerable queries
        r1_list, r5_list, r10_list, mrr_list = [], [], [], []
        latencies = []

        for qr in q_results:
            lat = qr.get("timings_ms", {}).get("total", 0.0)
            if lat:
                latencies.append(lat)

        for qr in ans_results:
            ranked = qr.get("ranked_conv_ids", [])
            rel = set(qr.get("relevant", []))
            r1_list.append(hit_at_k(ranked, rel, 1))
            r5_list.append(recall_at_k(ranked, rel, 5))
            r10_list.append(recall_at_k(ranked, rel, 10))
            mrr_list.append(reciprocal_rank(ranked, rel))

        m = cfg_res.get("metrics", {})
        recall_1 = sum(r1_list) / n_ans if n_ans else 0.0
        recall_5 = m.get("recall@5", sum(r5_list) / n_ans if n_ans else 0.0)
        recall_10 = m.get("recall@10", sum(r10_list) / n_ans if n_ans else 0.0)
        mrr_val = m.get("mrr", sum(mrr_list) / n_ans if n_ans else 0.0)
        ndcg_10 = m.get("ndcg@10", 0.0)

        # Gate metrics
        injected_ans = sum(1 for qr in ans_results if qr.get("injected", True))
        injected_no_ans = sum(1 for qr in no_ans_results if qr.get("injected", True))
        total_injected = injected_ans + injected_no_ans
        gate_precision = (injected_ans / total_injected) if total_injected > 0 else 1.0

        # Latencies
        sorted_lat = sorted(latencies) if latencies else [cfg_res.get("latency_mean_ms", 0.0)]
        p50_lat = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0
        p95_lat = cfg_res.get("latency_p95_ms", 0.0)

        # Per-category metrics (counts and percentages)
        by_cat = defaultdict(lambda: {"total": 0, "r5": [], "r10": [], "hit1": [], "mrr": []})
        for qr in ans_results:
            qid = qr["query_id"]
            cat = query_meta.get(qid, {}).get("type", "unknown")
            ranked = qr.get("ranked_conv_ids", [])
            rel = set(qr.get("relevant", []))
            by_cat[cat]["total"] += 1
            by_cat[cat]["hit1"].append(hit_at_k(ranked, rel, 1))
            by_cat[cat]["r5"].append(recall_at_k(ranked, rel, 5))
            by_cat[cat]["r10"].append(recall_at_k(ranked, rel, 10))
            by_cat[cat]["mrr"].append(reciprocal_rank(ranked, rel))

        cat_summary = {}
        for cat, stats in sorted(by_cat.items()):
            c_n = stats["total"]
            c_r5 = sum(stats["r5"]) / c_n if c_n else 0.0
            c_r10 = sum(stats["r10"]) / c_n if c_n else 0.0
            c_h1 = sum(stats["hit1"]) / c_n if c_n else 0.0
            c_mrr = sum(stats["mrr"]) / c_n if c_n else 0.0
            # Counts matching
            c_h1_count = sum(1 for x in stats["hit1"] if x > 0.0)
            c_r10_full = sum(1 for x in stats["r10"] if x >= 0.999)
            cat_summary[cat] = {
                "n": c_n,
                "hit1_count": c_h1_count,
                "hit1_pct": c_h1,
                "r5_pct": c_r5,
                "r10_count": c_r10_full,
                "r10_pct": c_r10,
                "mrr": c_mrr,
            }

        summaries.append({
            "config_name": cfg_name,
            "split": split_str,
            "total_n": total_n,
            "ans_n": n_ans,
            "no_ans_n": n_no_ans,
            "recall@1": recall_1,
            "recall@5": recall_5,
            "recall@10": recall_10,
            "mrr": mrr_val,
            "ndcg@10": ndcg_10,
            "gate_precision": gate_precision,
            "injected_ans": injected_ans,
            "injected_no_ans": injected_no_ans,
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "categories": cat_summary,
        })

    return summaries


def generate_markdown_report(
    results_base: Path = Path("eval/results"),
    queries_path: str | Path = "data/eval_v2/queries.jsonl",
    query_meta: dict[str, dict] | None = None,
) -> str:
    if query_meta is None:
        query_meta = load_query_metadata(queries_path)
    run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()])

    all_summaries = []
    for rd in run_dirs:
        summaries = analyze_run(rd, query_meta)
        all_summaries.extend(summaries)

    lines = [
        "# Retrieval Evaluation Comparison",
        "",
        "## Overall Benchmark Summary Table",
        "",
        "| Config | Split | n (Ans/NoAns) | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Gate Prec | P50 Latency | P95 Latency |",
        "| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for s in all_summaries:
        cfg = s["config_name"]
        split = s["split"]
        n_str = f"{s['total_n']} ({s['ans_n']}/{s['no_ans_n']})"
        r1 = f"{s['recall@1']:.4f}"
        r5 = f"{s['recall@5']:.4f}"
        r10 = f"{s['recall@10']:.4f}"
        mrr = f"{s['mrr']:.4f}"
        ndcg = f"{s['ndcg@10']:.4f}"
        g_prec = f"{s['gate_precision']:.1%}" if s['no_ans_n'] > 0 else "N/A"
        p50 = f"{s['p50_latency_ms']:.1f}ms"
        p95 = f"{s['p95_latency_ms']:.1f}ms"
        lines.append(
            f"| `{cfg}` | **{split}** | {n_str} | {r1} | {r5} | {r10} | {mrr} | {ndcg} | {g_prec} | {p50} | {p95} |"
        )

    lines.append("")
    lines.append("## Per-Category Performance Breakdown")
    lines.append("Reported as exact counts and percentages (Hit@1 and Recall@10):")
    lines.append("")

    for s in all_summaries:
        lines.append(f"### `{s['config_name']}` ({s['split']})")
        lines.append("| Category | n | Hit@1 (Count / %) | Recall@5 | Recall@10 (Full / %) | MRR |")
        lines.append("| :--- | :---: | :---: | ---: | :---: | ---: |")
        for cat, c_data in s["categories"].items():
            n = c_data["n"]
            h1_c = c_data["hit1_count"]
            h1_pct = f"{c_data['hit1_pct']:.1%}"
            r5_pct = f"{c_data['r5_pct']:.4f}"
            r10_c = c_data["r10_count"]
            r10_pct = f"{c_data['r10_pct']:.4f}"
            mrr_s = f"{c_data['mrr']:.4f}"
            lines.append(
                f"| **{cat}** | {n} | {h1_c}/{n} ({h1_pct}) | {r5_pct} | {r10_c}/{n} ({r10_pct}) | {mrr_s} |"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    report = generate_markdown_report()
    out_file = Path("eval/results/comparison.md")
    out_file.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWritten to {out_file}")


if __name__ == "__main__":
    main()

