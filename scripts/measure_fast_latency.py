"""Measure latency of fast mode on data/eval_v2 dev split and save results."""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from chatmem.eval.run_eval import load_queries, percentile_95
from chatmem.index.dense import get_qdrant_client
from chatmem.index.store import DEFAULT_DB_PATH, connect
from chatmem.retrieval.pipeline import PipelineConfig, QdrantConfig, Retriever


def main() -> None:
    queries_path = Path("data/eval_v2/queries.jsonl")
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found at {queries_path}")

    dev_queries = load_queries(queries_path, split="dev")
    print(f"Loaded {len(dev_queries)} dev queries from {queries_path}.")

    qdrant_cfg = QdrantConfig(mode="local", path="data/qdrant_local")
    qdrant_client = get_qdrant_client(qdrant_cfg)
    sqlite_conn = connect(DEFAULT_DB_PATH)

    pipeline_cfg = PipelineConfig(
        name="fast",
        strategy="qa",
        use_bm25=True,
        use_dense=True,
        use_rerank=False,
        dense_gate_threshold=0.70,
        final_top_k=10,
        qdrant=qdrant_cfg,
    )

    retriever = Retriever(
        config=pipeline_cfg,
        conn=sqlite_conn,
        client=qdrant_client,
    )

    # 1. Warmup query (excluded from metrics)
    print("Executing warmup query...")
    _ = retriever.search("warmup query to initialize dense model and fts index")
    print("Warmup complete.")

    # 2. Execute 3 runs
    runs_data = []
    run_medians = []
    run_p95s = []

    for run_idx in range(1, 4):
        print(f"Starting Run {run_idx}/3...")
        latencies: list[float] = []

        for q in dev_queries:
            t0 = time.perf_counter()
            _ = retriever.search(q.text)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(round(elapsed_ms, 2))

        med = statistics.median(latencies)
        p95 = percentile_95(latencies)
        run_medians.append(med)
        run_p95s.append(p95)

        runs_data.append(
            {
                "run": run_idx,
                "median_latency_ms": round(med, 2),
                "p95_latency_ms": round(p95, 2),
                "query_latencies_ms": latencies,
            }
        )
        print(f"  Run {run_idx}: Median = {med:.2f} ms, P95 = {p95:.2f} ms")

    summary_median = round(statistics.median(run_medians), 2)
    summary_p95 = round(statistics.median(run_p95s), 2)

    result_payload = {
        "dataset": "data/eval_v2",
        "split": "dev",
        "query_count": len(dev_queries),
        "mode": "fast",
        "strategy": "qa",
        "dense_gate_threshold": 0.70,
        "runs": runs_data,
        "summary": {
            "median_latency_ms": summary_median,
            "p95_latency_ms": summary_p95,
        },
    }

    out_file = Path("eval/results/fast_mode_latency.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(f"\nSaved latency report to {out_file}")
    print(f"Final Summary across 3 runs: Median = {summary_median} ms, P95 = {summary_p95} ms")

    retriever.close()


if __name__ == "__main__":
    main()

