"""Profiler script for CrossEncoder reranker alone.

Times separately:
1. Model load
2. First query (warmup)
3. Next 20 queries (measure median and p95 latency)
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import time
from pathlib import Path

import torch
from chatmem.retrieval.rerank import get_cross_encoder, rerank


def load_sample_candidates(db_path: Path, limit: int = 30) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, text FROM chunks WHERE strategy = 'qa' LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [(str(r[0]), str(r[1])) for r in rows]


def load_queries(queries_path: Path, count: int = 25) -> list[str]:
    queries = []
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = item.get("query", "").strip()
            if q:
                queries.append(q)
            if len(queries) >= count:
                break
    return queries


def main() -> None:
    db_path = Path("data/chatmem_metadata.db")
    queries_path = Path("data/eval_v2/queries_authored.jsonl")
    if not queries_path.exists():
        queries_path = Path("data/eval/queries.sample.jsonl")

    candidates = load_sample_candidates(db_path, limit=30)
    queries = load_queries(queries_path, count=25)

    print("=== CrossEncoder Profiler ===")
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Number of candidates: {len(candidates)}")
    print(f"Number of queries: {len(queries)}")

    # 1. Model load time
    t0 = time.perf_counter()
    _ = get_cross_encoder(model_name="BAAI/bge-reranker-base", max_length=256)
    load_time = time.perf_counter() - t0
    print(f"\n1. Model load time: {load_time:.3f} s")

    # 2. First query (warmup / cold start)
    first_query = queries[0]
    t0 = time.perf_counter()
    _ = rerank(first_query, candidates)
    first_query_time = time.perf_counter() - t0
    print(f"2. First query time: {first_query_time:.3f} s ({first_query_time * 1000:.1f} ms)")

    # 3. Next 20 queries
    timed_queries = queries[1:21]
    if len(timed_queries) < 20:
        # Repeat if fewer
        timed_queries = (timed_queries * (20 // len(timed_queries) + 1))[:20]

    durations: list[float] = []
    for i, q in enumerate(timed_queries, start=1):
        t0 = time.perf_counter()
        _ = rerank(q, candidates)
        dur = time.perf_counter() - t0
        durations.append(dur)

    durations_ms = [d * 1000.0 for d in durations]
    sorted_ms = sorted(durations_ms)
    median_ms = statistics.median(sorted_ms)
    mean_ms = statistics.mean(sorted_ms)
    p95_index = int(0.95 * len(sorted_ms))
    p95_ms = sorted_ms[min(p95_index, len(sorted_ms) - 1)]

    print("\n3. Next 20 queries:")
    print(f"   Min:    {sorted_ms[0]:.1f} ms")
    print(f"   Median: {median_ms:.1f} ms ({median_ms / 1000.0:.3f} s)")
    print(f"   Mean:   {mean_ms:.1f} ms ({mean_ms / 1000.0:.3f} s)")
    print(f"   P95:    {p95_ms:.1f} ms ({p95_ms / 1000.0:.3f} s)")
    print(f"   Max:    {sorted_ms[-1]:.1f} ms")


if __name__ == "__main__":
    main()
