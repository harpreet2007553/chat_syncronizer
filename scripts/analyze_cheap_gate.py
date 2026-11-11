"""Analysis script: evaluate cheap gating signals without a reranker on dev split.

Signals analyzed for each query's top result:
1. Dense cosine similarity of top chunk
2. BM25 score of top chunk
3. RRF score of top fused chunk
"""
from __future__ import annotations

import json
from pathlib import Path

from chatmem.index.dense import get_qdrant_client, search_dense
from chatmem.index.store import bm25_search, connect
from chatmem.retrieval.fusion import reciprocal_rank_fusion


def main() -> None:
    queries_path = Path("data/eval_v2/queries.jsonl")
    db_path = Path("data/chatmem_metadata.db")

    # Load dev queries
    conn = connect(db_path)
    qdrant = get_qdrant_client()

    dev_queries = []
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("split") == "dev":
                dev_queries.append(item)

    print(f"Loaded {len(dev_queries)} dev queries.")

    records = []
    for q in dev_queries:
        qid = q["qid"]
        query_text = q["query"]
        is_answerable = len(q.get("relevant", [])) > 0

        # 1. BM25 search
        bm25_results = bm25_search(conn, "qa", query_text, top_k=30)
        top_bm25_score = bm25_results[0][1] if bm25_results else 0.0

        # 2. Dense cosine search
        dense_results = search_dense(qdrant, "chunks_qa", query_text, top_k=30)
        top_dense_score = dense_results[0][1] if dense_results else 0.0

        # 3. RRF fusion
        fused = reciprocal_rank_fusion(
            {"bm25": bm25_results, "dense": dense_results},
            k=60,
            weights={"bm25": 1.0, "dense": 1.0},
            top_n=10,
        )
        top_rrf_score = fused[0].score if fused else 0.0

        records.append({
            "qid": qid,
            "query": query_text,
            "answerable": is_answerable,
            "type": q.get("type", "unknown"),
            "bm25": top_bm25_score,
            "dense": top_dense_score,
            "rrf": top_rrf_score,
        })
    conn.close()

    ans_records = [r for r in records if r["answerable"]]
    no_ans_records = [r for r in records if not r["answerable"]]
    n_ans = len(ans_records)
    n_no_ans = len(no_ans_records)

    print(f"\nDev split: {n_ans} answerable, {n_no_ans} no-answer.")

    # Sweep function
    for signal_name in ["dense", "bm25", "rrf"]:
        scores_ans = [r[signal_name] for r in ans_records]
        scores_no_ans = [r[signal_name] for r in no_ans_records]

        print("\n=======================================================")
        print(f"Signal: {signal_name.upper()}")
        print(f"Answerable stats: min={min(scores_ans):.4f}, median={sorted(scores_ans)[len(scores_ans)//2]:.4f}, max={max(scores_ans):.4f}")
        print(f"No-Answer stats:  min={min(scores_no_ans):.4f}, median={sorted(scores_no_ans)[len(scores_no_ans)//2]:.4f}, max={max(scores_no_ans):.4f}")

        # Sweep candidate thresholds
        all_scores = sorted(set(scores_ans + scores_no_ans))

        print("\nThreshold Sweep (sample points):")
        print("Threshold | Correct Abstain (NoAns) | False Injections | Ans Passed | Ans Lost | Precision")

        step = max(1, len(all_scores) // 10)
        sample_thresholds = [all_scores[i] for i in range(0, len(all_scores), step)]
        if all_scores[-1] not in sample_thresholds:
            sample_thresholds.append(all_scores[-1])

        for thresh in sample_thresholds:
            abstain_count = sum(1 for s in scores_no_ans if s < thresh)
            false_injections = n_no_ans - abstain_count
            ans_passed = sum(1 for s in scores_ans if s >= thresh)
            ans_lost = n_ans - ans_passed
            total_injected = ans_passed + false_injections
            precision = (ans_passed / total_injected) if total_injected > 0 else 1.0

            abstain_rate = abstain_count / n_no_ans

            print(f"{thresh:9.4f} | {abstain_count:2d}/{n_no_ans} ({abstain_rate*100:5.1f}%)     | {false_injections:2d}               | {ans_passed:2d}/{n_ans}    | {ans_lost:2d}       | {precision*100:5.1f}%")

        # Find best threshold that maximizes correct abstain while minimizing answerable lost
        # Ideal: high abstain rate (>= 80%) with minimal ans lost
        candidates = []
        for thresh in all_scores:
            abstain_count = sum(1 for s in scores_no_ans if s < thresh)
            false_injections = n_no_ans - abstain_count
            ans_passed = sum(1 for s in scores_ans if s >= thresh)
            ans_lost = n_ans - ans_passed
            total_injected = ans_passed + false_injections
            precision = (ans_passed / total_injected) if total_injected > 0 else 1.0
            candidates.append((abstain_count, -ans_lost, precision, thresh, false_injections, ans_passed))

        # Best balanced threshold
        # Sort by high abstain, low ans_lost
        candidates.sort(key=lambda x: (x[0] / n_no_ans >= 0.8, x[0] - x[1], x[2]), reverse=True)
        best = candidates[0]
        print(f"\nBest balanced threshold for {signal_name}: {best[3]:.4f}")
        print(f"  Abstain rate on no-answer: {best[0]}/{n_no_ans} ({best[0]/n_no_ans*100:.1f}%)")
        print(f"  False injections: {best[4]}")
        print(f"  Answerable kept: {best[5]}/{n_ans} ({best[5]/n_ans*100:.1f}%)")
        print(f"  Answerable lost: {n_ans - best[5]}")
        print(f"  Precision: {best[2]*100:.1f}%")


if __name__ == "__main__":
    main()

