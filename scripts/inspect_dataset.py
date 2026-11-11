"""Dataset inspection script for data/eval_v2.

Prints:
- number of conversations
- total messages
- messages per conversation (min / median / max)
- count of conversations with 20+ messages
"""
from __future__ import annotations

import statistics
from pathlib import Path

from chatmem.chunking.per_message import chunk_per_message
from chatmem.chunking.qa_pair import qa_chunks
from chatmem.chunking.sliding_window import window_chunks
from chatmem.ingest.normalize import load_all_exports


def main() -> None:
    eval_v2_dir = Path("data/eval_v2")
    if not eval_v2_dir.exists():
        print(f"Error: {eval_v2_dir} does not exist.")
        return

    convs = load_all_exports(eval_v2_dir)
    num_convs = len(convs)
    msg_counts = [len(c.messages) for c in convs]
    total_messages = sum(msg_counts)

    min_msgs = min(msg_counts) if msg_counts else 0
    max_msgs = max(msg_counts) if msg_counts else 0
    median_msgs = statistics.median(msg_counts) if msg_counts else 0
    convs_20plus = sum(1 for m in msg_counts if m >= 20)

    print("=== data/eval_v2 Dataset Inspection ===")
    print(f"Number of conversations: {num_convs}")
    print(f"Total messages: {total_messages}")
    print(f"Messages per conversation: min={min_msgs}, median={median_msgs}, max={max_msgs}")
    print(f"Conversations with 20+ messages: {convs_20plus}")

    # Inspect chunk outputs for each strategy
    pm_chunks = chunk_per_message(convs)
    ws_chunks = window_chunks(convs, window_size=3, overlap=1)
    qa_c = qa_chunks(convs)

    print("\n=== Chunk Counts from Parsed Conversations ===")
    print(f"per_message chunks: {len(pm_chunks)}")
    print(f"sliding_window chunks (window_size=3, overlap=1): {len(ws_chunks)}")
    print(f"qa_pair chunks: {len(qa_c)}")

    # Check distribution of message lengths across conversations
    from collections import Counter
    len_counter = Counter(msg_counts)
    print("\nMessage length distribution:")
    for length in sorted(len_counter.keys()):
        print(f"  {length} messages: {len_counter[length]} conversations")


if __name__ == "__main__":
    main()

