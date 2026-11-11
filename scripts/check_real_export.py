"""Privacy-safe validation of real and synthetic conversation exports.

Runs existing parsers on a given export file and prints an aggregated report.
NEVER prints message text, titles, conversation IDs, or private content.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

from chatmem.ingest.chatgpt import parse_chatgpt_conversation, parse_chatgpt_export
from chatmem.ingest.claude import parse_claude_conversation, parse_claude_export
from chatmem.schema import Conversation


CLAUDE_KNOWN_CONV_KEYS = {
    "uuid", "id", "conversation_id", "name", "title",
    "created_at", "create_time", "updated_at", "update_time",
    "chat_messages", "messages",
}

CHATGPT_KNOWN_CONV_KEYS = {
    "conversation_id", "id", "title", "create_time", "update_time",
    "mapping", "current_node",
}


def parse_timestamp(val: Any) -> datetime | None:
    """Safely parse datetime, ISO string, or numeric epoch timestamp into UTC datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return None
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return None
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except ValueError:
            pass
    return None


def detect_platform(path: Path, raw_data: Any) -> str:
    """Detect whether export is Claude or ChatGPT based on path or raw keys."""
    path_str = str(path).lower()
    if "claude" in path_str:
        return "claude"
    if "chatgpt" in path_str:
        return "chatgpt"

    # Inspect first conversation item
    items = raw_data
    if isinstance(raw_data, dict):
        items = raw_data.get("value") or raw_data.get("conversations") or raw_data.get("chats") or [raw_data]
    if isinstance(items, list) and items and isinstance(items[0], dict):
        first = items[0]
        if "mapping" in first or "current_node" in first:
            return "chatgpt"
        if "chat_messages" in first or "uuid" in first:
            return "claude"
    return "unknown"


def analyze_export(path: Path) -> dict[str, Any]:
    """Inspect and parse an export file, producing aggregate privacy-safe statistics."""
    report: dict[str, Any] = {
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "is_empty": False,
        "platform": "unknown",
        "raw_item_count": 0,
        "unread_top_level_keys": set(),
        "chatgpt_node_stats": {
            "null_message": 0,
            "non_text_or_skipped_by_type": Counter(),
            "branched_conversations": 0,
            "branch_points_total": 0,
            "total_nodes": 0,
            "chain_nodes": 0,
        },
        "parse_exceptions": [],
        "conversations_parsed": 0,
        "total_messages": 0,
        "messages_per_conv": {"min": 0, "median": 0.0, "max": 0},
        "zero_message_convs": 0,
        "single_role_convs": 0,
        "messages_by_role": Counter(),
        "empty_text_messages": 0,
        "timestamp_sanity": {
            "min_date": None,
            "max_date": None,
            "missing_or_invalid": 0,
        },
        "message_length_stats": {
            "max_length": 0,
            "over_20k_count": 0,
        },
    }

    if not path.exists() or path.stat().st_size == 0:
        report["is_empty"] = True
        return report

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as err:
        report["parse_exceptions"].append({
            "exception_type": type(err).__name__,
            "field": "file_json_root",
        })
        return report

    platform = detect_platform(path, raw_data)
    report["platform"] = platform

    # Extract conversation items
    conv_items: list[dict] = []
    if isinstance(raw_data, dict):
        for k in sorted(raw_data.keys()):
            if k not in {"value", "conversations", "chats"}:
                report["unread_top_level_keys"].add(f"root.{k}")
        wrapped = raw_data.get("value") or raw_data.get("conversations") or raw_data.get("chats")
        if isinstance(wrapped, list):
            conv_items = [x for x in wrapped if isinstance(x, dict)]
        elif "uuid" in raw_data or "mapping" in raw_data or "current_node" in raw_data:
            conv_items = [raw_data]
    elif isinstance(raw_data, list):
        conv_items = [x for x in raw_data if isinstance(x, dict)]

    report["raw_item_count"] = len(conv_items)

    # Inspect raw fields and ChatGPT tree structure
    known_keys = CLAUDE_KNOWN_CONV_KEYS if platform == "claude" else CHATGPT_KNOWN_CONV_KEYS
    all_raw_keys = set()
    for item in conv_items:
        all_raw_keys.update(item.keys())
        if platform == "chatgpt":
            mapping = item.get("mapping", {})
            if isinstance(mapping, dict):
                report["chatgpt_node_stats"]["total_nodes"] += len(mapping)
                branch_points = sum(
                    1 for node in mapping.values()
                    if isinstance(node, dict) and len(node.get("children", [])) > 1
                )
                if branch_points > 0:
                    report["chatgpt_node_stats"]["branched_conversations"] += 1
                    report["chatgpt_node_stats"]["branch_points_total"] += branch_points

                # Walk current_node chain
                chain_nodes = 0
                curr = item.get("current_node")
                visited = set()
                while curr and curr in mapping and curr not in visited:
                    visited.add(curr)
                    chain_nodes += 1
                    node = mapping[curr]
                    curr = node.get("parent") if isinstance(node, dict) else None
                report["chatgpt_node_stats"]["chain_nodes"] += chain_nodes

                for node in mapping.values():
                    if not isinstance(node, dict):
                        continue
                    msg = node.get("message")
                    if msg is None:
                        report["chatgpt_node_stats"]["null_message"] += 1
                        continue
                    content = msg.get("content") or {}
                    ctype = content.get("content_type", "unknown") if isinstance(content, dict) else type(content).__name__
                    parts = content.get("parts", []) if isinstance(content, dict) else []
                    has_text = any(isinstance(p, str) and p.strip() for p in parts)
                    if not has_text:
                        report["chatgpt_node_stats"]["non_text_or_skipped_by_type"][ctype] += 1

    report["unread_top_level_keys"].update(all_raw_keys - known_keys)

    # Run the actual parser
    conversations: list[Conversation] = []
    if platform == "claude":
        for idx, item in enumerate(conv_items):
            try:
                parsed = parse_claude_conversation(item)
                if parsed:
                    conversations.append(parsed)
            except Exception as err:
                report["parse_exceptions"].append({
                    "exception_type": type(err).__name__,
                    "field": f"conversation[{idx}]",
                })
    elif platform == "chatgpt":
        for idx, item in enumerate(conv_items):
            try:
                parsed = parse_chatgpt_conversation(item)
                if parsed:
                    conversations.append(parsed)
            except Exception as err:
                report["parse_exceptions"].append({
                    "exception_type": type(err).__name__,
                    "field": f"conversation[{idx}]",
                })
    else:
        # Fallback to export parser
        conversations = parse_claude_export(path) or parse_chatgpt_export(path)

    report["conversations_parsed"] = len(conversations)
    all_messages = [m for c in conversations for m in c.messages]
    report["total_messages"] = len(all_messages)

    if conversations:
        msg_counts = [len(c.messages) for c in conversations]
        report["messages_per_conv"] = {
            "min": min(msg_counts),
            "median": statistics.median(msg_counts),
            "max": max(msg_counts),
        }
        report["zero_message_convs"] = sum(1 for c in conversations if len(c.messages) == 0)
        report["single_role_convs"] = sum(
            1 for c in conversations if len({m.role for m in c.messages if m.role}) == 1
        )

    for m in all_messages:
        report["messages_by_role"][m.role] += 1
        if not m.text or not m.text.strip():
            report["empty_text_messages"] += 1

        t_len = len(m.text)
        if t_len > report["message_length_stats"]["max_length"]:
            report["message_length_stats"]["max_length"] = t_len
        if t_len > 20000:
            report["message_length_stats"]["over_20k_count"] += 1

        dt = parse_timestamp(m.timestamp)
        if dt is None:
            report["timestamp_sanity"]["missing_or_invalid"] += 1
        else:
            if report["timestamp_sanity"]["min_date"] is None or dt < report["timestamp_sanity"]["min_date"]:
                report["timestamp_sanity"]["min_date"] = dt
            if report["timestamp_sanity"]["max_date"] is None or dt > report["timestamp_sanity"]["max_date"]:
                report["timestamp_sanity"]["max_date"] = dt

    return report


def print_report(rep: dict[str, Any]) -> None:
    """Print the privacy-safe report in Markdown format."""
    print("=" * 60)
    print(f"REPORT: {rep['file_path']}")
    print("=" * 60)
    print(f"File Size: {rep['file_size_bytes']:,} bytes")
    if rep["is_empty"]:
        print("Status: Empty file (0 bytes). Parsed 0 conversations.")
        return

    print(f"Detected Platform: {rep['platform']}")
    print(f"Raw Conversations in JSON: {rep['raw_item_count']}")
    print(f"Successfully Parsed Conversations: {rep['conversations_parsed']}")
    print(f"Total Messages Parsed: {rep['total_messages']:,}")

    mpc = rep["messages_per_conv"]
    print(f"Messages Per Conversation: min={mpc['min']}, median={mpc['median']}, max={mpc['max']}")
    print(f"Conversations with 0 Messages: {rep['zero_message_convs']}")
    print(f"Conversations with Only 1 Role: {rep['single_role_convs']}")

    print("\nMessages by Role:")
    for role, count in sorted(rep["messages_by_role"].items()):
        print(f"  - {role}: {count:,}")

    print(f"\nEmpty Text Messages After Parsing: {rep['empty_text_messages']}")

    unread_keys = sorted(rep["unread_top_level_keys"])
    print(f"\nUnread Raw JSON Field Names ({len(unread_keys)}):")
    if unread_keys:
        for k in unread_keys:
            print(f"  - {k}")
    else:
        print("  (None)")

    if rep["platform"] == "chatgpt":
        c_stats = rep["chatgpt_node_stats"]
        print("\nChatGPT Tree & Node Statistics:")
        print(f"  - Total nodes across all mappings: {c_stats['total_nodes']}")
        print(f"  - Nodes on current_node path: {c_stats['chain_nodes']}")
        print(f"  - Branched conversations: {c_stats['branched_conversations']}")
        print(f"  - Total branch points (>1 children): {c_stats['branch_points_total']}")
        print(f"  - Null message nodes skipped: {c_stats['null_message']}")
        print("  - Non-text or skipped nodes by content_type:")
        for ctype, cnt in sorted(c_stats["non_text_or_skipped_by_type"].items()):
            print(f"      * {ctype}: {cnt}")

    print("\nTimestamp Sanity:")
    min_d = rep["timestamp_sanity"]["min_date"]
    max_d = rep["timestamp_sanity"]["max_date"]
    print(f"  - Earliest message timestamp: {min_d.isoformat() if min_d else 'N/A'}")
    print(f"  - Latest message timestamp:   {max_d.isoformat() if max_d else 'N/A'}")
    print(f"  - Missing or invalid timestamps: {rep['timestamp_sanity']['missing_or_invalid']}")

    l_stats = rep["message_length_stats"]
    print("\nMessage Length Statistics:")
    print(f"  - Longest message: {l_stats['max_length']:,} characters")
    print(f"  - Messages > 20,000 characters: {l_stats['over_20k_count']}")

    print("\nParse Exceptions:")
    if rep["parse_exceptions"]:
        for ex in rep["parse_exceptions"]:
            print(f"  - Exception: {ex['exception_type']} on field: {ex['field']}")
    else:
        print("  (0 exceptions encountered)")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Privacy-safe validation of conversation exports.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["data/raw/claude/conversations.json", "data/raw/chatgpt/conversations.json"],
        help="Paths to export files to check.",
    )
    args = parser.parse_args()

    for p_str in args.paths:
        p = Path(p_str)
        if not p.exists():
            print(f"Path does not exist: {p}", file=sys.stderr)
            continue
        rep = analyze_export(p)
        print_report(rep)


if __name__ == "__main__":
    main()
