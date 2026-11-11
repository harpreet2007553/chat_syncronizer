"""Inspect schema of data/raw/claude/conversations.json without printing private content.

Outputs only dictionary keys, data types, lengths, and counts.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path("data/raw/claude/conversations.json")
    if not path.exists():
        print("File does not exist:", path)
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Top-level type: {type(data).__name__}")
    if isinstance(data, dict):
        print(f"Top-level dict keys: {sorted(data.keys())}")
        if "value" in data:
            val = data["value"]
            print(f"data['value'] type: {type(val).__name__}")
            if isinstance(val, list):
                data = val
        elif "conversations" in data:
            data = data["conversations"]

    if isinstance(data, list):
        print(f"Total items in list: {len(data)}")
        if data:
            first = data[0]
            print(f"Item 0 type: {type(first).__name__}")
            if isinstance(first, dict):
                print(f"Item 0 keys: {sorted(first.keys())}")
                for k, v in first.items():
                    if isinstance(v, list):
                        print(f"  key '{k}': list of {len(v)} items")
                        if v and isinstance(v[0], dict):
                            print(f"    first item in '{k}' keys: {sorted(v[0].keys())}")
                            for subk, subv in v[0].items():
                                print(f"      subfield '{subk}': type={type(subv).__name__}")
                    else:
                        print(f"  key '{k}': type={type(v).__name__}")

            # Check if keys vary across items
            all_conv_keys = set()
            all_msg_keys = set()
            msg_key_names = set()
            for item in data:
                if isinstance(item, dict):
                    all_conv_keys.update(item.keys())
                    for k, v in item.items():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            msg_key_names.add(k)
                            for msg in v:
                                if isinstance(msg, dict):
                                    all_msg_keys.update(msg.keys())

            print(f"\nAll observed conversation keys across all items: {sorted(all_conv_keys)}")
            print(f"Message list key candidates: {sorted(msg_key_names)}")
            print(f"All observed message keys: {sorted(all_msg_keys)}")
    elif isinstance(data, dict):
        print(f"Top-level dict keys: {sorted(data.keys())}")


if __name__ == "__main__":
    main()
