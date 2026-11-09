"""Build, validate, and review the synthetic evaluation dataset for chatmem.

Pure Python, no network, no LLM calls.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from chatmem.ingest.normalize import load_all_exports

app = typer.Typer(help="Tools for building, validating, and reviewing chatmem eval_v2.")


# ============================================================================
# Helpers: Format Converters
# ============================================================================

def to_chatgpt_dict(conv: dict) -> dict:
    """Convert intermediate conversation dict to ChatGPT mapping-tree export format."""
    cid = conv["id"]
    title = conv.get("title", "Untitled")
    created = conv.get("created", 1673773200.0)
    if isinstance(created, str):
        try:
            created_ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except Exception:
            created_ts = 1673773200.0
    else:
        created_ts = float(created)

    mapping: dict[str, Any] = {}
    root_id = f"{cid}-root"
    mapping[root_id] = {
        "id": root_id,
        "message": None,
        "parent": None,
        "children": [],
    }

    prev_id = root_id
    msgs = conv.get("messages", [])
    for idx, m in enumerate(msgs):
        node_id = f"{cid}-msg-{idx}"
        mapping[prev_id]["children"].append(node_id)
        mapping[node_id] = {
            "id": node_id,
            "parent": prev_id,
            "children": [],
            "message": {
                "id": node_id,
                "author": {"role": m.get("role", "user")},
                "create_time": created_ts + idx * 60.0,
                "content": {
                    "content_type": "text",
                    "parts": [m.get("text", "")],
                },
            },
        }
        prev_id = node_id

    return {
        "id": cid,
        "conversation_id": cid,
        "title": title,
        "create_time": created_ts,
        "update_time": created_ts + len(msgs) * 60.0,
        "mapping": mapping,
        "current_node": prev_id,
    }


def to_claude_dict(conv: dict) -> dict:
    """Convert intermediate conversation dict to Claude chat_messages export format."""
    cid = conv["id"]
    title = conv.get("title", "Untitled")
    created = conv.get("created", 1673773200.0)
    if isinstance(created, (int, float)):
        created_str = datetime.fromtimestamp(created, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    else:
        created_str = str(created)

    chat_messages = []
    for idx, m in enumerate(conv.get("messages", [])):
        role = m.get("role", "user")
        sender = "human" if role == "user" else "assistant"
        chat_messages.append({
            "uuid": f"{cid}-msg-{idx}",
            "text": m.get("text", ""),
            "sender": sender,
            "created_at": created_str,
        })

    return {
        "uuid": cid,
        "name": title,
        "created_at": created_str,
        "updated_at": created_str,
        "chat_messages": chat_messages,
    }


def content_words(text: str) -> set[str]:
    """Extract lowercased content words excluding basic English stopwords."""
    stopwords = {
        "the", "a", "an", "is", "in", "it", "to", "of", "and", "or", "for", "with",
        "on", "at", "by", "from", "as", "that", "this", "my", "how", "why", "what",
        "can", "do", "does", "i", "you", "he", "she", "we", "they", "me", "him",
        "her", "us", "them", "are", "was", "were", "be", "been", "being", "have",
        "has", "had", "so", "but", "if", "not", "no", "just", "about", "your",
    }
    words = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()))
    return {w for w in words if w not in stopwords and len(w) > 1}


def compute_jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def assign_splits(queries: list[dict], seed: int = 1234) -> list[dict]:
    """Stratified 50/50 split assignment per query type with fixed seed."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for q in queries:
        by_type.setdefault(q["type"], []).append(q)

    result = []
    for qtype, group in sorted(by_type.items()):
        indices = list(range(len(group)))
        rng.shuffle(indices)
        half = len(group) // 2
        for rank, idx in enumerate(indices):
            q_copy = dict(group[idx])
            q_copy["split"] = "dev" if rank < half else "test"
            result.append(q_copy)

    result.sort(key=lambda x: x["qid"])
    return result


# ============================================================================
# Core Logic: Build, Validate, Review
# ============================================================================

def build_dataset(
    raw_dir: Path,
    out_dir: Path,
    sample_dir: Path | None = None,
) -> tuple[int, int]:
    """Compile raw batch files and authored queries into export JSON and queries.jsonl."""
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)

    # 1. Load sample original exports if available to copy c01..c14 unchanged
    sample_cg_map = {}
    sample_cl_map = {}
    if sample_dir and Path(sample_dir).exists():
        s_cg = Path(sample_dir) / "chatgpt" / "conversations.json"
        if s_cg.exists():
            for item in json.loads(s_cg.read_text(encoding="utf-8")):
                sample_cg_map[str(item.get("id") or item.get("conversation_id"))] = item
        s_cl = Path(sample_dir) / "claude" / "conversations.json"
        if s_cl.exists():
            for item in json.loads(s_cl.read_text(encoding="utf-8")):
                sample_cl_map[str(item.get("uuid"))] = item

    # 2. Collect all raw conversations from batch_*.json
    batch_files = sorted(raw_path.glob("batch_*.json"))
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.json files found in {raw_path}")

    all_convs: list[dict] = []
    for bf in batch_files:
        data = json.loads(bf.read_text(encoding="utf-8"))
        if isinstance(data, list):
            all_convs.extend(data)
        elif isinstance(data, dict):
            all_convs.append(data)

    # 3. Facts map
    facts_map: dict[str, list[str]] = {}
    fact_to_convs: dict[str, set[str]] = {}
    chatgpt_export: list[dict] = []
    claude_export: list[dict] = []

    for c in all_convs:
        cid = str(c["id"])
        platform = str(c.get("platform", "")).lower()
        facts = [str(f) for f in c.get("facts", [])]
        facts_map[cid] = facts

        for f in facts:
            fact_to_convs.setdefault(f, set()).add(cid)

        if platform == "chatgpt":
            if cid in sample_cg_map:
                chatgpt_export.append(sample_cg_map[cid])
            else:
                chatgpt_export.append(to_chatgpt_dict(c))
        elif platform == "claude":
            if cid in sample_cl_map:
                claude_export.append(sample_cl_map[cid])
            else:
                claude_export.append(to_claude_dict(c))
        else:
            raise ValueError(f"Unknown platform '{platform}' for conversation {cid}")

    # 4. Write platform directories
    cg_dir = out_path / "chatgpt"
    cg_dir.mkdir(parents=True, exist_ok=True)
    with (cg_dir / "conversations.json").open("w", encoding="utf-8") as f:
        json.dump(chatgpt_export, f, indent=2)

    cl_dir = out_path / "claude"
    cl_dir.mkdir(parents=True, exist_ok=True)
    with (cl_dir / "conversations.json").open("w", encoding="utf-8") as f:
        json.dump(claude_export, f, indent=2)

    with (out_path / "facts.json").open("w", encoding="utf-8") as f:
        json.dump(facts_map, f, indent=2)

    # 5. Process authored queries
    queries_file = raw_path / "queries_authored.jsonl"
    if not queries_file.exists():
        raise FileNotFoundError(f"Authored queries file not found: {queries_file}")

    queries_out: list[dict] = []
    with queries_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q_data = json.loads(line)
            qid = str(q_data["qid"])
            query_text = str(q_data["query"])
            qtype = str(q_data["type"]).strip().lower()
            needs = q_data.get("needs", [])
            split = q_data.get("split")

            # Compute relevant conversations by union over facts
            relevant_cids: set[str] = set()
            for need_fact in needs:
                matched = fact_to_convs.get(need_fact, set())
                relevant_cids.update(matched)

            if qtype.startswith("no_answer"):
                relevant_list = []
            else:
                relevant_list = sorted(list(relevant_cids))

            queries_out.append({
                "qid": qid,
                "query": query_text,
                "relevant": relevant_list,
                "type": qtype,
                "split": split,
            })

    queries_out.sort(key=lambda x: x["qid"])
    with (out_path / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in queries_out:
            f.write(json.dumps(q) + "\n")

    return len(all_convs), len(queries_out)


def validate_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    """Perform integrity, consistency, and quality checks on compiled dataset."""
    path = Path(dataset_dir)
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Round-trip parse check
    loaded_convs = load_all_exports(path)
    loaded_ids = {c.conversation_id for c in loaded_convs}

    cg_file = path / "chatgpt" / "conversations.json"
    cl_file = path / "claude" / "conversations.json"
    facts_file = path / "facts.json"
    queries_file = path / "queries.jsonl"

    if not cg_file.exists():
        errors.append(f"Missing {cg_file}")
    if not cl_file.exists():
        errors.append(f"Missing {cl_file}")
    if not facts_file.exists():
        errors.append(f"Missing {facts_file}")
    if not queries_file.exists():
        errors.append(f"Missing {queries_file}")

    if errors:
        return {}, errors, warnings

    cg_data = json.loads(cg_file.read_text(encoding="utf-8"))
    cl_data = json.loads(cl_file.read_text(encoding="utf-8"))
    facts_map = json.loads(facts_file.read_text(encoding="utf-8"))

    total_expected_convs = len(cg_data) + len(cl_data)
    if len(loaded_convs) != total_expected_convs:
        errors.append(
            f"Round-trip count mismatch: wrote {total_expected_convs} conversations, parsed {len(loaded_convs)}"
        )

    # 2. Conversation IDs unique & platform check
    seen_cids = set()
    for item in cg_data:
        cid = str(item.get("id") or item.get("conversation_id"))
        if cid in seen_cids:
            errors.append(f"Duplicate conversation ID in ChatGPT: {cid}")
        seen_cids.add(cid)
        if cid not in loaded_ids:
            errors.append(f"ChatGPT conversation {cid} dropped during parsing")

    for item in cl_data:
        cid = str(item.get("uuid"))
        if cid in seen_cids:
            errors.append(f"Duplicate conversation ID across exports: {cid}")
        seen_cids.add(cid)
        if cid not in loaded_ids:
            errors.append(f"Claude conversation {cid} dropped during parsing")

    # 3. Facts check
    fact_to_convs: dict[str, set[str]] = {}
    for cid, facts in facts_map.items():
        if not facts:
            warnings.append(f"Conversation {cid} has no facts tagged.")
        for f in facts:
            fact_to_convs.setdefault(f, set()).add(cid)

    # 4. Queries check
    queries: list[dict] = []
    seen_qids = set()
    types_by_split: dict[str, set[str]] = {"dev": set(), "test": set()}
    type_counts: Counter[str] = Counter()
    type_split_counts: Counter[tuple[str, str]] = Counter()

    with queries_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            q = json.loads(line)
            qid = str(q.get("qid"))
            if qid in seen_qids:
                errors.append(f"Duplicate query ID: {qid}")
            seen_qids.add(qid)
            queries.append(q)

            qtype = str(q.get("type", "")).strip().lower()
            split = str(q.get("split", "")).strip().lower()
            relevant = q.get("relevant", [])

            type_counts[qtype] += 1
            if split in ("dev", "test"):
                types_by_split[split].add(qtype)
                type_split_counts[(qtype, split)] += 1
            else:
                errors.append(f"Query {qid} has invalid split: '{split}' (expected dev or test)")

            if qtype.startswith("no_answer"):
                if relevant:
                    errors.append(f"no_answer query {qid} has non-empty relevant set: {relevant}")
            else:
                if not relevant:
                    errors.append(f"Answerable query {qid} ({qtype}) has empty relevant list")

    # Check every type present in both dev and test
    all_types = set(type_counts.keys())
    for t in all_types:
        if t not in types_by_split["dev"]:
            errors.append(f"Query type '{t}' missing from dev split")
        if t not in types_by_split["test"]:
            errors.append(f"Query type '{t}' missing from test split")

    # 5. Semantic quality checks & warnings
    conv_by_id = {c.conversation_id: c for c in loaded_convs}
    for q in queries:
        qid = q["qid"]
        qtext = q["query"]
        qtype = q["type"]
        rel_cids = q.get("relevant", [])

        if qtype == "paraphrase":
            q_words = content_words(qtext)
            for cid in rel_cids:
                conv = conv_by_id.get(cid)
                if not conv:
                    continue
                # Test against whole conversation
                conv_text = " ".join(m.text for m in conv.messages)
                c_words = content_words(conv_text)
                sim = compute_jaccard(q_words, c_words)

                # Also test against individual messages in case query is copied from one turn
                max_msg_sim = max(
                    (compute_jaccard(q_words, content_words(m.text)) for m in conv.messages),
                    default=0.0
                )

                if sim > 0.5 or max_msg_sim > 0.5:
                    warnings.append(
                        f"Paraphrase query {qid} has high Jaccard overlap (>0.5) against {cid} (sim={max(sim, max_msg_sim):.2f}): '{qtext}'"
                    )

        elif qtype == "keyword":
            # Warn if none of query's content tokens appear in relevant conversation
            q_tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-\.\'\@\=]+\b", qtext) if len(t) > 2]
            found = False
            for cid in rel_cids:
                conv = conv_by_id.get(cid)
                if not conv:
                    continue
                conv_text_lower = " ".join(m.text for m in conv.messages).lower()
                if any(t in conv_text_lower for t in q_tokens):
                    found = True
                    break
            if not found and rel_cids:
                warnings.append(
                    f"Keyword query {qid} tokens not found in any relevant conversation: '{qtext}'"
                )

        elif qtype == "multi_chat":
            if len(rel_cids) < 2:
                warnings.append(
                    f"multi_chat query {qid} has fewer than 2 relevant conversations ({len(rel_cids)}): {rel_cids}"
                )

    # Message length distribution stats
    msg_lens = [len(c.messages) for c in loaded_convs]
    stats = {
        "total_conversations": len(loaded_convs),
        "chatgpt_count": len(cg_data),
        "claude_count": len(cl_data),
        "total_queries": len(queries),
        "min_messages": min(msg_lens) if msg_lens else 0,
        "max_messages": max(msg_lens) if msg_lens else 0,
        "mean_messages": (sum(msg_lens) / len(msg_lens)) if msg_lens else 0.0,
        "long_convs_20plus": sum(1 for l in msg_lens if l >= 20),
        "type_counts": dict(type_counts),
        "type_split_counts": {f"{t}:{s}": c for (t, s), c in type_split_counts.items()},
    }

    return stats, errors, warnings


def build_review_markdown(
    dataset_dir: Path,
    out_path: Path,
    seed: int = 1234,
) -> str:
    """Generate review.md for human spot-checking with top lexical matches and sample queries."""
    path = Path(dataset_dir)
    convs = load_all_exports(path)
    conv_map = {c.conversation_id: c for c in convs}

    queries_file = path / "queries.jsonl"
    queries = [json.loads(line) for line in queries_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Index conversation texts and content words
    conv_texts = {}
    conv_word_sets = {}
    for c in convs:
        text = " ".join(f"{m.role}: {m.text}" for m in c.messages)
        conv_texts[c.conversation_id] = text
        conv_word_sets[c.conversation_id] = content_words(text)

    lines = [
        "# Benchmark Dataset Spot-Check Review",
        "",
        "> [!IMPORTANT]",
        "> Spot check at least 10 percent of labels by hand before trusting any metric.",
        "",
        "## Part 1: No-Answer Queries (Lexical Top-3 Matches)",
        "Confirm that none of the closest lexical matches actually contains the answer to the query.",
        "",
    ]

    no_ans_queries = [q for q in queries if str(q.get("type", "")).startswith("no_answer")]
    for q in no_ans_queries:
        qid = q["qid"]
        qtext = q["query"]
        qtype = q["type"]
        split = q.get("split", "all")
        q_words = content_words(qtext)

        # Score all conversations by token overlap
        scores = []
        for cid, c_words in conv_word_sets.items():
            overlap = len(q_words & c_words)
            scores.append((overlap, cid))

        scores.sort(key=lambda x: x[0], reverse=True)
        top3 = scores[:3]

        lines.append(f"### Query `{qid}` [{qtype} | {split}]")
        lines.append(f"**Query**: *\"{qtext}\"*")
        lines.append("")
        for rank, (score, cid) in enumerate(top3, start=1):
            c = conv_map[cid]
            preview = conv_texts[cid].replace("\n", " ")[:200]
            lines.append(f"{rank}. **[{cid}]** *\"{c.title}\"* (overlap={score} words)")
            lines.append(f"   > {preview}...")
        lines.append("")

    lines.append("## Part 2: Answerable Queries (Random 15 Sample)")
    lines.append("Confirm that the tagged conversations genuinely answer the query.")
    lines.append("")

    ans_queries = [q for q in queries if not str(q.get("type", "")).startswith("no_answer")]
    rng = random.Random(seed)
    sample_size = min(15, len(ans_queries))
    sample = rng.sample(ans_queries, sample_size)

    for q in sample:
        qid = q["qid"]
        qtext = q["query"]
        qtype = q["type"]
        split = q.get("split", "all")
        rel_cids = q.get("relevant", [])

        lines.append(f"### Query `{qid}` [{qtype} | {split}]")
        lines.append(f"**Query**: *\"{qtext}\"*")
        lines.append(f"**Relevant conversations**: `{rel_cids}`")
        lines.append("")
        for cid in rel_cids:
            c = conv_map.get(cid)
            if not c:
                continue
            preview = conv_texts[cid].replace("\n", " ")[:200]
            lines.append(f"- **[{cid}]** *\"{c.title}\"*")
            lines.append(f"  > {preview}...")
        lines.append("")

    md_content = "\n".join(lines)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md_content, encoding="utf-8")
    return md_content


# ============================================================================
# CLI Commands
# ============================================================================

@app.command()
def build(
    raw: Path = typer.Option(Path("data/eval_v2/raw"), "--raw", help="Path to raw directory with batch files."),
    out: Path = typer.Option(Path("data/eval_v2"), "--out", help="Output directory for generated dataset."),
    sample: Optional[Path] = typer.Option(Path("data/sample"), "--sample", help="Sample directory to copy c01..c14 unchanged."),
) -> None:
    """Compile raw batch files into ChatGPT and Claude export JSON, facts.json, and queries.jsonl."""
    typer.echo(f"Building dataset from {raw} into {out}...")
    num_c, num_q = build_dataset(raw_dir=raw, out_dir=out, sample_dir=sample)
    typer.echo(f"Successfully built {num_c} conversations and {num_q} queries in {out}")


@app.command()
def validate(
    dataset: Path = typer.Option(Path("data/eval_v2"), "--dataset", "-d", help="Path to compiled dataset directory."),
) -> None:
    """Validate dataset integrity, round-trip loading, and query constraints."""
    typer.echo(f"Validating dataset in {dataset}...")
    stats, errors, warnings = validate_dataset(dataset_dir=dataset)

    if stats:
        typer.echo("\n--- Dataset Statistics ---")
        typer.echo(f"Total Conversations: {stats['total_conversations']} (ChatGPT: {stats['chatgpt_count']}, Claude: {stats['claude_count']})")
        typer.echo(f"Total Queries: {stats['total_queries']}")
        typer.echo(f"Message Count Distribution: min={stats['min_messages']}, max={stats['max_messages']}, mean={stats['mean_messages']:.1f}")
        typer.echo(f"Conversations with >=20 messages: {stats['long_convs_20plus']}")
        typer.echo("\nQueries by Type:")
        for t, cnt in sorted(stats["type_counts"].items()):
            typer.echo(f"  - {t:<22}: {cnt}")
        typer.echo("\nQueries by Type x Split:")
        for ts, cnt in sorted(stats["type_split_counts"].items()):
            typer.echo(f"  - {ts:<28}: {cnt}")

    if warnings:
        typer.echo(f"\n--- Warnings ({len(warnings)}) ---")
        for w in warnings:
            typer.echo(f"  [WARN] {w}")

    if errors:
        typer.echo(f"\n--- Errors ({len(errors)}) ---", err=True)
        for e in errors:
            typer.echo(f"  [ERROR] {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("\nAll validation checks passed successfully!")


@app.command()
def review(
    dataset: Path = typer.Option(Path("data/eval_v2"), "--dataset", "-d", help="Path to compiled dataset directory."),
    out: Path = typer.Option(Path("data/eval_v2/review.md"), "--out", "-o", help="Path to review markdown output file."),
) -> None:
    """Generate review.md for human spot-checking."""
    typer.echo(f"Generating review document for {dataset} -> {out}...")
    build_review_markdown(dataset_dir=dataset, out_path=out, seed=1234)
    typer.echo(f"Review document written to {out}")


if __name__ == "__main__":
    app()
