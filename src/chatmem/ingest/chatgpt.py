from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from pydantic import ValidationError

from chatmem.schema import Conversation, Message
from chatmem.utils.logging import get_logger

logger = get_logger(__name__)

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "tool": "system",
}


def safe_datetime(val: Any, default: datetime | None = None) -> datetime:
    """Safely parse datetime, ISO string, or numeric epoch timestamp into UTC datetime."""
    if val is None:
        return default or datetime.now(timezone.utc)
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            return default or datetime.now(timezone.utc)
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return default or datetime.now(timezone.utc)
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except ValueError:
            pass
    return default or datetime.now(timezone.utc)


def parse_chatgpt_conversation(conv: dict) -> Conversation | None:
    """Parse a single ChatGPT conversation dictionary from an export into a Conversation object."""
    cid = str(conv.get("conversation_id") or conv.get("id") or "")
    if not cid:
        logger.warning("Skipping ChatGPT conversation with missing ID")
        return None

    title = str(conv.get("title") or "Untitled")
    create_time = safe_datetime(conv.get("create_time"))
    update_time = safe_datetime(conv.get("update_time"), default=create_time)

    mapping = conv.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}

    node_id = conv.get("current_node")
    if not node_id or node_id not in mapping:
        # Fallback: select leaf node with latest message timestamp or no children
        leaves = [
            nid for nid, n in mapping.items()
            if isinstance(n, dict) and not n.get("children")
        ]
        if leaves:
            def _leaf_ts(nid: str) -> float:
                msg = mapping[nid].get("message") or {}
                return float(msg.get("create_time") or 0.0)
            node_id = max(leaves, key=_leaf_ts)
        elif mapping:
            node_id = next(iter(mapping.keys()))

    # Walk from current_node up through parent to the root, then reverse for chronological order
    chain = []
    visited = set()

    while node_id and node_id in mapping and node_id not in visited:
        visited.add(node_id)
        node = mapping[node_id]
        chain.append(node)
        node_id = node.get("parent")

    chain.reverse()

    conv_messages: list[Message] = []
    for node in chain:
        msg = node.get("message")
        if not msg:
            continue

        content = msg.get("content") or {}
        text_parts: list[str] = []
        if isinstance(content, dict):
            parts = content.get("parts") or []
            for p in parts:
                if isinstance(p, str) and p.strip():
                    text_parts.append(p.strip())
                elif isinstance(p, dict):
                    # Multimodal or structured part with text
                    p_text = p.get("text")
                    if isinstance(p_text, str) and p_text.strip():
                        text_parts.append(p_text.strip())
            # Fallback to direct 'text' field (e.g. code interpreter or execution output)
            if not text_parts and content.get("text"):
                c_text = content.get("text")
                if isinstance(c_text, str) and c_text.strip():
                    text_parts.append(c_text.strip())
        elif isinstance(content, str) and content.strip():
            text_parts.append(content.strip())

        text = "\n".join(text_parts).strip()
        ts = safe_datetime(msg.get("create_time"), default=create_time)

        if not text:
            continue

        author = msg.get("author") or {}
        role_raw = author.get("role", "unknown")
        role = ROLE_MAP.get(role_raw, role_raw)

        try:
            m = Message(
                message_id=str(node.get("id") or msg.get("id")),
                conversation_id=cid,
                platform="chatgpt",
                role=role,
                title=title,
                text=text,
                timestamp=ts,
            )
            conv_messages.append(m)
        except ValidationError as err:
            logger.warning("Skipping invalid ChatGPT message in %s: %s", cid, err)

    try:
        return Conversation(
            conversation_id=cid,
            platform="chatgpt",
            title=title,
            creation_time=create_time,
            updation_time=update_time,
            messages=conv_messages,
        )
    except ValidationError as err:
        logger.warning("Skipping malformed ChatGPT conversation %s: %s", cid, err)
        return None


def parse_chatgpt_export(file_path: str | Path) -> list[Conversation]:
    """Load a ChatGPT export JSON file and parse all conversations."""
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        logger.warning("ChatGPT export file not found or empty: %s", path)
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        logger.error("Failed to decode JSON from %s: %s", path, err)
        return []

    if isinstance(data, dict):
        if isinstance(data.get("value"), list):
            data = data["value"]
        elif isinstance(data.get("conversations"), list):
            data = data["conversations"]
        elif "current_node" in data or "mapping" in data:
            data = [data]

    if not isinstance(data, list):
        logger.error("Expected a list of conversations in %s", path)
        return []

    conversations: list[Conversation] = []
    for item in data:
        conv = parse_chatgpt_conversation(item)
        if conv:
            conversations.append(conv)

    logger.info("Parsed %d ChatGPT conversations from %s", len(conversations), path)
    return conversations

