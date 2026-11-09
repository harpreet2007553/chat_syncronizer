import json
import logging
from pathlib import Path
from pydantic import ValidationError

from chatmem.schema import Conversation, Message
from chatmem.utils.logging import get_logger

logger = get_logger(__name__)

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "system": "system",
}


def parse_chatgpt_conversation(conv: dict) -> Conversation | None:
    """Parse a single ChatGPT conversation dictionary from an export into a Conversation object."""
    cid = str(conv.get("conversation_id") or conv.get("id") or "")
    if not cid:
        logger.warning("Skipping ChatGPT conversation with missing ID")
        return None

    title = str(conv.get("title") or "Untitled")
    create_time = conv.get("create_time", 0.0)
    update_time = conv.get("update_time", create_time)

    mapping = conv.get("mapping", {})
    if not isinstance(mapping, dict):
        mapping = {}

    # Walk from current_node up through parent to the root, then reverse for chronological order
    chain = []
    node_id = conv.get("current_node")
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

        parts = (msg.get("content") or {}).get("parts") or []
        text = "\n".join(str(p) for p in parts if isinstance(p, str)).strip()
        ts = msg.get("create_time") or create_time

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
    if not path.exists():
        logger.error("ChatGPT export file not found: %s", path)
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

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

