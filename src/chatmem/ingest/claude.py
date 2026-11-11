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


def parse_claude_conversation(conv: dict) -> Conversation | None:
    """Parse a single Claude conversation dictionary into a Conversation object."""
    cid = str(conv.get("uuid") or conv.get("id") or conv.get("conversation_id") or "")
    if not cid:
        logger.warning("Skipping Claude conversation with missing uuid/id")
        return None

    title = str(conv.get("name") or conv.get("title") or conv.get("summary") or "Untitled")
    created_at = safe_datetime(conv.get("created_at") or conv.get("create_time"))
    updated_at = safe_datetime(conv.get("updated_at") or conv.get("update_time"), default=created_at)

    conv_messages: list[Message] = []
    messages_raw = conv.get("chat_messages") or conv.get("messages") or []

    for item in messages_raw:
        text = str(item.get("text") or "").strip()

        # Fallback 1: Extract from structured content blocks (e.g. Claude 3/3.5 content array)
        if not text and item.get("content"):
            content = item.get("content")
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts: list[str] = []
                for b in content:
                    if isinstance(b, dict):
                        b_type = b.get("type")
                        if b_type == "text" and b.get("text"):
                            parts.append(str(b["text"]))
                    elif isinstance(b, str):
                        parts.append(b)
                text = "\n".join(p for p in parts if p).strip()

        # Fallback 2: Extract from attachments or files if message only contained attachments
        if not text and item.get("attachments"):
            parts = []
            for att in item.get("attachments", []):
                if isinstance(att, dict):
                    fn = att.get("file_name") or "file"
                    ec = att.get("extracted_content")
                    if ec:
                        parts.append(f"[File: {fn}]\n{ec}")
                    else:
                        parts.append(f"[File: {fn}]")
            text = "\n\n".join(parts).strip()

        if not text and item.get("files"):
            parts = []
            for f_item in item.get("files", []):
                if isinstance(f_item, dict):
                    fn = f_item.get("file_name") or "file"
                    parts.append(f"[File: {fn}]")
            text = "\n".join(parts).strip()

        if not text:
            continue

        sender = item.get("sender") or item.get("role") or (item.get("author") or {}).get("role", "unknown")
        role = ROLE_MAP.get(str(sender).lower(), str(sender).lower())
        msg_time = safe_datetime(
            item.get("created_at") or item.get("timestamp") or item.get("create_time"),
            default=created_at,
        )
        msg_id = str(item.get("uuid") or item.get("id") or item.get("message_id") or f"{cid}_{len(conv_messages)}")

        try:
            m = Message(
                message_id=msg_id,
                conversation_id=cid,
                platform="claude",
                role=role,
                title=title,
                text=text,
                timestamp=msg_time,
            )
            conv_messages.append(m)
        except ValidationError as err:
            logger.warning("Skipping invalid Claude message in %s: %s", cid, err)

    try:
        return Conversation(
            conversation_id=cid,
            platform="claude",
            title=title,
            creation_time=created_at,
            updation_time=updated_at,
            messages=conv_messages,
        )
    except ValidationError as err:
        logger.warning("Skipping malformed Claude conversation %s: %s", cid, err)
        return None


def parse_claude_export(file_path: str | Path) -> list[Conversation]:
    """Load a Claude export JSON file and parse all conversations.

    Handles both:
    1. A flat JSON list of conversation dicts
    2. A wrapped JSON object with a 'value', 'conversations', or 'chats' list (standard user export format)
    """
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        logger.warning("Claude export file not found or empty: %s", path)
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        logger.error("Failed to decode JSON from %s: %s", path, err)
        return []

    # Handle dictionary-wrapped exports (e.g. {"Count": 11, "value": [...]})
    if isinstance(data, dict):
        if isinstance(data.get("value"), list):
            data = data["value"]
        elif isinstance(data.get("conversations"), list):
            data = data["conversations"]
        elif isinstance(data.get("chats"), list):
            data = data["chats"]
        elif "uuid" in data or "chat_messages" in data:
            data = [data]
        else:
            logger.error("Expected a list or recognized dictionary wrapper in %s", path)
            return []

    if not isinstance(data, list):
        logger.error("Expected a list of conversations in %s", path)
        return []

    conversations: list[Conversation] = []
    for item in data:
        if isinstance(item, dict):
            conv = parse_claude_conversation(item)
            if conv:
                conversations.append(conv)

    logger.info("Parsed %d Claude conversations from %s", len(conversations), path)
    return conversations

