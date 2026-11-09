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


def parse_claude_conversation(conv: dict) -> Conversation | None:
    """Parse a single Claude conversation dictionary into a Conversation object."""
    cid = str(conv.get("uuid") or "")
    if not cid:
        logger.warning("Skipping Claude conversation with missing uuid")
        return None

    title = str(conv.get("name") or "Untitled")
    created_at = conv.get("created_at")
    updated_at = conv.get("updated_at") or created_at

    conv_messages: list[Message] = []
    for item in conv.get("chat_messages", []):
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        sender = item.get("sender", "unknown")
        role = ROLE_MAP.get(sender, sender)
        msg_time = item.get("created_at") or created_at
        msg_id = str(item.get("uuid") or f"{cid}_{len(conv_messages)}")

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
    """Load a Claude export JSON file and parse all conversations."""
    path = Path(file_path)
    if not path.exists():
        logger.error("Claude export file not found: %s", path)
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Expected a list of conversations in %s", path)
        return []

    conversations: list[Conversation] = []
    for item in data:
        conv = parse_claude_conversation(item)
        if conv:
            conversations.append(conv)

    logger.info("Parsed %d Claude conversations from %s", len(conversations), path)
    return conversations

