import logging
from pathlib import Path
from typing import Sequence

from chatmem.ingest.chatgpt import parse_chatgpt_conversation, parse_chatgpt_export
from chatmem.ingest.claude import parse_claude_conversation, parse_claude_export
from chatmem.schema import Conversation, Message
from chatmem.utils.logging import get_logger

logger = get_logger(__name__)

PARSERS = {
    "chatgpt": parse_chatgpt_export,
    "claude": parse_claude_export,
}


def load_platform_export(platform: str, file_path: str | Path) -> list[Conversation]:
    """Parse an export file for a specific platform."""
    parser = PARSERS.get(platform.lower())
    if not parser:
        raise ValueError(f"Unsupported platform: {platform}. Supported: {list(PARSERS.keys())}")
    return parser(file_path)


def load_all_exports(root_dir: str | Path) -> list[Conversation]:
    """Discover and parse conversation exports across all platforms under root_dir.

    Looks for subdirectories matching known platforms (e.g. 'chatgpt', 'claude')
    containing 'conversations.json'.
    """
    path = Path(root_dir)
    if not path.exists():
        logger.error("Root directory %s does not exist", path)
        return []

    all_conversations: list[Conversation] = []

    for platform_dir in sorted(p for p in path.iterdir() if p.is_dir()):
        platform = platform_dir.name.lower()
        parser = PARSERS.get(platform)
        if parser is None:
            continue

        conv_file = platform_dir / "conversations.json"
        if not conv_file.exists():
            continue

        convs = parser(conv_file)
        all_conversations.extend(convs)

    logger.info("Loaded a total of %d conversations from %s", len(all_conversations), path)
    return all_conversations


def extract_all_messages(conversations: Sequence[Conversation]) -> list[Message]:
    """Extract a flat list of Message objects across all conversations."""
    return [msg for conv in conversations for msg in conv.messages]

