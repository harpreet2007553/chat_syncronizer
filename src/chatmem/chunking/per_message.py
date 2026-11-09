import logging
from pathlib import Path
from typing import Sequence

from chatmem.ingest import extract_all_messages, load_all_exports
from chatmem.schema import Chunk, Conversation, Message
from chatmem.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample"


def chunk_per_message(
    data: Sequence[Conversation] | Sequence[Message] | dict[str, Conversation] | None = None,
) -> list[Chunk]:
    """Convert conversational messages into individual Chunk objects.

    Strategy: 1 message = 1 chunk.
    """
    if data is None:
        data = messages

    if isinstance(data, dict):
        # Extract messages from conversations dict
        messages_list = [m for c in data.values() for m in c.messages]
    elif isinstance(data, (list, tuple)) and data and isinstance(data[0], Conversation):
        messages_list = [m for c in data for m in c.messages]
    else:
        messages_list = list(data)

    per_message_chunks: list[Chunk] = []
    for idx, m in enumerate(messages_list):
        chunk = Chunk(
            chunk_id=f"chunk_pm_{idx}",
            conversation_id=m.conversation_id,
            platform=m.platform,
            title=m.title or "Untitled",
            text=f"{m.role.capitalize()}: {m.text}",
            message_ids=[m.message_id],
            time=m.timestamp,
        )
        per_message_chunks.append(chunk)

    logger.debug("Generated %d per-message chunks", len(per_message_chunks))
    return per_message_chunks


# Load sample data once using the official ingest module
loaded_convs = load_all_exports(DEFAULT_DATA_DIR)
conversations_list: dict[str, Conversation] = {
    c.conversation_id: c for c in loaded_convs
}
messages: list[Message] = extract_all_messages(loaded_convs)
chunks: list[Chunk] = chunk_per_message(messages)


if __name__ == "__main__":
    print(f"Loaded conversations: {len(conversations_list)}")
    print(f"Extracted messages: {len(messages)}")
    print(f"Generated per-message chunks: {len(chunks)}")
    if chunks:
        sample = chunks[0]
        print(f"\nSample Chunk [{sample.chunk_id}]:")
        print(f"  Title: {sample.title} ({sample.platform})")
        print(f"  Text: {sample.text}")
