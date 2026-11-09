import logging
from collections import defaultdict
from datetime import datetime
from typing import Sequence

from chatmem.schema import Chunk, Conversation, Message

logger = logging.getLogger(__name__)


def window_chunks(
    data: Sequence[Conversation] | dict[str, Conversation] | Sequence[Message] | None = None,
    window_size: int = 3,
    overlap: int = 1,
) -> list[Chunk]:
    """Slide a window of `window_size` messages over conversation turns,
    advancing by (window_size - overlap) messages at each step.

    Supports:
        - A list of Conversation objects
        - A dictionary of {conversation_id: Conversation}
        - A flat list of Message objects
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if not (0 <= overlap < window_size):
        raise ValueError("overlap must be >= 0 and strictly less than window_size")

    stride = window_size - overlap
    chunks: list[Chunk] = []
    chunk_idx = 0

    # 1. Normalize input into groups of messages per conversation:
    # conv_groups: list of tuple(conv_id, platform, title, list[Message])
    conv_groups: list[tuple[str, str, str, list[Message]]] = []

    if data is None:
        # Fallback to importing conversations_list from per_message if available
        try:
            from chatmem.chunking.per_message import conversations_list
            data = conversations_list
        except Exception as e:
            logger.warning("Could not auto-load conversations: %s", e)
            return []

    if isinstance(data, dict):
        conversations = list(data.values())
        for conv in conversations:
            # Ensure messages belong to this conversation
            msgs = [m for m in conv.messages if m.conversation_id == conv.conversation_id]
            # Fallback if conv.messages wasn't pre-filtered
            if not msgs and conv.messages:
                msgs = conv.messages
            conv_groups.append((conv.conversation_id, conv.platform, conv.title or "Untitled", msgs))

    elif isinstance(data, (list, tuple)) and data and isinstance(data[0], Conversation):
        for conv in data:
            msgs = [m for m in conv.messages if m.conversation_id == conv.conversation_id]
            if not msgs and conv.messages:
                msgs = conv.messages
            conv_groups.append((conv.conversation_id, conv.platform, conv.title or "Untitled", msgs))

    elif isinstance(data, (list, tuple)) and data and isinstance(data[0], Message):
        # Group flat messages by (conversation_id, platform)
        grouped: dict[tuple[str, str], list[Message]] = defaultdict(list)
        for msg in data:
            grouped[(msg.conversation_id, msg.platform)].append(msg)
        for (cid, platform), msgs in grouped.items():
            title = msgs[0].title if getattr(msgs[0], "title", None) else "Untitled"
            conv_groups.append((cid, platform, title, msgs))

    # 2. Generate sliding windows for each conversation
    for cid, platform, title, msgs in conv_groups:
        n = len(msgs)
        if n == 0:
            continue

        start = 0
        window_num = 0
        while start < n:
            end = min(start + window_size, n)
            window = msgs[start:end]

            text_parts = []
            for m in window:
                role_label = (m.role or "unknown").capitalize()
                text_parts.append(f"{role_label}: {m.text}")
            formatted_text = "\n\n".join(text_parts)

            chunk = Chunk(
                chunk_id=f"chunk_sw_{chunk_idx}",
                conversation_id=cid,
                platform=platform,
                title=title,
                text=formatted_text,
                message_ids=[m.message_id for m in window],
                time=window[0].timestamp,
            )
            chunks.append(chunk)
            chunk_idx += 1
            window_num += 1

            if end >= n:
                break
            start += stride

    return chunks


# Build sliding window chunks using available sample data
try:
    from chatmem.chunking.per_message import conversations_list
    Window_Chunks = window_chunks(conversations_list)
except Exception:
    Window_Chunks = []


if __name__ == "__main__":
    print(f"Generated {len(Window_Chunks)} sliding window chunks.")
    if Window_Chunks:
        sample = Window_Chunks[0]
        print(f"Sample Chunk ID: {sample.chunk_id}")
        print(f"Conversation ID: {sample.conversation_id} ({sample.platform})")
        print(f"Title: {sample.title}")
        print(f"Message IDs: {sample.message_ids}")
        print(f"Text:\n{sample.text}")



# Build sliding window chunks using available sample data
try:
    from chatmem.chunking.per_message import conversations_list
    Window_Chunks = window_chunks(conversations_list)
except Exception:
    Window_Chunks = []


if __name__ == "__main__":
    print(f"Generated {len(Window_Chunks)} sliding window chunks.")
    if Window_Chunks:
        sample = Window_Chunks[0]
        print(f"Sample Chunk ID: {sample.chunk_id}")
        print(f"Conversation ID: {sample.conversation_id} ({sample.platform})")
        print(f"Title: {sample.title}")
        print(f"Message IDs: {sample.message_ids}")
        print(f"Text:\n{sample.text}")