from chatmem.schema import Chunk, Conversation


def window_chunks(
    conversations: list[Conversation],
    window_size: int = 3,
    overlap: int = 1,
) -> list[Chunk]:
    """Slide a window of `window_size` messages over each conversation,
    moving forward by (window_size - overlap) messages each step."""
    if not 0 <= overlap < window_size:
        raise ValueError("overlap must be >= 0 and smaller than window_size")

    stride = window_size - overlap
    chunks: list[Chunk] = []

    for conv in conversations:
        msgs = conv.messages
        n = len(msgs)
        start, k = 0, 0

        while start < n:
            end = min(start + window_size, n)
            window = msgs[start:end]

            chunks.append(Chunk(
                chunk_id=f"{conv.platform}:{conv.conversation_id}:w{k}",
                conversation_id=conv.conversation_id,
                platform=conv.platform,
                title=conv.title,
                text="\n".join(f"{m.role}: {m.text}" for m in window),
                message_ids=[m.message_id for m in window],
                time=window[0].timestamp,
            ))

            if end == n:     
                break
            start += stride
            k += 1

    return chunks