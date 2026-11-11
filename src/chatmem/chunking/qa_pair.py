from collections import defaultdict
from chatmem.chunking.per_message import conversations_list
from chatmem.schema import Chunk, Conversation, Message


def qa_chunks(
    data: dict[str, Conversation] | list[Conversation] | list[Message] | None = None,
) -> list[Chunk]:
    """Create self-contained Q&A pair chunks from conversation turns.

    Pairs each User question with its corresponding Assistant response.
    """
    if data is None:
        data = conversations_list

    chunks: list[Chunk] = []
    idx = 0

    # Normalize to list of conversations
    if isinstance(data, dict):
        convs = list(data.values())
    elif isinstance(data, (list, tuple)) and data and isinstance(data[0], Conversation):
        convs = list(data)
    elif isinstance(data, (list, tuple)) and data and isinstance(data[0], Message):
        # Group flat messages by (conversation_id, platform)
        grouped = defaultdict(list)
        for m in data:
            grouped[(m.conversation_id, m.platform)].append(m)
        convs = []
        for (cid, platform), msgs in grouped.items():
            title = msgs[0].title if getattr(msgs[0], "title", None) else "Untitled"
            convs.append(
                Conversation(
                    conversation_id=cid,
                    platform=platform,
                    title=title,
                    creation_time=msgs[0].timestamp,
                    updation_time=msgs[-1].timestamp,
                    messages=msgs,
                )
            )
    else:
        convs = []

    for conv in convs:
        msgs = conv.messages
        i = 0
        while i < len(msgs):
            # Find user turn
            if msgs[i].role == "user":
                # Pair with assistant turn if followed immediately
                if i + 1 < len(msgs) and msgs[i + 1].role == "assistant":
                    user_msg = msgs[i]
                    asst_msg = msgs[i + 1]

                    chunk = Chunk(
                        chunk_id=f"chunk_qa_{idx}",
                        conversation_id=conv.conversation_id,
                        title=conv.title or "Untitled",
                        platform=conv.platform,
                        text=f"User: {user_msg.text}\n\nAssistant: {asst_msg.text}",
                        message_ids=[user_msg.message_id, asst_msg.message_id],
                        time=user_msg.timestamp,
                    )
                    chunks.append(chunk)
                    idx += 1
                    i += 2
                    continue
                else:
                    # Lone user question without assistant reply
                    user_msg = msgs[i]
                    chunk = Chunk(
                        chunk_id=f"chunk_qa_{idx}",
                        conversation_id=conv.conversation_id,
                        title=conv.title or "Untitled",
                        platform=conv.platform,
                        text=f"User: {user_msg.text}",
                        message_ids=[user_msg.message_id],
                        time=user_msg.timestamp,
                    )
                    chunks.append(chunk)
                    idx += 1
                    i += 1
                    continue
            i += 1

    return chunks


QA_Chunks: list[Chunk] = qa_chunks(conversations_list)


if __name__ == "__main__":
    print(f"Generated {len(QA_Chunks)} QA chunks.")
    if QA_Chunks:
        sample = QA_Chunks[0]
        print(f"Sample ID: {sample.chunk_id}")
        print(f"Title: {sample.title}")
        print(f"Text:\n{sample.text}")
