from chatmem.chunking.per_message import messages, conversations_list
from chatmem.schema import Chunk

# We will now create chunks of QA pairs

def qa_chunks(messages):
    chunks = []
    idx = 0

    for i in range(0, len(messages) - 1, 2):

        m1 = messages[i]
        m2 = messages[i + 1]

        if (
            m1.conversation_id != m2.conversation_id
            or m1.platform != m2.platform
            or m1.title != m2.title
        ):
            continue

        chunks.append(
            Chunk(
                chunk_id=f"chunk_{idx}",
                conversation_id=m1.conversation_id,
                title=m1.title,
                platform=m1.platform,
                text=f"{m1.text} {m2.text}",
                message_ids=[m1.message_id, m2.message_id],
                time=m1.timestamp,
            )
        )

        idx += 1

    return chunks
QA_Chunks = qa_chunks(messages)

