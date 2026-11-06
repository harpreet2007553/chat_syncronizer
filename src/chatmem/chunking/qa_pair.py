from chatmem.chunking.per_message import messages, conversations_list
from chatmem.schema import Chunk

# We will now create chunks of QA pairs

def qa_chunks(messages):
    chunks = []
    idx=0
    for i,m in enumerate(messages):
        if(i+1 == len(messages)) : break
        if(m.conversation_id != messages[i+1].conversation_id or m.platform != messages[i+1].platform or m.title != messages[i+1].title): continue
        chunk = Chunk(
            chunk_id= f"chunk_{idx}",
            conversation_id= m.conversation_id,
            title= m.title,
            platform=m.platform,
            text = m.text + " " + messages[i+1].text,
            message_ids=[m.message_id, messages[i+1].message_id],
            time = m.timestamp
        )
        idx += 1
        i += 1
        chunks.append(chunk)
    return chunks

QA_Chunks = qa_chunks(messages)

