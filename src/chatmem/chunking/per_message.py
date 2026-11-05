# import json
# from chatmem.schema import Chunk, Message, Conversation
# import os
# import logging
# from pathlib import Path



# logger = logging.getLogger(__name__)




# messages = []
# conversations = []

# def load_messages(dir):
#     count = 0
#     for f in dir:
#         if count == 1 : break
#         count += 1
#         chat_data = load_json(f"{DATA_DIR}/{f}/conversations.json")
#         if not chat_data:
#             logger.error("chat data is not loaded. Mostly problem in load_json function or in path given inside load_json")
#         for conv in chat_data:
#             messages_list = []
#             if conv["mapping"]:
#                 for i,val in conv["mapping"].items():
#                     if not val["message"]: continue
#                     m = Message(
#                         message_id= i,
#                         conversation_id=conv["id"],
#                         platform=f,
#                         role=val["message"]["author"]["role"] or "unknown",
#                         text=val["message"]["content"]["parts"][0],
#                         timestamp=val["message"]["create_time"]
#                     )
#                     messages_list.append(m)
#             c = Conversation(
#                 conversation_id=conv["id"], 
#                 platform=f, 
#                 messages=messages_list,
#                 title=conv["title"],
#                 creation_time=conv["create_time"],
#                 updation_time=conv["update_time"]
#                 )
#             conversations.append(c)     

# load_messages(dir)
# print(conversations)


from pydantic_core.core_schema import DataclassArgsSchema
from zoneinfo._common import load_data
import logging
from pathlib import Path
from pydantic import ValidationError
from chatmem.schema import Conversation, Message, Chunk
import os
import json 


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]   # chunking -> chatmem -> src -> project root
DATA_DIR = PROJECT_ROOT / "data" / "sample"

dir = os.listdir(DATA_DIR)
if not dir:
    logger.error("The given path is invalid or No files exist in the directory")

def load_json(file_path: str | Path) -> dict | list:
    path = Path(f"{file_path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        return data


ROLE_MAP = {"human": "user", "user": "user", "assistant": "assistant", "system": "system"}

messages = []
def parse_chatgpt(conv: dict) -> Conversation:
    cid = conv.get("conversation_id") or conv["id"]
    mapping = conv["mapping"]

    # walk from the last node to the root, then reverse (gives the real conversation order)
    chain, node_id = [], conv["current_node"]
    while node_id:
        node = mapping[node_id]
        chain.append(node)
        node_id = node["parent"]
    chain.reverse()
    
    for node in chain:
        msg = node.get("message")
        if not msg:
            continue
        parts = (msg.get("content") or {}).get("parts") or []
        text = "\n".join(p for p in parts if isinstance(p, str)).strip()
        ts = msg.get("create_time")
        if not text or ts is None:
            continue
        messages.append(Message(
            message_id=node["id"],
            conversation_id=cid,
            platform="chatgpt",
            role=ROLE_MAP.get(msg["author"]["role"], "unknown"),
            title= conv.get("title"),
            text=text,
            timestamp=ts,
        ))

    return Conversation(
        conversation_id=cid,
        platform="chatgpt",
        title=conv.get("title"),
        creation_time=conv["create_time"],
        updation_time=conv["update_time"],
        messages=messages,
    )


def parse_claude(conv: dict) -> Conversation:
    cid = conv["uuid"]
    for m in conv["chat_messages"]:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        messages.append(Message(
            message_id=m["uuid"],
            conversation_id=cid,
            platform="claude",
            role=ROLE_MAP.get(m["sender"], "unknown"),
            title=m["name"],
            text=text,
            timestamp=m["created_at"],
        ))

    return Conversation(
        conversation_id=cid,
        platform="claude",
        title=conv.get("name"),
        creation_time=conv["created_at"],
        updation_time=conv["updated_at"],
        messages=messages,
    )


PARSERS = {"chatgpt": parse_chatgpt, "claude": parse_claude}


def load_conversations(root: Path) -> list[Conversation]:
    conversations = []
    for platform_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        platform = platform_dir.name
        parser = PARSERS.get(platform)
        if parser is None:
            logger.warning("No parser for platform folder: %s", platform)
            continue

        raw = load_json(platform_dir / "conversations.json")
        for conv in raw:
            try:
                conversations.append(parser(conv))
            except (KeyError, ValidationError):
                logger.exception("Skipping a malformed %s conversation", platform)

        logger.info("Loaded %s conversations from %s", len(raw), platform)
    return conversations

conversations_list = {}
for f in dir:
    c = load_conversations(Path(f"{DATA_DIR}"))
    for i in c:
        conversations_list[f"{i.conversation_id}"] = i

print(len(conversations_list))
print(len(messages))

chunks = []
index = 0
for m in messages:
    chunk = Chunk(
        chunk_id = f"chunk_{index}",
        conversation_id = m.conversation_id,
        platform = m.platform,
        title = m.title,
        text = m.text,
        message_ids = m.message_id,
        time = m.timestamp 
    )
    chunks.append(chunk)
