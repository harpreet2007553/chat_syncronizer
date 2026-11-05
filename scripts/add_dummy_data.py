"""Generate dummy chat exports (ChatGPT-style and Claude-style) and a sample eval set.

Usage:
    python scripts/gen_dummy_data.py            # writes into data/sample and data/eval
    python scripts/gen_dummy_data.py data/raw   # writes into data/raw instead

All content is synthetic. Formats mimic the real exports closely enough to
exercise your parsers (tree-based `mapping` for ChatGPT, flat `chat_messages`
for Claude), but verify against your own real export before relying on them.
"""
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/sample")
EVAL_OUT = Path("./data/eval")
BASE = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)

# (conv_id, platform, title, days_offset, [(role, text), ...])
CONVS = [
    ("c01", "chatgpt", "Fix KeyError in pandas groupby", 0, [
        ("user", "I get KeyError: 'user_id' when I run df.groupby('user_id').agg(...). The column exists when I print df.columns."),
        ("assistant", "That usually means the column name has hidden whitespace or you already set it as the index. Check df.columns.tolist() for 'user_id ' with a trailing space, or call df.reset_index() before groupby."),
        ("user", "Yes, it was a trailing space from the CSV header. Is there a clean way to prevent this?"),
        ("assistant", "Normalize on load: df.columns = df.columns.str.strip().str.lower(). You can also pass skipinitialspace=True to pd.read_csv."),
    ]),
    ("c02", "chatgpt", "PyTorch CUDA out of memory", 2, [
        ("user", "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB while fine-tuning a 7B model on my laptop GPU."),
        ("assistant", "Reduce memory in this order: lower the batch size to 1 and use gradient accumulation, enable gradient checkpointing, load the model in 4-bit with bitsandbytes (QLoRA), and shorten max sequence length."),
        ("user", "I only have 8 GB VRAM. Will QLoRA fit a 7B model?"),
        ("assistant", "Often yes with 4-bit weights, LoRA rank 8-16, batch size 1 and a sequence length around 512. Expect slow training. A 3B model is more comfortable at 8 GB."),
    ]),
    ("c03", "claude", "Docker compose for Qdrant", 3, [
        ("user", "Give me a docker-compose.yml to run Qdrant locally with persistent storage."),
        ("assistant", "Use image qdrant/qdrant, expose ports 6333 and 6334, and mount a named volume to /qdrant/storage so data survives restarts. Add restart: unless-stopped."),
        ("user", "How do I check it is running?"),
        ("assistant", "Open http://localhost:6333/dashboard or curl http://localhost:6333/collections. You should get a JSON response with an empty collections list."),
    ]),
    ("c04", "claude", "Reciprocal rank fusion explained", 5, [
        ("user", "How does reciprocal rank fusion combine BM25 and vector search results?"),
        ("assistant", "RRF gives each document a score of 1/(k + rank) in every ranked list and sums them. The constant k, commonly 60, dampens the influence of top ranks. It needs only ranks, so you do not have to normalize BM25 and cosine scores."),
        ("user", "Why not just weight and add the normalized scores?"),
        ("assistant", "You can, but score distributions differ across retrievers and shift per query, so normalization is brittle. RRF is a robust default; tune a weighted sum only if you have eval data to justify it."),
    ]),
    ("c05", "chatgpt", "Paneer butter masala recipe", 6, [
        ("user", "Simple paneer butter masala recipe for 3 people?"),
        ("assistant", "Saute onion and tomato with cashews, blend into a puree, cook it with butter, kashmiri chilli, garam masala and a little cream, then add 250 g paneer cubes and simmer 5 minutes. Finish with kasuri methi."),
        ("user", "Can I skip the cashews?"),
        ("assistant", "Yes. Use a spoon of fresh cream or a few melon seeds for the same richness."),
    ]),
    ("c06", "claude", "Delhi to Jaipur weekend trip", 8, [
        ("user", "Plan a 2 day weekend trip from Delhi to Jaipur on a student budget."),
        ("assistant", "Take an early morning train to Jaipur. Day 1: Amer Fort, Jal Mahal, Hawa Mahal and the old city markets. Day 2: City Palace, Jantar Mantar and Nahargarh sunset. Stay in a hostel near Bani Park."),
        ("user", "What is the cheapest way back?"),
        ("assistant", "A night bus or a sleeper-class train on Sunday evening, booked a few days early."),
    ]),
    ("c07", "chatgpt", "Resume bullets for an ML project", 10, [
        ("user", "Help me write resume bullets for my pollution severity classification project."),
        ("assistant", "Use action plus metric: 'Built a gradient-boosted classifier for air pollution severity, improving macro-F1 from 0.61 to 0.78 over a logistic regression baseline.' Add data size, validation method and one deployment detail."),
        ("user", "I do not have deployment yet."),
        ("assistant", "Then say what you did do: 'Designed stratified cross-validation and error analysis across 5 severity classes.' Do not claim deployment you have not done."),
    ]),
    ("c08", "claude", "FastAPI CORS error from Chrome extension", 12, [
        ("user", "My extension fetch to my FastAPI server fails: Access to fetch at 'http://localhost:8000/ingest' from origin 'chrome-extension://abcd' has been blocked by CORS policy."),
        ("assistant", "Add CORSMiddleware to FastAPI and allow the extension origin: app.add_middleware(CORSMiddleware, allow_origins=['chrome-extension://<your-id>'], allow_methods=['*'], allow_headers=['*']). Also list localhost:8000 under host_permissions in manifest.json."),
        ("user", "It works from the background script but not from the content script."),
        ("assistant", "Content scripts run under the page origin, so route the request through the background service worker with chrome.runtime.sendMessage and fetch from there."),
    ]),
    ("c09", "chatgpt", "Chunking strategy for chat logs", 14, [
        ("user", "How should I chunk chat history for RAG? Fixed size feels wrong."),
        ("assistant", "Chat data has natural units. Try a question-answer pair as one chunk, or a sliding window of 3 turns with 1 turn of overlap. Keep metadata such as conversation id, timestamp and role so you can cite and filter."),
        ("user", "What about very long assistant answers with code?"),
        ("assistant", "Split long answers on code-block and paragraph boundaries, and prepend the user question to each piece so every chunk stays self-contained."),
    ]),
    ("c10", "claude", "Binary search edge cases", 15, [
        ("user", "Why does my binary search loop forever when low=0 and high=1?"),
        ("assistant", "If you use while low < high with mid = (low + high) // 2 and set low = mid, mid never advances. Use low = mid + 1 when the target is greater, or use the while low <= high form with mid +/- 1 updates."),
    ]),
    ("c11", "chatgpt", "Git rebase vs merge", 17, [
        ("user", "When should I use git rebase instead of merge?"),
        ("assistant", "Rebase your local feature branch onto main to keep a linear history before opening a PR. Use merge for shared branches, because rebasing rewrites commits and breaks collaborators' history."),
    ]),
    ("c12", "claude", "uv versus pip for Python projects", 19, [
        ("user", "Should I use uv or pip with venv for a new project?"),
        ("assistant", "uv is much faster and manages Python versions, the lockfile and the virtualenv together. Use uv init, uv add and uv run. Commit pyproject.toml and uv.lock, and keep .venv out of git."),
        ("user", "How do I pin the Python version?"),
        ("assistant", "Run uv python pin 3.12, which writes a .python-version file, then uv venv uses it automatically."),
    ]),
    ("c13", "chatgpt", "Guitar chords for beginners", 21, [
        ("user", "Which guitar chords should a beginner learn first?"),
        ("assistant", "Start with Em, Am, C, G and D. Many simple songs use only these. Practice clean changes between two chords for a few minutes daily."),
    ]),
    ("c14", "claude", "Choosing an embedding model", 23, [
        ("user", "bge-small or e5-base for retrieving my own chat history? I have a modest GPU."),
        ("assistant", "bge-small is fast and runs well on CPU, so it is a good baseline. e5-base is larger and often slightly better, but you must prefix inputs with 'query: ' and 'passage: '. Decide with your own eval set, measuring recall@5 and latency."),
        ("user", "Does a reranker make this choice less important?"),
        ("assistant", "Partly. A cross-encoder reranker over the top 30 candidates recovers many first-stage misses, but first-stage recall still caps the final result."),
    ]),
]


def ts(dt):
    return dt.timestamp()


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def chatgpt_export(convs):
    out = []
    for cid, _, title, day, msgs in convs:
        start = BASE + timedelta(days=day)
        root_id = str(uuid.uuid4())
        mapping = {root_id: {"id": root_id, "message": None, "parent": None, "children": []}}
        parent, last = root_id, root_id
        for i, (role, text) in enumerate(msgs):
            nid = str(uuid.uuid4())
            when = start + timedelta(minutes=2 * i)
            mapping[nid] = {
                "id": nid,
                "message": {
                    "id": nid,
                    "author": {"role": role, "metadata": {}},
                    "create_time": ts(when),
                    "content": {"content_type": "text", "parts": [text]},
                    "metadata": {},
                },
                "parent": parent,
                "children": [],
            }
            mapping[parent]["children"].append(nid)
            parent = last = nid
        out.append({
            "id": cid,
            "conversation_id": cid,
            "title": title,
            "create_time": ts(start),
            "update_time": ts(start + timedelta(minutes=2 * len(msgs))),
            "mapping": mapping,
            "current_node": last,
        })
    return out


def claude_export(convs):
    out = []
    for cid, _, title, day, msgs in convs:
        start = BASE + timedelta(days=day)
        chat_messages = []
        for i, (role, text) in enumerate(msgs):
            when = start + timedelta(minutes=2 * i)
            chat_messages.append({
                "uuid": str(uuid.uuid4()),
                "text": text,
                "content": [{"type": "text", "text": text}],
                "sender": "human" if role == "user" else "assistant",
                "created_at": iso(when),
                "updated_at": iso(when),
                "attachments": [],
                "files": [],
            })
        out.append({
            "uuid": cid,
            "name": title,
            "created_at": iso(start),
            "updated_at": iso(start + timedelta(minutes=2 * len(msgs))),
            "chat_messages": chat_messages,
        })
    return out


# Sample labeled queries. `relevant` lists conversation ids that should be retrieved;
# an empty list means retrieval should return nothing (tests the relevance gate).
QUERIES = [
    {"qid": "q01", "query": "why does groupby say KeyError even though the column is there", "relevant": ["c01"], "type": "paraphrase"},
    {"qid": "q02", "query": "CUDA out of memory Tried to allocate 2.00 GiB", "relevant": ["c02"], "type": "exact_string"},
    {"qid": "q03", "query": "how to run a vector database locally with docker and keep the data", "relevant": ["c03"], "type": "paraphrase"},
    {"qid": "q04", "query": "what is the k constant in RRF", "relevant": ["c04"], "type": "keyword"},
    {"qid": "q05", "query": "my extension cannot call localhost because of CORS from the content script", "relevant": ["c08"], "type": "paraphrase"},
    {"qid": "q06", "query": "how should I split long chat answers that contain code", "relevant": ["c09"], "type": "paraphrase"},
    {"qid": "q07", "query": "which embedding model and how do I choose between small and base", "relevant": ["c14"], "type": "paraphrase"},
    {"qid": "q08", "query": "set up python version and virtualenv with uv", "relevant": ["c12"], "type": "keyword"},
    {"qid": "q09", "query": "how to build a retrieval pipeline over my chat history", "relevant": ["c04", "c09", "c14"], "type": "multi_chat"},
    {"qid": "q10", "query": "best way to learn Spanish quickly", "relevant": [], "type": "no_answer"},
    {"qid": "q11", "query": "how do I deploy a kubernetes cluster on AWS", "relevant": [], "type": "no_answer"},
]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    chatgpt = [c for c in CONVS if c[1] == "chatgpt"]
    claude = [c for c in CONVS if c[1] == "claude"]
    write_json(OUT / "chatgpt" / "conversations.json", chatgpt_export(chatgpt))
    write_json(OUT / "claude" / "conversations.json", claude_export(claude))
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT / "queries.sample.jsonl", "w", encoding="utf-8") as f:
        for q in QUERIES:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"ChatGPT convs: {len(chatgpt)}, Claude convs: {len(claude)}, queries: {len(QUERIES)}")
    print(f"Wrote to {OUT} and {EVAL_OUT}")


if __name__ == "__main__":
    main()