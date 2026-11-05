# chatmem: cross-chatbot memory with hybrid-search RAG

> **For AI coding agents:** read this file fully before changing anything. The "Status" and "Conventions" sections tell you what exists, what does not, and how to work in this repo.

## What this project is

`chatmem` gives a user one private memory across all their AI chatbots (ChatGPT, Claude, Grok, Gemini, etc.). It ingests exported or captured chat history, indexes it, and, **only when a new prompt would benefit from past context**, retrieves the most relevant earlier conversations with a hybrid-search RAG pipeline and returns them with citations.

Built-in chatbot memory is siloed per vendor. chatmem is vendor-neutral and local-first.

The project has two goals, in this priority order:

1. **A production-style RAG system with real evaluation.** Retrieval quality must be *measured*, not assumed. This is the main deliverable.
2. **A browser extension** that captures chats and injects retrieved context into the prompt box. This is the last layer and is built after the retrieval core and the eval harness work.

## Core pipeline

```
raw exports -> parse -> normalize (Message schema) -> chunk -> index
                                                              |
query -> BM25 top-N  \                                        v
query -> dense top-N  -> RRF fusion -> cross-encoder rerank -> relevance gate -> results + citations
```

- **Parsing:** one parser per platform, all outputting the same normalized schema.
- **Chunking:** multiple strategies, compared by experiment: per-message, sliding window over turns (3 turns, 1 overlap), and question-answer pair (the expected best default for chat data).
- **Indexes:** BM25 (`rank_bm25` or SQLite FTS5) and a dense vector index (Qdrant, via `qdrant-client`), plus a metadata store in SQLite.
- **Fusion:** Reciprocal Rank Fusion, score = sum of 1/(k + rank) across lists, k = 60 by default. RRF uses ranks only, so BM25 and cosine scores are never normalized.
- **Reranker:** a cross-encoder (for example `bge-reranker-base`) applied to the top 20-50 fused candidates.
- **Relevance gate:** a score threshold that returns *nothing* when no chunk is relevant. Injecting irrelevant context is worse than injecting none.
- **Citations:** every returned chunk carries its platform, conversation id, title and timestamp.

## Tech stack

Python (managed with **uv**), FastAPI, Qdrant, `sentence-transformers` (embeddings and cross-encoder), `rank_bm25` or SQLite FTS5, pytest, ruff. The extension (later) is a Chrome Manifest V3 extension in plain JavaScript.

## Repository layout

```
chat-memory-rag/
├── README.md
├── pyproject.toml            # project name is "chatmem"; uv-managed
├── configs/                  # one YAML per experiment (bm25_only, dense_only, hybrid_rrf, hybrid_rerank)
├── data/
│   ├── raw/                  # real private exports. NEVER commit, NEVER read into logs or print
│   ├── processed/            # normalized conversations (JSONL)
│   ├── sample/               # synthetic data, safe to commit
│   │   ├── chatgpt/conversations.json
│   │   └── claude/conversations.json
│   └── eval/
│       └── queries.sample.jsonl
├── src/chatmem/              # the importable package
│   ├── schema.py             # Message, Chunk, Conversation dataclasses
│   ├── ingest/               # chatgpt.py, claude.py, normalize.py
│   ├── chunking/             # per_message.py, sliding_window.py, qa_pair.py
│   ├── index/                # bm25.py, dense.py, store.py
│   ├── retrieval/            # fusion.py, rerank.py, gate.py, pipeline.py
│   ├── api/                  # main.py (FastAPI), models.py
│   └── utils/                # config.py, logging.py
├── eval/                     # build_testset.py, metrics.py, run_eval.py, compare.py, results/
├── extension/                # Chrome MV3 extension (build last)
├── scripts/                  # gen_dummy_data.py, ingest_all.py, anonymize.py
├── tests/
├── notebooks/                # exploration only; no core logic
└── docs/                     # architecture diagram, findings.md
```

## Status (what exists today)

Be accurate about this. Do not assume planned items exist.

| Area | State |
|---|---|
| Project scaffold, uv environment | Done (package name `chatmem`) |
| Synthetic sample data (14 conversations) | Done: `data/sample/` |
| Sample labeled queries (11) | Done: `data/eval/queries.sample.jsonl` |
| `scripts/gen_dummy_data.py` | Done |
| `schema.py`, parsers, chunking | **Not started** |
| Indexes, fusion, reranker, gate, API | **Not started** |
| Eval harness (`metrics.py`, `run_eval.py`) | **Not started** |
| Browser extension | **Not started** (build last) |

Planned build order: schema and parsers, then chunking, then a BM25 and dense baseline, then metrics and a small eval set, then fusion, reranker and gate (each compared against the baseline), then API, then extension.

## Data formats

**ChatGPT export** (`conversations.json`): a list of conversations. Each has `title`, `create_time`, `update_time`, `current_node`, and a `mapping` tree: node id -> `{id, message, parent, children}`. A node's `message` is null for the root. Message text is in `message.content.parts`, role in `message.author.role`. To get the linear conversation, walk from `current_node` up through `parent`, then reverse. Skip nodes with null messages and non-text parts.

**Claude export** (`conversations.json`): a list of conversations. Each has `uuid`, `name`, `created_at`, `updated_at`, and a flat `chat_messages` list. Each message has `sender` (`human` or `assistant`), `text`, and `created_at`.

The files in `data/sample/` are **synthetic** and only *approximate* the real formats. Real exports contain extra fields and formats change, so parsers must tolerate missing or unknown fields and should be checked against a real export before being considered finished.

Normalized schema (target): `Message(conversation_id, platform, role, text, timestamp)`; `Chunk(chunk_id, conversation_id, platform, title, text, start_ts, end_ts, message_ids)`.

## Evaluation (central to the project)

- **Test set:** `data/eval/queries.*.jsonl`. Each line: `{"qid", "query", "relevant": [conversation ids], "type"}`. An empty `relevant` list means retrieval should return nothing (tests the gate). Query types: `paraphrase`, `exact_string`, `keyword`, `multi_chat`, `no_answer`.
- **Metrics:** recall@k (k = 1, 5, 10), MRR, nDCG@10, precision of the gate's inject-or-not decision, and latency per query.
- **Experiments:** BM25 only vs dense only vs hybrid; with and without reranker; chunking strategies; embedding models (for example `bge-small` vs `e5-base`). Each experiment is a YAML file in `configs/`, and results land in `eval/results/` and are merged into a markdown comparison table.
- Labels are at conversation level for now. Chunk-level labels may come later.

Never tune on the test set blindly: keep a held-out portion of queries for final numbers.

## Conventions for agents and contributors

- **Environment:** use `uv`. Run code with `uv run ...` from the repo root. Add dependencies with `uv add` (dev tools with `uv add --dev`). Do not use bare `pip install`.
- **Run from the repo root.** Paths in scripts are relative to it.
- **Package imports:** `from chatmem...`. Retrieval logic lives only in `src/chatmem/`; `eval/` and `api/` import it and must not reimplement it.
- **Config-driven pipeline:** `retrieval/pipeline.py` builds a retriever from a YAML in `configs/`. Do not hard-code experiment settings.
- **Privacy:** `data/raw/` and `data/processed/` hold real personal chats. Never commit them, print their contents, or send them to external APIs. Prefer local embedding and reranking models. Only synthetic or anonymized data goes in `data/sample/`.
- **Tests:** add pytest tests alongside new modules (parsers, chunking, fusion, metrics at minimum). Keep tests runnable on `data/sample/` only.
- **Style:** type hints, small functions, ruff-clean. Prefer dataclasses or pydantic models over bare dicts at module boundaries.
- **Honesty about results:** never hard-code or fabricate metrics. Every number in docs must come from a run in `eval/results/`.
- **Scope discipline:** do not start the extension or add new platforms before the retrieval core and eval harness work.

## Common commands

```bash
uv sync                                          # create/refresh the environment
uv run python scripts/gen_dummy_data.py          # regenerate sample data into data/sample
uv run pytest                                    # run tests
uv run python eval/run_eval.py --config configs/hybrid_rerank.yaml   # (planned)
uv run uvicorn chatmem.api.main:app --reload     # (planned)
```

## Open questions and design notes

- Best chunking unit for chat data (hypothesis: Q&A pair, with the user question prepended to long answer pieces).
- Whether e5-base beats bge-small enough to justify its cost; decide with eval numbers.
- Where to set the gate threshold; calibrate on the `no_answer` queries.
- Extension capture is fragile (site DOM changes), so keep capture code isolated per site in `extension/src/content/`.