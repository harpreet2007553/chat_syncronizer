# chatmem Local API Service

The `chatmem` local service is a lightweight, privacy-first REST API designed for pair-programming assistants, browser extensions, and agentic tools to retrieve relevant conversational history from personal chat exports (ChatGPT, Claude).

The service runs strictly on `localhost` (`127.0.0.1`) by default, processes all embeddings locally via embedded Qdrant and SQLite, and never transmits personal conversation data over any external network or telemetry service.

---

## 1. Quickstart & Service Lifecycle

### Starting the Server

Launch the service via the `chatmem` CLI:

```bash
uv run chatmem serve
```

Optional CLI parameters:
- `--config`, `-c`: Path to YAML configuration (default: `configs/api.yaml`).
- `--host`, `-h`: Interface address to bind to (overrides config).
- `--port`, `-p`: Port to listen on (default: `8000`).

Example:
```bash
uv run chatmem serve --config configs/api.yaml --port 8000
```

Startup banner:
```text
============================================================
 chatmem local API service
============================================================
 Host:            http://127.0.0.1:8000
 Default Mode:    fast
 Auth Token File: C:\Users\<user>\.chatmem\token
 Documentation:   docs/api.md
============================================================
```

### Process Concurrency & Embedded Storage Lock

The service uses an embedded local Qdrant engine (`qdrant-client` local mode) and SQLite (`data/chatmem_metadata.db`). Embedded Qdrant enforces a strict single-process file lock on its storage directory.
- The `chatmem serve` process owns the local storage exclusively during execution.
- Read operations are handled concurrently using thread pools.
- Write operations (`POST /ingest/messages` and `DELETE /conversations/...`) are serialized using an asynchronous mutex (`asyncio.Lock`).
- Shutting down the service (SIGINT / Ctrl+C) gracefully flushes and closes database handles.

---

## 2. Security & Privacy Guarantees

1. **Localhost Only**: The service refuses to bind to `0.0.0.0` or `::` unless `allow_all_interfaces: true` is explicitly configured.
2. **Bearer Token Authentication**: All endpoints (except `/health`) require an `Authorization: Bearer <token>` header. On first startup, the service generates a 256-bit cryptographically secure token and stores it at `~/.chatmem/token` with restricted file permissions (`0600`).
3. **Restricted CORS Policy**: Browser origins must be explicitly listed in `allowed_cors_origins`. Wildcard (`"*"`) origins are strictly forbidden and rejected at startup.
4. **Strict Privacy-Preserving Access Logging**:
   - Access logs record **only**: timestamp, HTTP method, route path, status code, response time in ms, and returned result count.
   - Request bodies, user queries, message contents, and retrieved text chunks are **never** logged.
5. **Request Size Guardrails**: Requests exceeding `max_request_bytes` (default 1 MB) are rejected immediately with HTTP 413. Individual messages exceeding `max_message_chars` (default 30,000) are rejected with HTTP 400.

---

## 3. Retrieval Modes & Trade-offs

| Mode | Pipeline & Gating | Typical Latency | Memory / Model Overhead | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`fast`** (default) | BM25 + Dense Search + RRF Fusion + Dense Cosine Gate | **~193 ms median**<br>(~262 ms P95) | Lightweight: BGE-small-en embedding model only | Interactive typing, live browser extension suggestions, real-time agent context injection |
| **`precise`** | BM25 + Dense Search + Cross-Encoder Reranker + Cross-Encoder Gate | ~800–1200 ms | Heavy: Cross-encoder reranker (`bge-reranker-base`) lazy-loaded into memory | In-depth research queries, code troubleshooting requiring semantic precision |

### Gate Threshold Calibration Notice

> [!WARNING]
> The default gate threshold in `fast` mode (`fast_gate_threshold: 0.70`) is an **uncalibrated placeholder default**. Because cosine similarity distributions vary depending on domain vocabulary and export volume, you should calibrate this threshold against your own validation queries before relying on it in production.

---

## 4. Benchmark Latency Measurements

Measured on `data/eval_v2` (`dev` split, 50 queries) over 3 complete evaluation sweeps (excluding initial cold-start warmup):

- **Query Count**: 50 queries (dev split)
- **Pipeline Strategy**: QA Pair (`strategy="qa"`)
- **Median Query Latency**: **192.94 ms**
- **95th Percentile Latency (P95)**: **261.72 ms**

Saved benchmark artifact: [`eval/results/fast_mode_latency.json`](file:///d:/chat-memory-rag/eval/results/fast_mode_latency.json)

---

## 5. Endpoints Reference

### 1. Health & Status Check

- **Method**: `GET /health`
- **Auth**: None required.
- **Description**: Returns operational status, model versions, index chunk counts, and indicates whether the heavy reranker is currently loaded into memory. Does not leak any conversational data.

#### Curl Example
```bash
curl -X GET http://127.0.0.1:8000/health
```

#### Response (`200 OK`)
```json
{
  "status": "ok",
  "version": "0.1.0",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "rerank_model": "BAAI/bge-reranker-base",
  "reranker_loaded": false,
  "conversations_indexed": 100,
  "chunks_indexed": 434
}
```

---

### 2. Search

- **Method**: `POST /search`
- **Auth**: `Bearer <token>`
- **Description**: Executes vector and lexical search across indexed chunks. Returns ranked results with scores, citations, gate decisions, and execution timings.

#### Request Body
```json
{
  "query": "How did I configure docker volumes for postgres?",
  "mode": "fast",
  "top_k": 5,
  "platform": "chatgpt",
  "since": "2025-01-01T00:00:00Z"
}
```

#### Curl Example
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Authorization: Bearer YOUR_SECRET_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How did I configure docker volumes for postgres?",
    "mode": "fast",
    "top_k": 3
  }'
```

#### Response (`200 OK`)
```json
{
  "injected": true,
  "reason": "ok",
  "gate_score": 0.8241,
  "score_scale": "rrf",
  "results": [
    {
      "chunk_id": "conv_982_chunk_0",
      "text": "User: How do I persist postgres data?\nAssistant: In your docker-compose.yml, declare a named volume:\n```yaml\nvolumes:\n  postgres_data:\n```",
      "score": 0.8241,
      "citation": {
        "platform": "chatgpt",
        "conversation_id": "conv_982",
        "title": "Postgres Docker Setup",
        "timestamp": "2025-03-12T14:22:10Z"
      }
    }
  ],
  "timings_ms": {
    "search": 42.15,
    "fusion": 0.82,
    "hydration": 2.11,
    "gate": 0.12,
    "total": 45.2
  }
}
```

---

### 3. Context Injection (Extension & Agent Prompting)

- **Method**: `POST /context`
- **Auth**: `Bearer <token>`
- **Description**: Returns a formatted, deterministic prompt block ready to prepend to LLM prompts. If the gate rejects the query as irrelevant, returns `injected: false` and `context: ""`. Truncation respects paragraph and sentence boundaries within `max_chars`.

#### Request Body
```json
{
  "query": "Where did we leave off on the database migration?",
  "mode": "fast",
  "max_chars": 2000
}
```

#### Curl Example
```bash
curl -X POST http://127.0.0.1:8000/context \
  -H "Authorization: Bearer YOUR_SECRET_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Where did we leave off on the database migration?",
    "max_chars": 1500
  }'
```

#### Response (`200 OK` - Injected)
```json
{
  "injected": true,
  "reason": "ok",
  "context": "--- PAST CONVERSATION CONTEXT (chatmem) ---\n[1] (chatgpt, 2025-05-10T11:00:00Z) Alembic migration\nUser: We finished adding alembic revision 004.\nAssistant: Next step is running alembic upgrade head.\n--- END CONTEXT ---",
  "chunks_used": 1,
  "total_chars": 242
}
```

#### Response (`200 OK` - Gate Abstain)
```json
{
  "injected": false,
  "reason": "below_threshold",
  "context": "",
  "chunks_used": 0,
  "total_chars": 0
}
```

---

### 4. Ingest Messages (Live Capture)

- **Method**: `POST /ingest/messages`
- **Auth**: `Bearer <token>`
- **Description**: Live capture endpoint for browser extensions and CLI capture hooks. Chunks messages using `qa_pair` strategy and incrementally updates SQLite and Qdrant. Idempotent: re-posting the same conversation replaces previous chunks without duplicating points.

#### Request Body
```json
{
  "platform": "claude",
  "conversation_id": "claude_conv_abc123",
  "title": "React Context Debugging",
  "messages": [
    {
      "role": "user",
      "text": "Why is my React Context re-rendering everything?",
      "timestamp": "2026-03-01T10:00:00Z"
    },
    {
      "role": "assistant",
      "text": "Because your provider value object is not memoized with useMemo.",
      "timestamp": "2026-03-01T10:00:05Z"
    }
  ]
}
```

#### Curl Example
```bash
curl -X POST http://127.0.0.1:8000/ingest/messages \
  -H "Authorization: Bearer YOUR_SECRET_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "claude",
    "conversation_id": "claude_conv_abc123",
    "title": "React Context Debugging",
    "messages": [
      {
        "role": "user",
        "text": "Why is my React Context re-rendering everything?"
      },
      {
        "role": "assistant",
        "text": "Because your provider value object is not memoized with useMemo."
      }
    ]
  }'
```

#### Response (`200 OK`)
```json
{
  "status": "ok",
  "conversation_id": "claude_conv_abc123",
  "platform": "claude",
  "chunks_indexed": 1,
  "messages_indexed": 2
}
```

---

### 5. Delete Conversation

- **Method**: `DELETE /conversations/{platform}/{conversation_id}`
- **Auth**: `Bearer <token>`
- **Description**: Completely purges the conversation, its messages, and all derived vector points from both SQLite and Qdrant.

#### Curl Example
```bash
curl -X DELETE http://127.0.0.1:8000/conversations/claude/claude_conv_abc123 \
  -H "Authorization: Bearer YOUR_SECRET_TOKEN"
```

#### Response (`200 OK`)
```json
{
  "status": "deleted",
  "conversation_id": "claude_conv_abc123",
  "platform": "claude",
  "chunks_deleted": 1
}
```

---

## 6. How Clients & Extensions Should Use `/context`

When integrating `chatmem` into a coding assistant or browser extension:

1. Prior to submitting the user's prompt to an external LLM, call `POST /context` with `{ "query": user_prompt, "max_chars": 2000 }`.
2. Inspect the response:
   - If `injected == true` and `context` is non-empty:
     Prepend `context` directly to the prompt or pass it as a system message.
   - If `injected == false`:
     Do **not** modify or inject anything into the prompt.
3. This ensures that only relevant past discussions are injected, conserving context window tokens and preventing hallucinations on irrelevant topics.

