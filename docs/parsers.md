# Export Parser Policies and Ingestion Specification

This document details the parsing specifications, format variations, and explicit handling policies for official chatbot export files in `chatmem`.

---

## 1. Supported Platform Formats

### Claude User Exports
- **Container Formats**:
  - Direct list of conversation objects (`[...]`).
  - Wrapped dictionary (`{"Count": N, "value": [...]}` or `{"conversations": [...]}`).
- **Conversation Identifier**: `uuid` $\to$ `id` $\to$ `conversation_id`.
- **Title**: `name` $\to$ `title` $\to$ `summary` $\to$ `"Untitled"`.
- **Timestamps**: Safe ISO-8601 parsing (`created_at`, `updated_at`) with fallback to current UTC timestamp.
- **Messages**: Flat list (`chat_messages` or `messages`).

### ChatGPT User Exports
- **Container Formats**:
  - Direct list of conversation objects (`[...]`).
  - Wrapped dictionary (`{"conversations": [...]}`).
- **Conversation Identifier**: `conversation_id` $\to$ `id`.
- **Title**: `title` $\to$ `"Untitled"`.
- **Timestamps**: Numeric Unix epoch seconds (`create_time`, `update_time`) normalized to UTC `datetime`.
- **Messages**: Graph/Tree mapping (`mapping` dictionary of nodes).

---

## 2. Ingestion Policies

### Policy 1: Branched ChatGPT Conversations
ChatGPT exports preserve conversation histories as tree graphs in the `mapping` dictionary. When users edit previous prompts or regenerate responses, nodes create multiple `children` (branch points).

- **Default Active Branch Policy**:
  - The parser follows `current_node` backwards through its `parent` pointers to the root, then reverses the list to form a chronological linear sequence.
  - This faithfully reconstructs the exact active branch visible to the user at the moment of export.
- **Missing or Invalid `current_node` Fallback**:
  - If `current_node` is missing, `null`, or points to a node outside `mapping`, the parser locates all leaf nodes (nodes with no children) and selects the leaf with the newest message `create_time`.
  - If no leaf nodes exist, it traverses backwards from the first available node.

### Policy 2: Non-Text and Structured Content
Chatbot exports contain heterogeneous content beyond plain text: tool calls, multimodal image pointers, code interpreter executions, and thinking/reasoning blocks.

- **Claude Non-Text Policy**:
  1. **Primary Text**: Extracted from message `text`.
  2. **Content Blocks**: If `text` is blank, inspects structured `content` blocks:
     - Text blocks (`type == "text"`) are joined with newlines.
     - Thinking blocks (`type == "thinking"`) are omitted to keep retrieval context focused on external communication rather than internal scratchpad reasoning.
     - Tool RPC payloads (`tool_use`, `tool_result`) are omitted unless user-facing text is present.
  3. **Attachments & Files**: When messages consist of file uploads without typed text, extracts `[File: <file_name>]\n<extracted_content>` from `attachments` or `[File: <file_name>]` from `files`.
- **ChatGPT Non-Text Policy**:
  1. **Parts Array**: Reads string elements and structured dicts containing textual keys (`{"text": "..."}`).
  2. **Direct Text**: If `parts` is empty, checks `content["text"]` (used by code interpreter inputs and execution outputs).
  3. **Multimodal Media**: Image asset pointers (`image_asset_pointer`) and binary references are skipped while preserving any surrounding user prompts or captions.
  4. **Roles**: System tool messages (`author.role == "tool"`) are normalized to `"system"`.

### Policy 3: Very Long Messages (>20,000 characters)
Real exports frequently contain pasted stack traces, log dumps, or lengthy source code files exceeding 20,000 characters.

- **Parser Ingestion Stage**:
  - The parser retains the full, untruncated message text during ingest and metadata storage.
  - Character and turn lengths are preserved faithfully to prevent unrecoverable data loss in the persistence layer.
- **Downstream Chunking Stage**:
  - Chunking modules (`per_message`, `sliding_window`, `qa_pair`) handle splitting or sliding window truncation into semantic retrieval units within model context windows (e.g. 512 tokens for `bge-small` / `bge-reranker-base`).

---

## 3. Privacy Guarantees
- Parsers and inspection tools only log aggregate metrics (counts, lengths, date ranges, and field names).
- Private conversation text, titles, user IDs, and message bodies are never written to test fixtures, logs, or reports.

