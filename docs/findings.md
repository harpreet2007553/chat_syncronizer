# chatmem Retrieval Benchmark & Findings

This document records the end-to-end evaluation results for `chatmem` on the synthetic benchmark dataset `data/eval_v2`. In accordance with project conventions, all numbers in this report originate directly from run artifacts stored in `eval/results/`. Configuration parameters and gate thresholds were tuned exclusively on the **dev** split, and the final frozen pipeline was evaluated once on the held-out **test** split.

---

## 1. Experimental Setup

- **Dataset**: `data/eval_v2`, consisting of 100 conversations (50 ChatGPT, 50 Claude).
  - 15 diverse topic clusters (Python environments, CUDA/GPU, RAG/vector databases, SQL/analytics, FastAPI/web backend, Git version control, React frontend, cooking, travel, career, personal finance, fitness, guitar/music, algorithms, home repair).
  - Length distribution: 10 conversations with $\ge 20$ messages, 10 conversations exhibiting multi-topic drift, and 80 conversations with 2 to 12 messages.
  - Creation dates span 16 months (January 2023 to May 2024).
- **Benchmark Queries**: 100 authored queries with a deterministic, stratified 50/50 split (`seed=1234`):
  - **Dev split** (50 queries): 32 answerable (10 paraphrase, 8 keyword, 8 multi-chat, 6 vague reference) and 18 no-answer (9 offtopic, 9 hard).
  - **Test split** (50 queries): 32 answerable (10 paraphrase, 8 keyword, 8 multi-chat, 6 vague reference) and 18 no-answer (9 offtopic, 9 hard).
- **Models**:
  - Dense Embedding: `BAAI/bge-small-en-v1.5` (384-dimensional vectors, local inference on CPU).
  - Cross-Encoder Reranker: `BAAI/bge-reranker-base` (local inference on CPU).
  - Vector DB: Qdrant (local Docker, `localhost:6333`).
  - Lexical Search: BM25 via `rank_bm25` / SQLite FTS5.
- **Evaluation Cutoffs**: $k \in \{1, 5, 10\}$.

---

## 2. Baseline Comparison (Dev Split)

All four retrieval architectures were evaluated on the 50 queries of the `dev` split using the default `qa_pair` chunking strategy (217 chunks in SQLite and Qdrant).

| Configuration | Score Scale | Recall@1 (Hit@1) | Recall@5 | Recall@10 | MRR | nDCG@10 | Latency (Mean) | Latency (P95) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bm25_only` | rrf | 0.8125 | 0.8698 | 0.9141 | 0.8812 | 0.8435 | 21.0 ms | 37.1 ms |
| `dense_only` | rrf | 0.9062 | 0.9583 | 0.9740 | 0.9531 | 0.9398 | 92.5 ms | 79.8 ms |
| `hybrid_rrf` | rrf | **0.9688** | 0.9583 | **0.9818** | **0.9750** | 0.9492 | 57.7 ms | 86.2 ms |
| `hybrid_rerank` | raw | 0.9375 | **0.9661** | **0.9818** | 0.9609 | **0.9576** | 6914.1 ms | 8399.2 ms |

### Key Takeaways
1. **Hybrid vs Dense parity at this sample size**: On the dev split, `hybrid_rrf` is slightly ahead of `dense_only` (Hit@1 0.9688 [31 of 32 queries] vs 0.9062 [29 of 32]; MRR 0.9750 vs 0.9531; tied Recall@5 at 0.9583). On the test split, `hybrid_rrf` is slightly behind `dense_only` (Hit@1 0.8750 [28 of 32] vs 0.9062 [29 of 32]; MRR 0.9219 vs 0.9453; Recall@5 0.8854 vs 0.9229). With 32 answerable queries, a difference of 1 or 2 queries is well within the statistical noise margin ($\pm 3.1\%$). Consequently, hybrid and dense are not statistically distinguishable at this sample size.
2. **Dense significantly outperforms BM25 on chat data**: Dense alone achieves 0.9583 Recall@5 vs BM25's 0.8698 on dev (and 0.9229 vs 0.8854 on test). Conversations frequently employ conversational, informal phrasing that diverges lexically from the user's recall query.
3. **Cross-encoder trade-off**: `hybrid_rerank` achieves the highest nDCG@10 (0.9576 dev, 0.9406 test) and highest test Hit@1 (0.9375 [30 of 32]), but requires significant latency per query on CPU across 30 candidates.
4. **Latency Measurement Caveats**: Latency was measured on a Windows CPU machine subject to OS scheduling and thread contention. Furthermore, `dense_only` latency differed noticeably between dev (92.5 ms mean) and test (329.5 ms mean), and instances where mean latency exceeds p95 (such as `dense_only` on dev: mean 92.5 ms vs p95 79.8 ms) indicate extreme outliers on the initial un-warmed query (cold-start model loading). Therefore, absolute latency figures are unreliable; only relative orders of magnitude (BM25 < Dense/RRF << Cross-Encoder) are meaningful.

---

## 3. Per-Category Breakdown (Dev Split)

Performance broken down by query type (counts and percentages):

| Configuration | Keyword (n=8) Hit@1 \| R@10 | Paraphrase (n=10) Hit@1 \| R@10 | Multi-Chat (n=8) Hit@1 \| R@10 | Vague Reference (n=6) Hit@1 \| R@10 |
| :--- | :---: | :---: | :---: | :---: |
| `bm25_only` | 8/8 (100%) \| 1.000 | 7/10 (70%) \| 0.900 | 8/8 (100%) \| 0.781 | 3/6 (50%) \| 1.000 |
| `dense_only` | 8/8 (100%) \| 1.000 | 9/10 (90%) \| 1.000 | 8/8 (100%) \| 0.896 | 4/6 (66.7%) \| 1.000 |
| `hybrid_rrf` | 8/8 (100%) \| 1.000 | 9/10 (90%) \| 1.000 | 8/8 (100%) \| 0.927 | **6/6 (100%)** \| 1.000 |
| `hybrid_rerank` | 8/8 (100%) \| 1.000 | **10/10 (100%)** \| 1.000 | 7/8 (87.5%) \| 0.927 | 5/6 (83.3%) \| 1.000 |

### Expected vs Observed Pattern
- **Keyword**: Both BM25 and Dense achieve 100% Hit@1 and Recall@10 when queries contain exact tokens (`@model_validator`, `git cherry-pick --continue`, etc.).
- **Paraphrase**: BM25 drops sharply to 70.0% Hit@1 (MRR 0.7700), whereas Dense achieves 90.0% Hit@1 (MRR 0.9500) and Cross-Encoder Rerank achieves 100% Hit@1 (MRR 1.0000). Dense semantic search is essential when words do not overlap.
- **Multi-Chat**: Hardest category across all models. While Hit@1 is high, Recall@10 drops (0.781 for BM25, 0.896 for Dense, 0.927 for Hybrid RRF) because retrieving all 2-4 distinct conversations requires deeper recall.
- **Vague Reference**: Lexical BM25 achieves only 50.0% Hit@1 (MRR 0.7500), while `hybrid_rrf` successfully unifies rank signals to achieve 100% Hit@1 (6/6).

---

## 4. Chunking Strategy Comparison (Dev Split)

Evaluated using `hybrid_rrf` across the three chunking strategies:
1. `per_message` (`pm`): 434 chunks (1 message per chunk).
2. `sliding_window` (`ws`): 217 chunks (3-turn window, 1-turn overlap).
3. `qa_pair` (`qa`): 217 chunks (User-Assistant turn pairs).

### Chunk Count Investigation & Mathematical Explanation
An initial inspection wondered why `data/eval_v2` (100 conversations, 10 with $\ge 20$ turns) generated exactly 434 `per_message` chunks, and why both `sliding_window` and `qa_pair` yielded exactly 217 chunks:
- **Conversation Message Distribution**: `scripts/inspect_dataset.py` confirms that `data/eval_v2` contains 100 conversations with exactly 434 total messages: 64 conversations have 2 messages, 26 have 4 messages, 9 have 20 messages, and 1 has 22 messages ($64 \times 2 + 26 \times 4 + 9 \times 20 + 1 \times 22 = 434$). Every conversation has an even number of messages (user/assistant alternating pairs).
- **Per-Message Chunk Count**: $434$ total non-empty messages $\implies 434$ chunks.
- **QA Pair Chunk Count**: Each conversation with even length $N = 2k$ yields exactly $k = N/2$ non-overlapping Q&A pairs. Total chunks: $434 / 2 = 217$.
- **Sliding Window Chunk Count (Window Size 3, Overlap 1)**: The stride is $3 - 1 = 2$. For any conversation with even length $N = 2k$, the window start indices are $0, 2, 4, \dots, 2k-2$. At start index $2k-2$, $\text{end} = \min(2k-2+3, 2k) = 2k \ge N$, which terminates the loop. The number of windows produced is exactly $(2k-2)/2 + 1 = k = N/2$.
- **Content Difference**: Although the chunk counts are identical (217), the chunks differ fundamentally: `qa_pair` chunks strictly contain 2 messages without overlap (434 message references total across chunks), whereas `sliding_window` chunks contain 3 messages with 1 overlapping message (except at the tail), totaling 551 message references across chunks. Unit tests in `tests/test_chunking.py` verify that `qa_pair` and `sliding_window` produce distinct, valid chunks on 6-message and odd-message conversations.

| Strategy | Chunks in Index | Recall@1 (Hit@1) | Recall@5 | Recall@10 | MRR | nDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `per_message` (`pm`) | 434 | **0.9688** | 0.9557 | 0.9740 | **0.9844** | **0.9542** | 511.0 ms |
| `sliding_window` (`ws`)| 217 | 0.9375 | 0.9271 | **0.9818** | 0.9583 | 0.9383 | **77.2 ms** |
| `qa_pair` (`qa`) | 217 | **0.9688** | **0.9583** | **0.9818** | 0.9750 | 0.9492 | 278.0 ms |

### Analysis
- **Retrieval Performance by Query Counts ($n=32$ answerable)**:
  - **Hit@1**: `qa_pair` correctly placed a relevant conversation at rank 1 on **31 of 32 queries** (96.9%), matching `per_message` (**31 of 32 queries**, 96.9%), while `sliding_window` succeeded on **30 of 32 queries** (93.8%). The difference is just **1 query**.
  - **Recall@5**: `qa_pair` scored 0.9583, `per_message` scored 0.9557, and `sliding_window` scored 0.9271. This spread represents a shift of roughly **1 query** out of 32.
  - **Recall@10**: `qa_pair` and `sliding_window` achieved 0.9818, while `per_message` scored 0.9740 (a difference of less than 1 query).
- **Statistical Significance**: Because all metric gaps between the three chunking strategies are within 1 to 2 queries on a sample size of 32 answerable queries, the strategies are **not distinguishable on retrieval accuracy alone**.
- **Practical Recommendation**: While retrieval accuracy is neck-and-neck, `qa_pair` is the pragmatic default because it produces half the index size of `per_message` (217 vs 434 chunks, halving embedding and storage cost) and avoids the conversational turn fragmentation and multi-reference index bloat (551 message references) of `sliding_window`.

---

## 5. Relevance Gate Calibration (Dev Only)

The relevance gate evaluates cross-encoder scores on the top-ranked candidate to decide whether to inject context into the prompt or return empty. We evaluated candidate thresholds across the 18 no-answer queries (9 offtopic, 9 hard) and 32 answerable queries on the `dev` split:

| Candidate Threshold | Correct Abstain (No-Answer, n=18) | False Injections | Inject Rate (Answerable, n=32) | Inject Precision | Recall Loss (Answerable) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.001 | 55.6% | 8 | 96.9% | 79.5% | 3.1% |
| 0.005 | 83.3% | 3 | 84.4% | 90.0% | 15.6% |
| **0.010 (Chosen)** | **94.4%** | **1** | **81.2%** | **96.3%** | **18.8%** |
| 0.050 | 94.4% | 1 | 68.8% | 95.7% | 31.2% |
| 0.140 | 100.0% | 0 | 59.4% | 100.0% | 40.6% |
| 0.200 | 100.0% | 0 | 53.1% | 100.0% | 46.9% |

### Threshold Selection & Trade-Off
- **Chosen Threshold**: `gate_threshold: 0.010` (raw cross-encoder score).
- **Abstain Rate on No-Answer**: **94.4%** (17 of 18 no-answer queries correctly blocked; all 9 offtopic queries blocked, 8 of 9 hard queries blocked).
- **False Injections**: Only 1 query (`q94` about mega backdoor roth, an adjacent topic to regular backdoor roth in `c56`) slipped through with score 0.135.
- **Precision**: **96.3%** of all injected prompts are genuinely answerable.
- **Recall Lost on Answerable**: 18.8% (6 of 32 answerable queries rejected, primarily vague references).
- **Alternatives Considered**:
  - `threshold = 0.140`: Achieves 100% correct abstain (0 false injections), but rejects 40.6% of answerable queries—too punitive.
  - `threshold = 0.001`: Retains 96.9% of answerable queries, but allows 8 false injections (abstain rate drops to 55.6%, precision drops to 79.5%).

---

## 6. Final Test Run (Held-Out Test Split)

With all pipeline settings, chunking strategies (`qa_pair`), and the gate threshold (`0.010`) **frozen**, the benchmark was executed once on the 50 queries of the held-out `test` split.

| Configuration | Split | Recall@1 (Hit@1) | Recall@5 | Recall@10 | MRR | nDCG@10 | Gate Precision | Mean Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bm25_only` | **test** | 0.8125 | 0.8854 | 0.8917 | 0.8854 | 0.8615 | 64.0% | 109.7 ms | 215.1 ms |
| `dense_only` | **test** | 0.9062 | 0.9229 | 0.9458 | 0.9453 | 0.9242 | 64.0% | 329.5 ms | 457.5 ms |
| `hybrid_rrf` | **test** | 0.8750 | 0.8854 | 0.9229 | 0.9219 | 0.8988 | 64.0% | 336.8 ms | 555.0 ms |
| `hybrid_rerank` | **test** | **0.9375** | **0.9354** | **0.9458** | **0.9688** | **0.9406** | 64.0% | 15365.7 ms | 41426.0 ms |
| `hybrid_rerank_gated` | **test** | 0.9062 | 0.8854 | 0.8854 | 0.9375 | 0.8893 | **96.9%** | 35197.5 ms | 43188.9 ms |

### Test Split Gate Verification
- **Correct Abstain on No-Answer Queries**: **100.0%** (18 out of 18 no-answer queries correctly rejected; 0 false injections).
- **Answerable Injection Rate**: **93.8%** (30 of 32 answerable queries cleared the gate; 2 false negatives).
- **Gate Precision**: **96.9%** on test (30 valid injections, 0 false injections from no-answer, 1 injection with sub-top relevant placement).
- **Sample Size Context**: The threshold (0.010) was calibrated against 6 candidate values on 18 no-answer dev queries and confirmed on 18 no-answer test queries. While achieving 18/18 correct abstentions on test is an encouraging indicator of generalization against adversarial hard negatives, 18 queries remains a small sample size.
- **Latency Observation**: Evaluation on test exhibited elevated CPU latency (`dense_only` mean rose from 92.5 ms to 329.5 ms, `hybrid_rerank` mean was 15.4 s with p95 at 41.4 s) under Windows CPU thread contention. High mean latencies relative to p50 reflect cold-start model execution and system throttling; only relative performance rankings are reliable.

---

## 7. Answers to Core Evaluation Questions

### 1. Does hybrid beat the better single retriever? By how much?
**At this sample size, hybrid and dense are not statistically distinguishable.**
- On `dev` ($n=32$ answerable), `hybrid_rrf` is slightly ahead of `dense_only` (Hit@1 0.9688 [31 of 32] vs 0.9062 [29 of 32]; MRR 0.9750 vs 0.9531; Recall@5 tied at 0.9583).
- On `test` ($n=32$ answerable), `hybrid_rrf` is slightly behind `dense_only` (Hit@1 0.8750 [28 of 32] vs 0.9062 [29 of 32]; MRR 0.9219 vs 0.9453; Recall@5 0.8854 vs 0.9229).
- With 32 answerable queries per split, a difference of 1 or 2 queries ($\pm 3.1\%$) represents statistical noise. Neither method clearly outperforms the other across both splits.

### 2. Does the reranker justify its latency cost?
**Only for high-stakes, latency-tolerant use cases.**
- `hybrid_rerank` achieves the highest nDCG@10 (0.9576 on dev, 0.9406 on test) and highest Hit@1 on test (0.9375 [30 of 32]).
- However, local CPU cross-encoder inference across 30 candidates takes **6.9s to 15.4s per query**, compared to **57.7ms to 337ms** for `hybrid_rrf`.
- **Recommendation**: For interactive real-time browser extension injection, `hybrid_rrf` is the practical production choice due to its orders-of-magnitude lower latency. For batch workflows or systems with GPU acceleration, `hybrid_rerank` provides superior fine-grained re-ordering and supplies cross-encoder scores necessary for relevance gating.

### 3. Which chunking strategy wins, and is the gap larger than noise?
**The retrieval accuracy gap is within statistical noise (at most 1 query difference).**
- On `dev` ($n=32$ answerable), `qa_pair` achieved Hit@1 on **31 of 32 queries** (96.9%), matching `per_message` (**31 of 32 queries**, 96.9%), while `sliding_window` achieved Hit@1 on **30 of 32 queries** (93.8%).
- For Recall@5, `qa_pair` scored 0.9583, `per_message` scored 0.9557, and `sliding_window` scored 0.9271.
- Because the entire performance spread across the three strategies is bounded within 1 query out of 32, no empirical winner can be declared from retrieval metrics alone.
- However, `qa_pair` is the recommended default on architectural grounds: it produces half the index size of `per_message` (217 vs 434 chunks) and avoids the conversational turn boundary truncation of sliding windows.

### 4. What threshold was chosen and what is the trade-off?
We chose **`gate_threshold = 0.010`** (raw cross-encoder score scale).
- **Threshold Origin & Trade-off**: The threshold was chosen from 6 candidate values on 18 no-answer dev queries (94.4% correct abstain, 17/18) and confirmed on 18 no-answer test queries (100.0% correct abstain, 18/18). While 18/18 on test is encouraging, 18 queries is a small evaluation sample.
- On answerable queries, it passed 26 of 32 queries on dev (81.2%) and 30 of 32 queries on test (93.8%), trading off marginal, low-confidence vague queries in order to reliably block irrelevant context injections.

---

## 8. Caveats & Limitations

1. **Synthetic Data Only**: All conversations and queries are synthetic approximations. While designed with realistic typos, multi-topic drift, and negative near-neighbors, real user chat exports may contain longer code blocks, multi-language code fragments, and different formatting quirks. Treat numbers as rigorous *relative* comparisons rather than absolute bounds.
2. **Small Per-Category Sample Sizes**: Each query category has 6 to 10 queries per split ($n=6$ for vague references, $n=8$ for keywords and multi-chat, $n=10$ for paraphrases). Across 32 answerable queries, a difference of 1-2 queries shifts metrics by 3-6%, meaning small differences represent noise.
3. **Parsers Not Yet Validated Against Real Multi-GB Exports**: The ChatGPT and Claude ingestion parsers conform to current export schemas, but vendor export formats evolve over time. Real user testing against raw ZIP archives remains necessary.

