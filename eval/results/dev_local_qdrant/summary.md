# Retrieval Evaluation Summary (Split: dev)

- **Split**: `dev`

| Config | Score Scale | Rerank Model | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | Latency Mean (ms) | Latency P95 (ms) | Warnings |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| dense_only | rrf | BAAI/bge-reranker-base | 0.9583 | 0.9740 | 0.9531 | 0.9398 | 0.9062 | 110.72 | 138.17 | none |
| hybrid_rrf | rrf | BAAI/bge-reranker-base | 0.9583 | 0.9818 | 0.9750 | 0.9492 | 0.9688 | 110.29 | 166.64 | none |
