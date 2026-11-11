# Retrieval Evaluation Summary (Split: dev)

- **Split**: `dev`

| Config | Score Scale | Rerank Model | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | Latency Mean (ms) | Latency P95 (ms) | Warnings |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| bm25_only | rrf | BAAI/bge-reranker-base | 0.8698 | 0.9141 | 0.8812 | 0.8435 | 0.8125 | 20.97 | 37.11 | none |
| dense_only | rrf | BAAI/bge-reranker-base | 0.9583 | 0.9740 | 0.9531 | 0.9398 | 0.9062 | 92.48 | 79.81 | none |
| hybrid_rrf | rrf | BAAI/bge-reranker-base | 0.9583 | 0.9818 | 0.9750 | 0.9492 | 0.9688 | 57.70 | 86.19 | none |
| hybrid_rerank | raw | BAAI/bge-reranker-base | 0.9661 | 0.9818 | 0.9609 | 0.9576 | 0.9375 | 6914.07 | 8399.20 | none |
