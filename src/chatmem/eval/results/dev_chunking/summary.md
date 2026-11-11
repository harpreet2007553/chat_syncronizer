# Retrieval Evaluation Summary (Split: dev)

- **Split**: `dev`

| Config | Score Scale | Rerank Model | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | Latency Mean (ms) | Latency P95 (ms) | Warnings |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| hybrid_rrf_pm | rrf | BAAI/bge-reranker-base | 0.9557 | 0.9740 | 0.9844 | 0.9542 | 0.9688 | 510.99 | 566.78 | none |
| hybrid_rrf_ws | rrf | BAAI/bge-reranker-base | 0.9271 | 0.9818 | 0.9583 | 0.9383 | 0.9375 | 77.20 | 106.89 | none |
| hybrid_rrf_qa | rrf | BAAI/bge-reranker-base | 0.9583 | 0.9818 | 0.9750 | 0.9492 | 0.9688 | 278.00 | 755.17 | none |
