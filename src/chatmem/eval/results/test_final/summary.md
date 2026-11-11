# Retrieval Evaluation Summary (Split: test)

- **Split**: `test`

| Config | Score Scale | Rerank Model | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@1 | Latency Mean (ms) | Latency P95 (ms) | Warnings |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| bm25_only | rrf | BAAI/bge-reranker-base | 0.8854 | 0.8917 | 0.8854 | 0.8615 | 0.8125 | 109.71 | 215.14 | none |
| dense_only | rrf | BAAI/bge-reranker-base | 0.9229 | 0.9458 | 0.9453 | 0.9242 | 0.9062 | 329.52 | 457.46 | none |
| hybrid_rrf | rrf | BAAI/bge-reranker-base | 0.8854 | 0.9229 | 0.9219 | 0.8988 | 0.8750 | 336.84 | 554.97 | none |
| hybrid_rerank | raw | BAAI/bge-reranker-base | 0.9354 | 0.9458 | 0.9688 | 0.9406 | 0.9375 | 15365.67 | 41425.96 | none |
| hybrid_rerank_gated | raw | BAAI/bge-reranker-base | 0.8854 | 0.8854 | 0.9375 | 0.8893 | 0.9062 | 35197.45 | 43188.91 | none |
