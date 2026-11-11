import math
import re
from collections import Counter
from typing import Sequence

from chatmem.schema import Chunk
from chatmem.utils.logging import get_logger

logger = get_logger(__name__)

# Basic alphanumeric tokenizer suitable for code identifiers and text
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[^\s\w]")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words and code symbols."""
    if not text:
        return []
    return [t.lower() for t in TOKEN_PATTERN.findall(text) if t.strip()]


class PurePythonBM25:
    """Zero-dependency BM25Okapi implementation for reliability."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_lens) / self.corpus_size) if self.corpus_size > 0 else 0.0

        # Term frequencies per document
        self.doc_freqs: list[Counter] = [Counter(doc) for doc in corpus]

        # Document frequencies for all terms across corpus
        df: Counter = Counter()
        for doc_counts in self.doc_freqs:
            for term in doc_counts:
                df[term] += 1

        # Precompute IDF for each term
        self.idf: dict[str, float] = {}
        for term, freq in df.items():
            # Standard Lucene/BM25Okapi IDF formula
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if self.corpus_size == 0 or self.avgdl == 0:
            return scores

        for q in query_tokens:
            q_idf = self.idf.get(q)
            if q_idf is None:
                continue

            for doc_idx, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(q, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lens[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[doc_idx] += q_idf * (numerator / denominator)

        return scores


# Try importing rank_bm25 if available, otherwise fallback to PurePythonBM25
try:
    from rank_bm25 import BM25Okapi as RankBM25Okapi

    _USE_RANK_BM25 = True
except ImportError:
    _USE_RANK_BM25 = False
    logger.info("rank-bm25 not installed; using built-in PurePythonBM25")


class BM25Index:
    """BM25 sparse / keyword index for fast lexical retrieval over Chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._bm25 = None

    def index_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Tokenize and build a BM25 index over the provided chunks."""
        self.chunks = list(chunks)
        if not self.chunks:
            self._bm25 = None
            return 0

        logger.info("Building BM25 index over %d chunks...", len(self.chunks))
        tokenized_corpus = [tokenize(c.text) for c in self.chunks]

        if _USE_RANK_BM25:
            self._bm25 = RankBM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self._bm25 = PurePythonBM25(tokenized_corpus, k1=self.k1, b=self.b)

        logger.info("Successfully indexed %d chunks in BM25", len(self.chunks))
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        platform_filter: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Search the BM25 index for chunks matching query terms.

        Args:
            query: The search query string.
            top_k: Maximum number of chunks to return.
            platform_filter: Optional platform filter (e.g. 'chatgpt' or 'claude').

        Returns:
            List of (Chunk, bm25_score) tuples sorted in descending order of score.
            Only chunks with a score > 0.0 are returned.
        """
        if self._bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Compute BM25 scores across all indexed chunks
        raw_scores = self._bm25.get_scores(query_tokens)

        # Pair chunks with scores and apply platform filter
        scored_pairs: list[tuple[Chunk, float]] = []
        for chunk, score in zip(self.chunks, raw_scores):
            if score <= 0.0:
                continue
            if platform_filter and chunk.platform != platform_filter:
                continue
            scored_pairs.append((chunk, float(score)))

        # Sort descending by score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]


if __name__ == "__main__":
    from chatmem.chunking.qa_pair import QA_Chunks

    print(f"Loaded {len(QA_Chunks)} QA chunks for BM25 testing.")

    bm25_index = BM25Index()
    bm25_index.index_chunks(QA_Chunks)

    test_queries = [
        "KeyError: 'user_id' pandas groupby",
        "docker-compose Qdrant persistent storage",
        "nonexistent search keyword test",
    ]

    for q in test_queries:
        print("\n==========================================")
        print(f"BM25 Query: '{q}'")
        results = bm25_index.search(q, top_k=2)
        if not results:
            print("No matching chunks found (score = 0).")
            continue

        for rank, (chunk, score) in enumerate(results, 1):
            print(f"\n[{rank}] Score: {score:.4f} | Chat: {chunk.title} ({chunk.platform})")
            print(f"Text snippet: {chunk.text[:120]}...")

