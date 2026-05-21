# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Statistical ranking strategies (BM25, Gzip)."""

import gzip
from typing import List, Tuple

import numpy as np

from .base import RankingStrategy
from .registry import RankingRegistry


@RankingRegistry.register("bm25")
class BM25Ranker(RankingStrategy):
    """
    BM25 (Okapi BM25) ranking algorithm.

    A bag-of-words retrieval function that ranks documents based on
    query term frequencies. Fast and effective for keyword matching.

    Requires: rank_bm25 library
    """

    name = "bm25"

    def __init__(self, **kwargs):
        pass

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        """
        Rank documents using BM25 scoring.

        Args:
            corpus: List of documents to rank.
            query: Query string.

        Returns:
            List of (index, score) tuples sorted by BM25 score (descending).
        """
        from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingImports]

        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)

        # Sort by score descending, return (index, 0) format for compatibility
        idx = [(int(ii), 0.0) for ii in (-doc_scores).argsort()]
        return idx


@RankingRegistry.register("gzip")
class GzipRanker(RankingStrategy):
    """
    Gzip compression-based ranking.

    Uses normalized compression distance (NCD) to measure similarity
    between documents and query. Based on the observation that similar
    texts compress better together.

    No external dependencies (uses standard library gzip).
    """

    name = "gzip"

    def __init__(self, **kwargs):
        pass

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        """
        Rank documents using gzip compression similarity.

        Args:
            corpus: List of documents to rank.
            query: Query string.

        Returns:
            List of (index, score) tuples sorted by compression similarity.
        """

        def get_score(x: str, y: str) -> float:
            """Compute normalized compression distance."""
            cx = len(gzip.compress(x.encode()))
            cy = len(gzip.compress(y.encode()))
            cxy = len(gzip.compress(f"{x} {y}".encode()))
            return (cxy - min(cx, cy)) / max(cx, cy)

        doc_scores = [get_score(doc, query) for doc in corpus]

        # Lower score = more similar, sort ascending
        idx = [(int(ii), 0.0) for ii in np.argsort(doc_scores)]
        return idx
