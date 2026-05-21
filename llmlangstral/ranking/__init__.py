# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Ranking strategies for LLMLangstral prompt compression.

This module provides a pluggable ranking system for ordering documents
or sentences by relevance to a query. Rankings are used to prioritize
content during compression.

Available ranking methods:
    - bm25: BM25 (Okapi BM25) lexical ranking
    - gzip: Compression-based similarity ranking
    - llmlingua: Perplexity-based ranking (native method)
    - longllmlingua: Alias for llmlingua
    - mistral: Mistral-based embeddings

Example:
    >>> from llmlangstral.ranking import RankingRegistry
    >>> ranker = RankingRegistry.get("bm25")
    >>> results = ranker.rank(["doc1", "doc2"], "query")
    >>> print(results)  # [(0, 0), (1, 0)] sorted by relevance

Adding custom rankers:
    >>> from llmlangstral.ranking import RankingRegistry, RankingStrategy
    >>> @RankingRegistry.register("my_ranker")
    ... class MyRanker(RankingStrategy):
    ...     def rank(self, corpus, query, **kwargs):
    ...         return [(i, 0) for i in range(len(corpus))]
"""

# Import all ranking implementations to trigger registration
from . import llmlingua  # noqa: F401
from . import mistral  # noqa: F401
from . import statistical  # noqa: F401
from .base import (
    ModelBasedRanker,
    PPLBasedRanker,
    RankingStrategy,
)
from .registry import RankingRegistry

__all__ = [
    "RankingStrategy",
    "ModelBasedRanker",
    "PPLBasedRanker",
    "RankingRegistry",
]
