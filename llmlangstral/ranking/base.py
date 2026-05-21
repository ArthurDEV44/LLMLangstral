# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Base classes for ranking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple


class RankingStrategy(ABC):
    """
    Abstract base class for all ranking algorithms.

    Ranking strategies are used to order context documents or sentences
    by relevance to a given query. This enables selective compression
    of the most relevant content.
    """

    name: str = "base"
    requires_model: bool = False
    requires_api_config: bool = False

    @abstractmethod
    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        """
        Rank corpus documents by relevance to query.

        Args:
            corpus: List of documents/sentences to rank.
            query: The query/question to rank against.
            **kwargs: Additional arguments (e.g., context_tokens_length).

        Returns:
            List of (index, score) tuples sorted by relevance (most relevant first).
            The index refers to the position in the original corpus.
            The score meaning varies by algorithm (higher = more relevant).
        """
        pass


class ModelBasedRanker(RankingStrategy, ABC):
    """
    Base class for rankers that use embedding models.

    Provides lazy loading and caching of models to avoid
    reloading when the same ranker is used multiple times.
    """

    requires_model: bool = True
    model_name: str = ""

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self._model = None
        self._tokenizer = None

    @property
    def model(self) -> Any:
        """Lazy load the model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the model. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _load_model")


class PPLBasedRanker(RankingStrategy, ABC):
    """
    Base class for rankers that use perplexity from the main LLM.

    Requires injection of a callable to compute conditional perplexity.
    """

    def __init__(
        self,
        ppl_fn: Optional[Callable[[str, str, str], float]] = None,
        **kwargs,
    ):
        """
        Initialize PPL-based ranker.

        Args:
            ppl_fn: Callable that computes conditional perplexity.
                    Signature: ppl_fn(text, question, condition_mode) -> float
        """
        self.ppl_fn = ppl_fn
