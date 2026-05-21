# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Base classes for multi-level filtering strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class FilterContext:
    """
    Shared context for all filter operations.

    Provides access to tokenizer, device, and callback functions
    without creating circular imports with PromptCompressor.

    Attributes:
        tokenizer: The tokenizer instance for text processing.
        device: Device string for tensor operations ("cuda", "cpu", etc.).
        max_position_embeddings: Maximum sequence length from model config.
        cache_bos_num: Number of BOS tokens to preserve in KV-cache.
        get_ppl_fn: Callback to compute perplexity.
        get_condition_ppl_fn: Callback to compute conditional perplexity.
        get_rank_results_fn: Callback to rank documents by relevance.
    """

    tokenizer: Any
    device: str
    max_position_embeddings: int
    cache_bos_num: int = 10
    get_ppl_fn: Optional[Callable[..., Any]] = None
    get_condition_ppl_fn: Optional[Callable[..., Any]] = None
    get_rank_results_fn: Optional[Callable[..., Any]] = None


class FilterBase(ABC):
    """
    Abstract base class for all filtering strategies.

    Filters reduce prompt content at different granularities:
    - Context level: Select most relevant documents
    - Sentence level: Select most relevant sentences
    - Token level: Select tokens based on perplexity
    """

    def __init__(self, ctx: FilterContext):
        """
        Initialize filter with shared context.

        Args:
            ctx: FilterContext with tokenizer, device, and callbacks.
        """
        self.ctx = ctx

    @property
    def tokenizer(self):
        """Get the tokenizer from context."""
        return self.ctx.tokenizer

    @property
    def device(self) -> str:
        """Get the device string from context."""
        return self.ctx.device

    @property
    def max_position_embeddings(self) -> int:
        """Get max position embeddings from context."""
        return self.ctx.max_position_embeddings

    def get_token_length(self, text: str) -> int:
        """
        Get the number of tokens in a text using the Mistral tokenizer.

        Args:
            text: Input text to tokenize.

        Returns:
            Number of tokens.
        """
        return len(self.tokenizer.encode(text))

    @abstractmethod
    def filter(self, *args, **kwargs) -> Any:
        """Apply the filtering strategy. Override in subclasses."""
        pass
