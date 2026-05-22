# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Base classes for compression components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


@dataclass
class CompressionResult:
    """
    Result of a compression operation.

    Attributes:
        compressed_prompt: The compressed prompt text.
        origin_tokens: Number of tokens in the original prompt.
        compressed_tokens: Number of tokens after compression.
        ratio: Compression ratio as formatted string (e.g., "2.5x").
        rate: Compression rate as formatted string (e.g., "40%").
        saving: Token savings description.
        compressed_prompt_list: Optional list of compressed segments.
        fn_labeled_original_prompt: Optional labeled version of original prompt.
    """

    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    ratio: str
    rate: str
    saving: str
    compressed_prompt_list: List[str] = field(default_factory=list)
    fn_labeled_original_prompt: str = ""

    @classmethod
    def from_compression(
        cls,
        compressed_prompt: str,
        origin_tokens: int,
        compressed_tokens: int,
        compressed_prompt_list: Optional[List[str]] = None,
        fn_labeled_original_prompt: str = "",
    ) -> "CompressionResult":
        """Factory computing ratio/rate/saving from token counts."""
        ratio = 1 if compressed_tokens == 0 else origin_tokens / compressed_tokens
        tokens_saved = max(origin_tokens - compressed_tokens, 0)
        return cls(
            compressed_prompt=compressed_prompt,
            origin_tokens=origin_tokens,
            compressed_tokens=compressed_tokens,
            ratio=f"{ratio:.1f}x",
            rate=f"{1 / ratio * 100:.1f}%",
            saving=f", Saving {tokens_saved} tokens.",
            compressed_prompt_list=compressed_prompt_list or [],
            fn_labeled_original_prompt=fn_labeled_original_prompt,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy dict format expected by callers."""
        d: dict[str, Any] = {
            "compressed_prompt": self.compressed_prompt,
            "origin_tokens": self.origin_tokens,
            "compressed_tokens": self.compressed_tokens,
            "ratio": self.ratio,
            "rate": self.rate,
            "saving": self.saving,
        }
        if self.compressed_prompt_list:
            d["compressed_prompt_list"] = self.compressed_prompt_list
        if self.fn_labeled_original_prompt:
            d["fn_labeled_original_prompt"] = self.fn_labeled_original_prompt
        return d


class BaseCompressor(ABC):
    """
    Abstract base class for all compressors.

    Compressors reduce prompt length while preserving semantic meaning,
    using perplexity-driven filtering over a Mistral causal LM
    (LLMLingua / LongLLMLingua, structured compression, JSON compression).
    """

    @abstractmethod
    def compress(
        self,
        context: Union[str, List[str]],
        **kwargs,
    ) -> CompressionResult:
        """
        Compress the given context.

        Args:
            context: Text or list of texts to compress.
            **kwargs: Algorithm-specific parameters.

        Returns:
            CompressionResult containing the compressed prompt and metadata.
        """
        pass
