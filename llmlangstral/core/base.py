# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Base classes for compression components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Union


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


class BaseCompressor(ABC):
    """
    Abstract base class for all compressors.

    Compressors reduce prompt length while preserving semantic meaning.
    Different implementations use various algorithms (LLMLingua, LLMLingua-2,
    structured compression, etc.).
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
