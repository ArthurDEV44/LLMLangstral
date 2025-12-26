# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Core components for LLMLangstral prompt compression.

This module provides foundational classes for model management,
tokenization, and compression interfaces.

Classes:
    - ModelManager: Centralized model loading with lazy initialization
    - CompressionResult: Dataclass for compression operation results
    - BaseCompressor: Abstract base class for all compressors
    - TokenizationMixin: Mixin providing tokenization and PPL utilities

Example:
    >>> from llmlangstral.core import ModelManager, CompressionResult
    >>> manager = ModelManager("mistralai/Mistral-7B-v0.3")
    >>> # Model loads lazily on first access
    >>> tokenizer = manager.tokenizer
    >>> model = manager.model
"""

from .base import BaseCompressor, CompressionResult
from .model_loader import ModelManager
from .tokenization import TokenizationMixin

__all__ = [
    "BaseCompressor",
    "CompressionResult",
    "ModelManager",
    "TokenizationMixin",
]
