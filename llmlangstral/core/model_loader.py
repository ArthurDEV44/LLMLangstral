# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Centralized model loading and management."""

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from ..utils import seed_everything


class ModelManager:
    """
    Centralized model loading with lazy initialization.

    Provides lazy loading and caching of models, tokenizers, and configs
    to avoid redundant loading operations. Models are loaded on first
    property access.

    Attributes:
        model_name: The HuggingFace model identifier.
        device_map: Device placement string ("cuda", "cpu", "mps", or device map).
        model_config: Optional model configuration overrides.

    Example:
        >>> manager = ModelManager("mistralai/Mistral-7B-v0.3")
        >>> # Nothing loaded yet
        >>> tokenizer = manager.tokenizer  # Loads model + tokenizer
        >>> model = manager.model  # Already cached, returns immediately
    """

    def __init__(
        self,
        model_name: str,
        device_map: str = "cuda",
        model_config: dict = None,
    ):
        """
        Initialize the ModelManager.

        Args:
            model_name: HuggingFace model identifier.
            device_map: Device to load model onto ("cuda", "cpu", "mps").
            model_config: Optional dict with model configuration overrides.
        """
        self.model_name = model_name
        self.device_map = device_map
        self.model_config = model_config.copy() if model_config else {}

        # Private cached instances (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._config = None
        self._device = None
        self._max_position_embeddings = None

        # LLMLingua-2 specific attributes
        self.max_batch_size = None
        self.max_seq_len = None
        self.max_force_token = None
        self.special_tokens = None
        self.added_tokens = None

    @property
    def model(self):
        """Get the loaded model (lazy loads on first access)."""
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self):
        """Get the loaded tokenizer (lazy loads on first access)."""
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    @property
    def config(self):
        """Get the model config (lazy loads on first access)."""
        if self._config is None:
            self._load()
        return self._config

    @property
    def device(self) -> str:
        """Get the resolved device string."""
        if self._device is None:
            self._load()
        return self._device

    @property
    def max_position_embeddings(self) -> int:
        """Get the maximum position embeddings from model config."""
        if self._max_position_embeddings is None:
            self._load()
        return self._max_position_embeddings

    def _load(self):
        """
        Load the model, tokenizer, and config.

        This method is called automatically on first property access.
        Handles device mapping, dtype selection, and model class detection.
        """
        model_config = self.model_config.copy()

        # Ensure trust_remote_code is set
        trust_remote_code = model_config.get("trust_remote_code", True)
        if "trust_remote_code" not in model_config:
            model_config["trust_remote_code"] = trust_remote_code

        # Load config
        self._config = AutoConfig.from_pretrained(self.model_name, **model_config)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **model_config)

        # Configure padding
        if model_config.get("pad_to_left", True):
            self._tokenizer.padding_side = "left"
            self._tokenizer.pad_token_id = (
                self._config.pad_token_id
                if self._config.pad_token_id
                else self._tokenizer.eos_token_id
            )

        # Determine model class based on architecture
        MODEL_CLASS = (
            AutoModelForTokenClassification
            if any(
                "ForTokenClassification" in ar for ar in self._config.architectures
            )
            else AutoModelForCausalLM
        )

        # Resolve device
        self._device = (
            self.device_map
            if any(key in self.device_map for key in ["cuda", "cpu", "mps"])
            else "cuda"
        )

        # Load model with appropriate settings
        if "cuda" in self.device_map or "cpu" in self.device_map:
            self._model = MODEL_CLASS.from_pretrained(
                self.model_name,
                torch_dtype=model_config.pop(
                    "torch_dtype",
                    "auto" if self.device_map == "cuda" else torch.float32,
                ),
                device_map=self.device_map,
                config=self._config,
                ignore_mismatched_sizes=True,
                **model_config,
            )
        else:
            self._model = MODEL_CLASS.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=model_config.pop("torch_dtype", "auto"),
                pad_token_id=self._tokenizer.pad_token_id,
                **model_config,
            )

        # Store max position embeddings
        self._max_position_embeddings = self._config.max_position_embeddings

    def init_llmlingua2(
        self,
        max_batch_size: int = 50,
        max_force_token: int = 100,
    ):
        """
        Initialize LLMLingua-2 specific settings.

        This adds special tokens for forced retention during compression
        and configures batch processing parameters.

        Args:
            max_batch_size: Maximum batch size for processing.
            max_force_token: Maximum number of force-retained tokens.
        """
        # Ensure model is loaded
        _ = self.model

        seed_everything(42)
        self.max_batch_size = max_batch_size
        self.max_seq_len = 512
        self.max_force_token = max_force_token

        # Extract special tokens
        self.special_tokens = set(
            [
                v
                for k, v in self._tokenizer.special_tokens_map.items()
                if k != "additional_special_tokens"
            ]
        )

        # Add custom tokens for force retention
        self.added_tokens = [f"[NEW{i}]" for i in range(max_force_token)]
        self._tokenizer.add_special_tokens(
            {"additional_special_tokens": self.added_tokens}
        )
        self._model.resize_token_embeddings(len(self._tokenizer))
