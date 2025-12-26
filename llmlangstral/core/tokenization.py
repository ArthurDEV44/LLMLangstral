# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Tokenization utilities and perplexity computation."""

import torch
from transformers.cache_utils import DynamicCache


class TokenizationMixin:
    """
    Mixin providing tokenization and perplexity computation utilities.

    This mixin is designed to be used with classes that have:
    - self.model: The language model
    - self.tokenizer: The tokenizer
    - self.device: The device string ("cuda", "cpu", etc.)
    - self.max_position_embeddings: Maximum sequence length

    The main method get_ppl() computes perplexity (or per-token loss)
    for given text, supporting KV-cache for efficient incremental processing.
    """

    def get_ppl(
        self,
        text: str,
        granularity: str = "sentence",
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None,
        condition_mode: str = "none",
        condition_pos_id: int = 0,
    ):
        """
        Compute perplexity or per-token loss for text.

        Args:
            text: Input text to compute PPL for.
            granularity: "sentence" for mean loss, "token" for per-token loss.
            input_ids: Optional pre-tokenized input IDs.
            attention_mask: Optional attention mask.
            past_key_values: Optional KV-cache for incremental processing.
            return_kv: Whether to return updated KV-cache.
            end: Optional end position for processing.
            condition_mode: Slicing mode for conditional PPL:
                - "none": Use all tokens
                - "before": Use tokens before condition_pos_id
                - "after": Use tokens after condition_pos_id
            condition_pos_id: Position for conditional slicing.

        Returns:
            If return_kv=False: Mean loss (sentence) or per-token loss tensor (token).
            If return_kv=True: Tuple of (loss, updated_past_key_values).
        """
        if input_ids is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt")
            input_ids = tokenized_text["input_ids"].to(self.device)
            attention_mask = tokenized_text["attention_mask"].to(self.device)

        if past_key_values is not None:
            # Handle both list format (legacy) and DynamicCache format (transformers 4.50+)
            if isinstance(past_key_values, list):
                past_length = past_key_values[0][0].shape[2]
                # Convert list to DynamicCache for newer transformers
                past_key_values_for_model = DynamicCache.from_legacy_cache(
                    past_key_values
                )
            else:
                past_length = past_key_values.get_seq_length()
                past_key_values_for_model = past_key_values
        else:
            past_length = 0
            past_key_values_for_model = None

        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)

        with torch.no_grad():
            response = self.model(
                input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values_for_model,
                use_cache=True,
            )
            # Convert DynamicCache back to list format for compatibility
            new_past_key_values = response.past_key_values
            if isinstance(new_past_key_values, DynamicCache):
                past_key_values = new_past_key_values.to_legacy_cache()
            else:
                past_key_values = new_past_key_values

        shift_logits = response.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1 : end].contiguous()

        # Flatten the tokens
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)

        if condition_mode == "before":
            loss = loss[:condition_pos_id]
        elif condition_mode == "after":
            loss = loss[condition_pos_id:]

        res = loss.mean() if granularity == "sentence" else loss
        return (res, past_key_values) if return_kv else res
