# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from typing import Any, List, Optional, Union

import torch
from transformers.cache_utils import DynamicCache

from .core import BaseCompressor, CompressionResult, ModelManager
from .filters import (
    ContextLevelFilter,
    FilterContext,
    SentenceLevelFilter,
    TokenLevelFilter,
)
from .mistral_config import DEFAULT_MODEL
from .ranking import RankingRegistry
from .utils import (
    concate_segment_info,
    process_structured_json_data,
    remove_consecutive_commas,
    segment_structured_context,
)


class PromptCompressor(BaseCompressor):
    """
    PromptCompressor is designed for compressing prompts based on a given Mistral language model.

    This class initializes with the language model and its configuration, preparing it for prompt compression tasks.
    The architecture is based on the paper "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models".
    Jiang, Huiqiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. arXiv preprint arXiv:2310.05736 (2023).

    Args:
        model_name (str, optional): The name of the Mistral model to be loaded. Default is "mistralai/Mistral-7B-v0.3".
        device_map (str, optional): The device to load the model onto, e.g., "cuda" for GPU. Default is "cuda".
        model_config (dict, optional): A dictionary containing the configuration parameters for the model. Default is None (no overrides).

    Example:
        >>> compress_method = PromptCompressor(model_name="mistralai/Mistral-7B-v0.3", device_map="cpu")
        >>> context = ["This is the first context sentence.", "Here is another context sentence."]
        >>> result = compress_method.compress_prompt(context, use_context_level_filter=True, target_token=5)
        >>> print(result["compressed_prompt"])
        # This will print the compressed version of the context.

    Note:
        The `PromptCompressor` class requires the Hugging Face Transformers library and an appropriate environment to load and run the models.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_map: str = "cuda",
        model_config: Optional[dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.retrieval_model = None
        self.retrieval_model_name = None
        self.cache_bos_num = 10
        self.prefix_bos_num = 100

        # Use ModelManager for centralized model loading
        self._model_manager = ModelManager(model_name, device_map, model_config or {})
        self.context_idxs = []
        self._filter_ctx = None  # Lazy initialized

    # Filter context for delegation
    @property
    def _filters(self) -> FilterContext:
        """Lazy-initialize filter context with callbacks."""
        if self._filter_ctx is None:
            self._filter_ctx = FilterContext(
                tokenizer=self.tokenizer,
                device=self.device,
                max_position_embeddings=self.max_position_embeddings,
                cache_bos_num=self.cache_bos_num,
                get_ppl_fn=self.get_ppl,
                get_condition_ppl_fn=self.get_condition_ppl,
                get_rank_results_fn=self.get_rank_results,
            )
        return self._filter_ctx

    # Delegation properties for backward compatibility
    @property
    def model(self):
        """Get the loaded model (delegated to ModelManager)."""
        return self._model_manager.model

    @property
    def tokenizer(self):
        """Get the loaded tokenizer (delegated to ModelManager)."""
        return self._model_manager.tokenizer

    @property
    def device(self):
        """Get the device string (delegated to ModelManager)."""
        return self._model_manager.device

    @property
    def max_position_embeddings(self):
        """Get max position embeddings (delegated to ModelManager)."""
        return self._model_manager.max_position_embeddings

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
        assert self.tokenizer is not None, "Tokenizer must be loaded"
        if input_ids is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt")
            input_ids = tokenized_text["input_ids"].to(self.device)
            attention_mask = tokenized_text["attention_mask"].to(self.device)
        if past_key_values is not None:
            # Handle both list format (legacy) and DynamicCache format (transformers 4.50+)
            if isinstance(past_key_values, list):
                past_length = past_key_values[0][0].shape[2]
                # Convert list to DynamicCache for newer transformers
                past_key_values_for_model = DynamicCache.from_legacy_cache(past_key_values)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                past_length = past_key_values.get_seq_length()
                past_key_values_for_model = past_key_values
        else:
            past_length = 0
            past_key_values_for_model = None
        if end is None:
            end = input_ids.shape[1]
        end = min(end, past_length + self.max_position_embeddings)
        assert self.model is not None, "Model must be loaded before computing PPL"
        assert attention_mask is not None, "attention_mask must be set"
        with torch.no_grad():
            response = self.model(
                input_ids[:, past_length:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=past_key_values_for_model,
                use_cache=True,
            )
            # Convert DynamicCache back to list format for compatibility with rest of code
            new_past_key_values = response.past_key_values
            if isinstance(new_past_key_values, DynamicCache):
                past_key_values = new_past_key_values.to_legacy_cache()  # pyright: ignore[reportAttributeAccessIssue]
            else:
                past_key_values = new_past_key_values

        shift_logits = response.logits[..., :-1, :].contiguous()  # type: ignore[index]
        shift_labels = input_ids[..., past_length + 1 : end].contiguous()
        # Flatten the tokens
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)  # type: ignore[index]
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

    def __call__(self, *args, **kwargs):
        return self.compress_prompt(*args, **kwargs)

    def compress(
        self,
        context: Union[str, List[str]],
        **kwargs,
    ) -> CompressionResult:
        """Implement BaseCompressor.compress() as a thin wrapper around compress_prompt().

        Returns a CompressionResult dataclass rather than the legacy dict;
        callers expecting the dict shape should keep using compress_prompt().
        """
        if isinstance(context, str):
            context = [context]
        result_dict = self.compress_prompt(context, **kwargs)
        return CompressionResult(
            compressed_prompt=result_dict["compressed_prompt"],
            origin_tokens=result_dict["origin_tokens"],
            compressed_tokens=result_dict["compressed_tokens"],
            ratio=result_dict["ratio"],
            rate=result_dict["rate"],
            saving=result_dict["saving"],
            compressed_prompt_list=result_dict.get("compressed_prompt_list", []),
            fn_labeled_original_prompt=result_dict.get(
                "fn_labeled_original_prompt", ""
            ),
        )

    def compress_json(
        self,
        json_data: dict[str, Any],
        json_config: Union[str, dict[str, Any]],
        instruction: str = "",
        question: str = "",
        rate: float = 0.5,
        target_token: float = -1,
        iterative_size: int = 200,
        use_sentence_level_filter: bool = False,
        use_keyvalue_level_filter: bool = False,
        use_token_level_filter: bool = True,
        keep_split: bool = False,
        keep_first_sentence: int = 0,
        keep_last_sentence: int = 0,
        keep_sentence_number: int = 0,
        high_priority_bonus: int = 100,
        context_budget: str = "+100",
        token_budget_ratio: float = 1.4,
        condition_in_question: str = "none",
        reorder_keyvalue: str = "original",
        condition_compare: bool = False,
        rank_method: str = "llmlingua",
    ):
        context, force_context_ids = process_structured_json_data(
            json_data, json_config
        )
        compressed_res = self.structured_compress_prompt(
            context=context,
            instruction=instruction,
            question=question,
            rate=rate,
            target_token=target_token,
            iterative_size=iterative_size,
            force_context_ids=force_context_ids,
            use_sentence_level_filter=use_sentence_level_filter,
            use_context_level_filter=use_keyvalue_level_filter,
            use_token_level_filter=use_token_level_filter,
            keep_split=keep_split,
            keep_first_sentence=keep_first_sentence,
            keep_last_sentence=keep_last_sentence,
            keep_sentence_number=keep_sentence_number,
            high_priority_bonus=high_priority_bonus,
            context_budget=context_budget,
            token_budget_ratio=token_budget_ratio,
            condition_in_question=condition_in_question,
            reorder_context=reorder_keyvalue,
            condition_compare=condition_compare,
            add_instruction=False,
            rank_method=rank_method,
            concate_question=False,
            strict_preserve_uncompressed=False,
        )
        compressed_json_text = remove_consecutive_commas(
            compressed_res["compressed_prompt"]
        )
        compressed_res["compressed_prompt"] = json.loads(compressed_json_text)
        return compressed_res

    def structured_compress_prompt(
        self,
        context: List[str],
        instruction: str = "",
        question: str = "",
        rate: float = 0.5,
        target_token: float = -1,
        iterative_size: int = 200,
        force_context_ids: Optional[List[int]] = None,
        force_context_number: Optional[int] = None,
        use_sentence_level_filter: bool = False,
        use_context_level_filter: bool = True,
        use_token_level_filter: bool = True,
        keep_split: bool = False,
        keep_first_sentence: int = 0,
        keep_last_sentence: int = 0,
        keep_sentence_number: int = 0,
        high_priority_bonus: int = 100,
        context_budget: str = "+100",
        token_budget_ratio: float = 1.4,
        condition_in_question: str = "none",
        reorder_context: str = "original",
        dynamic_context_compression_ratio: float = 0.0,
        condition_compare: bool = False,
        add_instruction: bool = False,
        rank_method: str = "llmlingua",
        concate_question: bool = True,
        strict_preserve_uncompressed: bool = True,
    ):
        """
        Compresses the given prompt context based on a specified structure.

        Each element of context should be segmented using one or more non-nested '<llmlingua></llmlingua>' tags.
        Each '<llmlingua>' tag can include optional parameters 'rate' and 'compress' (e.g., '<llmlingua, rate=0.3, compress=True>'),
        indicating the compression rate for that segment. Default values are 'rate=rate' and 'compress=True'.
        When 'compress' is set to False, it overrides the 'rate' parameter, resulting in no compression for that segment.

        Args:
            context (List[str]): List of context strings divided by '<llmlingua></llmlingua>' tags with optional compression settings.
            instruction (str, optional): Additional instruction text to be included in the prompt. Default is an empty string.
            question (str, optional): A specific question that the prompt is addressing. Default is an empty string.
            rate (float, optional): The compression rate is defined the same as in paper "Language Modeling Is Compression".
                Delétang, Grégoire, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern,
                Jordi Grau-Moya et al. "Language modeling is compression." arXiv preprint arXiv:2309.10668 (2023):
                .. math::\text{Compression Rate} = \frac{\text{Compressed Size}}{\text{Raw Size}}
                Default is 0.5. The actual compression rate is generally lower than the specified target, but there can be
                fluctuations due to differences in tokenizers. If specified, it should be a float less than or equal
                to 1.0, representing the target compression rate. ``rate``, is applicable only within the context-level filter
                and the sentence-level filter. In the token-level filter, the rate for each segment overrides the global rate.
                However, for segments where no specific rate is defined, the global rate serves as the default value. The final
                compression rate of the entire text is a composite result of multiple compression rates applied across different sections.
            target_token (float, optional): The global maximum number of tokens to be achieved. Default is -1, indicating no
                specific target. The actual number of tokens after compression should generally be less than the specified target_token,
                but there can be fluctuations due to differences in tokenizers. If specified, compression will be based on the target_token as
                the sole criterion, overriding the ``rate``. ``target_token``, is applicable only within the context-level
                filter and the sentence-level filter. In the token-level filter, the rate for each segment overrides the global target token.
                However, for segments where no specific rate is defined, the global rate calculated from global target token serves
                as the default value. The final target token of the entire text is a composite result of multiple compression rates
                applied across different sections.
            iterative_size (int, optional): The number of tokens to consider in each iteration of compression. Default is 200.
            force_context_ids (List[int], optional): List of specific context IDs to always include in the compressed result. Default is None.
            force_context_number (int, optional): The number of context sections to forcibly include. Default is None.
            use_sentence_level_filter (bool, optional): Whether to apply sentence-level filtering in compression. Default is False.
            use_context_level_filter (bool, optional): Whether to apply context-level filtering in compression. Default is True.
            use_token_level_filter (bool, optional): Whether to apply token-level filtering in compression. Default is True.
            keep_split (bool, optional): Whether to preserve the original separators without compression. Default is False.
            keep_first_sentence (int, optional): Number of sentences to forcibly preserve from the start of the context. Default is 0.
            keep_last_sentence (int, optional): Number of sentences to forcibly preserve from the end of the context. Default is 0.
            keep_sentence_number (int, optional): Total number of sentences to forcibly preserve in the compression. Default is 0.
            high_priority_bonus (int, optional): Bonus score for high-priority sentences to influence their likelihood of being retained. Default is 100.
            context_budget (str, optional): Token budget for the context-level filtering, expressed as a string to indicate flexibility. Default is "+100".
            token_budget_ratio (float, optional): Ratio to adjust token budget during sentence-level filtering. Default is 1.4.
            condition_in_question (str, optional): Specific condition to apply to question in the context. Default is "none".
            reorder_context (str, optional): Strategy for reordering context in the compressed result. Default is "original".
            dynamic_context_compression_ratio (float, optional): Ratio for dynamically adjusting context compression. Default is 0.0.
            condition_compare (bool, optional): Whether to enable condition comparison during token-level compression. Default is False.
            add_instruction (bool, optional): Whether to add the instruction to the prompt prefix. Default is False.
            rank_method (str, optional): Method used for ranking elements during compression. Default is "llmlingua".
            concate_question (bool, optional): Whether to concatenate the question to the compressed prompt. Default is True.

        Returns:
            dict: A dictionary containing:
                - "compressed_prompt" (str): The resulting compressed prompt.
                - "origin_tokens" (int): The original number of tokens in the input.
                - "compressed_tokens" (int): The number of tokens in the compressed output.
                - "ratio" (str): The compression ratio achieved, calculated as the original token number divided by the token number after compression.
                - "rate" (str): The compression rate achieved, in a human-readable format.
                - "saving" (str): Number of tokens saved by compression.
        """
        if not context:
            context = [" "]
        if isinstance(context, str):
            context = [context]
        assert self.tokenizer is not None, "Tokenizer must be loaded"
        context = [
            self.tokenizer.decode(self.tokenizer(c, add_special_tokens=False).input_ids)
            for c in context
        ]
        context_tokens_length = [self.get_token_length(c) for c in context]
        instruction_tokens_length, question_tokens_length = self.get_token_length(
            instruction
        ), self.get_token_length(question)
        if target_token == -1:
            target_token = (
                (
                    instruction_tokens_length
                    + question_tokens_length
                    + sum(context_tokens_length)
                )
                * rate
                - instruction_tokens_length
                - (question_tokens_length if concate_question else 0)
            )
        else:
            rate = target_token / sum(context_tokens_length)
        (
            context,
            context_segs,
            context_segs_rate,
            context_segs_compress,
        ) = self.segment_structured_context(context, rate)
        return self.compress_prompt(
            context,
            instruction,
            question,
            rate,
            target_token,
            iterative_size,
            force_context_ids,
            force_context_number,
            use_sentence_level_filter,
            use_context_level_filter,
            use_token_level_filter,
            keep_split,
            keep_first_sentence,
            keep_last_sentence,
            keep_sentence_number,
            high_priority_bonus,
            context_budget,
            token_budget_ratio,
            condition_in_question,
            reorder_context,
            dynamic_context_compression_ratio,
            condition_compare,
            add_instruction,
            rank_method,
            concate_question,
            context_segs=context_segs,
            context_segs_rate=context_segs_rate,
            context_segs_compress=context_segs_compress,
            strict_preserve_uncompressed=strict_preserve_uncompressed,
        )

    def compress_prompt(
        self,
        context: List[str],
        instruction: str = "",
        question: str = "",
        rate: float = 0.5,
        target_token: float = -1,
        iterative_size: int = 200,
        force_context_ids: Optional[List[int]] = None,
        force_context_number: Optional[int] = None,
        use_sentence_level_filter: bool = False,
        use_context_level_filter: bool = True,
        use_token_level_filter: bool = True,
        keep_split: bool = False,
        keep_first_sentence: int = 0,
        keep_last_sentence: int = 0,
        keep_sentence_number: int = 0,
        high_priority_bonus: int = 100,
        context_budget: str = "+100",
        token_budget_ratio: float = 1.4,
        condition_in_question: str = "none",
        reorder_context: str = "original",
        dynamic_context_compression_ratio: float = 0.0,
        condition_compare: bool = False,
        add_instruction: bool = False,
        rank_method: str = "llmlingua",
        concate_question: bool = True,
        context_segs: Optional[List[List[str]]] = None,
        context_segs_rate: Optional[List[List[float]]] = None,
        context_segs_compress: Optional[List[List[bool]]] = None,
        target_context: int = -1,
        context_level_rate: float = 1.0,
        context_level_target_token: int = -1,
        strict_preserve_uncompressed: bool = True,
    ):
        """
        Compresses the given context.

        Args:
            context (List[str]): List of context strings that form the basis of the prompt.
            instruction (str, optional): Additional instruction text to be included in the prompt. Default is an empty string.
            question (str, optional): A specific question that the prompt is addressing. Default is an empty string.
            rate (float, optional): The maximum compression rate target to be achieved. The compression rate is defined
                the same as in paper "Language Modeling Is Compression". Delétang, Grégoire, Anian Ruoss, Paul-Ambroise Duquenne,
                Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya et al. "Language modeling is compression."
                arXiv preprint arXiv:2309.10668 (2023):
                .. math::\text{Compression Rate} = \frac{\text{Compressed Size}}{\text{Raw Size}}
                Default is 0.5. The actual compression rate is generally lower than the specified target, but there can be
                fluctuations due to differences in tokenizers. If specified, it should be a float less than or equal
                to 1.0, representing the target compression rate.
            target_token (float, optional): The maximum number of tokens to be achieved. Default is -1, indicating no specific target.
                The actual number of tokens after compression should generally be less than the specified target_token, but there can
                be fluctuations due to differences in tokenizers. If specified, compression will be based on the target_token as
                the sole criterion, overriding the ``rate``.
            iterative_size (int, optional): The number of tokens to consider in each iteration of compression. Default is 200.
            force_context_ids (List[int], optional): List of specific context IDs to always include in the compressed result. Default is None.
            force_context_number (int, optional): The number of context sections to forcibly include. Default is None.
            use_sentence_level_filter (bool, optional): Whether to apply sentence-level filtering in compression. Default is False.
            use_context_level_filter (bool, optional): Whether to apply context-level filtering in compression. Default is True.
            use_token_level_filter (bool, optional): Whether to apply token-level filtering in compression. Default is True.
            keep_split (bool, optional): Whether to preserve the original separators without compression. Default is False.
            keep_first_sentence (int, optional): Number of sentences to forcibly preserve from the start of the context. Default is 0.
            keep_last_sentence (int, optional): Number of sentences to forcibly preserve from the end of the context. Default is 0.
            keep_sentence_number (int, optional): Total number of sentences to forcibly preserve in the compression. Default is 0.
            high_priority_bonus (int, optional): Bonus score for high-priority sentences to influence their likelihood of being retained. Default is 100.
            context_budget (str, optional): Token budget for the context-level filtering, expressed as a string to indicate flexibility. Default is "+100".
            token_budget_ratio (float, optional): Ratio to adjust token budget during sentence-level filtering. Default is 1.4.
            condition_in_question (str, optional): Specific condition to apply to question in the context. Default is "none".
            reorder_context (str, optional): Strategy for reordering context in the compressed result. Default is "original".
            dynamic_context_compression_ratio (float, optional): Ratio for dynamically adjusting context compression. Default is 0.0.
            condition_compare (bool, optional): Whether to enable condition comparison during token-level compression. Default is False.
            add_instruction (bool, optional): Whether to add the instruction to the prompt prefix. Default is False.
            rank_method (str, optional): Method used for ranking elements during compression. Default is "llmlingua".
            concate_question (bool, optional): Whether to concatenate the question to the compressed prompt. Default is True.

            target_context (int, optional): The maximum number of contexts to be achieved. Default is -1, indicating no specific target.
            context_level_rate (float, optional): The minimum compression rate target to be achieved in context level. Default is 1.0.
            context_level_target_token (float, optional): The maximum number of tokens to be achieved in context level compression.
                Default is -1, indicating no specific target. Only used in the coarse-to-fine compression senario.
            force_context_ids (List[int], optional): List of specific context IDs to always include in the compressed result. Default is None.
        Returns:
            dict: A dictionary containing:
                - "compressed_prompt" (str): The resulting compressed prompt.
                - "origin_tokens" (int): The original number of tokens in the input.
                - "compressed_tokens" (int): The number of tokens in the compressed output.
                - "ratio" (str): The compression ratio achieved, calculated as the original token number divided by the token number after compression.
                - "rate" (str): The compression rate achieved, in a human-readable format.
                - "saving" (str): Number of tokens saved by compression.
        """
        assert (
            rate <= 1.0
        ), "Error: 'rate' must not exceed 1.0. The value of 'rate' indicates compression rate and must be within the range [0, 1]."

        if not context:
            context = [" "]
        if isinstance(context, str):
            context = [context]
        assert not (
            rank_method == "longllmlingua" and not question
        ), "In the LongLLMLingua, it is necessary to set a question."
        if condition_compare and "_condition" not in condition_in_question:
            condition_in_question += "_condition"
        if rank_method == "longllmlingua":
            if condition_in_question == "none":
                condition_in_question = "after"
        elif rank_method == "llmlingua":
            condition_in_question = (
                "none"
                if "_condition" not in condition_in_question
                else "none_condition"
            )
        origin_tokens = self.get_token_length(
            "\n\n".join([instruction] + context + [question]).strip()
        )
        context_tokens_length = [self.get_token_length(c) for c in context]
        instruction_tokens_length, question_tokens_length = self.get_token_length(
            instruction
        ), self.get_token_length(question)
        if target_token == -1:
            target_token = (
                (
                    instruction_tokens_length
                    + question_tokens_length
                    + sum(context_tokens_length)
                )
                * rate
                - instruction_tokens_length
                - (question_tokens_length if concate_question else 0)
            )
        condition_flag = "_condition" in condition_in_question
        condition_in_question = condition_in_question.replace("_condition", "")

        if len(context) > 1 and use_context_level_filter:
            context, dynamic_ratio, context_used = self.control_context_budget(
                context,
                context_tokens_length,
                target_token,
                force_context_ids,
                force_context_number,
                question,
                condition_in_question,
                reorder_context=reorder_context,
                dynamic_context_compression_ratio=dynamic_context_compression_ratio,
                rank_method=rank_method,
                context_budget=context_budget,
                context_segs=context_segs,
                context_segs_rate=context_segs_rate,
                context_segs_compress=context_segs_compress,
                strict_preserve_uncompressed=strict_preserve_uncompressed,
            )
            if context_segs is not None:
                assert context_segs_rate is not None
                assert context_segs_compress is not None
                context_segs = [context_segs[idx] for idx in context_used]
                context_segs_rate = [context_segs_rate[idx] for idx in context_used]
                context_segs_compress = [
                    context_segs_compress[idx] for idx in context_used
                ]
        else:
            dynamic_ratio = [0.0] * len(context)

        segments_info = []
        if use_sentence_level_filter:
            context, segments_info = self.control_sentence_budget(
                context,
                target_token,
                keep_first_sentence=keep_first_sentence,
                keep_last_sentence=keep_last_sentence,
                keep_sentence_number=keep_sentence_number,
                high_priority_bonus=high_priority_bonus,
                token_budget_ratio=token_budget_ratio,
                question=question,
                condition_in_question=condition_in_question,
                rank_method=rank_method,
                context_segs=context_segs,
                context_segs_rate=context_segs_rate,
                context_segs_compress=context_segs_compress,
            )
        elif context_segs is not None:
            assert context_segs_rate is not None
            assert context_segs_compress is not None
            for context_idx in range(len(context)):
                segments_info.append(
                    [
                        (len(seg_text), seg_rate, seg_compress)
                        for seg_text, seg_rate, seg_compress in zip(
                            context_segs[context_idx],
                            context_segs_rate[context_idx],
                            context_segs_compress[context_idx],
                        )
                    ]
                )
        segments_info = [
            self.concate_segment_info(segment_info) for segment_info in segments_info
        ]

        if condition_flag:
            prefix = question + "\n\n" + instruction if add_instruction else question
            if (
                self.get_token_length(prefix + "\n\n") + iterative_size * 2
                > self.max_position_embeddings
            ):
                assert self.tokenizer is not None, "Tokenizer must be loaded"
                tokens = self.tokenizer(prefix, add_special_tokens=False).input_ids
                prefix = self.tokenizer.decode(
                    tokens[: self.prefix_bos_num]
                    + tokens[
                        len(tokens)
                        - self.max_position_embeddings
                        + 2
                        + self.prefix_bos_num
                        + 2 * iterative_size :
                    ]
                )
            start = self.get_prefix_length(prefix + "\n\n", context[0])
            context = [prefix] + context
        else:
            start = 0

        if use_token_level_filter:
            compressed_context = self.iterative_compress_prompt(
                context,
                target_token,
                iterative_size=iterative_size,
                keep_split=keep_split,
                start=start,
                dynamic_ratio=dynamic_ratio,
                condition_compare=condition_compare,
                segments_info=segments_info,
            )
            assert self.tokenizer is not None, "Tokenizer must be loaded"
            compressed_prompt = (
                self.tokenizer.batch_decode(compressed_context[0])[0]
                .replace("<s> ", "")
                .replace("<s>", "")
            )
        else:
            if condition_flag:
                context = context[1:]
            compressed_prompt = "\n\n".join(context)

        res = []
        if instruction:
            res.append(instruction)
        if compressed_prompt.strip():
            res.append(compressed_prompt)
        if question and concate_question:
            res.append(question)

        compressed_prompt = "\n\n".join(res)

        compressed_tokens = self.get_token_length(compressed_prompt)
        return CompressionResult.from_compression(
            compressed_prompt=compressed_prompt,
            origin_tokens=origin_tokens,
            compressed_tokens=compressed_tokens,
        ).to_dict()

    def get_token_length(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> int:
        assert self.tokenizer is not None, "Tokenizer must be loaded"
        return len(
            self.tokenizer(text, add_special_tokens=add_special_tokens).input_ids
        )

    def get_prefix_length(self, prefix: str, text: str) -> int:
        assert self.tokenizer is not None, "Tokenizer must be loaded"
        possible_prefix_token = max(self.get_token_length(prefix, False) - 3, 1)
        full_input_ids = self.tokenizer(
            prefix + text[:100], add_special_tokens=False
        ).input_ids
        i = possible_prefix_token
        for i in range(possible_prefix_token, len(full_input_ids)):
            cur_prefix = self.tokenizer.decode(full_input_ids[:i])
            if cur_prefix == prefix:
                break
        return i

    def get_condition_ppl(
        self,
        text: str,
        question: str,
        condition_in_question: str = "none",
        granularity: str = "sentence",
    ):
        if condition_in_question == "none":
            return self.get_ppl(text, granularity=granularity)
        elif condition_in_question == "before":
            return self.get_ppl(
                question + text,
                granularity=granularity,
                condition_mode="after",
                condition_pos_id=self.get_token_length(question) - 1,
            )
        elif condition_in_question == "after":
            return self.get_ppl(
                text + question,
                granularity=granularity,
                condition_mode="after",
                condition_pos_id=self.get_token_length(text) - 1,
            )

    # =========================================================================
    # Filtering methods - delegated to filters/ module
    # =========================================================================

    def control_context_budget(
        self,
        context: List[str],
        context_tokens_length: List[int],
        target_token: float,
        force_context_ids: Optional[List[int]] = None,
        force_context_number: Optional[int] = None,
        question: str = "",
        condition_in_question: str = "none",
        reorder_context: str = "original",
        dynamic_context_compression_ratio: float = 0.0,
        rank_method: str = "longllmlingua",
        context_budget: str = "+100",
        context_segs: Optional[List[List[str]]] = None,
        context_segs_rate: Optional[List[List[float]]] = None,
        context_segs_compress: Optional[List[List[bool]]] = None,
        strict_preserve_uncompressed: bool = True,
    ):
        """Delegate to ContextLevelFilter."""
        filter_obj = ContextLevelFilter(self._filters)
        res, dynamic_ratio, used, context_idxs_new = filter_obj.filter(
            context=context,
            context_tokens_length=context_tokens_length,
            target_token=target_token,
            force_context_ids=force_context_ids,
            force_context_number=force_context_number,
            question=question,
            condition_in_question=condition_in_question,
            reorder_context=reorder_context,
            dynamic_context_compression_ratio=dynamic_context_compression_ratio,
            rank_method=rank_method,
            context_budget=context_budget,
            context_segs=context_segs,
            context_segs_rate=context_segs_rate,
            context_segs_compress=context_segs_compress,
            strict_preserve_uncompressed=strict_preserve_uncompressed,
        )
        self.context_idxs.append(context_idxs_new)
        return res, dynamic_ratio, used

    def control_sentence_budget(
        self,
        context: List[str],
        target_token: float,
        keep_first_sentence: int = 0,
        keep_last_sentence: int = 0,
        keep_sentence_number: int = 0,
        high_priority_bonus: int = 100,
        token_budget_ratio: float = 1.4,
        question: str = "",
        condition_in_question: str = "none",
        rank_method: str = "longllmlingua",
        context_segs: Optional[List[List[str]]] = None,
        context_segs_rate: Optional[List[List[float]]] = None,
        context_segs_compress: Optional[List[List[bool]]] = None,
    ):
        """Delegate to SentenceLevelFilter."""
        filter_obj = SentenceLevelFilter(self._filters)
        return filter_obj.filter(
            context=context,
            target_token=target_token,
            keep_first_sentence=keep_first_sentence,
            keep_last_sentence=keep_last_sentence,
            keep_sentence_number=keep_sentence_number,
            high_priority_bonus=high_priority_bonus,
            token_budget_ratio=token_budget_ratio,
            question=question,
            condition_in_question=condition_in_question,
            rank_method=rank_method,
            context_segs=context_segs,
            context_segs_rate=context_segs_rate,
            context_segs_compress=context_segs_compress,
        )

    def iterative_compress_prompt(
        self,
        context: List[str],
        target_token: float,
        iterative_size: int = 200,
        keep_split: bool = False,
        split_token_id: int = 13,
        start: int = 0,
        dynamic_ratio: Optional[list[float]] = None,
        condition_compare: bool = False,
        segments_info: Optional[List[List[tuple[int, float, bool]]]] = None,
    ):
        """Delegate to TokenLevelFilter."""
        filter_obj = TokenLevelFilter(self._filters)
        return filter_obj.filter(
            context=context,
            target_token=target_token,
            iterative_size=iterative_size,
            keep_split=keep_split,
            split_token_id=split_token_id,
            start=start,
            dynamic_ratio=dynamic_ratio,
            condition_compare=condition_compare,
            segments_info=segments_info,
        )

    def recover(
        self,
        original_prompt: str,
        compressed_prompt: str,
        response: str,
    ):
        assert self.tokenizer is not None, "Tokenizer must be loaded"
        tokenizer = self.tokenizer

        def match_from_compressed(response_word):
            response_input_ids = tokenizer(
                response_word, add_special_tokens=False
            )["input_ids"]
            response_set, response_c = set(response_input_ids), defaultdict(list)
            for idx in range(M):
                if original_input_ids[idx] in response_set:
                    response_c[original_input_ids[idx]].append(idx)
            res, res_min, res_c = None, float("inf"), 1
            n = len(response_input_ids)
            for start_pos in response_c[response_input_ids[0]]:
                x, y, c = 0, start_pos, 1
                for x in range(1, n):
                    idx = bisect.bisect_right(response_c[response_input_ids[x]], y)
                    if (
                        idx >= len(response_c[response_input_ids[x]])
                        or response_c[response_input_ids[x]][idx] - y > 10
                    ):
                        continue
                    c += 1
                    y = response_c[response_input_ids[x]][idx]
                if c > res_c:
                    res_c = c
                    res_min = y - start_pos + 1
                    res = (start_pos, y + 1)
                elif c == res_c and y - start_pos + 1 < res_min:
                    res_min = y - start_pos + 1
                    res = (start_pos, y + 1)

            if res is None:
                return response_word
            # while l > 0 and not tokenizer.convert_ids_to_tokens(original_input_ids[l]).startswith("_"):
            #     l -= 1
            # while r < M - 1 and not tokenizer.convert_ids_to_tokens(original_input_ids[l]).startswith("_"):
            #     l -= 1
            return tokenizer.decode(original_input_ids[res[0] : res[1]])

        response_words = response.split(" ")

        original_input_ids = tokenizer(original_prompt, add_special_tokens=False)[
            "input_ids"
        ]
        N, M = len(response_words), len(original_input_ids)
        recovered_response_words = []
        pos = 0
        while pos < N:
            if response_words[pos] not in compressed_prompt:
                recovered_response_words.append(response_words[pos])
                pos += 1
                continue
            r = pos
            while (
                r + 1 < N
                and " ".join(response_words[pos : r + 2]) in compressed_prompt
            ):
                r += 1

            match_words = match_from_compressed(
                " ".join(response_words[pos : r + 1])
            )
            recovered_response_words.append(match_words)
            pos = r + 1
        return " ".join(recovered_response_words)

    def get_rank_results(
        self,
        context: list[str],
        question: str,
        rank_method: str,
        condition_in_question: str,
        context_tokens_length: list[int],
    ):
        """
        Rank context documents by relevance to the question.

        Delegates to the RankingRegistry for pluggable ranking strategies.

        Args:
            context: List of documents/sentences to rank.
            question: Query string for ranking.
            rank_method: Name of the ranking strategy (e.g., "bm25", "llmlingua").
            condition_in_question: Conditioning mode for PPL-based ranking.
            context_tokens_length: Token lengths for each context (used by llmlingua).

        Returns:
            List of (index, score) tuples sorted by relevance.
        """
        # Build kwargs for ranker instantiation
        ranker_kwargs: dict[str, Any] = {
            "device": self.device,
        }

        # LLMLingua/LongLLMLingua need the PPL function
        if rank_method in ["llmlingua", "longllmlingua"]:
            ranker_kwargs["ppl_fn"] = self.get_condition_ppl

        # Get ranker instance from registry
        ranker = RankingRegistry.get(rank_method, **ranker_kwargs)

        # Call rank with appropriate kwargs
        return ranker.rank(
            corpus=context,
            query=question,
            condition_in_question=condition_in_question,
            context_tokens_length=context_tokens_length,
        )

    def segment_structured_context(self, context, global_rate):
        return segment_structured_context(context, global_rate)

    def concate_segment_info(self, segment_info):
        return concate_segment_info(segment_info)
