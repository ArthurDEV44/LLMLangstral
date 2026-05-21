# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Context-level filtering for document selection."""

from typing import Any, List, Optional, Tuple

from .base import FilterBase


class ContextLevelFilter(FilterBase):
    """
    Filter contexts by relevance to query within token budget.

    Ranks documents using configurable ranking methods and selects
    the most relevant ones that fit within the target token budget.
    Supports dynamic compression ratios for position-based weighting.
    """

    def filter(
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
    ) -> Tuple[List[str], List[float], List[int], List[Any]]:
        """
        Filter contexts by relevance and budget.

        Args:
            context: List of context documents to filter.
            context_tokens_length: Token count for each context.
            target_token: Target token budget.
            force_context_ids: Indices of contexts to always include.
            force_context_number: Maximum number of contexts to include.
            question: Query for relevance ranking.
            condition_in_question: Conditioning mode ("none", "before", "after").
            reorder_context: Reordering strategy ("original", "two_stage").
            dynamic_context_compression_ratio: Position-based ratio adjustment.
            rank_method: Ranking algorithm to use.
            context_budget: Budget adjustment expression (e.g., "+100").
            context_segs: Structured segments per context.
            context_segs_rate: Compression rates per segment.
            context_segs_compress: Compression flags per segment.
            strict_preserve_uncompressed: Force include uncompressed segments.

        Returns:
            Tuple of (filtered_contexts, dynamic_ratios, used_indices).
        """
        # Rank contexts by relevance
        assert self.ctx.get_rank_results_fn is not None
        demostrations_sort = self.ctx.get_rank_results_fn(
            context,
            question,
            rank_method,
            condition_in_question,
            context_tokens_length,
        )

        if target_token < 0:
            target_token = 100
        target_token = eval("target_token" + context_budget)

        res = []
        used = force_context_ids if force_context_ids is not None else []

        # Force include contexts with uncompressed segments
        if context_segs is not None and strict_preserve_uncompressed:
            assert context_segs_compress is not None
            for idx, _ in enumerate(context):
                if False in context_segs_compress[idx] and idx not in used:
                    used.append(idx)

        # Track context indices for ranking
        context_idxs = [x for idx, (x, _) in enumerate(demostrations_sort)]

        # Select contexts within budget
        for idx, _ in demostrations_sort:
            if idx >= len(context_tokens_length):
                continue
            target_token -= context_tokens_length[idx]
            if idx not in used:
                used.append(idx)
            if target_token < 0 or (
                force_context_number is not None and len(res) >= force_context_number
            ):
                break

        original_used = used

        # Reorder contexts based on strategy
        if reorder_context == "original":
            used = sorted(used)
        elif reorder_context == "two_stage":
            left, right = [_ for idx, _ in enumerate(used) if idx % 2 == 0], [
                _ for idx, _ in enumerate(used) if idx % 2 == 1
            ]
            used = left + right[::-1]

        # Calculate dynamic compression ratios
        if dynamic_context_compression_ratio > 0:
            N = len(used)
            dynamic_ratio = [
                i * (abs(dynamic_context_compression_ratio) / (N - 1)) if N > 1 else 0
                for i in range(-(N - 1), N, 2)
            ][::-1]
            dynamic_ratio_map = {i: j for i, j in zip(original_used, dynamic_ratio)}
            dynamic_ratio = [dynamic_ratio_map[i] for i in used]
        else:
            dynamic_ratio = [0.0] * len(used)

        res = [context[idx] for idx in used if idx < len(context)]
        return res, dynamic_ratio, used, context_idxs
