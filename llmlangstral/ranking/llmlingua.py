# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""LLMLingua perplexity-based ranking strategies."""

from typing import Callable, List, Optional, Tuple

from .base import PPLBasedRanker
from .registry import RankingRegistry


# Decorators apply bottom-up: "longllmlingua" registers first, then "llmlingua"
# wraps and sets cls.name = "llmlingua" as the canonical identifier. Both keys
# end up in RankingRegistry pointing at this same class; the difference in
# behavior between the two modes is driven by condition_in_question in
# PromptCompressor, not by a separate ranker class.
@RankingRegistry.register("llmlingua")
@RankingRegistry.register("longllmlingua")
class LLMLinguaRanker(PPLBasedRanker):
    """
    LLMLingua/LongLLMLingua perplexity-based ranking.

    Ranks documents by conditional perplexity with respect to a query.
    Uses the main language model (not a separate embedding model).

    This is the native ranking method for LLMLingua compression.
    Lower perplexity = more relevant to the query.

    Note: Both "llmlingua" and "longllmlingua" map to this ranker.
    The difference is in how condition_in_question is handled by
    the caller (PromptCompressor).
    """

    name = "llmlingua"

    def __init__(
        self,
        ppl_fn: Optional[Callable[[str, str, str], float]] = None,
        **kwargs,
    ):
        """
        Initialize LLMLingua ranker.

        Args:
            ppl_fn: Callable to compute conditional perplexity.
                    Signature: ppl_fn(text, question, condition_mode) -> tensor
                    where condition_mode is "none", "before", or "after".
        """
        super().__init__(ppl_fn=ppl_fn, **kwargs)

    def rank(
        self,
        corpus: List[str],
        query: str,
        condition_in_question: str = "none",
        context_tokens_length: Optional[List[int]] = None,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        """
        Rank documents by conditional perplexity.

        Args:
            corpus: List of documents to rank.
            query: Query string.
            condition_in_question: Conditioning mode ("none", "before", "after").
            context_tokens_length: Token lengths for each document (unused but kept for API compatibility).

        Returns:
            List of (index, ppl_score) tuples sorted by perplexity.
        """
        if self.ppl_fn is None:
            raise ValueError(
                "LLMLinguaRanker requires ppl_fn to be provided. "
                "Pass the get_condition_ppl method from PromptCompressor."
            )

        # Append context to query for better conditioning
        augmented_query = query + " We can get the answer to this question in the given documents."

        # Compute PPL for each document
        context_ppl = []
        for doc in corpus:
            ppl = self.ppl_fn(doc, augmented_query, condition_in_question)
            # Handle tensor output
            if hasattr(ppl, "cpu"):
                ppl = ppl.cpu().item()  # pyright: ignore[reportAttributeAccessIssue]
            context_ppl.append(ppl)

        # Sort direction depends on conditioning mode
        # - "none": lower PPL = less predictable = less relevant → sort descending
        # - "after"/"before": lower PPL = more predictable given query = more relevant → sort ascending
        sort_direct = -1 if condition_in_question == "none" else 1

        sorted_results = sorted(
            enumerate(context_ppl),
            key=lambda x: sort_direct * x[1],
        )

        return sorted_results
