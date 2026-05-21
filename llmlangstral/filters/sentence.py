# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Sentence-level filtering for fine-grained content selection."""

from collections import defaultdict
from typing import Any, List, Optional, Tuple

import nltk

from .base import FilterBase


class SentenceLevelFilter(FilterBase):
    """
    Filter sentences by relevance to query within token budget.

    Splits contexts into sentences, ranks them by relevance or PPL,
    and selects the most relevant ones within the token budget.
    Supports keeping first/last sentences and structured segments.
    """

    def filter(
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
    ) -> Tuple[List[str], List[List[Tuple[Any, ...]]]]:
        """
        Filter sentences by relevance and budget.

        Args:
            context: List of context documents.
            target_token: Target token budget.
            keep_first_sentence: Number of first sentences to always keep.
            keep_last_sentence: Number of last sentences to always keep.
            keep_sentence_number: Sentences to keep per document.
            high_priority_bonus: PPL bonus for priority sentences.
            token_budget_ratio: Budget multiplier for sentence selection.
            question: Query for relevance ranking.
            condition_in_question: Conditioning mode.
            rank_method: Ranking algorithm ("longllmlingua" or other).
            context_segs: Structured segments per context.
            context_segs_rate: Compression rates per segment.
            context_segs_compress: Compression flags per segment.

        Returns:
            Tuple of (filtered_contexts, segments_info).
        """

        def keep_sentence(dem_idx: int, sent_keep: int):
            """Mark best sentences with priority bonus."""
            idxs = sorted(dem_g[dem_idx], key=lambda x: sentence_ppl[x])[:sent_keep]
            for idx in idxs:
                sentence_ppl[idx] += high_priority_bonus

        def sync_sentence(sentences, text):
            """Synchronize sentence boundaries with original text."""
            seen_text = 0
            sentence_num = len(sentences)
            new_sentences = []
            for i, s in enumerate(sentences):
                assert s == text[seen_text : seen_text + len(s)]
                if i == sentence_num - 1:
                    new_sentences.append(text[seen_text:])
                    break
                next_sentence_start = text.find(
                    sentences[i + 1][:5], seen_text + len(s)
                )
                new_sentences.append(text[seen_text:next_sentence_start])
                seen_text = next_sentence_start
            assert "".join(new_sentences) == text
            return new_sentences

        # Tokenize contexts into sentences
        sentences = [nltk.sent_tokenize(c) for c in context]
        sentences = [sync_sentence(s, c) for s, c in zip(sentences, context)]

        # Map sentences to documents
        dem_g, s2de, idx = defaultdict(set), defaultdict(int), 0
        for idx_d, s in enumerate(sentences):
            for _ in s:
                dem_g[idx_d].add(idx)
                s2de[idx] = idx_d
                idx += 1

        # Handle structured compression segments
        sen2seg_ratio = {}
        if context_segs is not None:
            assert context_segs_rate is not None
            assert context_segs_compress is not None
            idx = 0
            for idx_d, sentences_each_context in enumerate(sentences):
                segments_length = [len(s) for s in context_segs[idx_d]]
                seg_idx, cur_seg_seen = 0, 0
                for sentence in sentences_each_context:
                    sentence_seg_ratio = []
                    remain = len(sentence)
                    while remain:
                        if segments_length[seg_idx] - cur_seg_seen <= remain:
                            new_seg_len = segments_length[seg_idx] - cur_seg_seen
                            sentence_seg_ratio.append(
                                (
                                    new_seg_len,
                                    context_segs_rate[idx_d][seg_idx],
                                    context_segs_compress[idx_d][seg_idx],
                                )
                            )
                            seg_idx += 1
                            cur_seg_seen = 0
                            remain -= new_seg_len
                        else:
                            sentence_seg_ratio.append(
                                (
                                    remain,
                                    context_segs_rate[idx_d][seg_idx],
                                    context_segs_compress[idx_d][seg_idx],
                                )
                            )
                            cur_seg_seen += remain
                            remain = 0
                    sen2seg_ratio[idx] = sentence_seg_ratio
                    idx += 1

        context_sentences = [s for ii in sentences for s in ii]
        sentence_tokens_length = [
            self.get_token_length(sentence) for sentence in context_sentences
        ]
        N = len(context_sentences)
        flags = list(range(len(context_sentences)))

        # Handle single sentence case
        if len(sentence_tokens_length) == 1:
            segments_info = []
            if context_segs is not None:
                segments_info.append(sen2seg_ratio[0])
            return context, segments_info

        # Rank sentences
        if rank_method == "longllmlingua":
            assert self.ctx.get_condition_ppl_fn is not None
            sentence_ppl = [
                self.ctx.get_condition_ppl_fn(
                    sentence, question, condition_in_question
                )
                .cpu()
                .item()
                for sentence in context_sentences
            ]
            # Apply priority bonuses
            if keep_first_sentence:
                sentence_ppl[:keep_first_sentence] = [
                    ii + high_priority_bonus
                    for ii in sentence_ppl[:keep_first_sentence]
                ]
            if keep_last_sentence:
                sentence_ppl[-keep_last_sentence:] = [
                    ii + high_priority_bonus
                    for ii in sentence_ppl[-keep_last_sentence:]
                ]
            if keep_sentence_number:
                for dem_idx in range(len(sentences)):
                    keep_sentence(dem_idx, keep_sentence_number)
            sort_direct = -1 if condition_in_question == "none" else 1
            sent_sort = sorted(
                enumerate(sentence_ppl), key=lambda x: sort_direct * x[1]
            )
        else:
            assert self.ctx.get_rank_results_fn is not None
            sent_sort = self.ctx.get_rank_results_fn(
                context_sentences,
                question,
                rank_method,
                condition_in_question,
                [0] * len(context_sentences),
            )

        # Select sentences within budget
        sentence_flags = [False] * N
        if target_token < 0:
            target_token = 100
        target_token *= token_budget_ratio

        for idx, _ in sent_sort:
            idx = flags[idx]
            target_token -= sentence_tokens_length[idx]
            sentence_flags[idx] = True
            if target_token < 0:
                break

        # Force include uncompressed segments
        if context_segs is not None:
            for idx in range(N):
                preserved = [sen_seg_info[2] for sen_seg_info in sen2seg_ratio[idx]]
                if False in preserved:
                    sentence_flags[idx] = True

        # Reconstruct filtered contexts
        idx = 0
        res = []
        new_segments_info = []
        for s in sentences:
            tmp = [jj for ii, jj in enumerate(s) if sentence_flags[idx + ii]]
            res.append("".join(tmp))
            if context_segs is not None:
                segment_ratio = []
                for ii in range(len(s)):
                    if sentence_flags[idx + ii]:
                        segment_ratio.extend(sen2seg_ratio[idx + ii])
                new_segments_info.append(segment_ratio)
            idx += len(s)

        return res, new_segments_info
