# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Token-level filtering for fine-grained compression."""

from typing import Any, List, Optional, Tuple

import torch

from .base import FilterBase


class TokenLevelFilter(FilterBase):
    """
    Filter tokens based on perplexity for fine-grained compression.

    Iteratively processes text in chunks, computing per-token perplexity
    and selecting tokens above a dynamic threshold. Supports KV-cache
    compression for long sequences and conditional comparison mode.
    """

    def filter(
        self,
        context: List[str],
        target_token: float,
        iterative_size: int = 200,
        keep_split: bool = False,
        split_token_id: int = 13,
        start: int = 0,
        dynamic_ratio: Optional[List[Any]] = None,
        condition_compare: bool = False,
        segments_info: Optional[List[List[Tuple[Any, ...]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iteratively compress tokens based on PPL.

        Args:
            context: List of context documents to compress.
            target_token: Target token count after compression.
            iterative_size: Window size for iterative processing.
            keep_split: Whether to preserve split tokens.
            split_token_id: Token ID for split character (default: newline).
            start: Start position for compression.
            dynamic_ratio: Per-context compression ratio adjustments.
            condition_compare: Use conditional PPL comparison.
            segments_info: Structured segment information.

        Returns:
            Tuple of (compressed_input_ids, compressed_attention_mask).
        """
        # Calculate iterative compression ratios
        if segments_info is None or segments_info == []:
            iterative_ratios = self.get_dynamic_compression_ratio(
                context, target_token, iterative_size, dynamic_ratio, start
            )
        else:
            iterative_ratios = self.get_structured_dynamic_compression_ratio(
                context, iterative_size, dynamic_ratio, start, segments_info
            )

        # Tokenize context
        context_text = "\n\n".join(context)
        tokenized_text = self.tokenizer(
            context_text, return_tensors="pt", add_special_tokens=False
        )
        input_ids = tokenized_text["input_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)

        compressed_input_ids, compressed_attention_mask = input_ids, attention_mask
        self_compressed_input_ids: Optional[torch.Tensor] = None
        self_compressed_attention_mask: Optional[torch.Tensor] = None
        self_loss: Optional[torch.Tensor] = None

        if condition_compare:
            self_input_ids, self_attention_mask = (
                input_ids[:, start:],
                attention_mask[:, start:],
            )
            self_compressed_input_ids, self_compressed_attention_mask = (
                self_input_ids,
                self_attention_mask,
            )

        end = min(iterative_size + start, compressed_input_ids.shape[1])
        threshold, keep_flag = None, None

        # Build keep_flag for split token preservation
        if keep_split:
            input_ids_numpy = input_ids.cpu().detach().numpy()[0]
            n_ids = len(input_ids_numpy)
            keep_flag = [
                int(
                    (
                        ii > 0
                        and input_ids_numpy[ii] == split_token_id
                        and input_ids_numpy[ii - 1] == split_token_id
                    )
                    or (
                        ii < n_ids - 1
                        and input_ids_numpy[ii] == split_token_id
                        and input_ids_numpy[ii + 1] == split_token_id
                    )
                )
                for ii in range(n_ids)
            ]
            keep_flag = torch.tensor(keep_flag).to(self.device)

        past_key_values, past_loss, ready_end = None, None, 0
        self_past_key_values, self_past_loss, self_ready_end = None, None, 0
        pop_compressed_input_ids, pop_self_compressed_input_ids = None, None
        idx = 0

        while end <= compressed_input_ids.shape[1]:
            # KV-Cache Compression when exceeding max position embeddings
            if end > self.max_position_embeddings and past_key_values is not None:
                e, s = end - self.max_position_embeddings, min(
                    self.ctx.cache_bos_num + start, self.max_position_embeddings
                )
                if pop_compressed_input_ids is None:
                    pop_compressed_input_ids = compressed_input_ids[:, :e]
                else:
                    pop_compressed_input_ids = torch.cat(
                        [pop_compressed_input_ids, compressed_input_ids[:, :e]], dim=-1
                    )
                compressed_input_ids = compressed_input_ids[:, e:]
                compressed_attention_mask = compressed_attention_mask[:, e:]
                past_key_values = [
                    [
                        torch.cat([k[..., :s, :], k[..., s + e :, :]], dim=-2),
                        torch.cat([v[..., :s, :], v[..., s + e :, :]], dim=-2),
                    ]
                    for k, v in past_key_values
                ]
                if keep_flag is not None:
                    keep_flag = keep_flag[e:]
                end, ready_end = end - e, ready_end - e

                if condition_compare:
                    assert self_past_key_values is not None
                    assert self_compressed_input_ids is not None
                    assert self_compressed_attention_mask is not None
                    s = min(s, self_past_key_values[0][0].shape[2] - e)
                    self_ready_end -= e
                    if pop_self_compressed_input_ids is None:
                        pop_self_compressed_input_ids = self_compressed_input_ids[:, :e]
                    else:
                        pop_self_compressed_input_ids = torch.cat(
                            [
                                pop_self_compressed_input_ids,
                                self_compressed_input_ids[:, :e],
                            ],
                            dim=-1,
                        )
                    self_compressed_input_ids = self_compressed_input_ids[:, e:]
                    self_compressed_attention_mask = self_compressed_attention_mask[
                        :, e:
                    ]
                    self_past_key_values = [
                        [
                            torch.cat([k[..., :s, :], k[..., s + e :, :]], dim=-2),
                            torch.cat([v[..., :s, :], v[..., s + e :, :]], dim=-2),
                        ]
                        for k, v in self_past_key_values
                    ]

            # Compute PPL for current window
            assert self.ctx.get_ppl_fn is not None
            loss, past_key_values = self.ctx.get_ppl_fn(
                "",
                "token",
                compressed_input_ids,
                compressed_attention_mask,
                past_key_values=past_key_values,
                return_kv=True,
                end=end if idx else None,
            )

            if loss.shape[0] == 0:
                break

            # Accumulate past losses
            if past_loss is not None:
                if end - 1 > len(past_loss):
                    past_loss = torch.cat(
                        [past_loss, torch.zeros_like(loss)[: end - 1 - len(past_loss)]]
                    )
                past_loss[ready_end : end - 1] = loss
                loss = past_loss
            else:
                past_loss = loss

            if idx:
                past_key_values = [
                    [k[:, :, : end - iterative_size], v[:, :, : end - iterative_size]]
                    for k, v in past_key_values
                ]
            else:
                past_key_values = None

            # Handle conditional comparison mode
            if condition_compare:
                assert self_compressed_input_ids is not None
                assert self_compressed_attention_mask is not None
                assert self.ctx.get_ppl_fn is not None
                self_loss, self_past_key_values = self.ctx.get_ppl_fn(
                    "",
                    "token",
                    self_compressed_input_ids,
                    self_compressed_attention_mask,
                    past_key_values=self_past_key_values,
                    return_kv=True,
                    end=end - start if idx else None,
                )
                assert self_loss is not None
                if self_past_loss is not None:
                    if end - start - 1 > len(self_past_loss):
                        self_past_loss = torch.cat(
                            [
                                self_past_loss,
                                torch.zeros_like(self_loss)[
                                    : end - 1 - start - len(self_past_loss)
                                ],
                            ]
                        )
                    self_past_loss[self_ready_end : end - start - 1] = self_loss
                    self_loss = self_past_loss
                else:
                    self_past_loss = self_loss

                if idx:
                    self_past_key_values = [
                        [
                            k[:, :, : end - iterative_size - start],
                            v[:, :, : end - iterative_size - start],
                        ]
                        for k, v in self_past_key_values
                    ]
                else:
                    self_past_key_values = None

                self_ready_end = (
                    end - start - iterative_size if not (start and idx == 0) else 0
                )

            ready_end = end - iterative_size if not (start and idx == 0) else 0

            # Process each ratio segment
            for delta_end, ratio in iterative_ratios[idx]:
                loss = past_loss
                if condition_compare:
                    self_loss = self_past_loss
                    assert self_loss is not None
                    threshold = self.get_estimate_threshold_base_distribution(
                        self_loss[: loss[start:].shape[0]] - loss[start:], ratio, False
                    )
                else:
                    threshold = self.get_estimate_threshold_base_distribution(
                        loss, ratio, False
                    )

                (
                    compressed_input_ids,
                    compressed_attention_mask,
                    keep_flag,
                    end,
                    past_loss,
                    self_past_loss,
                    self_compressed_input_ids,
                    self_compressed_attention_mask,
                ) = self.get_compressed_input(
                    loss,
                    compressed_input_ids,
                    compressed_attention_mask,
                    end - iterative_size + delta_end,
                    iterative_size=delta_end,
                    threshold=threshold,
                    keep_flag=keep_flag,
                    split_token_id=split_token_id,
                    start=start,
                    self_loss=self_loss if condition_compare else None,
                    self_input_ids=(
                        self_compressed_input_ids if condition_compare else None
                    ),
                    self_attention_mask=(
                        self_compressed_attention_mask if condition_compare else None
                    ),
                )
                end += iterative_size
            idx += 1

        # Concatenate popped tokens
        if pop_compressed_input_ids is not None:
            compressed_input_ids = torch.cat(
                [pop_compressed_input_ids, compressed_input_ids], dim=-1
            )

        return compressed_input_ids[:, start:], compressed_attention_mask[:, start:]

    def get_compressed_input(
        self,
        loss,
        input_ids,
        attention_mask,
        end=200,
        iterative_size=200,
        threshold=0.5,
        keep_flag=None,
        split_token_id: int = 13,
        start: int = 0,
        self_loss=None,
        self_input_ids=None,
        self_attention_mask=None,
    ):
        """
        Select tokens based on PPL threshold.

        Args:
            loss: Per-token loss values.
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            end: End position for current window.
            iterative_size: Window size.
            threshold: PPL threshold for token selection.
            keep_flag: Flags for tokens to always keep.
            split_token_id: Token ID for split character.
            start: Start position.
            self_loss: Conditional loss values.
            self_input_ids: Conditional input IDs.
            self_attention_mask: Conditional attention mask.

        Returns:
            Tuple of compressed tensors and updated state.
        """
        if self_loss is not None:
            need_idx = torch.concat(
                [
                    loss[:start] > 0,
                    self_loss[: loss[start:].shape[0]] - loss[start:] > threshold,
                    loss[:1] > 0,
                ]
            )
        else:
            need_idx = torch.concat([loss > threshold, loss[:1] > 0])

        need_idx[end:] = 1
        need_idx[: end - iterative_size] = 1
        loss = loss[need_idx[:-1]]

        if self_loss is not None:
            if need_idx.shape[0] < self_loss.shape[0] + start + 1:
                need_idx = torch.cat(
                    [
                        need_idx,
                        torch.ones(
                            self_loss.shape[0] - need_idx.shape[0] + start + 1,
                            dtype=torch.bool,
                        ).to(need_idx.device),
                    ]
                )
            self_loss = self_loss[need_idx[start:-1]]

        if need_idx.shape[0] < input_ids.shape[1]:
            need_idx = torch.cat(
                [
                    need_idx,
                    torch.ones(
                        input_ids.shape[1] - need_idx.shape[0], dtype=torch.bool
                    ).to(need_idx.device),
                ]
            )
        elif need_idx.shape[0] > input_ids.shape[1]:
            need_idx = need_idx[: input_ids.shape[1]]

        if keep_flag is not None:
            need_idx[keep_flag == 1] = 1

        last = -1
        if keep_flag is not None:
            for ii in range(max(0, end - iterative_size), end):
                if need_idx[ii] != 1:
                    continue
                now = input_ids[0][ii].detach().cpu().item()
                if (
                    now == split_token_id
                    and last == split_token_id
                    and keep_flag[ii].detach().cpu().item() == 0
                ):
                    need_idx[ii] = 0
                else:
                    last = now

        compressed_input_ids = input_ids[attention_mask == 1][need_idx].unsqueeze(0)
        compressed_attention_mask = attention_mask[attention_mask == 1][
            need_idx
        ].unsqueeze(0)

        if self_loss is not None:
            assert self_input_ids is not None
            assert self_attention_mask is not None
            self_compressed_input_ids = self_input_ids[self_attention_mask == 1][
                need_idx[start:]
            ].unsqueeze(0)
            self_compressed_attention_mask = self_attention_mask[
                self_attention_mask == 1
            ][need_idx[start:]].unsqueeze(0)
        else:
            self_compressed_input_ids, self_compressed_attention_mask = None, None

        if keep_flag is not None:
            if len(keep_flag) > len(need_idx):
                keep_flag = torch.cat(
                    [
                        keep_flag[:start],
                        keep_flag[start : len(need_idx) + start][need_idx],
                        keep_flag[start + len(need_idx) :],
                    ]
                )
            else:
                keep_flag = keep_flag[need_idx]

        end -= (need_idx[:end] == 0).sum()

        return (
            compressed_input_ids,
            compressed_attention_mask,
            keep_flag,
            end,
            loss,
            self_loss,
            self_compressed_input_ids,
            self_compressed_attention_mask,
        )

    def get_estimate_threshold_base_distribution(
        self, ppl, ratio: float, condition_flag: bool = False
    ):
        """
        Estimate PPL threshold for target compression ratio.

        Args:
            ppl: Tensor of PPL values.
            ratio: Target compression ratio.
            condition_flag: Whether to use ascending sort.

        Returns:
            Threshold value for token selection.
        """
        if ratio == 1.0:
            return float("-inf")
        ppl = ppl[ppl != 10000]
        target_token = max(0, min(len(ppl) - 1, int(len(ppl) * ratio) - 1))
        return (
            ppl.sort(descending=not condition_flag)
            .values[target_token]
            .detach()
            .cpu()
            .item()
        )

    def get_dynamic_compression_ratio(
        self,
        context: List[str],
        target_token: float,
        iterative_size: int,
        dynamic_ratio: Optional[List[Any]],
        start: int,
        seg_info: Optional[List[List[Tuple[Any, ...]]]] = None,
    ):
        """
        Compute per-context compression ratios for iterative processing.

        Args:
            context: List of context documents.
            target_token: Target token count.
            iterative_size: Window size.
            dynamic_ratio: Per-context ratio adjustments.
            start: Start position.
            seg_info: Segment information (unused in this method).

        Returns:
            List of iterative compression ratios per window.
        """

        assert dynamic_ratio is not None

        def get_ratio(base: float, delta: float):
            return max(min(1, base + delta), 0)

        context_length = [self.get_token_length(ii) + 2 for ii in context]
        if start:
            context_length = context_length[1:]
        tau = target_token / (sum(context_length) + 1)

        res, idx, last, last_target = [], 0, 1, []
        while idx < len(context_length):
            if last + context_length[idx] >= iterative_size:
                last_target.append(
                    (iterative_size - last, get_ratio(tau, dynamic_ratio[idx]))
                )
                res.append(last_target)
                last = last + context_length[idx] - iterative_size
                if last > iterative_size:
                    k = last // iterative_size
                    res.extend(
                        [[(iterative_size, get_ratio(tau, dynamic_ratio[idx]))]] * k
                    )
                    last -= k * iterative_size

                last_target = (
                    [(last, get_ratio(tau, dynamic_ratio[idx]))] if last else []
                )
            else:
                last += context_length[idx]
                last_target.append(
                    (context_length[idx], get_ratio(tau, dynamic_ratio[idx]))
                )
            idx += 1

        if last_target:
            res.append(last_target)

        return res

    def get_structured_dynamic_compression_ratio(
        self,
        context: List[str],
        iterative_size: int,
        dynamic_ratio: Optional[List[Any]],
        start: int,
        seg_info: Optional[List[List[Tuple[Any, ...]]]] = None,
    ):
        """
        Map structured segments to token-level compression ratios.

        Args:
            context: List of context documents.
            iterative_size: Window size.
            dynamic_ratio: Per-context ratio adjustments (unused).
            start: Start position.
            seg_info: Segment information with (length, rate, compress) tuples.

        Returns:
            List of iterative compression ratios per window.
        """
        assert seg_info is not None
        if start:
            pure_context = context[1:]
        else:
            pure_context = context

        global_dynamic_rate, global_dynamic_compress, segments = [], [], []
        for context_idx, text in enumerate(pure_context):
            text_seen = 0
            for seg_idx, (seg_len, seg_rate, seg_compress) in enumerate(
                seg_info[context_idx]
            ):
                seg_text = text[text_seen : text_seen + seg_len]
                if (
                    seg_idx == len(seg_info[context_idx]) - 1
                    and context_idx != len(pure_context) - 1
                ):
                    seg_text += "\n\n"
                segments.append(seg_text)
                if seg_compress:
                    global_dynamic_rate.append(seg_rate)
                else:
                    global_dynamic_rate.append(1.0)
                global_dynamic_compress.append(seg_compress)
                text_seen += seg_len

        origin_text = "\n\n".join(pure_context)
        assert len("".join(segments)) == len(origin_text)
        assert len(segments) == len(global_dynamic_rate) == len(global_dynamic_compress)

        text_input_ids = self.tokenizer(
            "\n\n".join(context), add_special_tokens=False
        ).input_ids[start:]
        assert self.tokenizer.decode(text_input_ids) == origin_text

        dynamic_compression_ratio = self.token_segment(
            text_input_ids,
            iterative_size,
            segments,
            global_dynamic_rate,
            global_dynamic_compress,
        )

        return dynamic_compression_ratio

    def token_segment(
        self,
        text_input_ids: List[int],
        iterative_size: int,
        segments: List[str],
        global_dynamic_rate: List[float],
        global_dynamic_compress: List[bool],
    ):
        """
        Map compression rates to token ranges.

        Args:
            text_input_ids: List of token IDs.
            iterative_size: Window size.
            segments: List of text segments.
            global_dynamic_rate: Compression rate per segment.
            global_dynamic_compress: Compression flag per segment.

        Returns:
            List of (token_count, rate) tuples per iterative window.
        """
        decode_window = 3
        seg_idx, seg_seen, token_seen_num, last_rate = 0, 0, 0, -1
        dynamic_compression_rate, local_compresssion_rate = [], []

        for i in range(len(text_input_ids)):
            if i < decode_window:
                id_pre, id_cur = text_input_ids[:i], text_input_ids[: i + 1]
            else:
                id_pre, id_cur = (
                    text_input_ids[i - decode_window + 1 : i],
                    text_input_ids[i - decode_window + 1 : i + 1],
                )

            cur_word = self.tokenizer.decode(id_cur)[
                len(self.tokenizer.decode(id_pre)) :
            ]
            cur_word_len = len(cur_word)

            if cur_word_len and cur_word_len >= len(segments[seg_idx]) - seg_seen:
                possible_rate, possible_compress = [], []
                while (
                    cur_word_len and cur_word_len >= len(segments[seg_idx]) - seg_seen
                ):
                    possible_rate.append(global_dynamic_rate[seg_idx])
                    possible_compress.append(global_dynamic_compress[seg_idx])
                    cur_word_len -= len(segments[seg_idx]) - seg_seen
                    seg_idx += 1
                    seg_seen = 0
                if cur_word_len:
                    possible_rate.append(global_dynamic_rate[seg_idx])
                    possible_compress.append(global_dynamic_compress[seg_idx])
                new_rate = 1.0 if False in possible_compress else min(possible_rate)
            else:
                new_rate = global_dynamic_rate[seg_idx]

            if new_rate != last_rate and i - token_seen_num:
                local_compresssion_rate.append((i - token_seen_num, last_rate))
                token_seen_num = i
            last_rate = new_rate
            seg_seen += cur_word_len

            if (i + 1) % iterative_size == 0:
                if token_seen_num != i + 1:
                    local_compresssion_rate.append((i + 1 - token_seen_num, last_rate))
                    token_seen_num = i + 1
                dynamic_compression_rate.append(local_compresssion_rate[:])
                local_compresssion_rate = []

        if token_seen_num != len(text_input_ids):
            local_compresssion_rate.append(
                (len(text_input_ids) - token_seen_num, last_rate)
            )
        if local_compresssion_rate != []:
            dynamic_compression_rate.append(local_compresssion_rate[:])

        return dynamic_compression_rate
