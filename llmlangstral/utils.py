from __future__ import annotations

import json
import os
import random
import re
from typing import Any, List, Tuple

import numpy as np
import torch
import yaml


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_structured_json_data(json_data, json_config):
    if isinstance(json_config, str):
        with open(json_config, "r") as file:
            json_config = yaml.safe_load(file)
    elif not isinstance(json_config, dict):
        raise ValueError(
            "Invalid json config file. It should be a dictionary or a path to a yaml file."
        )
    assert set(json_data.keys()) == set(
        json_config.keys()
    ), "Keys in json data and json config file do not match."
    context = ["<llmlingua, compress=False>{</llmlingua>"]
    forced_context_ids = [0]
    for i, (k, v) in enumerate(json_data.items()):
        if not json_config[k]["pair_remove"]:
            forced_context_ids.append(i + 1)
        rate, compress, value_type = (
            json_config[k]["rate"],
            json_config[k]["compress"],
            json_config[k]["value_type"],
        )
        if not compress:
            rate = 1
        context.append(precess_jsonKVpair(k, v, value_type, rate))
    context[-1] = context[-1][:-14] + "</llmlingua>"
    context.append("<llmlingua, compress=False>}</llmlingua>")
    forced_context_ids.append(len(json_data) + 1)

    return context, forced_context_ids


def precess_jsonKVpair(k, v, value_type, rate):
    if rate == 1:
        return (
            "<llmlingua, compress=False>"
            + f"{json.dumps({k: v})[1:-1]}, "
            + "</llmlingua>"
        )
    if value_type == "str" or value_type == "string":
        v = str(v)
        new_v = (
            f"</llmlingua><llmlingua, rate={rate}>"
            + v
            + "</llmlingua><llmlingua, compress=False>"
        )
        return (
            "<llmlingua, compress=False>"
            + f"{json.dumps({k: new_v})[1:-1]}, "
            + "</llmlingua>"
        )
    elif value_type in ["int", "float", "integer", "number"]:
        if value_type in ["int", "integer"]:
            v = int(v)
        if value_type in ["float", "number"]:
            v = float(v)
        return (
            "<llmlingua, compress=False>"
            + f'"{k}": </llmlingua><llmlingua, rate={rate}>{v}</llmlingua><llmlingua, compress=False>, </llmlingua>'
        )
    elif value_type == "bool" or value_type == "boolean":
        if v in ["True", "true", "TRUE", True]:
            v = "true"
        elif v in ["False", "false", "FALSE", False]:
            v = "false"
        else:
            raise ValueError(f"Invalid boolean value: {v}")
        new_v = (
            f"</llmlingua><llmlingua, rate={rate}>"
            + v
            + "</llmlingua><llmlingua, compress=False>"
        )
        return (
            "<llmlingua, compress=False>"
            + f"{json.dumps({k: new_v})[1:-1]}, "
            + "</llmlingua>"
        )
    elif value_type == "list" or value_type == "List":
        return (
            "<llmlingua, compress=False>"
            + f'"{k}": {process_sequence_data(rate, "[", "]", v)}'
        )
    elif value_type == "dict" or value_type == "dictionary":
        return (
            "<llmlingua, compress=False>"
            + f'"{k}": {process_sequence_data(rate, "[", "]", v, is_dict=True)}'
        )
    elif value_type == "set":
        raise ValueError(f"Invalid value type: {value_type}")
        # return '<llmlingua, compress=False>' + f'"{k}": {process_sequence_data(rate, "{", "}", v)}'
    elif value_type == "tuple":
        return (
            "<llmlingua, compress=False>"
            + f'"{k}": {process_sequence_data(rate, "(", ")", v)}'
        )
    else:
        raise ValueError(f"Invalid value type: {value_type}")


def process_sequence_data(rate, start, end, sequence, is_dict=False):
    res = f'{start}"'
    n = len(sequence)
    if not is_dict:
        for i, item in enumerate(sequence):
            item = str(item)
            res += f"</llmlingua><llmlingua, rate={rate}>{item}</llmlingua><llmlingua, compress=False>"
            if i != n - 1:
                res += '", "'
    else:
        for i, (k, v) in enumerate(sequence.items()):
            item = f"{k}: {v}"
            item.replace('"', "'")
            res += f"</llmlingua><llmlingua, rate={rate}>{item}</llmlingua><llmlingua, compress=False>"
            if i != n - 1:
                res += '", "'
    res += f'"{end}, </llmlingua>'
    return res


def remove_consecutive_commas(text):
    text = re.sub(r",\s*", ",", text)
    text = re.sub(r",+", ",", text)
    return text


def segment_structured_context(
    context: List[str],
    global_rate: float,
):
    new_context, context_segs, context_segs_rate, context_segs_compress = (
        [],
        [],
        [],
        [],
    )
    for text in context:
        if not text.startswith("<llmlingua"):
            text = "<llmlingua>" + text
        if not text.endswith("</llmlingua>"):
            text = text + "</llmlingua>"

        # Regular expression to match <llmlingua, rate=x, compress=y>content</llmlingua>, allowing rate and compress in any order
        pattern = r"<llmlingua\s*(?:,\s*rate\s*=\s*([\d\.]+))?\s*(?:,\s*compress\s*=\s*(True|False))?\s*(?:,\s*rate\s*=\s*([\d\.]+))?\s*(?:,\s*compress\s*=\s*(True|False))?\s*>([^<]+)</llmlingua>"
        matches = re.findall(pattern, text)

        # Extracting segment contents
        segments = [match[4] for match in matches]

        # Extracting rate and compress, considering their possible positions
        segs_rate = [
            float(match[0]) if match[0] else (float(match[2]) if match[2] else None)
            for match in matches
        ]
        segs_compress = [
            (
                match[1] == "True"
                if match[1]
                else (match[3] == "True" if match[3] else None)
            )
            for match in matches
        ]

        segs_compress = [
            compress if compress is not None else True for compress in segs_compress
        ]
        segs_rate = [
            rate if rate else (global_rate if compress else 1.0)
            for rate, compress in zip(segs_rate, segs_compress)
        ]
        assert (
            len(segments) == len(segs_rate) == len(segs_compress)
        ), "The number of segments, rates, and compress flags should be the same."
        assert all(
            seg_rate <= 1.0 for seg_rate in segs_rate
        ), "Error: 'rate' must not exceed 1.0. The value of 'rate' indicates compression rate and must be within the range [0, 1]."

        new_context.append("".join(segments))
        context_segs.append(segments)
        context_segs_rate.append(segs_rate)
        context_segs_compress.append(segs_compress)

    return new_context, context_segs, context_segs_rate, context_segs_compress


def concate_segment_info(
    segment_info: List[List[Tuple[Any, ...]]],
):
    new_segment_info = []
    for i, (seg_len, seg_ratio, seg_compress) in enumerate(segment_info):
        if (
            new_segment_info
            and new_segment_info[-1][1] == seg_ratio
            and new_segment_info[-1][2] == seg_compress
        ):
            new_segment_info[-1] = (
                new_segment_info[-1][0] + seg_len,
                seg_ratio,
                seg_compress,
            )
        else:
            new_segment_info.append((seg_len, seg_ratio, seg_compress))
    return new_segment_info
