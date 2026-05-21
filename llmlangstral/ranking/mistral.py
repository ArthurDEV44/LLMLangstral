# Copyright (c) 2023-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Mistral embedding-based ranking strategy."""

from typing import List, Optional, Tuple

import numpy as np

from .base import ModelBasedRanker
from .registry import RankingRegistry


@RankingRegistry.register("mistral")
class MistralRanker(ModelBasedRanker):
    """
    Mistral-based embedding ranker.

    Uses intfloat/e5-mistral-7b-instruct for dense embeddings.
    Prefixes texts with "query: " for retrieval-optimized embeddings.

    Requires: sentence_transformers library
    """

    name = "mistral"

    def __init__(self, device: str = "cuda", embedding_model: Optional[str] = None, **kwargs):
        """
        Initialize Mistral ranker.

        Args:
            device: Device to run model on ("cuda" or "cpu").
            embedding_model: Override default embedding model.
                             Defaults to EMBEDDING_MODEL from mistral_config.
        """
        super().__init__(device=device, **kwargs)
        if embedding_model:
            self.model_name = embedding_model
        else:
            # Import here to avoid circular imports
            from ..mistral_config import EMBEDDING_MODEL

            self.model_name = EMBEDDING_MODEL

    def _load_model(self):
        try:
            from sentence_transformers import (
                SentenceTransformer,  # pyright: ignore[reportMissingImports]
            )
        except ImportError as exc:
            raise ImportError(
                "rank_method='mistral' requires sentence_transformers. "
                "Install it with: pip install 'llmlangstral[mistral-ranker]'"
            ) from exc

        self._model = SentenceTransformer(self.model_name)

    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs,
    ) -> List[Tuple[int, float]]:
        try:
            from sentence_transformers import util  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "rank_method='mistral' requires sentence_transformers. "
                "Install it with: pip install 'llmlangstral[mistral-ranker]'"
            ) from exc

        # Prefix with "query: " as recommended for e5-mistral
        doc_embeds = self.model.encode(
            ["query: " + doc for doc in corpus],
            normalize_embeddings=True,
        )
        query_embed = self.model.encode(
            "query: " + query,
            normalize_embeddings=True,
        )

        doc_scores = -util.dot_score(doc_embeds, query_embed).cpu().numpy().reshape(-1)
        idx = [(int(ii), 0.0) for ii in np.argsort(doc_scores)]
        return idx
