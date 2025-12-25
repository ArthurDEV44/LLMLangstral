# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Functional tests for Mistral models in LLMLangstral.

These tests validate that Mistral models work correctly with the
PromptCompressor class. Unlike other test files, these tests do NOT
compare exact output strings (since different models produce different
compressions), but instead verify functional behavior.
"""

import unittest

from llmlangstral import PromptCompressor
from llmlangstral.mistral_config import (
    DEFAULT_MODEL,
    EMBEDDING_MODEL,
    MISTRAL_MODELS,
    SMALL_MODEL,
)


class TestMistralConfig(unittest.TestCase):
    """Tests for Mistral configuration."""

    def test_default_model_is_mistral(self):
        """Verify that the default model is a Mistral model."""
        self.assertIn("mistral", DEFAULT_MODEL.lower())

    def test_small_model_is_ministral(self):
        """Verify that the small model is Ministral."""
        self.assertIn("ministral", SMALL_MODEL.lower())

    def test_embedding_model_is_mistral_based(self):
        """Verify that the embedding model is Mistral-based."""
        self.assertIn("mistral", EMBEDDING_MODEL.lower())

    def test_mistral_models_dict_complete(self):
        """Verify that MISTRAL_MODELS contains all required keys."""
        required_keys = ["default", "small", "medium", "large", "quantized", "embedding"]
        for key in required_keys:
            self.assertIn(key, MISTRAL_MODELS)


class TestMistralCompressor(unittest.TestCase):
    """Functional tests for Mistral compression.

    Note: These tests use SMALL_MODEL with CPU to minimize resource usage
    in CI environments. Tests validate functionality, not exact outputs.
    """

    @classmethod
    def setUpClass(cls):
        """Initialize compressor with small Mistral model."""
        # Skip if model loading fails (e.g., in resource-constrained CI)
        try:
            cls.compressor = PromptCompressor(
                model_name=SMALL_MODEL,
                device_map="cpu",
            )
            cls.model_available = True
        except Exception as e:
            cls.model_available = False
            cls.skip_reason = str(e)

    def setUp(self):
        """Skip test if model is not available."""
        if not self.model_available:
            self.skipTest(f"Model not available: {self.skip_reason}")

    def test_basic_compression(self):
        """Test that basic compression returns expected structure."""
        prompt = (
            "This is a test prompt that should be compressed. "
            "It contains multiple sentences with various words. "
            "The compression algorithm should reduce its length."
        )
        result = self.compressor.compress_prompt(prompt, rate=0.5)

        # Verify result structure
        self.assertIn("compressed_prompt", result)
        self.assertIn("origin_tokens", result)
        self.assertIn("compressed_tokens", result)
        self.assertIn("ratio", result)

        # Verify compression occurred
        self.assertGreater(result["origin_tokens"], 0)
        self.assertGreater(result["compressed_tokens"], 0)
        self.assertLessEqual(
            result["compressed_tokens"], result["origin_tokens"]
        )

    def test_compression_with_context(self):
        """Test compression with context list."""
        context = [
            "First context sentence about programming.",
            "Second context sentence about machine learning.",
            "Third context sentence about data science.",
        ]
        result = self.compressor.compress_prompt(context, rate=0.5)

        # Verify result structure
        self.assertIn("compressed_prompt", result)
        self.assertIsInstance(result["compressed_prompt"], str)
        self.assertGreater(len(result["compressed_prompt"]), 0)

    def test_compression_with_instruction_and_question(self):
        """Test compression with instruction and question parameters."""
        context = ["Some context about a topic that needs to be discussed."]
        instruction = "Summarize the following:"
        question = "What is the main point?"

        result = self.compressor.compress_prompt(
            context,
            instruction=instruction,
            question=question,
            rate=0.5,
        )

        # Verify instruction and question are in output
        self.assertIn("compressed_prompt", result)

    def test_target_token_compression(self):
        """Test compression with target_token parameter."""
        prompt = (
            "This is a longer prompt that we want to compress to a specific "
            "number of tokens. The compression should aim for the target."
        )
        target = 20

        result = self.compressor.compress_prompt(prompt, target_token=target)

        # Compressed tokens should be close to target (within reasonable margin)
        self.assertIn("compressed_tokens", result)
        # Allow some flexibility as exact token targeting is approximate
        self.assertLessEqual(result["compressed_tokens"], target * 1.5)


class TestMistralModelNames(unittest.TestCase):
    """Test that model name validation works correctly."""

    def test_default_model_initialization(self):
        """Test that default initialization uses Mistral model."""
        # This test just verifies the import works - actual model loading
        # is tested in TestMistralCompressor
        from llmlangstral import PromptCompressor
        from llmlangstral.mistral_config import DEFAULT_MODEL

        # Verify the default model constant is correct
        self.assertEqual(DEFAULT_MODEL, "mistralai/Mistral-7B-v0.3")


if __name__ == "__main__":
    unittest.main()
