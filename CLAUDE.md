# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMLangstral is a French fork of Microsoft's LLMLingua prompt compression library, migrated to use Mistral AI models. It reduces prompt length while maintaining semantic information for LLMs, achieving up to 20x compression with minimal performance loss.

**Key Variants:**
- **LLMLangstral** - Base compression using Mistral models (Mistral-7B, Ministral-3B)
- **LongLLMLangstral** - Handles long contexts, mitigates "lost in the middle" issue
- **LLMLangstral-2** - Fast distilled model (3x-6x faster), uses XLM-RoBERTa (no Mistral equivalent exists)
- **SecurityLingua** - Jailbreak attack detection via security-aware compression

## Common Commands

```bash
# Install dependencies (dev)
pip install -e ".[dev]"

# Run all tests (parallel)
make test
# or directly:
pytest -n auto --dist=loadfile -s -v ./tests/

# Run a single test file
pytest tests/test_llmlangstral.py -v

# Run a specific test
pytest tests/test_llmlangstral.py::LLMLangstralTester::test_compress_prompt -v

# Code formatting and linting
make style
# or individually:
black llmlangstral tests
isort llmlangstral tests
flake8 llmlangstral tests
```

## Architecture

### Core Entry Point
All compression methods are accessed through `PromptCompressor` class in `llmlangstral/prompt_compressor.py`:

```python
from llmlangstral import PromptCompressor

# LLMLangstral (default)
compressor = PromptCompressor()
result = compressor.compress_prompt(prompt, instruction="", question="", target_token=200)

# LLMLangstral-2
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True
)
result = compressor.compress_prompt(prompt, rate=0.33, force_tokens=['\n', '?'])

# SecurityLingua
compressor = PromptCompressor(
    model_name="SecurityLingua/securitylingua-xlm-s2s",
    use_slingua=True
)
```

### Key Compression Methods
- `compress_prompt()` - Main compression (all variants)
- `compress_prompt_llmlingua2()` - LLMLangstral-2 specific
- `structured_compress_prompt()` - XML tag-based granular control
- `compress_json()` - JSON-specific with per-field config

### Prompt Structure Concept
LLMLangstral divides prompts into components with different compression sensitivity:
- **Instruction** (HIGH sensitivity) - Task description, placed first
- **Context** (LOW sensitivity) - Documents, examples, demonstrations
- **Question** (HIGH sensitivity) - User query, placed last

### Structured Compression Tags
Use XML-style tags for per-section compression control:
```python
"<llmlingua, compress=False>Keep this</llmlingua>"
"<llmlingua, rate=0.5>Compress to 50%</llmlingua>"
```

## Code Style

- Python 3.8+
- Black: line-length=88
- isort: profile="black", known_first_party=["llmlangstral"]
- Flake8: max-line-length=119

## Directory Structure

- `llmlangstral/` - Main package with `PromptCompressor` class
- `llmlangstral/mistral_config.py` - Centralized Mistral model registry
- `tests/` - Unit tests (test_llmlangstral.py, test_llmlangstral2.py, test_longllmlangstral.py, test_mistral.py)
- `examples/` - Jupyter notebooks (RAG, CoT, Code, OnlineMeeting)
- `experiments/llmlangstral2/` - LLMLangstral-2 training pipeline (data_collection, model_training, evaluation)
- `experiments/securitylingua/` - SecurityLingua training

## Model Options

Models are defined in `llmlangstral/mistral_config.py`:

- Default: `mistralai/Mistral-7B-v0.3`
- Small: `mistralai/Ministral-3-3B-Instruct-2512`
- Medium: `mistralai/Ministral-3-8B-Instruct-2512`
- Large: `mistralai/Mistral-Large-3`
- Quantized (< 8GB GPU): `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`
- Embedding: `intfloat/e5-mistral-7b-instruct`
- LLMLangstral-2: `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (XLM-RoBERTa, not Mistral)
- CI Testing: `openaccess-ai-collective/tiny-mistral` (~1M params, Mistral architecture)
