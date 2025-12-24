# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMLingua is a prompt compression library from Microsoft that reduces prompt length while maintaining semantic information for LLMs. It achieves up to 20x compression with minimal performance loss.

**Key Variants:**
- **LLMLingua** - Base compression using small language models (Mistral-7B, Ministral-3B)
- **LongLLMLingua** - Handles long contexts, mitigates "lost in the middle" issue
- **LLMLingua-2** - Fast distilled model (3x-6x faster), uses BERT/XLM-RoBERTa
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
pytest tests/test_llmlingua.py -v

# Run a specific test
pytest tests/test_llmlingua.py::LLMLinguaTester::test_compress_prompt -v

# Code formatting and linting
make style
# or individually:
black llmlingua tests
isort -rc llmlingua tests
flake8 llmlingua tests
```

## Architecture

### Core Entry Point
All compression methods are accessed through `PromptCompressor` class in `llmlingua/prompt_compressor.py`:

```python
from llmlingua import PromptCompressor

# LLMLingua (default)
compressor = PromptCompressor()
result = compressor.compress_prompt(prompt, instruction="", question="", target_token=200)

# LLMLingua-2
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
- `compress_prompt_llmlingua2()` - LLMLingua-2 specific
- `structured_compress_prompt()` - XML tag-based granular control
- `compress_json()` - JSON-specific with per-field config

### Prompt Structure Concept
LLMLingua divides prompts into components with different compression sensitivity:
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
- isort: profile="black", known_first_party=["llmlingua"]
- Flake8: max-line-length=119

## Directory Structure

- `llmlingua/` - Main package with `PromptCompressor` class
- `tests/` - Unit tests (test_llmlingua.py, test_llmlingua2.py, test_longllmlingua.py)
- `examples/` - Jupyter notebooks (RAG, CoT, Code, OnlineMeeting)
- `experiments/llmlingua2/` - LLMLingua-2 training pipeline
- `experiments/securitylingua/` - SecurityLingua training

## Model Options

- Default: `mistralai/Mistral-7B-v0.3`
- Smaller: `mistralai/Ministral-3-3B-Instruct-2512`
- Medium: `mistralai/Ministral-3-8B-Instruct-2512`
- LLMLingua-2: `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`
- Quantized (< 8GB GPU): `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`
- Embedding: `intfloat/e5-mistral-7b-instruct`
