# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Configuration des modèles Mistral AI pour LLMLangstral."""

MISTRAL_MODELS = {
    "default": "mistralai/Mistral-7B-v0.3",
    "small": "mistralai/Ministral-3-3B-Instruct-2512",
    "medium": "mistralai/Ministral-3-8B-Instruct-2512",
    "large": "mistralai/Mistral-Large-3",
    "quantized": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "embedding": "intfloat/e5-mistral-7b-instruct",
}

# Note: LLMLangstral-2 utilise toujours le modèle Microsoft XLM-RoBERTa car
# aucun modèle Mistral équivalent pour token classification n'existe.
# Voir: microsoft/llmlingua-2-xlm-roberta-large-meetingbank
LLMLANGSTRAL2_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"

# Modèle léger pour tests CI (architecture Mistral, ~1M params)
TEST_MODEL = "openaccess-ai-collective/tiny-mistral"

# Alias pour migration progressive
DEFAULT_MODEL = MISTRAL_MODELS["default"]
SMALL_MODEL = MISTRAL_MODELS["small"]
QUANTIZED_MODEL = MISTRAL_MODELS["quantized"]
EMBEDDING_MODEL = MISTRAL_MODELS["embedding"]
