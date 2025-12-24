# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Configuration des modèles Mistral AI pour LLMLingstral."""

MISTRAL_MODELS = {
    "default": "mistralai/Mistral-7B-v0.3",
    "small": "mistralai/Ministral-3-3B-Instruct-2512",
    "medium": "mistralai/Ministral-3-8B-Instruct-2512",
    "large": "mistralai/Mistral-Large-3",
    "quantized": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "embedding": "intfloat/e5-mistral-7b-instruct",
}

# Modèles LLMLingua-2 Mistral (noms de destination après entraînement)
MISTRAL_LINGUA2_MODELS = {
    "large": "mistralai/mistral-lingua-2-7b-meetingbank",   # À entraîner
    "small": "mistralai/mistral-lingua-2-3b-meetingbank",   # À entraîner
}

# Modèle de base pour l'entraînement token classification
TRAINING_BASE_MODEL = "mistralai/Ministral-3-3B-Instruct-2512"

# Alias pour migration progressive
DEFAULT_MODEL = MISTRAL_MODELS["default"]
SMALL_MODEL = MISTRAL_MODELS["small"]
QUANTIZED_MODEL = MISTRAL_MODELS["quantized"]
EMBEDDING_MODEL = MISTRAL_MODELS["embedding"]
