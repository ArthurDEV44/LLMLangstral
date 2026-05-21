# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Configuration des modèles Mistral AI pour LLMLangstral.

AVERTISSEMENT — Modèles non publiés sur HuggingFace Hub
-------------------------------------------------------
Trois identifiants ci-dessous sont des PLACEHOLDERS pour des modèles
qui n'existent pas (encore) publiquement sur huggingface.co :

  * mistralai/Ministral-3-3B-Instruct-2512  (clé "small")
  * mistralai/Ministral-3-8B-Instruct-2512  (clé "medium")
  * mistralai/Mistral-Large-3               (clé "large")

Charger un PromptCompressor avec ces identifiants lèvera une OSError
("Repository not found"). Les tests fonctionnels (`test_mistral.py`)
gèrent cette absence via skipTest dans setUpClass.

Modèles RÉELS, utilisables immédiatement :
  * mistralai/Mistral-7B-v0.3                    (clé "default")
  * TheBloke/Mistral-7B-Instruct-v0.2-GPTQ       (clé "quantized")
  * intfloat/e5-mistral-7b-instruct              (clé "embedding")

Avant d'ajouter un nouvel identifiant ici, vérifier son existence sur
https://huggingface.co/<identifier>.
"""

MISTRAL_MODELS = {
    "default": "mistralai/Mistral-7B-v0.3",
    # ⚠ placeholder — non publié, voir l'avertissement en tête de fichier
    "small": "mistralai/Ministral-3-3B-Instruct-2512",
    # ⚠ placeholder — non publié, voir l'avertissement en tête de fichier
    "medium": "mistralai/Ministral-3-8B-Instruct-2512",
    # ⚠ placeholder — non publié, voir l'avertissement en tête de fichier
    "large": "mistralai/Mistral-Large-3",
    "quantized": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "embedding": "intfloat/e5-mistral-7b-instruct",
}

# Modèle léger pour tests CI (architecture Mistral, ~1M params)
TEST_MODEL = "openaccess-ai-collective/tiny-mistral"

# Alias pour migration progressive
DEFAULT_MODEL = MISTRAL_MODELS["default"]
SMALL_MODEL = MISTRAL_MODELS["small"]
QUANTIZED_MODEL = MISTRAL_MODELS["quantized"]
EMBEDDING_MODEL = MISTRAL_MODELS["embedding"]
