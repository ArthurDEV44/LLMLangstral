# AGENTS.md

Instructions pour les agents IA de codage (Codex, Cursor, Aider, Continue, etc.) travaillant sur ce dépôt. Suit la [convention agents.md](https://agents.md/).

## TL;DR

LLMLangstral est un fork français de Microsoft LLMLingua migré vers Mistral AI. Bibliothèque Python de compression de prompts LLM (jusqu'à 20x). Le code est en cours de refactor modulaire (`core/`, `filters/`, `ranking/`). Point d'entrée unique : la classe `PromptCompressor`.

```bash
pip install -e ".[dev]"   # installer en mode dev
make test                 # lancer les tests (parallèle)
make style                # black + isort + flake8
```

## Structure du projet

```
llmlangstral/
├── prompt_compressor.py      # Classe PromptCompressor (point d'entrée unique)
├── mistral_config.py         # Registre des modèles Mistral
├── utils.py                  # Helpers (TokenClfDataset, seed_everything)
├── version.py                # VERSION = "0.2.2"
├── core/                     # Modèle, ABC, dataclass de résultat
│   ├── base.py               # CompressionResult, BaseCompressor (ABC)
│   └── model_loader.py       # ModelManager (lazy-load HuggingFace)
├── filters/                  # Filtres multi-niveaux
│   ├── base.py               # FilterContext, FilterBase (ABC)
│   ├── context.py            # ContextLevelFilter (passage)
│   ├── sentence.py           # SentenceLevelFilter (phrase)
│   ├── token.py              # TokenLevelFilter (élagage par PPL)
│   └── llmlingua2.py         # LLMLingua2Compressor (XLM-RoBERTa) — voir Pièges
└── ranking/                  # Stratégies de ranking (plugin registry)
    ├── registry.py           # RankingRegistry (@register("nom"))
    ├── base.py               # RankingStrategy, ModelBasedRanker, APIBasedRanker, PPLBasedRanker
    ├── statistical.py        # BM25Ranker, GzipRanker
    ├── neural.py             # SentBert, BGE, BGEReranker, BGELLMEmbedder, Jinza
    ├── llmlingua.py          # LLMLinguaRanker (alias "llmlingua" + "longllmlingua")
    ├── api_based.py          # OpenAIRanker, VoyageAIRanker, CohereRanker
    └── mistral.py            # MistralRanker (intfloat/e5-mistral-7b-instruct)

tests/         # Suites pytest (test_llmlangstral, _longllmlangstral, _llmlangstral2, _mistral)
examples/      # Notebooks Jupyter : RAG, CoT, Code, OnlineMeeting, Retrieval
experiments/   # Pipelines d'entraînement (llmlangstral2/, securitylingua/)
```

## Variantes de compression

| Variante | Activation | Modèle | Caractéristique |
|---|---|---|---|
| **LLMLangstral** | défaut | `mistralai/Mistral-7B-v0.3` | Compression itérative par perplexité |
| **LongLLMLangstral** | `use_context_level_filter=True` | idem | Longs contextes, lutte contre "lost in the middle" |
| **LLMLangstral-2** | `use_llmlingua2=True` | `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | 3-6x plus rapide |
| **SecurityLingua** | `use_slingua=True` | `SecurityLingua/securitylingua-xlm-s2s` | Détection de jailbreak |

## Exemple d'utilisation

```python
from llmlangstral import PromptCompressor

# Variante par défaut
compressor = PromptCompressor()
result = compressor.compress_prompt(
    prompt,
    instruction="",
    question="",
    target_token=200,
)

# Variante rapide (LLMLangstral-2)
compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
)
result = compressor.compress_prompt(
    prompt,
    rate=0.33,
    force_tokens=["\n", "?"],
)
```

### Structure conceptuelle du prompt

LLMLangstral découpe un prompt en trois composants à sensibilité différente :

- **Instruction** (sensibilité HAUTE) — placée en premier
- **Context** (sensibilité BASSE) — documents, démos, exemples ; compressible agressivement
- **Question** (sensibilité HAUTE) — placée à la fin

### Tags structurés

```python
"<llmlingua, compress=False>Garder intact</llmlingua>"
"<llmlingua, rate=0.5>Compresser à 50%</llmlingua>"
```

## Commandes de développement

```bash
# Installation
pip install -e ".[dev]"

# Tests (parallèles, tous les fichiers)
make test
# équivalent : pytest -n auto --dist=loadfile -s -v ./tests/

# Test d'un fichier
pytest tests/test_llmlangstral.py -v

# Test précis
pytest tests/test_llmlangstral.py::LLMLangstralTester::test_compress_prompt -v

# Format + lint (Black 88 + isort + flake8 119)
make style
```

CI : `.github/workflows/unittest.yml` — matrice Ubuntu/macOS/Windows × Python 3.9/3.10/3.11. Requiert le secret `HF_TOKEN` pour télécharger les modèles.

## Conventions de code

- **Python** : `>=3.8.0` (déclaré), CI testé sur 3.9/3.10/3.11
- **Black** : `line-length=88`, `target-version=['py38']`
- **isort** : `profile="black"`, `known_first_party=["llmlangstral"]`
- **Flake8** : `max-line-length=119`, ignore `E203, E501, E741, W503, W605`
- **Docstrings** : style Google/numpy (`Args:`, `Returns:`, `Raises:`, `Example:`)
- **Type hints** : systématiques dans les nouvelles classes

> ⚠ **Tension 3.8 vs 3.9** : du code utilise `dict[str, Any]` (3.9+) alors que `python_requires=">=3.8"`. Si tu vises 3.8 réel, ajoute `from __future__ import annotations` ou utilise `Dict` depuis `typing`.

## Architecture — patterns clés

### Plugin registry pour ranking

```python
from llmlangstral.ranking.registry import RankingRegistry
from llmlangstral.ranking.base import RankingStrategy

@RankingRegistry.register("mon_ranker")
class MonRanker(RankingStrategy):
    def rank(self, query, docs, **kwargs):
        ...
```

Importe la classe dans `ranking/__init__.py` avec `# noqa: F401` pour déclencher l'enregistrement à l'import du package.

### Pattern filtre

Hérite de `FilterBase` (`filters/base.py:38`), accepte un `FilterContext` (qui porte tokenizer, device, callbacks PPL/ranking), implémente `filter()`.

### Lazy model loading

`ModelManager` (`core/model_loader.py:19`) charge le modèle HuggingFace à la première utilisation. `PromptCompressor` expose `model`, `tokenizer`, `device` via des `@property` qui délèguent au manager.

## Modèles Mistral

`llmlangstral/mistral_config.py` :

```python
MISTRAL_MODELS = {
    "default":   "mistralai/Mistral-7B-v0.3",
    "small":     "mistralai/Ministral-3-3B-Instruct-2512",
    "medium":    "mistralai/Ministral-3-8B-Instruct-2512",
    "large":     "mistralai/Mistral-Large-3",
    "quantized": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "embedding": "intfloat/e5-mistral-7b-instruct",
}
LLMLANGSTRAL2_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
TEST_MODEL          = "openaccess-ai-collective/tiny-mistral"   # CI uniquement
DEFAULT_MODEL       = MISTRAL_MODELS["default"]
```

> ⚠ **Modèles fantômes** : `Ministral-3-3B-Instruct-2512`, `Ministral-3-8B-Instruct-2512`, `Mistral-Large-3` **n'existent pas** sur HuggingFace Hub. Vérifie sur `huggingface.co` avant d'ajouter un identifiant. Les tests qui ciblent ces modèles fantômes sont `skipTest`.

## Pièges connus

1. **`llmlangstral/filters/llmlingua2.py` non tracké git** — importé depuis `filters/__init__.py:37`, présent sur disque, mais absent de l'index. À `git add` lors du prochain commit.
2. **`llmlangstral/core/tokenization.py` supprimé non stagé** — apparaît `D` dans `git status`. À `git rm` proprement.
3. **`setup.cfg` contient `known_first_party = sdtools`** — héritage de l'ancien nom du package. `pyproject.toml` a la priorité.
4. **`Makefile install` cassé** — cible `dist/sdtools*` (ancien nom). Utilise `pip install -e ".[dev]"`.
5. **`test_llmlangstral2.py` télécharge ~1.1 GB sans cache** à chaque run CI. Considérer un cache HF.
6. **`PromptCompressor` encore volumineux** (~1098 lignes après refactor). Le refactor n'est pas fini — extraire vers `core/`, `filters/`, `ranking/` au lieu d'enfler ce fichier.

## Workflow attendu d'un agent

### Avant d'éditer `prompt_compressor.py`

Demande-toi : la logique appartient-elle à `core/`, `filters/`, ou `ranking/` ? Le but du refactor en cours est de **réduire** ce fichier, pas de le grossir.

### Ajouter un ranker

1. Crée la classe dans le module `ranking/` adapté (`statistical.py`, `neural.py`, etc.)
2. Décore avec `@RankingRegistry.register("nom")`
3. Importe dans `ranking/__init__.py` avec `# noqa: F401`
4. Ajoute un test unitaire dans `tests/` utilisant `tiny-mistral`

### Ajouter un filtre

1. Hérite de `FilterBase` (`filters/base.py`)
2. Accepte un `FilterContext` en input
3. Implémente `filter()` selon le contrat attendu par `prompt_compressor.py`
4. Test unitaire avec `tiny-mistral`

### Ajouter un modèle Mistral

1. **Vérifie qu'il existe** sur HuggingFace Hub (`https://huggingface.co/<id>`)
2. Édite `llmlangstral/mistral_config.py`
3. Ajoute un test dans `tests/test_mistral.py` avec `skipTest` si le modèle n'est pas téléchargeable

### Avant de commit

- `make style` — sinon le CI casse
- `make test` localement si tu as touché à du code de compression
- Ne commit pas avec des secrets HF dans les fichiers de config

## Documents internes à consulter

- **`PLAN.md`** — Plan de refactoring modulaire v0.3.0 (étapes terminées + restantes)
- **`MIGRATION_PLAN_MISTRAL.md`** — Mapping détaillé LLaMA→Mistral ligne par ligne
- **`DOCUMENT.md`** — Doc technique des principes de compression
- **`README.md`** — Présentation publique

## Limites des agents

- **Ne pas inventer d'identifiants de modèles HuggingFace** — vérifie d'abord
- **Ne pas committer `filters/llmlingua2.py` ou supprimer `core/tokenization.py`** sans confirmation explicite — l'état git du refactor est délicat
- **Ne pas désactiver les hooks pre-commit ni les flags `--no-verify`** sans demander
- **Ne pas réintroduire de logique métier dans `prompt_compressor.py`** — c'est l'inverse du refactor en cours
- **Ne pas modifier `setup.cfg` ni `pyproject.toml`** sans annoncer les conséquences (CI, release)
