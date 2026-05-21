# CLAUDE.md

Instructions pour Claude Code (claude.ai/code) lorsqu'il travaille dans ce dépôt.

## Vue d'ensemble du projet

**LLMLangstral** est un fork français de Microsoft LLMLingua, migré 100% vers les modèles Mistral AI. La bibliothèque compresse les prompts LLM jusqu'à 20x tout en préservant l'information sémantique, réduisant ainsi latence et coûts d'inférence.

### Une variante Mistral

| Mode | Activation | Usage |
|---|---|---|
| **LLMLangstral** (base) | `PromptCompressor()` | Compression itérative par perplexité avec Mistral causal (`Mistral-7B-v0.3`, `Ministral-3B`) |
| **LongLLMLangstral** | `compress_prompt(..., use_context_level_filter=True)` | Longs contextes, mitige le "lost in the middle" — même backbone Mistral |

Une seule classe d'entrée : `PromptCompressor` (`llmlangstral/prompt_compressor.py`). Le constructeur n'accepte plus que `model_name`, `device_map`, `model_config`.

## Architecture modulaire (v0.3.0)

Le refactor a éclaté l'ancien "God class" `PromptCompressor` en sous-modules plugins. Le fichier `prompt_compressor.py` ne fait plus que ~924 lignes et délègue.

```
llmlangstral/
├── __init__.py            # Exporte PromptCompressor, MISTRAL_MODELS, DEFAULT_MODEL
├── prompt_compressor.py   # Point d'entrée — orchestre core/filters/ranking
├── mistral_config.py      # Registre des modèles Mistral (voir plus bas)
├── utils.py               # seed_everything + helpers JSON/segments
├── version.py             # VERSION = "0.3.0"
│
├── core/                  # Chargement modèle + ABC + résultat
│   ├── base.py            # CompressionResult (dataclass) + BaseCompressor (ABC)
│   └── model_loader.py    # ModelManager — lazy-load HuggingFace
│
├── filters/               # Filtres de compression multi-niveaux
│   ├── base.py            # FilterContext (dataclass) + FilterBase (ABC)
│   ├── context.py         # ContextLevelFilter — niveau passage
│   ├── sentence.py        # SentenceLevelFilter — niveau phrase
│   └── token.py           # TokenLevelFilter — niveau token (élagage PPL)
│
└── ranking/               # Stratégies de ranking via plugin registry
    ├── registry.py        # RankingRegistry — décorateur @register("nom")
    ├── base.py            # RankingStrategy, ModelBasedRanker, PPLBasedRanker
    ├── statistical.py     # BM25Ranker, GzipRanker
    ├── llmlingua.py       # LLMLinguaRanker (alias "llmlingua" + "longllmlingua")
    └── mistral.py         # MistralRanker — intfloat/e5-mistral-7b-instruct
```

### Pattern de plugin (ranking)

Chaque ranker s'enregistre via décorateur :

```python
@RankingRegistry.register("bm25")
class BM25Ranker(RankingStrategy): ...
```

Les imports avec `# noqa: F401` dans `ranking/__init__.py` déclenchent l'enregistrement. Pour ajouter un ranker : créer la classe + décorer + importer dans `__init__.py`.

Le registre contient exactement 5 clés : `bm25`, `gzip`, `llmlingua`, `longllmlingua`, `mistral`. Tout ranker non-Mistral est interdit en v0.3.x.

### Orchestration

- `PromptCompressor.__init__` instancie `ModelManager` (`core/model_loader.py`) et expose `model`/`tokenizer`/`device` via `@property` qui délèguent au manager.
- `FilterContext` est lazy-construit à la première utilisation (property `_filters`) et porte les callbacks `get_ppl`, `get_condition_ppl`, `get_rank_results`.
- `get_rank_results()` dispatch vers `RankingRegistry.get(method)`.

## Point d'entrée — API publique

```python
from llmlangstral import PromptCompressor

# Compression Mistral standard
compressor = PromptCompressor()  # défaut: mistralai/Mistral-7B-v0.3, device_map="cuda"
result = compressor.compress_prompt(
    context=["doc1", "doc2"],
    instruction="",
    question="",
    target_token=200,
)

# Mode long-contexte (LongLLMLangstral)
result = compressor.compress_prompt(
    context=[...],
    rate=0.5,
    use_context_level_filter=True,
    rank_method="longllmlingua",
)
```

### Méthodes clés (`llmlangstral/prompt_compressor.py`)

| Méthode | Rôle |
|---|---|
| `compress_prompt()` | Compression Mistral universelle (32 kwargs publics) |
| `structured_compress_prompt()` | Tags XML `<llmlingua rate=...>` |
| `compress_json()` | Compression JSON par champ |
| `iterative_compress_prompt()` | Multi-passe |
| `get_rank_results()` | Dispatch ranking via registry |

### Concept de structure de prompt

Les prompts sont divisés en trois composants à sensibilité différente :

- **Instruction** (sensibilité HAUTE) — description de tâche, placée en premier
- **Context** (sensibilité BASSE) — documents, exemples, démos (compressible agressivement)
- **Question** (sensibilité HAUTE) — requête utilisateur, placée à la fin

### Tags structurés

Le wire-format est conservé (compat utilisateur) :

```python
"<llmlingua, compress=False>Texte à préserver intact</llmlingua>"
"<llmlingua, rate=0.5>Compresser à 50%</llmlingua>"
```

> Décision : tag `<llmlingua, ...>` non renommé en `<llmlangstral, ...>` — c'est un protocole d'entrée, pas une marque. Pattern fork-compat (Neovim/MariaDB).

## Configuration des modèles Mistral

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
TEST_MODEL    = "openaccess-ai-collective/tiny-mistral"  # CI uniquement
DEFAULT_MODEL = MISTRAL_MODELS["default"]
```

> ⚠ **Modèles fantômes** : `Ministral-3-3B-Instruct-2512`, `Ministral-3-8B-Instruct-2512` et `Mistral-Large-3` **n'existent pas** sur HuggingFace Hub à ce jour. Seuls `Mistral-7B-v0.3` et `TheBloke/...-GPTQ` sont réels. Les tests qui ciblent les autres modèles passent par `skipTest` si indisponibles. **N'invente pas** d'identifiants supplémentaires — vérifie sur HF Hub avant de référencer un modèle.

## Capacité retirée — LLMLingua-2 / SecurityLingua

La v0.3.0 supprime le chemin XLM-RoBERTa (LLMLingua-2 et SecurityLingua) ainsi que 9 rankers non-Mistral (BGE, SentBert, Jinza, OpenAI, VoyageAI, Cohere). **Conséquence assumée** : perte du chemin "fast token-classification" 3-6x plus rapide qu'une compression par perplexité. Aucun équivalent Mistral n'existe à ce jour (Mistral n'a pas de modèle encoder pré-entraîné pour token classification).

- **Migration** : retire les kwargs `use_llmlingua2`, `use_slingua`, `open_api_config`, `llmlingua2_config` de `PromptCompressor(...)` et les 8 kwargs `return_word_label`, `word_sep`, `label_sep`, `token_to_word`, `force_tokens`, `force_reserve_digit`, `drop_consecutive`, `chunk_end_tokens` de `compress_prompt(...)`.
- **Archive** : les pipelines d'entraînement (`experiments/llmlangstral2/`, `experiments/securitylingua/`) sont préservés sur la branche `legacy/experiments`. Voir US-013.
- **Réversibilité** : tag `v0.2.2` (à créer avant merge v0.3.0) ou `pip install llmlangstral==0.2.2`.

## Commandes courantes

```bash
# Installation (mode dev)
pip install -e ".[dev]"

# Lancer tous les tests (parallèle)
make test
# équivalent : pytest -n auto --dist=loadfile -s -v ./tests/

# Un seul fichier de test
pytest tests/test_llmlangstral.py -v

# Un test précis
pytest tests/test_llmlangstral.py::LLMLangstralTester::test_compress_prompt -v

# Format et lint (Black 88 + isort + flake8 119)
make style
```

## Tests

| Fichier | Modèle utilisé | Couvre |
|---|---|---|
| `tests/test_llmlangstral.py` | `tiny-mistral` (CPU) | Base, `compress_prompt`, structured, JSON |
| `tests/test_longllmlangstral.py` | `tiny-mistral` (CPU) | Context filter, multi-doc, reorder |
| `tests/test_mistral.py` | `Ministral-3-3B` (skip si absent) | Config Mistral, fonctionnel |

Pas de `conftest.py`. CI : `.github/workflows/unittest.yml` (Ubuntu/macOS/Windows × Python 3.9/3.10/3.11), nécessite un secret `HF_TOKEN`.

## Conventions de code

- **Python** : `>=3.9` ciblé (3.8 abandonné post-v0.3.0), CI sur 3.9/3.10/3.11
- **Black** : `line-length=88`, `target-version=['py38']`
- **isort** : `profile="black"`, `known_first_party=["llmlangstral"]`
- **Flake8** : `max-line-length=119`, ignore `E203, E501, E741, W503, W605`
- **Docstrings** : style Google/numpy (sections `Args:`, `Returns:`, `Raises:`, `Example:`)
- **Type hints** : utilisés systématiquement dans les nouvelles classes

> ⚠ Le code utilise la syntaxe `dict[str, Any]` (3.9+) sous `from __future__ import annotations`. Ne pas réintroduire des annotations non-quotées sans le `from __future__`.

## Pièges connus de ce repo

1. **`PromptCompressor` reste un peu fourre-tout** (~940 lignes après refactor). Plan : continuer à extraire vers `core/filters/ranking/`. Voir `PLAN.md`.
2. **Quality gate v0.3.0** : `grep -rE "use_llmlingua2|use_slingua|LLMLingua2Compressor|OpenAIRanker|VoyageAIRanker|CohereRanker|BGERanker|SentBertRanker|JinzaRanker|APIBasedRanker|BGEReranker|BGELLMEmbedderRanker|LLMLANGSTRAL2_MODEL|init_llmlingua2" llmlangstral/ tests/` doit retourner zéro. Vérifie post-modif.
3. **Modèle fantôme dans `test_mistral.py`** : la classe `TestMistralCompressor` cible `SMALL_MODEL` qui n'existe pas encore sur HuggingFace Hub → tous ses tests sont systématiquement skippés en CI. Voir docstring de la classe pour l'activation future.

### Pièges précédemment signalés, désormais résolus

- ~~`setup.cfg` contenait `known_first_party = sdtools`~~ → corrigé : `known_first_party = llmlangstral`.
- ~~`Makefile` `install` cherchait `dist/sdtools*`~~ → corrigé : `dist/llmlangstral*`.
- ~~`tiktoken`/GPT-3.5-turbo utilisé pour compter `origin_tokens` et `compressed_tokens`~~ → remplacé par le tokenizer Mistral via `self.get_token_length()`. Le ratio de compression rapporté est désormais cohérent avec le backbone Mistral.
- ~~`PromptCompressor` ne dérivait pas de `BaseCompressor`~~ → hérite désormais de l'ABC et expose `compress()` retournant un `CompressionResult` (les anciens callers continuent d'utiliser `compress_prompt()` pour le dict legacy).
- ~~Branche `AutoModelForTokenClassification` dans `model_loader.py`~~ → retirée : v0.3.x est causal-LM-only.
- ~~`model_config: dict[str, Any] = {}` (mutable default)~~ → corrigé en `Optional[dict[str, Any]] = None`.

## Documents internes à consulter

- **`tasks/prd-full-mistral-refactor.md`** — PRD de la v0.3.0 mono-Mistral (stories US-001 à US-015)
- **`tasks/prd-full-mistral-refactor-status.json`** — Status JSON machine-readable des stories
- **`PLAN.md`** — Plan de refactoring modulaire v0.3.0 (état actuel, étapes restantes)
- **`MIGRATION_PLAN_MISTRAL.md`** — Mapping détaillé LLaMA→Mistral, ligne par ligne
- **`DOCUMENT.md`** — Doc technique des principes de compression
- **`README.md`** — Présentation publique du fork

## Examples & expériences

- `examples/` — notebooks Jupyter Mistral-only : `RAG.ipynb`, `RAGLlamaIndex.ipynb`, `CoT.ipynb`, `Code.ipynb`, `OnlineMeeting.ipynb`, `Retrieval.ipynb` (le notebook `LLMLangstral2.ipynb` est retiré en US-013, archivé sur `legacy/experiments`).
- Les pipelines d'entraînement non-Mistral (`experiments/llmlangstral2/`, `experiments/securitylingua/`) sont déplacés vers la branche `legacy/experiments` (US-013) pour préserver la valeur historique sans charger le tronc.

## Quand tu travailles sur ce repo

- **Avant de modifier `prompt_compressor.py`** : vérifie d'abord si la logique appartient à `core/`, `filters/`, ou `ranking/`. Le but du refactor est de **réduire** ce fichier.
- **Pour ajouter un ranker** : crée la classe dans le module `ranking/` approprié, décore avec `@RankingRegistry.register("nom")`, ajoute l'import dans `ranking/__init__.py` avec `# noqa: F401`. **Seuls les rankers compatibles Mistral sont acceptés en v0.3.x** (pas de retour BGE/OpenAI/etc.).
- **Pour ajouter un filtre** : hérite de `FilterBase`, prends `FilterContext` en input, retourne un objet conforme à l'interface attendue par `prompt_compressor.py`.
- **Pour ajouter un modèle Mistral** : édite `llmlangstral/mistral_config.py` **et** vérifie qu'il existe sur HuggingFace Hub avant.
- **Tests** : tout nouveau filtre/ranker doit avoir un test unitaire utilisant `tiny-mistral` pour rester rapide en CI.
- **Lint avant commit** : `make style` (sinon le CI casse).
