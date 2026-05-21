[PRD]
# PRD: Refactor Full-Mistral — LLMLangstral v0.3.0

## Changelog

| Version | Date | Author | Summary |
|---------|------|--------|---------|
| 1.0 | 2026-05-21 | Arthur Jean (via write-prd) | Initial draft — refactor v0.3.0 mono-provider Mistral |

## Problem Statement

LLMLangstral est aujourd'hui présenté comme un fork Mistral-only de Microsoft LLMLingua, mais l'audit du codebase (mai 2026) révèle un écart structurel entre la promesse et l'implémentation :

1. **Le projet n'est pas Mistral-only.** 9 rankers sur 13 ciblent d'autres providers (BGE, SentBert, Jina, OpenAI, VoyageAI, Cohere). 3 variantes sur 4 utilisent XLM-RoBERTa (LLMLingua-2, SecurityLingua) au lieu de Mistral. La constante `LLMLANGSTRAL2_MODEL` pointe explicitement vers `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`.

2. **Surface API encombrée par des paramètres non-Mistral.** `PromptCompressor.__init__` expose `use_llmlingua2`, `use_slingua`, `llmlingua2_config`, `open_api_config` ; `compress_prompt` expose 8 kwargs (`return_word_label`, `word_sep`, `label_sep`, `token_to_word`, `force_tokens`, `force_reserve_digit`, `drop_consecutive`, `chunk_end_tokens`) qui sont silencieusement transmis au seul chemin LLMLingua-2 sans effet sur le chemin principal Mistral.

3. **Dette technique aggravée par le refactor modulaire v0.3.0 inachevé.** Le helper `utils.py` contient 4 fonctions (`TokenClfDataset`, `is_begin_of_new_word`, `get_pure_token`, `replace_added_token`) exclusivement consommées par `filters/llmlingua2.py`. Le module `ranking/base.py` déclare `APIBasedRanker` qui sert exclusivement aux 3 rankers API supprimables.

4. **Documentation incohérente.** `CLAUDE.md` annonce 4 variantes ; `README.md` parle de Mistral ; `mistral_config.py` documente que LLMLingua-2 reste sous XLM-RoBERTa. Le `setup.cfg` contient encore `known_first_party = sdtools` (ancien nom hérité).

**Why now :** La conversation utilisateur du 2026-05-21 a explicitement validé la direction "100% Mistral" après audit. Le refactor v0.3.0 modulaire (`core/`, `filters/`, `ranking/`) déjà entamé crée la fenêtre idéale pour matérialiser la rupture en un bump de version cohérent (0.2.2 → 0.3.0). Différer signifie accumuler de nouvelles dépendances cross-provider et rendre le coût du refactor exponentiel.

## Overview

Le refactor "Full-Mistral" est un **breaking change v0.3.0** qui aligne l'implémentation avec le positionnement du fork. Approche en 3 axes :

**Axe 1 — Suppression chirurgicale.** Suppression atomique (code + tests + docs dans un seul commit par story) de tout chemin XLM-RoBERTa (LLMLingua-2, SecurityLingua) et de tous les rankers non-Mistral (BGE, SentBert, Jina, OpenAI, VoyageAI, Cohere). Pas de DeprecationWarning : le research (Larson 2024) confirme que les avertissements silencieux échouent à atteindre les downstream maintainers, et SemVer 0.x autorise les hard-removals dans une release mineure.

**Axe 2 — Nettoyage de l'API publique.** `PromptCompressor.__init__` perd `use_llmlingua2`, `use_slingua`, `llmlingua2_config`, `open_api_config`. `compress_prompt` perd les 8 kwargs LLMLingua-2-only. Les imports `from llmlangstral import LLMLingua2Compressor` cassent franchement (ImportError) plutôt que silencieusement.

**Axe 3 — Préservation du protocole.** Les tags structurés `<llmlingua, rate=...>` sont **conservés tels quels** car ils constituent un protocole d'entrée utilisateur (pas une marque), suivant le pattern Neovim/MariaDB de compat wire-format après fork. Les classes `LLMLinguaRanker`, `RankingRegistry`, et la classe `BM25Ranker`/`GzipRanker` (sans dépendance ML) restent — leur nom est interne et leur valeur orthogonale au refactor.

**Limitation assumée :** LLMLingua-2 est 3-6x plus rapide que le chemin perplexity-based (paper arXiv 2403.12968). Aucun équivalent Mistral n'existe à la date du PRD. Cette perte de capacité est documentée dans le README et les pipelines d'entraînement de l'expériment sont archivés sur une branche `legacy/experiments` pour réversibilité future.

## Goals

| Goal | Month-1 Target | Month-6 Target |
|------|---------------|----------------|
| Zéro symbole non-Mistral dans `llmlangstral/` (mesuré par grep automatisé) | 100% (0 occurrence) | 100% |
| Réduction du nombre de rankers enregistrés | 13 → 5 (`bm25`, `gzip`, `llmlingua`, `longllmlingua`, `mistral`) | 5 maintenus |
| Réduction de la surface API publique de `PromptCompressor` | Retirer 4 kwargs `__init__` + 8 kwargs `compress_prompt` | Pas de nouveau kwarg non-Mistral introduit |
| Taille du fichier `prompt_compressor.py` | < 950 lignes (vs 1098 actuelles) | < 900 lignes |
| Test suite verte sur Python 3.9/3.10/3.11 | `make test` 100% pass | Idem + Python 3.12 |
| Documentation cohérente (CLAUDE.md, README.md, DOCUMENT.md, Transparency_FAQ.md) | 0 référence runtime à LLMLingua-2 / XLM-RoBERTa | Idem |

## Target Users

### Développeur Python utilisant LLMLangstral pour compresser des prompts Mistral
- **Role :** Ingénieur backend ou ML, intègre la compression dans un pipeline RAG ou agent qui consomme l'API Mistral
- **Behaviors :** Installe via `pip install llmlangstral`, charge `PromptCompressor()`, appelle `compress_prompt()` avant l'inférence Mistral
- **Pain points :** Doit aujourd'hui parser une API encombrée par 4 variantes différentes ; ne sait pas si `use_llmlingua2=True` est compatible avec son cas ; télécharge ~1.1 GB de poids XLM-RoBERTa qu'il n'utilisera jamais
- **Current workaround :** Lit le code source pour confirmer quel chemin son code emprunte ; ignore les kwargs LLMLingua-2-only en espérant qu'ils ne cassent rien
- **Success looks like :** Une seule API (`compress_prompt(prompt, target_token=...)`), aucun téléchargement de modèle non-Mistral, signature de fonction claire

### Contributeur du fork LLMLangstral
- **Role :** Maintenir et étendre le fork, ajouter des rankers Mistral spécialisés ou des filtres
- **Behaviors :** Lit CLAUDE.md, PLAN.md, MIGRATION_PLAN_MISTRAL.md ; ouvre des PRs ciblées sur `core/`, `filters/`, `ranking/`
- **Pain points :** Le découpage actuel mélange code Mistral et XLM-RoBERTa ; impossible d'ajouter un nouveau ranker Mistral sans comprendre les contraintes de `LLMLingua2Compressor`
- **Current workaround :** Évite de toucher au code touché par les multiples variantes
- **Success looks like :** Architecture cohérente où chaque module a une raison d'être Mistral-relative, registry plugin simple à étendre

### Mainteneur de package downstream (utilisateur indirect)
- **Role :** Maintient une lib ou une application qui dépend de `llmlangstral`
- **Behaviors :** Pin la version, surveille les breaking changes via changelog
- **Pain points :** Aujourd'hui aucun changelog versionné des breaking ; les imports `LLMLingua2Compressor` etc. cassent silencieusement si supprimés sans signal
- **Current workaround :** Pin strict (`llmlangstral==0.2.2`)
- **Success looks like :** Changelog v0.3.0 explicite ; ImportError clair (pas silent failure) ; migration guide ; branche `legacy/v0.2.x` préservée

## Research Findings

### Competitive Context
- **openai-python (mono-provider) :** Thin, prévisible, idiomatique au provider unique. Modèle de référence pour la simplicité d'usage.
- **LangChain / LiteLLM (multi-provider) :** Couche d'adapter qui leak les divergences provider en error handling et response shapes. Coût de maintenance élevé.
- **Microsoft LLMLingua (upstream) :** Multi-provider hérité, capacités larges mais surface API trouble.
- **Market gap :** Aucune lib dédiée à la compression de prompts spécifiquement optimisée Mistral. Le fork LLMLangstral peut occuper ce créneau opinionated.

### Best Practices Applied
- **SemVer 0.x = liberté de hard-removal** (semver.org) → Pas de fenêtre de dépréciation pour 0.2.2 → 0.3.0
- **DeprecationWarning silencieux ne marche pas** (Seth Larson 2024) → Préférer un changelog explicite + version pin
- **Atomic deletion** : code + tests + docs + extras dans un commit par story → évite stale references et test drift
- **Wire-format compatibility après fork** (Neovim/MariaDB) → Conserver les tags `<llmlingua, ...>` intactes

*Full research sources available in this PRD's git history (Phase 2 of write-prd execution, 2026-05-21).*

## Assumptions & Constraints

### Assumptions (to validate)
- **L'utilisateur principal accepte la perte de capacité LLMLingua-2** — basé sur la conversation explicite du 2026-05-21 ("Fait toute les étapes le but et que ce soit full mistral")
- **Aucun consumer downstream stable n'existe encore** — le projet est en `0.2.2`, marqué "Development Status :: 3 - Alpha" dans setup.py, classifié comme un fork pré-1.0
- **Le wire-format `<llmlingua, ...>` est utilisé en production par les utilisateurs actuels** — non vérifié empiriquement, mais documenté dans DOCUMENT.md et testé dans `test_longllmlangstral.py` → conservation par défaut
- **`tiktoken` reste un compromis acceptable pour le token counting** — la migration vers le tokenizer Mistral introduirait des écarts numériques observables par les utilisateurs (token counts différents dans le dict retourné)
- **Les pipelines d'entraînement dans `experiments/` ont une valeur historique** — la branche `legacy/experiments` les préserve sans charger le tronc principal

### Hard Constraints
- **Python 3.9+ minimum après le refactor** (déjà entamé via `from __future__ import annotations`, voir commit précédent)
- **Backwards-compat de `compress_prompt()` pour les kwargs non-LLMLingua-2** : signature exacte préservée pour les paramètres conservés (24 kwargs)
- **Tags structurés `<llmlingua, rate=X>`, `<llmlingua, compress=False>` inchangés** : tout test passant sur 0.2.2 et utilisant ces tags doit continuer à passer
- **CI verte sur Ubuntu, macOS, Windows × Python 3.9, 3.10, 3.11** (matrice actuelle de `.github/workflows/unittest.yml`)
- **`make style`** (Black 88 + isort + flake8 119) doit passer après chaque story
- **`make test`** doit passer après chaque story (sauf US-015 qui valide le tout)

## Quality Gates

These commands must pass for every user story:
- `make style` - Format check (Black 88 + isort) + lint (flake8 119)
- `make test` - Full pytest suite with parallel execution (`pytest -n auto --dist=loadfile`)
- `python -c "import llmlangstral; print(llmlangstral.__version__)"` - Smoke import test
- `python -m py_compile $(git ls-files 'llmlangstral/**/*.py' 'tests/**/*.py')` - Syntax compile check
- `! grep -rE "use_llmlingua2|use_slingua|LLMLingua2Compressor|OpenAIRanker|VoyageAIRanker|CohereRanker|BGERanker|SentBertRanker|JinzaRanker|APIBasedRanker|BGEReranker|BGELLMEmbedderRanker|LLMLANGSTRAL2_MODEL|init_llmlingua2" llmlangstral/ tests/ 2>/dev/null` - Dead-symbol grep must return empty (story-scoped : applicable une fois sa portée terminée)

## Epics & User Stories

### EP-001: Suppression du chemin LLMLingua-2 (XLM-RoBERTa)

Suppression complète de la variante LLMLingua-2 et de son cousin SecurityLingua qui partagent le backend XLM-RoBERTa. Inclut le code de compression, l'initialisation du ModelManager, la configuration model, et les tests associés.

**Definition of Done :** Plus aucune référence runtime à `LLMLingua2Compressor`, `use_llmlingua2`, `use_slingua`, `init_llmlingua2`, `LLMLANGSTRAL2_MODEL` dans `llmlangstral/` et `tests/`. Le grep guard du Quality Gate retourne zéro pour ces symboles.

#### US-001: Supprimer `filters/llmlingua2.py` et ses tests
**Description :** En tant que mainteneur, je veux supprimer atomiquement le fichier `llmlangstral/filters/llmlingua2.py` et son test `tests/test_llmlangstral2.py` afin de retirer le chemin de compression XLM-RoBERTa en un seul commit, évitant ainsi le risque de test drift CI (téléchargement de 1.1 GB sur un code orphelin).

**Priority:** P0
**Size:** S (2 pts)
**Dependencies:** None

**Acceptance Criteria:**
- [ ] Le fichier `llmlangstral/filters/llmlingua2.py` est supprimé via `git rm`
- [ ] Le fichier `tests/test_llmlangstral2.py` est supprimé via `git rm`
- [ ] La ligne `from .llmlingua2 import LLMLingua2Compressor` est retirée de `llmlangstral/filters/__init__.py:37`
- [ ] La chaîne `"LLMLingua2Compressor"` est retirée du `__all__` de `llmlangstral/filters/__init__.py:45`
- [ ] `python -c "from llmlangstral.filters.llmlingua2 import LLMLingua2Compressor"` lève `ModuleNotFoundError` (unhappy path : import explicite échoue franchement, pas silencieusement)
- [ ] `python -c "from llmlangstral.filters import LLMLingua2Compressor"` lève `ImportError`
- [ ] `make style` et `make test` passent

#### US-002: Retirer les flags `use_llmlingua2` / `use_slingua` de `PromptCompressor`
**Description :** En tant que mainteneur, je veux retirer les paramètres `use_llmlingua2`, `use_slingua`, `llmlingua2_config`, `open_api_config` du constructeur de `PromptCompressor` afin d'aligner la surface API avec le positionnement Mistral-only.

**Priority:** P0
**Size:** M (3 pts)
**Dependencies:** Blocked by US-001

**Acceptance Criteria:**
- [ ] Les paramètres `use_llmlingua2`, `use_slingua`, `llmlingua2_config`, `open_api_config` sont retirés de la signature `PromptCompressor.__init__` (lignes 67-75)
- [ ] Les attributs d'instance `self.use_llmlingua2`, `self.use_slingua`, `self.open_api_config`, `self._llmlingua2` sont retirés
- [ ] Le bloc `if use_llmlingua2 or use_slingua:` aux lignes 93-94 est retiré
- [ ] La property `_llm2` (lignes 737-746) et la méthode `compress_prompt_llmlingua2` (lignes 748-...) sont retirées
- [ ] Le branchement `if self.use_llmlingua2:` au début de `compress_prompt` (ligne 546) est retiré ainsi que le bloc d'appel `return self.compress_prompt_llmlingua2(...)` (lignes 547-565)
- [ ] `PromptCompressor(use_llmlingua2=True)` lève `TypeError: __init__() got an unexpected keyword argument 'use_llmlingua2'` (unhappy path explicite)
- [ ] La docstring de `PromptCompressor` (lignes 32-65) est mise à jour : retrait des références LLMLingua-2 et de l'example `model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank"`
- [ ] `make style` et `make test` passent

#### US-003: Retirer `init_llmlingua2()` et les properties LLMLingua-2-only de `ModelManager`
**Description :** En tant que mainteneur, je veux retirer la méthode `init_llmlingua2()` et les 5 attributs LLMLingua-2-only de `ModelManager` afin que la classe n'expose plus que des concepts Mistral pertinents.

**Priority:** P0
**Size:** S (2 pts)
**Dependencies:** Blocked by US-002

**Acceptance Criteria:**
- [ ] La méthode `ModelManager.init_llmlingua2()` (`core/model_loader.py:177-216`) est supprimée
- [ ] Les attributs `max_batch_size`, `max_seq_len`, `max_force_token`, `special_tokens`, `added_tokens` sont retirés de `ModelManager.__init__` (lignes 65-69)
- [ ] Les properties de délégation associées dans `PromptCompressor` (lignes 133-156) sont retirées
- [ ] L'import de `seed_everything` dans `model_loader.py:16` est retiré s'il n'est plus utilisé (vérification grep)
- [ ] `python -c "from llmlangstral.core import ModelManager; m = ModelManager('mistralai/Mistral-7B-v0.3'); m.init_llmlingua2()"` lève `AttributeError` (unhappy path)
- [ ] `make style` et `make test` passent

#### US-004: Retirer `LLMLANGSTRAL2_MODEL` de `mistral_config.py`
**Description :** En tant que mainteneur, je veux retirer la constante `LLMLANGSTRAL2_MODEL` qui pointe vers un modèle XLM-RoBERTa afin que le module de configuration ne référence que des identifiants Mistral.

**Priority:** P0
**Size:** XS (1 pt)
**Dependencies:** Blocked by US-002

**Acceptance Criteria:**
- [ ] La ligne `LLMLANGSTRAL2_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"` (mistral_config.py:43) est supprimée
- [ ] Le commentaire d'introduction lignes 40-42 est supprimé
- [ ] `python -c "from llmlangstral.mistral_config import LLMLANGSTRAL2_MODEL"` lève `ImportError` (unhappy path : import explicite échoue)
- [ ] Aucun import résiduel de `LLMLANGSTRAL2_MODEL` dans le repo (grep zéro hit)
- [ ] `make style` et `make test` passent

---

### EP-002: Suppression des rankers non-Mistral

Suppression des 9 rankers qui dépendent de modèles ou providers non-Mistral : BGE (BAAI), SentBert (multi-qa-mpnet), Jina, OpenAI, VoyageAI, Cohere. Les rankers `bm25`, `gzip`, `llmlingua`/`longllmlingua`, `mistral` sont conservés.

**Definition of Done :** Les modules `ranking/neural.py` et `ranking/api_based.py` n'existent plus. `RankingRegistry._strategies` contient exactement 5 clés : `bm25`, `gzip`, `llmlingua`, `longllmlingua`, `mistral`. La classe `APIBasedRanker` n'existe plus.

#### US-005: Supprimer `ranking/neural.py` (BGE, SentBert, Jinza, BGEReranker, BGELLMEmbedderRanker)
**Description :** En tant que mainteneur, je veux supprimer le module `ranking/neural.py` qui contient 5 rankers dépendant de modèles non-Mistral (BAAI BGE, multi-qa-mpnet, jinaai), afin de réduire la surface du registry et éliminer les imports lazy à `sentence-transformers` non utilisés en chemin Mistral.

**Priority:** P0
**Size:** S (2 pts)
**Dependencies:** None (parallèle à EP-001)

**Acceptance Criteria:**
- [ ] Le fichier `llmlangstral/ranking/neural.py` est supprimé via `git rm`
- [ ] La ligne `from . import neural  # noqa: F401` est retirée de `llmlangstral/ranking/__init__.py:53`
- [ ] La docstring du module ranking (lignes 11-25) est mise à jour : retrait des entrées `sentbert`, `bge`, `bge_reranker`, `bge_llmembedder`, `jinza`
- [ ] `RankingRegistry.get("bge")` retourne `None` ou lève (selon l'implémentation actuelle — à vérifier) — comportement documenté dans le test
- [ ] Test : `RankingRegistry._strategies` ne contient pas les clés `sentbert`, `bge`, `bge_reranker`, `bge_llmembedder`, `jinza` (unhappy path : leur usage échoue franchement)
- [ ] `make style` et `make test` passent

#### US-006: Supprimer `ranking/api_based.py` (OpenAI, VoyageAI, Cohere)
**Description :** En tant que mainteneur, je veux supprimer le module `ranking/api_based.py` afin de retirer toute dépendance optionnelle aux APIs externes (OpenAI, VoyageAI, Cohere) et matérialiser le positionnement self-hosted Mistral-only.

**Priority:** P0
**Size:** XS (1 pt)
**Dependencies:** None (parallèle à US-005)

**Acceptance Criteria:**
- [ ] Le fichier `llmlangstral/ranking/api_based.py` est supprimé via `git rm`
- [ ] La ligne `from . import api_based  # noqa: F401` est retirée de `llmlangstral/ranking/__init__.py:49`
- [ ] La docstring du module ranking : retrait des entrées `openai`, `voyageai`, `cohere`
- [ ] Test : `RankingRegistry._strategies` ne contient pas les clés `openai`, `voyageai`, `cohere`
- [ ] `make style` et `make test` passent

#### US-007: Retirer `APIBasedRanker` de `ranking/base.py` et nettoyer `ranking/__init__.py`
**Description :** En tant que mainteneur, je veux retirer la classe abstraite `APIBasedRanker` (orpheline après US-006) et nettoyer le `__all__` de `ranking/__init__.py` afin de ne pas exposer des classes sans implémentation.

**Priority:** P0
**Size:** XS (1 pt)
**Dependencies:** Blocked by US-005, US-006

**Acceptance Criteria:**
- [ ] La classe `APIBasedRanker` (`ranking/base.py:74-84`) est supprimée
- [ ] L'import `APIBasedRanker` dans `ranking/__init__.py:41` est retiré
- [ ] La chaîne `"APIBasedRanker"` est retirée du `__all__` de `ranking/__init__.py:58`
- [ ] `python -c "from llmlangstral.ranking import APIBasedRanker"` lève `ImportError`
- [ ] La docstring du module `ranking/__init__.py` est mise à jour pour décrire 5 rankers (au lieu de 13)
- [ ] `make style` et `make test` passent

---

### EP-003: Nettoyage des helpers utilitaires

Suppression des 4 fonctions dans `utils.py` exclusivement consommées par le `LLMLingua2Compressor` supprimé.

**Definition of Done :** `utils.py` ne contient plus que les helpers utilisés par le chemin Mistral principal (`seed_everything`, `process_structured_json_data`, `precess_jsonKVpair`, `process_sequence_data`, `remove_consecutive_commas`, `segment_structured_context`, `concate_segment_info`).

#### US-008: Supprimer `TokenClfDataset`, `is_begin_of_new_word`, `get_pure_token`, `replace_added_token` de `utils.py`
**Description :** En tant que mainteneur, je veux retirer les 4 helpers exclusivement consommés par le `LLMLingua2Compressor` (supprimé en EP-001) afin d'éliminer du code mort et simplifier `utils.py`.

**Priority:** P0
**Size:** S (2 pts)
**Dependencies:** Blocked by US-001

**Acceptance Criteria:**
- [ ] La classe `TokenClfDataset` (`utils.py:14-78`) est supprimée
- [ ] Les fonctions `is_begin_of_new_word` (lignes 93-120), `replace_added_token` (lignes 125-128), `get_pure_token` (lignes 131-141) sont supprimées
- [ ] Les imports `from torch.utils.data import Dataset` et `import string` sont retirés s'ils ne sont plus utilisés (vérification grep dans le fichier)
- [ ] La fonction `seed_everything` (lignes 81-88) est conservée (utilisée potentiellement ailleurs — vérifier et garder par sécurité)
- [ ] `python -c "from llmlangstral.utils import TokenClfDataset"` lève `ImportError`
- [ ] `python -c "from llmlangstral.utils import is_begin_of_new_word"` lève `ImportError`
- [ ] `python -c "from llmlangstral.utils import get_pure_token"` lève `ImportError`
- [ ] `make style` et `make test` passent

---

### EP-004: Nettoyage de l'API publique de `compress_prompt`

Suppression des kwargs de `compress_prompt` qui ne sont consommés que par le chemin LLMLingua-2 supprimé.

**Definition of Done :** La signature de `compress_prompt` ne contient plus aucun paramètre LLMLingua-2-only. Toute tentative d'invocation avec ces kwargs lève `TypeError` clair.

#### US-009: Retirer les 8 kwargs LLMLingua-2-only de `compress_prompt`
**Description :** En tant que développeur utilisant `compress_prompt`, je veux que la signature ne contienne que les paramètres effectivement consommés par le chemin Mistral afin de ne pas être induit en erreur par des kwargs ignorés silencieusement.

**Priority:** P0
**Size:** M (3 pts)
**Dependencies:** Blocked by US-002

**Acceptance Criteria:**
- [ ] Les kwargs `return_word_label`, `word_sep`, `label_sep`, `token_to_word`, `force_tokens`, `force_reserve_digit`, `drop_consecutive`, `chunk_end_tokens` sont retirés de la signature de `compress_prompt`
- [ ] La docstring de `compress_prompt` (lignes 434-545) est mise à jour : retrait de la documentation des 8 kwargs
- [ ] Les références dans la docstring aux clés de retour `"compressed_prompt_list"` (ligne 536) et `"fn_labeled_original_prompt"` (ligne 537) sont retirées ou marquées "réservé"
- [ ] `compress_prompt(["foo"], force_tokens=["\n"])` lève `TypeError: compress_prompt() got an unexpected keyword argument 'force_tokens'` (unhappy path explicite)
- [ ] `make style` et `make test` passent

#### US-010: Mettre à jour la docstring de `PromptCompressor` (exemple + lien arXiv)
**Description :** En tant que développeur lisant la documentation inline, je veux que l'example dans la docstring de `PromptCompressor` utilise un modèle Mistral réel afin de ne pas être induit en erreur par un example XLM-RoBERTa.

**Priority:** P0
**Size:** XS (1 pt)
**Dependencies:** Blocked by US-002

**Acceptance Criteria:**
- [ ] L'example ligne 57 `>>> compress_method = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, )` est remplacé par un example Mistral : `>>> compress_method = PromptCompressor(model_name="mistralai/Mistral-7B-v0.3", device_map="cpu")`
- [ ] Les références aux papers LLMLingua-2 (arXiv 2403.12968) dans la docstring sont retirées
- [ ] La section "Args" mentionne uniquement les paramètres conservés
- [ ] `make style` passe (Black + isort + flake8 sur le fichier modifié)

---

### EP-005: Documentation, archivage et finalisation v0.3.0

Mise à jour de toute la documentation utilisateur et interne, archivage des expériences d'entraînement non-Mistral sur une branche dédiée, et bump de version.

**Definition of Done :** `CLAUDE.md`, `README.md`, `DOCUMENT.md`, `Transparency_FAQ.md` ne référencent plus aucun chemin non-Mistral runtime. La version est bumpée à `0.3.0` avec un changelog. Le quality gate grep retourne zéro pour tout symbole supprimé.

#### US-011: Mettre à jour `CLAUDE.md` (table des variantes + sections architecture)
**Description :** En tant que contributeur lisant `CLAUDE.md` pour comprendre le projet, je veux que le document reflète l'architecture mono-variante Mistral afin de ne pas être induit en erreur par la table "Quatre variantes".

**Priority:** P1
**Size:** S (2 pts)
**Dependencies:** Blocked by EP-001, EP-002, EP-003, EP-004

**Acceptance Criteria:**
- [ ] La section "Quatre variantes" est remplacée par "Une variante Mistral" (LLMLangstral base + LongLLMLangstral via flag `use_context_level_filter=True`)
- [ ] La table d'orchestration retire les références à `LLMLingua-2`, `SecurityLingua`
- [ ] La section "Configuration des modèles Mistral" retire la constante `LLMLANGSTRAL2_MODEL` et l'avertissement associé
- [ ] Le tableau "Méthodes clés" retire la ligne `compress_prompt_llmlingua2()`
- [ ] La section "Pièges connus" est mise à jour : retrait des entrées obsolètes (llmlingua2.py non tracké, tokenization.py supprimé)
- [ ] Ajout d'une section "Capacité retirée" documentant la suppression de LLMLingua-2 (gap 3-6x performance) avec lien vers `legacy/experiments`

#### US-012: Mettre à jour `README.md`, `DOCUMENT.md`, `Transparency_FAQ.md`
**Description :** En tant qu'utilisateur découvrant le projet via le README, je veux que la documentation publique reflète clairement le positionnement Mistral-only.

**Priority:** P1
**Size:** M (3 pts)
**Dependencies:** Blocked by US-011

**Acceptance Criteria:**
- [ ] `README.md` : section "Variantes" retirée ou simplifiée à la variante Mistral unique
- [ ] `README.md` : ajout d'une section "Migration depuis v0.2.x" listant les breaking changes (4 kwargs `__init__`, 8 kwargs `compress_prompt`, suppressions de classes)
- [ ] `README.md` : ajout d'une note sur la limitation LLMLingua-2 (perte 3-6x vitesse, pas d'équivalent Mistral à date)
- [ ] `DOCUMENT.md` : retrait des examples `use_llmlingua2=True` (lignes 61, 68, 271, 281)
- [ ] `Transparency_FAQ.md` : retrait des références LLMLingua-2 (lignes 167, 174)
- [ ] `MIGRATION_PLAN_MISTRAL.md` : mise à jour pour refléter l'état post-refactor
- [ ] `setup.cfg` : `known_first_party = sdtools` corrigé en `llmlangstral`, retrait de `sklearn` de `known_third_party`

#### US-013: Archiver `experiments/llmlangstral2/` et `experiments/securitylingua/` sur branche `legacy/experiments`
**Description :** En tant que contributeur futur souhaitant entraîner un compressor XLM-RoBERTa ou un détecteur SecurityLingua, je veux pouvoir retrouver les pipelines d'entraînement sur une branche git dédiée afin de préserver la valeur historique du code sans encombrer le tronc principal.

**Priority:** P1
**Size:** S (2 pts)
**Dependencies:** None (peut être fait en parallèle de EP-001 à EP-004)

**Acceptance Criteria:**
- [ ] Une branche `legacy/experiments` est créée depuis le HEAD actuel avant suppression
- [ ] Sur la branche `main`, les répertoires `experiments/llmlangstral2/` et `experiments/securitylingua/` sont supprimés via `git rm -r`
- [ ] Le fichier `examples/LLMLangstral2.ipynb` est supprimé sur `main`
- [ ] Le fichier `README.md` mentionne la branche `legacy/experiments` comme lieu d'archive
- [ ] La branche `legacy/experiments` est poussée sur le remote (si remote disponible)

#### US-014: Bump version 0.2.2 → 0.3.0 + changelog + `git rm core/tokenization.py`
**Description :** En tant que mainteneur publiant la release, je veux bumper la version et écrire un changelog complet afin que les utilisateurs downstream comprennent les breaking changes.

**Priority:** P0
**Size:** XS (1 pt)
**Dependencies:** Blocked by EP-001, EP-002, EP-003, EP-004, US-011, US-012

**Acceptance Criteria:**
- [ ] `llmlangstral/version.py` : `VERSION = "0.3.0"`
- [ ] Un fichier `CHANGELOG.md` est créé (ou mis à jour s'il existe) avec une section `## [0.3.0] - 2026-XX-XX` détaillant : (a) Removed (LLMLingua-2, SecurityLingua, 9 rankers, 12 kwargs API), (b) Changed (signature `PromptCompressor.__init__`, signature `compress_prompt`), (c) Migration notes
- [ ] `git rm llmlangstral/core/tokenization.py` (déjà supprimé du disque, à stager si pas déjà fait)
- [ ] La date dans le changelog est la date effective du merge
- [ ] `git tag v0.3.0` est créé (optionnel — décision mainteneur, non bloquant pour la story)

#### US-015: Quality gate final — full test suite + smoke imports + dead-symbol grep
**Description :** En tant que mainteneur acceptant la PR de merge, je veux valider que toutes les suppressions sont effectives et que rien n'est cassé via un quality gate exhaustif.

**Priority:** P0
**Size:** S (2 pts)
**Dependencies:** Blocked by US-001 à US-014

**Acceptance Criteria:**
- [ ] `make style` passe (Black 88, isort, flake8 119)
- [ ] `make test` passe en intégralité sur Python 3.9, 3.10, 3.11 (matrice CI)
- [ ] Le grep guard suivant retourne zéro ligne : `grep -rE "use_llmlingua2|use_slingua|LLMLingua2Compressor|OpenAIRanker|VoyageAIRanker|CohereRanker|BGERanker|SentBertRanker|JinzaRanker|APIBasedRanker|BGEReranker|BGELLMEmbedderRanker|LLMLANGSTRAL2_MODEL|init_llmlingua2" llmlangstral/ tests/`
- [ ] `python -c "import llmlangstral; print(llmlangstral.__version__)"` retourne `0.3.0`
- [ ] `python -c "from llmlangstral.ranking import RankingRegistry; print(sorted(RankingRegistry._strategies.keys()))"` retourne exactement `['bm25', 'gzip', 'llmlingua', 'longllmlingua', 'mistral']`
- [ ] `python -m py_compile $(git ls-files 'llmlangstral/**/*.py' 'tests/**/*.py')` passe sans erreur
- [ ] Le test smoke `pytest tests/test_mistral.py -v` continue à passer (avec ses skips appropriés sur modèles fantômes)
- [ ] Unhappy path test : `python -c "from llmlangstral.filters import LLMLingua2Compressor"` lève `ImportError` (et non `ModuleNotFoundError` car le package `filters` existe encore)
- [ ] Stale egg-info : `rm -rf llmlangstral.egg-info && pip install -e ".[dev]"` régénère sans erreur

## Functional Requirements

- FR-01: Le système doit fournir une classe unique `PromptCompressor` dont le constructeur accepte uniquement les paramètres `model_name`, `device_map`, `model_config` (signature minimaliste)
- FR-02: Le système doit charger uniquement des modèles compatibles avec l'écosystème Mistral (Mistral-7B-v0.3, Ministral-3B, e5-mistral-7b-instruct, tiny-mistral pour CI)
- FR-03: Le système ne doit PAS importer, instancier, ou télécharger de modèles BERT, XLM-RoBERTa, BGE, SentBert, Jina, OpenAI, VoyageAI, ou Cohere lors d'un usage normal
- FR-04: Le système doit lever `ImportError` ou `TypeError` clair (pas de silent failure) si un consommateur tente d'utiliser un symbole supprimé
- FR-05: Le système doit conserver le protocole de tags structurés `<llmlingua, rate=X>`, `<llmlingua, compress=False>` (compat wire-format pour utilisateurs existants)
- FR-06: Le système doit conserver le pattern plugin `RankingRegistry.register(name)` pour permettre des extensions futures (mais sans exposer de rankers non-Mistral par défaut)
- FR-07: Le système doit publier la version `0.3.0` avec un changelog explicite documentant les breaking changes

## Non-Functional Requirements

- **Performance :** Pas de régression mesurable sur le chemin Mistral principal. `compress_prompt(["foo"], rate=0.5)` avec `tiny-mistral` doit s'exécuter en < 10 secondes sur CPU (baseline actuel de `tests/test_llmlangstral.py`).
- **Maintenance :** `prompt_compressor.py` doit passer sous la barre des 950 lignes (vs 1098 actuelles) ; suppression nette de >150 lignes attendue.
- **Compatibilité Python :** Support effectif de Python 3.9, 3.10, 3.11 (CI). Python 3.8 explicitement abandonné (déjà entamé via `from __future__ import annotations`).
- **Taille du package :** Réduction de la surface installable. Le module `ranking/` passe de 6 fichiers (380+ lignes) à 4 fichiers (~200 lignes). Le module `filters/` passe de 5 fichiers à 4.
- **Test coverage :** Maintenir 100% pass sur la suite tests existante après suppression de `test_llmlangstral2.py`. Couvrir explicitement les unhappy paths via assertions `pytest.raises(TypeError)` et `pytest.raises(ImportError)`.
- **Documentation :** Zéro référence runtime à LLMLingua-2 / XLM-RoBERTa dans `CLAUDE.md`, `README.md`, `DOCUMENT.md` après merge. Vérifiable par grep.
- **Migration time :** Un utilisateur sur 0.2.2 doit pouvoir migrer vers 0.3.0 en moins de 15 minutes en suivant le guide de migration du README (4 kwargs `__init__` + 8 kwargs `compress_prompt` à retirer, 0 nouveau concept à apprendre).

## Edge Cases & Error States

| # | Scenario | Trigger | Expected Behavior | User Message |
|---|----------|---------|-------------------|--------------|
| 1 | Import explicite d'un symbole supprimé | `from llmlangstral.filters import LLMLingua2Compressor` | `ImportError` immédiat | `ImportError: cannot import name 'LLMLingua2Compressor' from 'llmlangstral.filters'` |
| 2 | Kwarg supprimé passé à `__init__` | `PromptCompressor(use_llmlingua2=True)` | `TypeError` immédiat | `TypeError: __init__() got an unexpected keyword argument 'use_llmlingua2'` |
| 3 | Kwarg supprimé passé à `compress_prompt` | `compress_prompt(["foo"], force_tokens=["\n"])` | `TypeError` immédiat | `TypeError: compress_prompt() got an unexpected keyword argument 'force_tokens'` |
| 4 | Ranker supprimé demandé au registry | `RankingRegistry.get("openai")` | Lève `KeyError` ou retourne `None` (comportement existant à confirmer) | `KeyError: 'openai'` ou `None` selon implémentation |
| 5 | Import d'un module supprimé | `import llmlangstral.ranking.neural` | `ModuleNotFoundError` | `ModuleNotFoundError: No module named 'llmlangstral.ranking.neural'` |
| 6 | Modèle XLM-RoBERTa passé à `PromptCompressor` | `PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")` | Le chargement HuggingFace tente le download, échoue avec un message HF ou charge en mode token-classification ; non bloquant — comportement = celui de `AutoModelForCausalLM.from_pretrained` sur un modèle non-causal | Message HF natif (pas notre responsabilité) |
| 7 | Migration partielle (consumer pin 0.2.x mais réimporte 0.3.0) | `pip install llmlangstral` sans pin → 0.3.0 récupéré | Le code utilisant `use_llmlingua2` casse avec `TypeError` au runtime | Le changelog 0.3.0 doit signaler ce risque ; consumer doit pin explicitement |
| 8 | Stale `egg-info` post-pull | `git pull` puis `import llmlangstral` sans réinstaller | Peut référencer des fichiers supprimés → `ImportError` selon Python | README/CHANGELOG documente `rm -rf llmlangstral.egg-info && pip install -e ".[dev]"` |
| 9 | Test CI cache HuggingFace : ancien modèle XLM-RoBERTa téléchargé reste dans `~/.cache/huggingface` | Cache local non purgé | Aucun impact runtime (le code ne le charge plus) | Documenté dans le CHANGELOG comme observation |
| 10 | Branche `legacy/experiments` perdue (non poussée sur remote) | Mainteneur supprime la branche locale par erreur | Récupérable via `git reflog` pendant 90 jours | Documenter dans le README que la branche doit être poussée immédiatement |

## Risks & Mitigations

| # | Risk | Probability | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | Capability gap LLMLingua-2 (3-6x plus rapide) jamais comblé | High | High | Documentation explicite dans README ; branche `legacy/experiments` préservée avec training pipelines ; ouverture future possible vers un Mistral fine-tuné pour token classification |
| 2 | Utilisateurs downstream non avertis cassent silencieusement | Medium | Medium | Changelog explicite v0.3.0 ; ImportError/TypeError clairs (pas silent) ; README migration guide ; recommandation pin `llmlangstral>=0.3.0` ou `llmlangstral<0.3.0` |
| 3 | Suppression partielle laisse du code mort (helpers utils.py, imports stale) | Low | Medium | Quality gate US-015 avec grep guard automatisé ; coverage run avant/après pour repérer le code orphelin |
| 4 | `tiktoken` reste comme dette technique (provider non-Mistral) | Medium | Low | Documenté comme follow-up P2 (story US-... pour v0.4.0) ; impact actuel = token counting uniquement, pas runtime |
| 5 | `examples/*.ipynb` continuent d'importer `openai` (downstream LLM, pas ranker) | Low | Low | Audit confirmé : tous les imports `openai` dans `examples/` sont pour l'LLM downstream, pas pour `OpenAIRanker`. Aucune action requise. |
| 6 | Tests `tests/test_longllmlangstral.py` utilisent encore des tags `<llmlingua, ...>` | Low | Low | Audit confirmé : tags conservés (compat wire-format intentionnelle). Tests passent inchangés. |
| 7 | `setup.py` Development Status reste "3 - Alpha" alors qu'on bump 0.3.0 | Low | Low | Considérer passage à "4 - Beta" en US-014 (décision mainteneur, non bloquant) |
| 8 | Branche `legacy/experiments` collecte de la rouille et devient obsolète | Medium | Low | Acceptable — c'est l'objectif d'une branche d'archive. Documenter dans README qu'elle est en lecture seule. |

## Non-Goals

Explicit boundaries — what this version does NOT include:

- **NE PAS implémenter un compressor Mistral-fast équivalent à LLMLingua-2.** Le fine-tuning d'un Mistral encoder pour token-classification dépasse le scope de ce refactor — c'est un travail de R&D distinct, candidat pour v0.4.x ou v0.5.x.
- **NE PAS introduire de nouveau provider ML dans `ranking/`.** Le pattern plugin reste, mais seuls des rankers Mistral-compatibles seront acceptés en v0.3.x.
- **NE PAS migrer `tiktoken` vers le tokenizer Mistral.** La compatibilité numérique du `origin_tokens` / `compressed_tokens` retournés est conservée pour ce release. Migration possible en v0.4.0.
- **NE PAS renommer le wire-format tag `<llmlingua, ...>` en `<llmlangstral, ...>`.** C'est un protocole d'entrée utilisateur ; cohérence avec le pattern fork-compat (Neovim/MariaDB). Décision recherche-fondée.
- **NE PAS retirer `BM25Ranker` et `GzipRanker`** (algorithmes statistiques sans dépendance modèle). Ils sont orthogonaux au positionnement Mistral et utiles comme baselines.
- **NE PAS retirer la classe abstraite `RankingRegistry`** ni le pattern plugin. Utile pour extensions futures, sans coût.
- **NE PAS ouvrir de DeprecationWarning pendant une fenêtre de transition.** Research-fondée (Larson 2024) : ne marche pas pour les libs Python. Hard removal autorisé par SemVer 0.x.
- **NE PAS toucher au code `core/`, `filters/context.py`, `filters/sentence.py`, `filters/token.py`** au-delà du strict nécessaire (suppression d'imports orphelins). Le refactor modulaire v0.3.0 est conservé tel quel.

## Files NOT to Modify

- `llmlangstral/core/base.py` — ABC `BaseCompressor` et `CompressionResult` ; utilisés par tous les chemins conservés
- `llmlangstral/core/model_loader.py` (sauf `init_llmlingua2()` et ses 5 properties — US-003) — reste l'entrée canonique pour charger Mistral
- `llmlangstral/filters/base.py` — `FilterContext`, `FilterBase` ABC ; orthogonaux au refactor
- `llmlangstral/filters/context.py`, `sentence.py`, `token.py` — chemin Mistral principal, intacts
- `llmlangstral/ranking/registry.py` — pattern plugin générique
- `llmlangstral/ranking/base.py` (sauf `APIBasedRanker` — US-007) — ABC partagées par les rankers conservés
- `llmlangstral/ranking/statistical.py` (BM25Ranker, GzipRanker) — algorithmes sans dépendance modèle
- `llmlangstral/ranking/llmlingua.py` (LLMLinguaRanker pour `llmlingua`/`longllmlingua`) — utilise le backbone Mistral via PPL
- `llmlangstral/ranking/mistral.py` (MistralRanker) — coeur du positionnement
- `tests/test_llmlangstral.py`, `tests/test_longllmlangstral.py`, `tests/test_mistral.py` — conservés (audit confirme : aucun usage de symbole supprimé sauf params LLMLingua-2-only éventuels)
- `LICENSE`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md` — fichiers de gouvernance
- `.github/workflows/unittest.yml` — CI matrix conservée

## Technical Considerations

Frame as questions for engineering input — not mandates :

- **Architecture :** Le pattern `RankingRegistry` reste-t-il utile à 5 rankers (vs 13) ? Recommandé : **oui** — coût zéro, ouverture future. Engineering à confirmer.
- **API Design :** Faut-il préserver `compress_prompt_llmlingua2` comme alias raisant `NotImplementedError` pendant une version ? Recommandé : **non** — research montre que les avertissements/aliases discrets ne sont pas remarqués. Préférer hard removal + changelog. Engineering à valider.
- **Dependencies :** `tiktoken` reste dans `INSTALL_REQUIRES`. Migration vers tokenizer Mistral = follow-up story en v0.4.0 ? Trade-off : parité numérique du `origin_tokens` retourné vs cohérence provider. Recommandé : reporter à v0.4.0.
- **Migration :** Faut-il un module compat `llmlangstral.compat` avec des classes-stubs (`LLMLingua2Compressor` raising NotImplementedError) ? Recommandé : **non** (cf. ci-dessus), mais sondage utilisateurs possible.
- **Test fixtures :** Les golden strings dans `test_llmlangstral.py` et `test_longllmlangstral.py` resteront-elles cohérentes après ce refactor ? Audit confirme oui (chemin Mistral inchangé). Engineering à re-confirmer post-merge.
- **Rollback plan :** En cas de régression critique post-0.3.0, la branche `legacy/experiments` ne contient pas le code livré v0.2.2. Recommandé : taguer `v0.2.2` avant le merge v0.3.0 pour permettre un rollback via `pip install llmlangstral==0.2.2`.

## Success Metrics

| Metric | Baseline (current) | Target | Timeframe | How Measured |
|--------|-------------------|--------|-----------|-------------|
| Symboles non-Mistral runtime dans `llmlangstral/` | ~25 (LLMLingua2Compressor, BGE, SentBert, etc.) | 0 | Post-US-015 (Month 0) | `grep -rE "<dead-symbols>" llmlangstral/ \| wc -l` |
| Nombre de rankers enregistrés | 13 | 5 | Post-US-015 | `python -c "from llmlangstral.ranking import RankingRegistry; print(len(RankingRegistry._strategies))"` |
| Nombre de kwargs publics de `compress_prompt` | 32 | 24 (8 retirés) | Post-US-009 | `inspect.signature(PromptCompressor.compress_prompt).parameters` |
| Taille de `prompt_compressor.py` | 1098 lignes | < 950 lignes | Post-US-015 | `wc -l llmlangstral/prompt_compressor.py` |
| Taille totale `llmlangstral/` (.py uniquement) | ~3500 lignes | < 2800 lignes | Post-US-015 | `find llmlangstral -name '*.py' -exec wc -l {} + \| tail -1` |
| Pass rate des tests sur Python 3.9/3.10/3.11 | 100% | 100% | Post-US-015 | CI `.github/workflows/unittest.yml` |
| Temps d'exécution `make test` (chemin tiny-mistral) | ~120s (estimé) | ≤ 120s ou ≤ 60s (sans llmlingua2 download 1.1 GB) | Post-US-015 | `time make test` |
| Issues utilisateurs liées à breaking changes | N/A (release future) | < 5 ouvertes dans les 30 jours post-merge | Month-1 | GitHub issues label `migration-help` |
| Adoption v0.3.0 dans PyPI downloads | N/A | > 50% des downloads en 30 jours | Month-1 | PyPI stats |

## Open Questions

- **OQ-1 :** Le mainteneur (Arthur Jean) confirme-t-il que la suppression de LLMLingua-2 sans replacement Mistral-fast est acceptable ? *Status : confirmé via conversation 2026-05-21 — résolu.*
- **OQ-2 :** Faut-il créer une issue GitHub dédiée pour tracker le développement d'un Mistral-encoder pour token classification (replacement LLMLingua-2) ? *À décider par le mainteneur, recommandé.*
- **OQ-3 :** La branche `legacy/experiments` doit-elle être protégée (no-force-push, no-delete) sur GitHub ? *À décider par le mainteneur. Recommandé : oui (read-only).*
- **OQ-4 :** Faut-il taguer `v0.2.2` avant le merge v0.3.0 pour rollback facile ? *Recommandé oui (one-liner). À exécuter par le mainteneur.*
- **OQ-5 :** Faut-il purger le cache CI HuggingFace après le merge pour éviter que les workflows téléchargent encore `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` ? *À vérifier dans `.github/workflows/unittest.yml` — pas bloquant.*
- **OQ-6 :** Le rename interne de `LLMLinguaRanker` → `MistralLinguaRanker` est-il souhaité ? Le research suggère oui (Could Have), mais introduit un nouveau breaking change pour les utilisateurs custom. *À décider en v0.4.0, pas bloquant pour v0.3.0.*
- **OQ-7 :** `tests/test_llmlangstral.py` et `tests/test_longllmlangstral.py` doivent-ils être renommés (retrait du préfixe `test_ll` pour cohérence) ? *Non bloquant, refactor cosmétique.*
[/PRD]
