# Plan de Refactoring Modulaire — LLMLangstral

> **Version cible** : 0.3.0 (mono-Mistral)
> **Statut** : ✅ DONE — phases 1 à 3 livrées dans le commit `b860a0b`. Refactor mono-Mistral livré dans `f5cbb0e`.
> **Dernière mise à jour** : 2026-05-21

---

## Contexte

Avant le refactor, `llmlangstral/prompt_compressor.py` (~2489 lignes) était un "God class" avec 8+ responsabilités. La v0.3.0 a éclaté ce fichier en sous-modules `core/`, `filters/`, `ranking/` et l'a réduit à **942 lignes**. Le même refactor supprime LLMLingua-2 / SecurityLingua et tous les rankers non-Mistral pour aligner le projet sur l'objectif "mono-Mistral".

---

## Objectifs (atteints)

1. **Séparation des responsabilités** — 1 module = 1 responsabilité. ✅
2. **Façade unique** — `PromptCompressor` reste le seul point d'entrée public. ✅
3. **Extensibilité ranking** — plugin registry via `@RankingRegistry.register("nom")`. ✅
4. **Testabilité** — chaque ranker/filtre testable isolément. ✅
5. **Mono-Mistral** — aucune dépendance encoder, aucun ranker non-Mistral. ✅

---

## Architecture livrée

```
llmlangstral/
├── __init__.py            # Exporte PromptCompressor, MISTRAL_MODELS, DEFAULT_MODEL
├── prompt_compressor.py   # Façade (942 lignes) — orchestre core/filters/ranking
├── mistral_config.py      # MISTRAL_MODELS + DEFAULT_MODEL
├── utils.py               # seed_everything + helpers JSON/segments
├── version.py             # VERSION = "0.3.0"
│
├── core/
│   ├── base.py            # CompressionResult (dataclass) + BaseCompressor (ABC)
│   └── model_loader.py    # ModelManager — lazy-load HF causal-LM
│
├── filters/
│   ├── base.py            # FilterContext + FilterBase (ABC)
│   ├── context.py         # ContextLevelFilter
│   ├── sentence.py        # SentenceLevelFilter
│   └── token.py           # TokenLevelFilter (PPL token pruning + KV rolling)
│
└── ranking/
    ├── registry.py        # RankingRegistry — décorateur @register("nom")
    ├── base.py            # RankingStrategy, ModelBasedRanker, PPLBasedRanker
    ├── statistical.py     # BM25Ranker, GzipRanker
    ├── llmlingua.py       # LLMLinguaRanker (alias "llmlingua" + "longllmlingua")
    └── mistral.py         # MistralRanker (intfloat/e5-mistral-7b-instruct)
```

Le registre contient exactement 5 clés : `bm25`, `gzip`, `llmlingua`, `longllmlingua`, `mistral`.

---

## Phases

### Phase 1 — Module `ranking/` ✅ DONE (`a7df588`)

Extraction des stratégies de ranking en module plugin (`RankingRegistry`). Seuls les rankers Mistral-compatibles sont conservés : `bm25`, `gzip`, `llmlingua`, `longllmlingua`, `mistral`. Les rankers neuraux non-Mistral (BGE/SentBert/Jinza) et API (OpenAI/Cohere/VoyageAI) ont été supprimés en US-009/US-010.

### Phase 2 — Module `core/` ✅ DONE (`9d91f25`)

Création de `BaseCompressor` (ABC), `CompressionResult` (dataclass) et `ModelManager` (lazy-load HF). `PromptCompressor` hérite désormais de `BaseCompressor` et expose `model`/`tokenizer`/`device` via `@property` qui délèguent au manager.

### Phase 3 — Module `filters/` ✅ DONE (`b860a0b`)

Extraction de `control_context_budget` / `control_sentence_budget` / `iterative_compress_prompt` vers `filters/{context,sentence,token}.py`. Injection de dépendances via `FilterContext` portant les callbacks `get_ppl_fn`, `get_condition_ppl_fn`, `get_rank_results_fn`.

### Phase 4 — Refactor mono-Mistral ✅ DONE (`f5cbb0e`)

Suppression de :

- LLMLingua-2 (XLM-RoBERTa token classification path)
- SecurityLingua
- Rankers non-Mistral : BGERanker, BGEReranker, BGELLMEmbedderRanker, SentBertRanker, JinzaRanker, OpenAIRanker, CohereRanker, VoyageAIRanker
- Kwargs publics liés : `use_llmlingua2`, `use_slingua`, `open_api_config`, `llmlingua2_config` (constructeur), et 8 kwargs de `compress_prompt` (`return_word_label`, `word_sep`, `label_sep`, `token_to_word`, `force_tokens`, `force_reserve_digit`, `drop_consecutive`, `chunk_end_tokens`)

Archive sur branche `legacy/experiments`. Réversibilité via tag `v0.2.2` ou `pip install llmlangstral==0.2.2`.

### Phase 5 — API transformers 5.x ✅ DONE (2026-05-21)

`DynamicCache.from_legacy_cache` / `to_legacy_cache` ont été supprimés en transformers 5.x. `get_ppl` round-trippe désormais entre format legacy `[[k, v], ...]` (utilisé par `TokenLevelFilter` pour le KV-cache rolling) et `DynamicCache(config=model.config)` à la frontière du forward pass. Import mis à jour : `from transformers import DynamicCache` (au lieu de `transformers.cache_utils`).

`get_condition_ppl` lève désormais `ValueError` sur valeur inconnue de `condition_in_question` (au lieu de retourner silencieusement `None`).

---

## Phases non livrées (non bloquantes)

Les phases suivantes du plan initial ne sont **pas** prévues pour l'instant — la façade actuelle suffit à la lisibilité et l'extension reste possible via les modules `filters/` et `ranking/`.

- **Phase 6 (compression/)** — séparer `compress_prompt`, `structured_compress_prompt`, `compress_json` en classes distinctes. Pas urgent ; le fichier façade fait 942 lignes, en deçà du seuil "God class".
- **Phase 7 (budget/ et recovery/)** — extraire `get_dynamic_compression_ratio`, `get_structured_dynamic_compression_ratio`, `recover()`. Idem, gain marginal.

Si le fichier façade repasse au-dessus de ~1200 lignes (par exemple en ajoutant un mode de compression majeur), reprendre l'extraction Phase 6.

---

## Critères de succès — état réel

- [x] Quality gate v0.3.0 : `grep -rE "use_llmlingua2|use_slingua|LLMLingua2Compressor|OpenAIRanker|VoyageAIRanker|CohereRanker|BGERanker|SentBertRanker|JinzaRanker|APIBasedRanker|BGEReranker|BGELLMEmbedderRanker|LLMLANGSTRAL2_MODEL|init_llmlingua2" llmlangstral/ tests/` retourne **zéro** résultat.
- [x] `PromptCompressor.__init__` n'accepte que `model_name`, `device_map`, `model_config`.
- [x] `RankingRegistry` contient exactement 5 clés : `bm25`, `gzip`, `llmlingua`, `longllmlingua`, `mistral`.
- [x] `PromptCompressor` hérite de `BaseCompressor` et expose `compress()` retournant `CompressionResult`.
- [x] Tests `test_llmlangstral.py` et `test_longllmlangstral.py` passent sur `tiny-mistral` (CPU).
- [ ] Couverture de tests > 80 % — pas mesurée.
- [ ] Benchmark perf avant/après — non réalisé (fork sans utilisateur en production aujourd'hui).

---

## Pour aller plus loin

- Reprendre Phase 6 dès que `prompt_compressor.py` dépasse ~1200 lignes.
- Ajouter un mode `flash_attention_2` pour Mistral-7B-v0.3.
- Activer les vrais modèles `Ministral-3-3B-Instruct-2512` / `Ministral-3-8B-Instruct-2512` / `Mistral-Large-3` dès qu'ils sont publiés sur HF Hub (actuellement skippés dans `test_mistral.py`).
