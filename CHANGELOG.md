# Changelog

All notable changes to LLMLangstral are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.x = unstable, breaking changes allowed in MINOR bumps).

## [0.3.0] - 2026-05-21

Refactor "Full-Mistral" — LLMLangstral devient un fork **100% Mistral**. Release breaking change suite à l'audit du 2026-05-21. Tous les chemins non-Mistral (LLMLingua-2 XLM-RoBERTa, SecurityLingua, rankers BGE / SentBert / Jinza / OpenAI / VoyageAI / Cohere) ont été retirés du tronc principal et archivés sur la branche `legacy/experiments`.

### Removed

**Variantes de compression** :

- `LLMLingua2Compressor` (`llmlangstral/filters/llmlingua2.py`) — chemin XLM-RoBERTa token classification, 3-6x plus rapide que le chemin par perplexité. Aucun équivalent Mistral n'existe à ce jour.
- Le tests file `tests/test_llmlangstral2.py` (téléchargeait ~1.1 GB de poids XLM-RoBERTa à chaque run CI).
- La constante `LLMLANGSTRAL2_MODEL` de `llmlangstral/mistral_config.py`.
- La méthode `ModelManager.init_llmlingua2()` et les 5 attributs LLMLingua-2-only (`max_batch_size`, `max_seq_len`, `max_force_token`, `special_tokens`, `added_tokens`).
- Les 5 properties de délégation correspondantes sur `PromptCompressor`.
- La méthode `PromptCompressor.compress_prompt_llmlingua2()` et la property `_llm2`.

**Rankers non-Mistral (9 supprimés)** :

- `SentBertRanker` (`multi-qa-mpnet-base-dot-v1`)
- `BGERanker` (`BAAI/bge-large-en-v1.5`)
- `BGEReranker` (`BAAI/bge-reranker-large`)
- `BGELLMEmbedderRanker` (`BAAI/llm-embedder`)
- `JinzaRanker` (`jinaai/jina-embeddings-v2-base-en`)
- `OpenAIRanker` (API OpenAI Embeddings)
- `VoyageAIRanker` (API VoyageAI)
- `CohereRanker` (API Cohere Rerank)
- `APIBasedRanker` (ABC orpheline après suppression des 3 rankers API)

Le registre `RankingRegistry._strategies` contient désormais exactement **5 clés** : `bm25`, `gzip`, `llmlingua`, `longllmlingua`, `mistral`.

**Helpers utilitaires LLMLingua-2-only (`utils.py`)** :

- Classe `TokenClfDataset` (~65 lignes)
- Fonction `is_begin_of_new_word`
- Fonction `replace_added_token`
- Fonction `get_pure_token`
- Imports devenus inutilisés : `import string`, `from torch.utils.data import Dataset`

**Module orphelin** :

- `llmlangstral/core/tokenization.py` (déjà supprimé du disque dans une révision précédente, maintenant aussi stagé git).

### Changed

**`PromptCompressor.__init__(...)` perd 4 kwargs** :

- `open_api_config`
- `use_llmlingua2`
- `use_slingua`
- `llmlingua2_config`

Signature post-refactor :

```python
PromptCompressor(
    model_name: str = "mistralai/Mistral-7B-v0.3",
    device_map: str = "cuda",
    model_config: dict = {},
)
```

**`PromptCompressor.compress_prompt(...)` perd 8 kwargs** :

- `return_word_label`
- `word_sep`
- `label_sep`
- `token_to_word`
- `force_tokens`
- `force_reserve_digit`
- `drop_consecutive`
- `chunk_end_tokens`

Le `Returns` dict ne contient plus `compressed_prompt_list` ni `fn_labeled_original_prompt` (clés LLMLingua-2-only).

**Surface API publique** :

- `from llmlangstral import PromptCompressor` ne charge plus aucun modèle XLM-RoBERTa au runtime.
- `from llmlangstral.ranking import APIBasedRanker` → `ImportError`.
- `from llmlangstral.utils import TokenClfDataset` → `ImportError`.
- `PromptCompressor(use_llmlingua2=True)` → `TypeError`.

**Documentation** :

- `CLAUDE.md`, `README.md`, `DOCUMENT.md`, `Transparency_FAQ.md` réécrits pour ne référencer que le chemin Mistral.
- `MIGRATION_PLAN_MISTRAL.md` complété d'un préambule v0.3.0 documentant les décisions exécutées vs rejetées.
- `setup.cfg` : `known_first_party = sdtools` corrigé en `llmlangstral` ; `sklearn` retiré de `known_third_party`.

### Preserved (intentionnel)

- **Wire-format tags structurés** `<llmlingua, rate=X>`, `<llmlingua, compress=False>` conservés — protocole d'entrée utilisateur, pattern fork-compat (Neovim/MariaDB).
- **Pattern plugin `RankingRegistry`** maintenu pour extensions futures (mais seuls rankers Mistral-compatibles acceptés en v0.3.x).
- **`tiktoken` (gpt-3.5-turbo)** comme tokenizer pour `origin_tokens` — parité numérique avec v0.2.x. Migration vers tokenizer Mistral candidate pour v0.4.0.
- **`seed_everything` dans `utils.py`** conservé (utilité orthogonale au refactor).

### Migration depuis v0.2.x

Suivez le guide dans `README.md` section "Migration depuis v0.2.x". Étapes essentielles :

1. Retirer les 4 kwargs de `PromptCompressor(...)` et les 8 kwargs de `compress_prompt(...)` listés ci-dessus.
2. Si vous importiez `LLMLingua2Compressor`, `LLMLANGSTRAL2_MODEL`, `APIBasedRanker`, `TokenClfDataset`, `OpenAIRanker` (ou tout autre symbole supprimé) : `ImportError` au runtime — remplacer par un chemin Mistral.
3. Si vous utilisiez le chemin "compression rapide" LLMLingua-2 : aucun équivalent Mistral à ce jour. Restez sur 0.2.x (`pip install "llmlangstral<0.3.0"`) ou cherchez le pipeline d'entraînement sur la branche `legacy/experiments`.
4. Post-pull editable install : `rm -rf llmlangstral.egg-info && pip install -e ".[dev]"` pour régénérer le metadata.

### Archive

La branche `legacy/experiments` (poussée sur `origin`) préserve les pipelines d'entraînement supprimés :

- `experiments/llmlangstral2/` — data collection (GPT-4 distillation), model training (RoBERTa & Mistral), evaluation (BBH, GSM8K, LongBench, MeetingBank, ZeroSCROLLS)
- `experiments/securitylingua/` — entraînement détecteur jailbreak
- `examples/LLMLangstral2.ipynb` — notebook démo LLMLingua-2

Recommandation : protéger la branche en lecture-seule sur GitHub (no force-push, no-delete).

### Quality gates

- `make style` (Black 88 + isort + flake8 119) : passe sur tout `llmlangstral/` et `tests/`.
- Registre des rankers : exactement `['bm25', 'gzip', 'llmlingua', 'longllmlingua', 'mistral']`.
- Test suite : 3 fichiers (`test_llmlangstral.py`, `test_longllmlangstral.py`, `test_mistral.py`), 13 tests collectés.
- `prompt_compressor.py` : 924 lignes (vs 2222 dans `b860a0b`, vs 1098 dans le working tree pré-US-001).

---

## [0.2.2] - 2026-02-18

Dernière release de la branche v0.2.x. État pré-refactor full-Mistral. Reste disponible via `pip install "llmlangstral<0.3.0"` pour les consumers qui dépendent du chemin LLMLingua-2.
