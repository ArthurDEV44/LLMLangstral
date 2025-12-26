# Plan de Refactoring Modulaire - LLMLangstral

> **Date** : 2025-12-26
> **Version cible** : 0.3.0
> **Statut** : En cours de planification

---

## Contexte

Le fichier `llmlangstral/prompt_compressor.py` (~2489 lignes) est devenu un "God Class" avec 8+ responsabilités distinctes. Cette architecture monolithique impacte :

- **Maintenabilité** : Modifications risquées, effets de bord fréquents
- **Testabilité** : Impossible de tester les composants isolément
- **Extensibilité** : Ajouter un algorithme de ranking = modifier la classe principale
- **Onboarding** : Courbe d'apprentissage élevée pour les contributeurs

---

## Objectifs

1. **Séparation des responsabilités** : 1 module = 1 responsabilité
2. **Backward compatibility** : API publique `PromptCompressor` inchangée
3. **Extensibilité** : Plugin pattern pour les algorithmes de ranking
4. **Testabilité** : Couverture > 80% avec tests unitaires isolés
5. **Performance** : Aucune régression (même vitesse, même consommation mémoire)

---

## Architecture Cible

```
llmlangstral/
├── __init__.py                    # API publique (Façade)
├── version.py                     # (existant)
├── mistral_config.py              # (existant)
├── utils.py                       # (existant, à enrichir)
│
├── core/
│   ├── __init__.py
│   ├── base.py                    # BaseCompressor (ABC), CompressionResult
│   ├── model_loader.py            # ModelManager
│   └── tokenization.py            # TokenizationMixin, PPL computation
│
├── compression/
│   ├── __init__.py
│   ├── llmlingua.py               # LLMLinguaCompressor
│   ├── llmlingua2.py              # LLMLingua2Compressor
│   ├── longllmlingua.py           # LongLLMLinguaCompressor
│   ├── structured.py              # StructuredCompressor (XML tags)
│   └── json_compressor.py         # JSONCompressor
│
├── ranking/
│   ├── __init__.py                # RankingRegistry export
│   ├── base.py                    # RankingStrategy (ABC)
│   ├── registry.py                # RankingRegistry (plugin pattern)
│   ├── statistical.py             # BM25Ranker, GzipRanker
│   ├── neural.py                  # SentBertRanker, BGERanker, JinzaRanker
│   ├── api_based.py               # OpenAIRanker, CohereRanker, VoyageRanker
│   ├── llmlingua.py               # LLMLinguaRanker, LongLLMLinguaRanker
│   └── mistral.py                 # MistralEmbeddingRanker
│
├── filters/
│   ├── __init__.py
│   ├── base.py                    # BaseFilter (ABC)
│   ├── context_filter.py          # ContextLevelFilter
│   ├── sentence_filter.py         # SentenceLevelFilter
│   └── token_filter.py            # TokenLevelFilter
│
├── budget/
│   ├── __init__.py
│   ├── context_budget.py          # ContextBudgetController
│   ├── sentence_budget.py         # SentenceBudgetController
│   └── dynamic_ratio.py           # DynamicCompressionRatioCalculator
│
└── recovery/
    ├── __init__.py
    └── text_recovery.py           # TextRecoveryEngine
```

---

## Phases de Migration

### Phase 1 : Extraction du module `ranking/` (Priorité: HAUTE) ✅ COMPLÉTÉ

**Objectif** : Extraire les 13 algorithmes de ranking en module séparé avec Strategy Pattern.

**Fichiers créés** :
- [x] `llmlangstral/ranking/__init__.py`
- [x] `llmlangstral/ranking/base.py`
- [x] `llmlangstral/ranking/registry.py`
- [x] `llmlangstral/ranking/statistical.py` (bm25, gzip)
- [x] `llmlangstral/ranking/neural.py` (sentbert, bge, bge_reranker, bge_llmembedder, jinza)
- [x] `llmlangstral/ranking/api_based.py` (openai, cohere, voyageai)
- [x] `llmlangstral/ranking/llmlingua.py` (llmlingua, longllmlingua)
- [x] `llmlangstral/ranking/mistral.py` (mistral embedding)

**Code cible** :

```python
# ranking/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple

class RankingStrategy(ABC):
    """Base class for all ranking algorithms."""

    name: str = "base"
    requires_model: bool = False

    @abstractmethod
    def rank(
        self,
        corpus: List[str],
        query: str,
        **kwargs
    ) -> List[Tuple[int, float]]:
        """
        Rank corpus documents by relevance to query.

        Returns:
            List of (index, score) tuples sorted by relevance.
        """
        pass


# ranking/registry.py
class RankingRegistry:
    """Plugin registry for ranking strategies."""

    _strategies: dict[str, type[RankingStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a ranking strategy."""
        def decorator(strategy_cls: type[RankingStrategy]):
            cls._strategies[name] = strategy_cls
            strategy_cls.name = name
            return strategy_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> RankingStrategy:
        """Get an instance of a registered strategy."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown ranking strategy: {name}")
        return cls._strategies[name](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered strategies."""
        return list(cls._strategies.keys())
```

**Tests** :
- [x] Tests de non-régression passent (10/10)
- [ ] `tests/ranking/test_registry.py` (optionnel, à ajouter)
- [ ] `tests/ranking/test_statistical.py` (optionnel, à ajouter)

**Résultats** :
- `prompt_compressor.py` réduit de **~270 lignes** (2489 → 2219)
- 8 fichiers créés dans `llmlangstral/ranking/`
- Backward compatibility maintenue

**Effort réel** : 1 jour
**Risque** : Faible (extraction pure, pas de modification de logique)

---

### Phase 2 : Extraction du module `core/` (Priorité: HAUTE) ✅ COMPLÉTÉ

**Objectif** : Centraliser le chargement de modèles et la gestion des tokens.

**Fichiers créés** :
- [x] `llmlangstral/core/__init__.py`
- [x] `llmlangstral/core/base.py` (BaseCompressor, CompressionResult)
- [x] `llmlangstral/core/model_loader.py` (ModelManager)
- [x] `llmlangstral/core/tokenization.py` (TokenizationMixin)

**Code cible** :

```python
# core/base.py
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Union

@dataclass
class CompressionResult:
    """Result of a compression operation."""
    compressed_prompt: str
    origin_tokens: int
    compressed_tokens: int
    ratio: str
    rate: str
    saving: str

    # Optional fields for specific compressors
    compressed_prompt_list: List[str] = None
    fn_labeled_original_prompt: str = None


class BaseCompressor(ABC):
    """Abstract base class for all compressors."""

    @abstractmethod
    def compress(
        self,
        context: Union[str, List[str]],
        **kwargs
    ) -> CompressionResult:
        """Compress the given context."""
        pass


# core/model_loader.py
class ModelManager:
    """Centralized model loading and caching."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "cuda",
        model_config: dict = None,
    ):
        self.model_name = model_name
        self.device_map = device_map
        self.model_config = model_config or {}

        self._model = None
        self._tokenizer = None
        self._config = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """Lazy loading of model and tokenizer."""
        # ... extraction de load_model() actuel
        pass
```

**Résultats** :
- 4 fichiers créés dans `llmlangstral/core/`
- `prompt_compressor.py` utilise délégation vers `ModelManager`
- Propriétés de backward compatibility ajoutées
- Syntaxe validée (py_compile OK)

**Effort réel** : 1 jour
**Risque** : Faible

---

### Phase 3 : Extraction du module `filters/` (Priorité: MOYENNE)

**Objectif** : Séparer les 3 niveaux de filtrage en classes distinctes.

**Fichiers à créer** :
- [ ] `llmlangstral/filters/__init__.py`
- [ ] `llmlangstral/filters/base.py`
- [ ] `llmlangstral/filters/context_filter.py`
- [ ] `llmlangstral/filters/sentence_filter.py`
- [ ] `llmlangstral/filters/token_filter.py`

**Méthodes à extraire de `prompt_compressor.py`** :
- `control_context_budget()` → `ContextLevelFilter`
- `control_sentence_budget()` → `SentenceLevelFilter`
- `iterative_compress_prompt()` → `TokenLevelFilter`
- `get_compressed_input()` → `TokenLevelFilter`

**Effort estimé** : 2-3 jours
**Risque** : Moyen (logique complexe, beaucoup d'interdépendances)

---

### Phase 4 : Extraction du module `compression/` (Priorité: MOYENNE)

**Objectif** : Séparer les variantes de compression en classes distinctes.

**Fichiers à créer** :
- [ ] `llmlangstral/compression/__init__.py`
- [ ] `llmlangstral/compression/llmlingua.py`
- [ ] `llmlangstral/compression/llmlingua2.py`
- [ ] `llmlangstral/compression/structured.py`
- [ ] `llmlangstral/compression/json_compressor.py`

**Méthodes à extraire** :
- `compress_prompt()` (partie LLMLingua) → `LLMLinguaCompressor`
- `compress_prompt_llmlingua2()` → `LLMLingua2Compressor`
- `structured_compress_prompt()` → `StructuredCompressor`
- `compress_json()` → `JSONCompressor`

**Effort estimé** : 3-4 jours
**Risque** : Élevé (cœur de la logique métier)

---

### Phase 5 : Extraction des modules `budget/` et `recovery/` (Priorité: BASSE)

**Fichiers à créer** :
- [ ] `llmlangstral/budget/__init__.py`
- [ ] `llmlangstral/budget/dynamic_ratio.py`
- [ ] `llmlangstral/recovery/__init__.py`
- [ ] `llmlangstral/recovery/text_recovery.py`

**Méthodes à extraire** :
- `get_dynamic_compression_ratio()` → `DynamicRatioCalculator`
- `get_structured_dynamic_compression_ratio()` → `DynamicRatioCalculator`
- `recover()` → `TextRecoveryEngine`

**Effort estimé** : 1-2 jours
**Risque** : Faible

---

### Phase 6 : Façade et Backward Compatibility (Priorité: CRITIQUE)

**Objectif** : Maintenir l'API publique `PromptCompressor` identique.

**Modifications** :
- [ ] Refactorer `__init__.py` pour exposer la nouvelle architecture
- [ ] `PromptCompressor` devient une Façade qui délègue aux modules

**Code cible** :

```python
# __init__.py
from .core.model_loader import ModelManager
from .core.base import CompressionResult
from .compression import LLMLinguaCompressor, LLMLingua2Compressor
from .ranking import RankingRegistry
from .mistral_config import DEFAULT_MODEL, MISTRAL_MODELS

class PromptCompressor:
    """
    Unified API for prompt compression.

    This class provides backward compatibility with the original API
    while delegating to the new modular architecture internally.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_map: str = "cuda",
        model_config: dict = None,
        open_api_config: dict = None,
        use_llmlingua2: bool = False,
        use_slingua: bool = False,
        llmlingua2_config: dict = None,
    ):
        self._model_manager = ModelManager(
            model_name=model_name,
            device_map=device_map,
            model_config=model_config or {},
        )

        self.use_llmlingua2 = use_llmlingua2
        self.use_slingua = use_slingua
        self.open_api_config = open_api_config or {}

        # Select appropriate compressor
        if use_llmlingua2 or use_slingua:
            self._compressor = LLMLingua2Compressor(
                model_manager=self._model_manager,
                config=llmlingua2_config or {},
            )
        else:
            self._compressor = LLMLinguaCompressor(
                model_manager=self._model_manager,
            )

    def compress_prompt(self, context, **kwargs) -> dict:
        """
        Compress the given context (backward compatible API).

        Returns dict instead of CompressionResult for compatibility.
        """
        result = self._compressor.compress(context, **kwargs)
        return result.__dict__

    # Delegate other methods...
    def structured_compress_prompt(self, context, **kwargs):
        return self._compressor.structured_compress(context, **kwargs)

    def compress_json(self, json_data, json_config, **kwargs):
        return self._compressor.compress_json(json_data, json_config, **kwargs)


__all__ = [
    "PromptCompressor",
    "CompressionResult",
    "MISTRAL_MODELS",
    "DEFAULT_MODEL",
    "RankingRegistry",
]
```

**Effort estimé** : 1 jour
**Risque** : Faible (pure délégation)

---

### Phase 7 : Tests et Documentation (Priorité: HAUTE)

**Tests à créer** :
- [ ] `tests/core/test_model_loader.py`
- [ ] `tests/core/test_tokenization.py`
- [ ] `tests/compression/test_llmlingua.py`
- [ ] `tests/compression/test_llmlingua2.py`
- [ ] `tests/filters/test_context_filter.py`
- [ ] `tests/filters/test_sentence_filter.py`
- [ ] `tests/filters/test_token_filter.py`
- [ ] `tests/test_backward_compatibility.py` (CRITIQUE)

**Documentation** :
- [ ] Mettre à jour `CLAUDE.md` avec la nouvelle architecture
- [ ] Ajouter docstrings à tous les modules publics
- [ ] Créer `docs/architecture.md` avec diagrammes

**Effort estimé** : 2-3 jours
**Risque** : -

---

## Récapitulatif des Efforts

| Phase | Description | Effort | Risque | Priorité | Statut |
|-------|-------------|--------|--------|----------|--------|
| 1 | Module `ranking/` | 1j | Faible | HAUTE | ✅ FAIT |
| 2 | Module `core/` | 1j | Faible | HAUTE | ✅ FAIT |
| 3 | Module `filters/` | 2-3j | Moyen | MOYENNE | ⏳ |
| 4 | Module `compression/` | 3-4j | Élevé | MOYENNE | ⏳ |
| 5 | Modules `budget/` + `recovery/` | 1-2j | Faible | BASSE | ⏳ |
| 6 | Façade backward-compatible | 1j | Faible | CRITIQUE | ⏳ |
| 7 | Tests + Documentation | 2-3j | - | HAUTE | ⏳ |
| **Total** | | **11-17j** | | | |

---

## Critères de Succès

- [ ] Tous les tests existants passent (backward compatibility)
- [ ] Couverture de tests > 80%
- [ ] Aucune régression de performance (benchmark avant/après)
- [ ] `prompt_compressor.py` < 200 lignes (Façade uniquement)
- [ ] Chaque module < 500 lignes
- [ ] Documentation à jour

---

## Risques et Mitigations

| Risque | Impact | Probabilité | Mitigation |
|--------|--------|-------------|------------|
| Régression fonctionnelle | Élevé | Moyenne | Tests de non-régression exhaustifs |
| Régression performance | Moyen | Faible | Benchmarks avant/après chaque phase |
| Breaking changes API | Élevé | Faible | Façade backward-compatible |
| Complexité accrue | Moyen | Moyenne | Documentation claire, exemples |

---

## Notes

- Commencer par Phase 1 (ranking) car c'est le plus isolé
- Toujours garder `prompt_compressor.py` fonctionnel pendant la migration
- Utiliser des feature flags si nécessaire pour basculer entre ancien/nouveau code
- Faire des PR atomiques par phase pour faciliter la review
