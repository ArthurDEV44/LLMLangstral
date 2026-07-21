# A propos de LLMLangstral

**LLMLangstral** est un fork francais de [Microsoft LLMLingua](https://github.com/microsoft/LLMLingua), construit exclusivement autour des modeles **Mistral AI**.

## Presentation

LLMLangstral compresse les prompts LLM jusqu'a **20x** en supprimant les tokens a faible contenu informatif tout en preservant le sens. Moins de tokens envoyes a l'API signifie moins de latence et des couts reduits.

La compression repose sur un modele Mistral local (par defaut `Mistral-7B-v0.3`) qui evalue la perplexite segment par segment. Aucune donnee ne quitte votre machine.

## Deux modes

- **LLMLangstral** — compression standard par perplexite, adapte a la plupart des cas d'usage
- **LongLLMLangstral** — mode long-contexte avec filtrage multi-niveaux (passage → phrase → token) et reordonnancement pour attenuer le probleme du "lost in the middle"

## Architecture (v0.3.0)

```
llmlangstral/
├── prompt_compressor.py   # Point d'entree unique (PromptCompressor)
├── core/                  # Chargement modele + classes de base
├── filters/               # Filtres contexte / phrase / token
└── ranking/               # Strategies de ranking (registre de plugins)
```

Le refactor v0.3.0 a eclate l'ancien monolithe en sous-modules : `core/` (modele, ABCs), `filters/` (3 niveaux de compression), `ranking/` (5 strategies enfichables via decorateur `@register`).

## Utilisation minimale

```python
from llmlangstral import PromptCompressor

compressor = PromptCompressor()
result = compressor.compress_prompt(
    context=["Document 1", "Document 2"],
    question="Ma question",
    target_token=200,
)
print(result["compressed_prompt"])
```

## Stack technique

- Python 3.9+
- Backbone : `transformers` + `accelerate` + `torch` + `sentencepiece`
- Rankers optionnels : `rank_bm25`, `sentence_transformers`
- Qualite : Black, isort, flake8, pytest

## Licence

MIT (heritee de Microsoft LLMLingua).
