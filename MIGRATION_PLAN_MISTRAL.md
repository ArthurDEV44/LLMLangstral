# Plan de Migration LLMLingua vers Mistral AI

## Objectif

Migrer le fork LLMLingstral pour utiliser exclusivement des modèles Mistral AI en remplacement des modèles Microsoft, Meta (LLaMA), et autres modèles tiers.

---

## 1. Inventaire des Modèles Actuels

### 1.1 Modèles Causals (LLMLingua/LongLLMLingua)

| Modèle Actuel | Fichier | Ligne | Usage |
|---------------|---------|-------|-------|
| `NousResearch/Llama-2-7b-hf` | `llmlingua/prompt_compressor.py` | 73 | Défaut LLMLingua |
| `microsoft/phi-2` | README.md, DOCUMENT.md | - | Alternative légère |
| `TheBloke/Llama-2-7b-Chat-GPTQ` | README.md, DOCUMENT.md | - | Version quantifiée |
| `lgaalves/gpt2-dolly` | `tests/test_llmlingua.py` | 68 | Tests |

### 1.2 Modèles Token Classification (LLMLingua-2)

| Modèle Actuel | Fichier | Ligne | Usage |
|---------------|---------|-------|-------|
| `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | `llmlingua/prompt_compressor.py` | 61 | LLMLingua-2 principal |
| `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` | README.md | 218 | LLMLingua-2 small |
| `FacebookAI/xlm-roberta-large` | `experiments/llmlingua2/model_training/train_roberta.py` | 27 | Base entraînement |

### 1.3 Modèles d'Embedding/Ranking

| Modèle Actuel | Fichier | Ligne | Méthode |
|---------------|---------|-------|---------|
| `multi-qa-mpnet-base-dot-v1` | `llmlingua/prompt_compressor.py` | 1852 | sentbert |
| `BAAI/bge-large-en-v1.5` | `llmlingua/prompt_compressor.py` | 1887 | bge |
| `BAAI/bge-reranker-large` | `llmlingua/prompt_compressor.py` | 1902 | bge_reranker |
| `BAAI/llm-embedder` | `llmlingua/prompt_compressor.py` | 1934 | bge_llmembedder |
| `jinaai/jina-embeddings-v2-base-en` | `llmlingua/prompt_compressor.py` | 1990 | jinza |

### 1.4 SecurityLingua

| Modèle Actuel | Fichier | Usage |
|---------------|---------|-------|
| `SecurityLingua/securitylingua-xlm-s2s` | README.md, experiments/ | Détection jailbreak |

### 1.5 Tokenizers

| Modèle Actuel | Fichier | Ligne | Usage |
|---------------|---------|-------|-------|
| `gpt-3.5-turbo` | `llmlingua/prompt_compressor.py` | 89 | Comptage tokens OAI |
| `gpt-4` | `experiments/llmlingua2/` | - | Data collection |

---

## 2. Substitutions Mistral Proposées

### 2.1 Modèles Causals (Perplexité)

| Actuel | Remplacement Mistral | Paramètres | Avantages |
|--------|---------------------|------------|-----------|
| `NousResearch/Llama-2-7b-hf` | **`mistralai/Mistral-7B-v0.3`** | 7B | Performance supérieure, contexte 32K, Apache 2.0 |
| `microsoft/phi-2` | **`mistralai/Ministral-3-3B-Instruct-2512`** | 3B | Ultra-léger, 8GB VRAM en FP8 |
| `TheBloke/Llama-2-7b-Chat-GPTQ` | **`TheBloke/Mistral-7B-Instruct-v0.2-GPTQ`** | 7B quantifié | <8GB VRAM, GPTQ optimisé |
| `lgaalves/gpt2-dolly` (tests) | **`mistralai/Ministral-3-3B-Instruct-2512`** | 3B | Tests rapides |

**Alternatives selon les ressources:**
- GPU limitée (< 8GB): `mistralai/Ministral-3-3B-Instruct-2512`
- GPU standard (16-24GB): `mistralai/Mistral-7B-v0.3` ou `mistralai/Ministral-3-8B-Instruct-2512`
- GPU haute performance: `mistralai/Mistral-Large-3` (41B actifs, MoE)

### 2.2 Modèles Token Classification (LLMLingua-2)

**Problème**: Mistral ne fournit pas de modèles token classification pré-entraînés.

**Solution**: Entraîner un nouveau modèle basé sur l'architecture Mistral.

| Actuel | Approche Migration |
|--------|-------------------|
| `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | Entraîner `MistralForTokenClassification` sur les données MeetingBank |
| `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` | Entraîner `Ministral-3B` avec tête classification |

**Modèle de base recommandé pour fine-tuning:**
```python
# Option 1: Utiliser l'architecture Mistral native
from transformers import MistralForTokenClassification
model = MistralForTokenClassification.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512",
    num_labels=2  # keep/remove
)

# Option 2: Distillation vers un modèle plus petit
# Utiliser les données de microsoft/MeetingBank-LLMCompressed
```

### 2.3 Modèles d'Embedding/Ranking

| Actuel | Remplacement Mistral | Notes |
|--------|---------------------|-------|
| `multi-qa-mpnet-base-dot-v1` | **`intfloat/e5-mistral-7b-instruct`** | Embeddings basés sur Mistral-7B |
| `BAAI/bge-large-en-v1.5` | **`intfloat/e5-mistral-7b-instruct`** | SentenceTransformer compatible |
| `BAAI/bge-reranker-large` | **API `mistral-embed`** ou fine-tune Mistral | Via Mistral API |
| `jinaai/jina-embeddings-v2-base-en` | **`intfloat/e5-mistral-7b-instruct`** | 4096 dimensions |

**Nouvelle méthode de ranking proposée:**
```python
# Ajouter rank_method="mistral"
# Utilisant intfloat/e5-mistral-7b-instruct
```

### 2.4 SecurityLingua

| Actuel | Approche Migration |
|--------|-------------------|
| `SecurityLingua/securitylingua-xlm-s2s` | Entraîner sur `mistralai/Ministral-3-3B-Instruct-2512` avec données jailbreak |

### 2.5 Tokenizers

| Actuel | Remplacement | Notes |
|--------|--------------|-------|
| `tiktoken` (gpt-3.5/gpt-4) | **`transformers` tokenizer Mistral** | BPE compatible |

---

## 3. Plan d'Implémentation

### Phase 1: Infrastructure (Semaine 1-2) ✅ COMPLÉTÉE

#### 1.1 Mise à jour des dépendances
```python
# setup.py - Modifier INSTALL_REQUIRES
INSTALL_REQUIRES = [
    "transformers>=4.40.0",  # Support Mistral 3
    "accelerate>=0.27.0",
    "torch>=2.1.0",
    "sentencepiece",  # Tokenizer Mistral
    "protobuf",
    "nltk",
    "numpy",
    # Retirer tiktoken ou le garder en option
]
```

#### 1.2 Créer constantes Mistral
```python
# llmlingua/mistral_config.py (nouveau fichier)
MISTRAL_MODELS = {
    "default": "mistralai/Mistral-7B-v0.3",
    "small": "mistralai/Ministral-3-3B-Instruct-2512",
    "medium": "mistralai/Ministral-3-8B-Instruct-2512",
    "large": "mistralai/Mistral-Large-3",
    "quantized": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    "embedding": "intfloat/e5-mistral-7b-instruct",
}
```

### Phase 2: Migration Core (Semaine 3-4) ✅ COMPLÉTÉE

#### 2.1 Modifier `prompt_compressor.py`

**Fichier**: `llmlingua/prompt_compressor.py`

**Lignes à modifier:**

```python
# Ligne 73 - Changer le modèle par défaut
# AVANT:
model_name: str = "NousResearch/Llama-2-7b-hf"
# APRÈS:
model_name: str = "mistralai/Mistral-7B-v0.3"

# Ligne 61 - Modèle LLMLingua-2 (temporaire jusqu'à fine-tuning)
# Garder XLM-RoBERTa jusqu'à ce que le modèle Mistral soit entraîné
# Puis remplacer par: "mistralai/mistral-lingua-2-meetingbank" (à créer)
```

#### 2.2 Ajouter support tokenizer Mistral

```python
# llmlingua/prompt_compressor.py - Nouvelle méthode
def _get_mistral_tokenizer(self, model_name: str):
    """Charge le tokenizer Mistral avec configuration correcte."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
```

#### 2.3 Modifier les méthodes de ranking

**Fichier**: `llmlingua/prompt_compressor.py` (lignes 1850-2070)

```python
# Ajouter nouvelle méthode de ranking Mistral
elif rank_method == "mistral":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    # ... implémentation similaire à sentbert
```

### Phase 3: Entraînement Modèles (Semaine 5-8) ✅ COMPLÉTÉE

#### 3.1 Entraîner MistralLingua-2

**Objectif**: Créer `mistralai/mistral-lingua-2-meetingbank`

**Étapes:**
1. Télécharger dataset `microsoft/MeetingBank-LLMCompressed`
2. Adapter le script `experiments/llmlingua2/model_training/train_roberta.py`
3. Entraîner `MistralForTokenClassification` sur les données

**Script modifié:**
```python
# experiments/llmlingua2/model_training/train_mistral.py (nouveau)
from transformers import (
    MistralForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

model = MistralForTokenClassification.from_pretrained(
    "mistralai/Ministral-3-3B-Instruct-2512",
    num_labels=2,
    torch_dtype=torch.bfloat16
)
```

#### 3.2 Entraîner MistralSecurityLingua

**Objectif**: Créer modèle de détection jailbreak basé sur Mistral

**Base**: Utiliser `SecurityLingua/securitylingua-jailbreak-pairs` dataset

### Phase 4: Tests & Documentation (Semaine 9-10) ✅ COMPLÉTÉE

#### 4.1 Mise à jour des tests

**Fichier**: `tests/test_llmlingua.py`
```python
# Ligne 68 - Changer modèle de test
# AVANT:
llm_lingua = PromptCompressor(model_name="lgaalves/gpt2-dolly")
# APRÈS:
llm_lingua = PromptCompressor(model_name="mistralai/Ministral-3-3B-Instruct-2512")
```

#### 4.2 Nouveaux tests Mistral
```python
# tests/test_mistral.py (nouveau)
class MistralLinguaTester(unittest.TestCase):
    def test_mistral_7b_compression(self):
        compressor = PromptCompressor(model_name="mistralai/Mistral-7B-v0.3")
        # ...

    def test_ministral_3b_compression(self):
        compressor = PromptCompressor(model_name="mistralai/Ministral-3-3B-Instruct-2512")
        # ...
```

#### 4.3 Mise à jour documentation

**Fichiers à modifier:**
- `README.md`
- `DOCUMENT.md`
- `CLAUDE.md`
- `examples/*.ipynb`

---

## 4. Fichiers à Modifier

| Fichier | Modifications |
|---------|--------------|
| `llmlingua/prompt_compressor.py` | Modèle défaut, tokenizer, ranking methods |
| `llmlingua/utils.py` | Support tokenizer Mistral |
| `llmlingua/__init__.py` | Exporter nouvelles constantes |
| `setup.py` | Dépendances |
| `tests/test_llmlingua.py` | Modèle de test |
| `tests/test_llmlingua2.py` | Modèle LLMLingua-2 |
| `experiments/llmlingua2/model_training/` | Scripts entraînement Mistral |
| `experiments/securitylingua/` | Migration SecurityLingua |
| `README.md` | Documentation |
| `DOCUMENT.md` | Guide utilisateur |
| `examples/*.ipynb` | Notebooks exemples |

---

## 5. Tableau de Correspondance Final

| Composant | Modèle Original | Modèle Mistral | Status |
|-----------|-----------------|----------------|--------|
| **LLMLingua Default** | Llama-2-7B | `mistralai/Mistral-7B-v0.3` | Direct swap |
| **LLMLingua Small** | phi-2 | `mistralai/Ministral-3-3B-Instruct-2512` | Direct swap |
| **LLMLingua Quantized** | Llama-2-7B-GPTQ | `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` | Direct swap |
| **LLMLingua-2 Large** | xlm-roberta-large | Fine-tune Mistral-7B | Entraînement requis |
| **LLMLingua-2 Small** | bert-multilingual | Fine-tune Ministral-3B | Entraînement requis |
| **Embedding sentbert** | mpnet-base | `intfloat/e5-mistral-7b-instruct` | Direct swap |
| **Embedding bge** | bge-large | `intfloat/e5-mistral-7b-instruct` | Direct swap |
| **Reranker** | bge-reranker | API Mistral ou fine-tune | À développer |
| **SecurityLingua** | xlm-roberta-s2s | Fine-tune Ministral-3B | Entraînement requis |
| **Tokenizer** | tiktoken | Mistral tokenizer | Direct swap |

---

## 6. Risques et Mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Performance dégradée token classification | Élevé | Entraînement soigné avec hyperparamètres optimisés |
| Incompatibilité tokenizer | Moyen | Tests extensifs sur datasets variés |
| VRAM insuffisante Mistral-7B | Moyen | Proposer Ministral-3B comme alternative |
| Perte multilingue | Élevé | Ministral 3 supporte 40+ langues nativement |

---

## 7. Ressources Requises

### Matériel
- GPU: A100 40GB ou équivalent pour entraînement
- Stockage: ~100GB pour modèles et datasets

### Datasets
- `microsoft/MeetingBank-LLMCompressed` (LLMLingua-2)
- `SecurityLingua/securitylingua-jailbreak-pairs` (SecurityLingua)

### Temps estimé
- Phase 1: 2 semaines
- Phase 2: 2 semaines
- Phase 3: 4 semaines (entraînement)
- Phase 4: 2 semaines
- **Total: ~10 semaines**

---

## 8. Sources

- [Mistral AI HuggingFace](https://huggingface.co/mistralai)
- [Introducing Mistral 3](https://mistral.ai/news/mistral-3)
- [Ministral 3 Collection](https://huggingface.co/collections/mistralai/ministral-3)
- [E5-Mistral-7B-Instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct)
- [TheBloke Mistral GPTQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ)
- [Mistral Embeddings API](https://docs.mistral.ai/capabilities/embeddings)
- [MistralForTokenClassification](https://huggingface.co/docs/transformers/en/model_doc/mistral)
