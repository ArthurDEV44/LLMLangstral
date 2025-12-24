# LLMLangstral

**Réduisez vos coûts d'API jusqu'à 20x en compressant intelligemment vos prompts.**

---

## Le problème

Les grands modèles de langage (LLM) comme GPT-4, Claude ou Mistral facturent à l'utilisation de tokens. Plus votre prompt est long, plus vous payez cher. De plus, chaque modèle a une limite de contexte : au-delà, il ne peut plus traiter votre texte.

Concrètement, cela signifie :

- Des factures d'API qui explosent sur des projets à fort volume
- L'impossibilité de traiter des documents longs en une seule requête
- Des performances dégradées quand le modèle "oublie" les informations au milieu d'un long contexte

---

## La solution

LLMLangstral analyse votre texte et supprime les mots non essentiels tout en préservant le sens. Le résultat : un prompt plus court qui transmet la même information au LLM.

**Exemple concret :**

| Avant compression | Après compression |
|-------------------|-------------------|
| 2000 tokens | 200 tokens |
| Coût : 0,06 $ | Coût : 0,006 $ |

Le LLM reçoit moins de tokens mais comprend toujours le contexte. Vous économisez 90% sur cet appel.

---

## Comment ça fonctionne

LLMLangstral utilise un petit modèle de langage local (basé sur l'architecture Mistral) pour évaluer l'importance de chaque mot. Les mots à faible valeur informative sont retirés, les mots clés sont conservés.

Le processus se déroule en trois étapes :

1. **Analyse** — Le texte est découpé en segments (phrases, paragraphes)
2. **Évaluation** — Chaque segment reçoit un score d'importance
3. **Compression** — Les segments les moins importants sont supprimés ou abrégés

Tout se passe localement sur votre machine. Aucune donnée n'est envoyée à un service externe pendant la compression.

---

## Les variantes disponibles

LLMLangstral propose plusieurs méthodes de compression selon vos besoins :

### LLMLingua

La méthode de base. Elle utilise la perplexité (une mesure de "surprise" du modèle) pour identifier les tokens importants. Efficace pour la plupart des cas d'usage.

### LongLLMLingua

Optimisée pour les longs documents. Elle résout le problème du "lost in the middle" où les LLM ont tendance à oublier les informations situées au centre d'un long texte. Particulièrement utile pour le RAG (Retrieval-Augmented Generation).

### LLMLingua-2

La version la plus rapide. Elle utilise un modèle de classification de tokens (basé sur XLM-RoBERTa) entraîné par distillation depuis GPT-4. Résultat : 3 à 6 fois plus rapide que LLMLingua standard.

### SecurityLingua

Dédiée à la sécurité. Elle détecte les tentatives de jailbreak en compressant le prompt pour révéler les intentions malveillantes cachées. Coût de détection 100 fois inférieur aux solutions classiques.

---

## Cas d'usage

### Réduction des coûts d'API

Compressez systématiquement vos prompts avant de les envoyer à GPT-4 ou Claude. Économies typiques : 50 à 90% selon le type de contenu.

### Traitement de documents longs

Résumez des rapports, des transcriptions de réunions ou des bases de connaissances qui dépassent la limite de contexte du modèle cible.

### Amélioration du RAG

Dans un pipeline RAG, compressez les documents récupérés avant de les injecter dans le prompt. Cela permet d'inclure plus de contexte pertinent sans dépasser les limites.

### Accélération de l'inférence

Moins de tokens à traiter signifie une réponse plus rapide du LLM. Utile pour les applications temps réel.

### Protection contre les attaques

Avec SecurityLingua, détectez les prompts malveillants avant qu'ils n'atteignent votre LLM principal.

---

## Modèles utilisés

LLMLangstral s'appuie principalement sur des modèles Mistral AI :

| Usage | Modèle | Taille |
|-------|--------|--------|
| Compression standard | Mistral 7B v0.3 | 7 milliards de paramètres |
| Compression légère | Ministral 3B | 3 milliards de paramètres |
| Ressources limitées | Mistral 7B GPTQ | Version quantifiée, moins de 8 Go de VRAM |
| Ranking de documents | E5-Mistral 7B | Embeddings pour le tri par pertinence |
| Compression rapide | XLM-RoBERTa | Classification de tokens (LLMLingua-2) |

---

## Avantages clés

**Économique** — Réduction drastique des coûts d'API sans perte de qualité perceptible.

**Portable** — Fonctionne avec n'importe quel LLM cible (OpenAI, Anthropic, Mistral, modèles open source).

**Local** — La compression s'effectue sur votre infrastructure. Vos données restent privées.

**Flexible** — Contrôle fin du taux de compression par section du prompt.

**Rapide** — LLMLingua-2 traite les prompts en millisecondes.

---

## Limites

- La compression peut occasionnellement supprimer des informations pertinentes
- Les modèles de compression nécessitent un GPU pour des performances optimales (CPU possible mais plus lent)
- LLMLingua-2 (la version rapide) reste basée sur XLM-RoBERTa, pas sur Mistral

---

## Origine du projet

LLMLangstral est un fork du projet LLMLingua de Microsoft Research, adapté pour utiliser principalement des modèles Mistral AI. Les travaux de recherche originaux ont été publiés à EMNLP 2023 et ACL 2024.

---

## Ressources

- Documentation technique détaillée : voir le fichier DOCUMENT.md
- Exemples pratiques : dossier examples/
- FAQ : fichier Transparency_FAQ.md

---

## Licence

MIT License — Utilisation libre pour projets personnels et commerciaux.
