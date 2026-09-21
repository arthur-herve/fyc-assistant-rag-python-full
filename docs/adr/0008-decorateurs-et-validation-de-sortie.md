# ADR 0008 — Les garde-fous sont des décorateurs de ports ; la forme de la sortie est une règle métier

**Date** : 11/09/2026 · **Statut** : acceptée · complétée par l'ADR 0009 (21/09/2026)

## Contexte

Le banc d'essai du 11/09 a montré qwen3 déversant son raisonnement en anglais dans la réponse,
avec un « [1] » dedans : la vérification des citations l'a accepté. Il fallait un garde-fou de
fond, et plus généralement un moyen d'ajouter cache, journal et nouvelles tentatives sans toucher
au cas d'usage ni à l'adaptateur HTTP (S4.1).

## Décision

- Un décorateur implémente le même port que l'objet qu'il enveloppe. Les techniques
  (`CachedEmbedder`, `LoggingEmbedder`/`LoggingGenerator`, `RetryingEmbedder`/`RetryingGenerator`)
  vivent dans `infrastructure/decorators.py` ; celui qui porte une règle métier
  (`OutputValidatingGenerator`) vit dans `application/guards.py` et s'appuie sur
  `domain/output_rules.py` (vide, trop long, autre langue, raisonnement déversé).
- L'empilement est décidé dans `composition.py` (`decorate()`) à partir de `[decorators]` de la
  configuration, de l'intérieur vers l'extérieur : tentatives, journal, cache, validation.
- Une sortie rejetée est une tentative ratée pour `AskQuestion` (tracée `<rejetée : …>`), qui
  réessaie puis renvoie « non sourcé ».

## Conséquences

- Les cas d'usage ne voient que les ports ; le test d'architecture vérifie que seule la
  composition importe l'infrastructure.
- Les règles de forme sont des heuristiques (marqueurs de raisonnement, ratio de mots anglais) :
  elles peuvent se tromper, elles sont testées sur les cas réels rencontrés, et elles sont dans le
  domaine parce qu'elles disent ce qu'est une réponse acceptable pour le métier.
- Un cache d'embeddings est un artefact lié au modèle (S4.2) : le nôtre est en mémoire ; un cache
  persistant devrait porter l'identifiant du modèle dans sa clé.

## Écarté

Mettre la validation dans le service IA (« c'est le modèle qui bavarde ») : le service ne sait
pas ce qu'est une réponse acceptable pour l'application. Mettre la validation dans `AskQuestion`
directement : elle y serait invisible et non substituable ; l'exercice « ajouter un décorateur »
n'aurait plus de sens.
