# ADR 0007 — Bibliothèque standard uniquement dans l'application

**Date** : 11/09/2026 · **Statut** : acceptée

## Contexte

Le cours doit tourner sur un portable ordinaire, sans matériel spécialisé, pour des apprenants
qui découvrent le sujet. Chaque dépendance est une installation qui peut échouer et une API qui
peut changer pendant les cinq ans d'accessibilité du dépôt.

## Décision

`assistant/` et `ai_service/` n'utilisent que la bibliothèque standard de Python 3.11+ :
`http.server`, `urllib`, `json`, `tomllib`, `hashlib`, `xml.etree`, `logging`. Exception
explicite : `sentence-transformers`, optionnel, côté service IA seulement. Les vrais modèles
sont servis par Ollama, un programme externe.

## Conséquences

- Tests, démarrage hors-ligne et import du corpus fonctionnent sans `pip install`.
- L'index est un fichier JSON à recherche exhaustive : quelques milliers de morceaux, pas plus.
  C'est une limite nommée (S4.3), pas un oubli ; une base vectorielle se brancherait derrière le
  même port `VectorIndex`.
- Le code est plus long qu'avec un cadriciel (serveur HTTP, argparse), mais lisible en entier.

## Écarté

FastAPI + numpy (choix d'`AssistantQR` pour son service Python) : plus court et plus rapide,
mais une chaîne d'installation de plus pour les apprenants, et du code qui cache ce que le cours
veut montrer.
