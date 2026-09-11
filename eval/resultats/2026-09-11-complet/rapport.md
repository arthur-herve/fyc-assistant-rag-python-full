# Banc d'essai — 2026-09-11 18:07

24 questions · 3 passage(s) par question · prompt `v1+9cf8bd83` · découpage `{"max_chars": 800, "overlap_chars": 120, "include_title": true}` · top_k=4 · température=0.2

## Recherche (sans génération)

| Embeddings | Modèle servi | Dim. | Morceaux | Indexation (s) | Hit@1 | Hit@k | Score top-1 médian (répondables) | (hors corpus) | Seuil configuré | Seuil suggéré | Séparation | Seuil utilisé |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nomic | `ollama:nomic-embed-text@0a109f422b47` | 768 | 15 | 0.20 | 0.90 | 0.95 | 0.72 | 0.64 | 0.55 | 0.59 | 0.83 | 0.55 |
| bge-m3 | `ollama:bge-m3@790764642607` | 1024 | 15 | 6.60 | 1.00 | 1.00 | 0.67 | 0.41 | 0.45 | 0.46 | 0.96 | 0.45 |

## Réponses (avec génération)

| Embeddings | Génération | Répond (répondables) | Bonne source | Mots-clés | Non sourcé | Refus justes (hors corpus) | Fuites d'accès | Stabilité | Tentatives | Latence médiane (ms) | p90 (ms) | Erreurs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nomic | extractive | 1.00 | 0.95 | 0.90 | 0.00 | 0.00 | 0 | 1.00 | 1.00 | 42 | 57 | 0 |
| nomic | llama3-2-3b | 0.93 | 1.00 | 0.96 | 0.24 | 0.00 | 0 | 0.93 | 1.26 | 784 | 2094 | 0 |
| nomic | qwen3-4b | 0.91 | 0.94 | 0.94 | 0.10 | 0.00 | 0 | 0.90 | 1.24 | 19690 | 80204 | 0 |
| bge-m3 | extractive | 1.00 | 1.00 | 0.90 | 0.00 | 0.60 | 0 | 1.00 | 1.00 | 95 | 125 | 0 |
| bge-m3 | llama3-2-3b | 1.00 | 0.95 | 0.95 | 0.03 | 0.60 | 0 | 0.99 | 1.06 | 546 | 1299 | 0 |
| bge-m3 | qwen3-4b | 0.90 | 1.00 | 1.00 | 0.11 | 0.60 | 0 | 0.92 | 1.21 | 13231 | 60158 | 0 |

## Lecture

- **Hit@1 / Hit@k** : part des questions répondables dont un document attendu arrive en tête / figure dans les k passages retrouvés. Sur un petit corpus, Hit@k est vite saturé : regarder Hit@1.
- **Seuil suggéré** : sépare au mieux les questions répondables des questions hors corpus. Calibré sur ces mêmes questions, il est optimiste.
- **Bonne source** : parmi les réponses données, part qui cite un document attendu.
- **Mots-clés** : part des mots-clés attendus présents dans la réponse (indicateur grossier).
- **Non sourcé** : le modèle n'a pas cité correctement ses sources malgré les tentatives.
- **Fuites d'accès** : doit toujours valoir 0, le filtrage est fait avant le modèle.
- **Stabilité** : pour une même question, part des passages qui donnent le même statut et les mêmes documents cités (1 = parfaitement stable). Nécessite au moins 2 passages.
