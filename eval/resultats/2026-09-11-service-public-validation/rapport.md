# Banc d'essai — 2026-09-11 18:23

16 questions · 1 passage(s) par question · prompt `v1+9cf8bd83` · découpage `{"max_chars": 800, "overlap_chars": 120, "include_title": true}` · top_k=4 · température=0.2

## Recherche (sans génération)

| Embeddings | Modèle servi | Dim. | Morceaux | Indexation (s) | Hit@1 | Hit@k | Score top-1 médian (répondables) | (hors corpus) | Seuil configuré | Seuil suggéré | Séparation | Seuil utilisé |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nomic | `ollama:nomic-embed-text@0a109f422b47` | 768 | 3505 | 60.60 | 0.69 | 0.85 | 0.81 | 0.74 | 0.73 | 0.76 | 0.94 | 0.73 |
| bge-m3 | `ollama:bge-m3@790764642607` | 1024 | 3505 | 182.70 | 0.77 | 0.92 | 0.72 | 0.61 | 0.65 | 0.62 | 0.94 | 0.65 |

## Réponses (avec génération)

| Embeddings | Génération | Répond (répondables) | Bonne source | Mots-clés | Non sourcé | Refus justes (hors corpus) | Fuites d'accès | Stabilité | Tentatives | Latence médiane (ms) | p90 (ms) | Erreurs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nomic | extractive | 1.00 | 0.69 | 0.31 | 0.00 | 0.33 | 0 | — | 1.00 | 240 | 311 | 0 |
| bge-m3 | extractive | 0.92 | 0.92 | 0.42 | 0.00 | 1.00 | 0 | — | 1.00 | 354 | 417 | 0 |

## Lecture

- **Hit@1 / Hit@k** : part des questions répondables dont un document attendu arrive en tête / figure dans les k passages retrouvés. Sur un petit corpus, Hit@k est vite saturé : regarder Hit@1.
- **Seuil suggéré** : sépare au mieux les questions répondables des questions hors corpus. Calibré sur ces mêmes questions, il est optimiste.
- **Bonne source** : parmi les réponses données, part qui cite un document attendu.
- **Mots-clés** : part des mots-clés attendus présents dans la réponse (indicateur grossier).
- **Non sourcé** : le modèle n'a pas cité correctement ses sources malgré les tentatives.
- **Fuites d'accès** : doit toujours valoir 0, le filtrage est fait avant le modèle.
- **Stabilité** : pour une même question, part des passages qui donnent le même statut et les mêmes documents cités (1 = parfaitement stable). Nécessite au moins 2 passages.
