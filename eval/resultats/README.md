# Résultats de référence (mesurés le 11 septembre 2026)

Machine : Windows 11, Ryzen 7 5800H, 15,4 Go de RAM, RTX 3070 Laptop 8 Go, Ollama 0.34, Python 3.13.
Chaque dossier contient le rapport brut produit par l'outil (`rapport.md`, plus `resultats.csv`,
`synthese.json` ou `instantanes/`) ; les index reconstruits ne sont pas conservés. Les fichiers
`LECTURE*.md` sont les commentaires pour le cours.

| Dossier | Outil | Ce qu'on y mesure | Lecture |
|---|---|---|---|
| `2026-09-11-complet/` | banc d'essai, corpus Solvéo (15 morceaux), 24 questions × 3 passages | nomic / bge-m3 × extractif / llama3.2:3b / qwen3:4b ; seuils suggérés | `LECTURE.md` |
| `2026-09-11-service-public/` | banc d'essai, corpus réel (3 505 morceaux), 42 questions × 3 | nomic / bge-m3 × extractif / llama3.2:3b ; seuils suggérés | `LECTURE.md` |
| `2026-09-11-service-public-validation/` | banc d'essai, 16 questions jamais vues | le seuil calibré tient-il ? | dans la lecture ci-dessus |
| `2026-09-11-service-public-qwen/` | banc d'essai, corpus réel, 42 questions × 1 | qwen3:4b derrière nomic et bge-m3 (budget de réflexion 1 200) | `LECTURE-experiences.md` |
| `exp-changement-embeddings-2026-09-11/` | expérience | bge-m3 → nomic : refus sans réindexation, puis dérive | `LECTURE-experiences.md` |
| `exp-prompt-v2-2026-09-11/` | expérience | prompt v1 → v2, tout le reste constant | idem |
| `exp-stabilite-2026-09-11/` | expérience | rien ne change, 3 passages | idem |
| `exp-changement-generateur-2026-09-11/` | expérience, 12 questions | llama3.2:3b → qwen3:4b, budget de réflexion 1 200 | idem |
| `exp-changement-generateur-2026-09-11-budget3000/` | expérience, 12 questions | idem, budget 3 000 | idem |
| `exp-cace-decoupage-2026-09-11/` | expérience | morceaux 800 → 300, tout le reste constant | idem |

Reproduire : les commandes sont en tête de chaque rapport et dans le README du dépôt (sections
« banc d'essai » et « expériences reproductibles »). Les chiffres changeront avec la machine, la
version d'Ollama, les poids des modèles et la date du corpus : c'est le sujet du cours.
