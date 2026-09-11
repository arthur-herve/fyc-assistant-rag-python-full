# Lecture du banc d'essai du 11 septembre 2026 — corpus Solvéo

Premier banc d'essai avec de vrais modèles. Commande, lancée depuis la racine du projet :

```bash
python -m assistant benchmark --embedding nomic bge-m3 --generation extractive llama3-2-3b qwen3-4b --runs 3
```

Machine : Windows 11, Ryzen 7 5800H, 15,4 Go de RAM, RTX 3070 Laptop 8 Go, Ollama 0.34.
Corpus : 9 documents fictifs Solvéo, 15 morceaux. 24 questions (19 répondables, 2 accès refusés,
3 hors corpus) × 3 passages. Prompt `v1+9cf8bd83`, découpage 800 / 120, top_k 4, température 0,2.
Les chiffres bruts sont dans `rapport.md`, `synthese.json` et `resultats.csv`.

## Ce qu'on retient pour le cours

1. **Le seuil de pertinence appartient au modèle d'embeddings, pas à l'application (S3.2, CACE).**
   Avec `nomic`, les questions hors corpus obtiennent un score top-1 médian de 0,64 contre 0,72 pour
   les questions répondables : le seuil de 0,55 hérité de la configuration hors-ligne ne rejette rien
   (0 refus juste sur 15). Avec `bge-m3`, les deux populations sont à 0,41 et 0,67 : le seuil de 0,45
   sépare correctement (séparation 0,96). Le banc propose 0,59 pour `nomic` et 0,46 pour `bge-m3` ;
   ces valeurs sont reportées dans `config/app.toml`. Changer de modèle d'embeddings a donc obligé à
   recalibrer un paramètre *métier* (« je ne sais pas »), en plus de réindexer.

2. **La qualité de la recherche décide de ce que le générateur peut faire (S2.3, S4.1).**
   Même générateur `llama3.2:3b` : 24 % de réponses non sourcées derrière `nomic`, 3 % derrière
   `bge-m3`. Derrière `nomic`, les questions hors corpus atteignent le modèle (seuil trop bas) ; il
   répond alors « les passages ne permettent pas de répondre » sans citer, et la règle métier
   « pas de réponse non sourcée » prend le relais. Le garde-fou déterministe a servi ; il n'aurait
   pas dû être sollicité.

3. **Changer de générateur est une ligne de configuration… et 20 fois plus de latence (S3.3, S4.1).**
   `llama3.2:3b` répond en 0,5 à 0,8 s (médiane), `qwen3:4b` en 13 à 20 s, avec un p90 à 60-80 s
   et un maximum à 102 s : son mode réflexion consomme jusqu'à 1 200 jetons avant la réponse.
   Le même index a servi aux deux, sans réindexation — le verdict « le générateur est un détail »
   tient pour les données, pas pour l'expérience utilisateur ni pour les seuils de délai.

4. **La particularité d'un modèle fuit jusqu'à la réponse si le service IA ne l'arrête pas (S4.1).**
   Avant correction, `think = false` faisait déverser par qwen3 son raisonnement en anglais dans le
   texte de la réponse (400 jetons de « Okay, let's see… »). Le raisonnement contenait « [1] » :
   la vérification déterministe des citations l'a accepté, et le banc affichait 100 % de bonnes
   sources. La correction est dans `ai_service/backends/ollama.py` (réflexion activée, renvoyée à
   part par Ollama, budget de jetons séparé). Leçon : une vérification de forme n'est pas une
   vérification de fond ; depuis, le décorateur de validation (ADR 0008, `domain/output_rules.py`)
   refuse une réponse dans la mauvaise langue, trop longue ou qui raisonne.

5. **Le non-déterminisme se mesure (S3.1).** Stabilité 0,93 (nomic + llama) et 0,99 (bge-m3 + llama)
   : sur 3 passages, une même question change de statut ou de sources dans 1 à 7 % des cas, à
   température 0,2, sans graine. Le générateur hors-ligne `extractive` est à 1,00 : c'est la
   référence déterministe. `qwen3:4b` : 0,90 à 0,92.

6. **Les droits d'accès n'ont jamais fui** (0 sur 432 appels) : le filtrage se fait avant le prompt,
   le modèle ne voit pas ce que l'utilisateur n'a pas le droit de lire. Les questions « accès refusé »
   ne sont pas toujours refusées pour autant : un document public voisin peut fournir une réponse
   partielle (`acces-salaire-refuse` est répondue 2 fois sur 3 derrière `bge-m3`, sans citer la
   grille des salaires). Le banc distingue bien les deux : fuite = 0, refus juste = 0,60.

7. **Limites de ce banc** : 15 morceaux et 24 questions, c'est un jeu de démonstration, pas une
   évaluation ; le seuil suggéré est calibré sur les questions qu'il évalue (optimiste) ; une
   question répondable (`it-mdp`, mot de passe) est ratée 3 fois sur 3 par la recherche `nomic`
   (hit@1 = 0,90). Le banc sur le corpus réel Service-Public (322 fiches, 3 505 morceaux, jeu de
   calibration puis de validation) est dans `../2026-09-11-service-public/`.
