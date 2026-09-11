# Lecture du banc d'essai du 11 septembre 2026 — corpus Service-Public

Corpus réel : 322 fiches Service-Public.gouv.fr (thème Travail - Formation, secteur privé),
3 505 morceaux. Deux jeux de questions : **calibration** (42 questions, ce dossier) puis
**validation** (16 questions jamais vues, dossier `../2026-09-11-service-public-validation/`).
Même machine que le banc Solvéo (RTX 3070 8 Go, Ollama 0.34). Le générateur `qwen3:4b` est
mesuré à part (`../2026-09-11-service-public-qwen/`, 1 passage) à cause de sa latence.

```bash
python -m assistant benchmark --config config/app-ollama.toml --questions eval/questions-service-public.json \
    --embedding nomic bge-m3 --generation extractive llama3-2-3b --runs 3
python -m assistant benchmark --config config/app-ollama.toml --questions eval/questions-service-public-validation.json \
    --embedding nomic bge-m3 --generation extractive --runs 1
```

## Recherche : calibration puis validation

| | nomic (calibration) | nomic (validation) | bge-m3 (calibration) | bge-m3 (validation) |
|---|---|---|---|---|
| Hit@1 | 0,78 | 0,69 | **0,97** | **0,77** |
| Hit@4 | 0,94 | 0,85 | 0,97 | 0,92 |
| Score top-1 médian répondables / hors corpus | 0,80 / 0,71 | 0,81 / 0,74 | 0,74 / 0,55 | 0,72 / 0,61 |
| Seuil suggéré (séparation) | 0,73 (0,91) | 0,76 (0,94) | 0,65 (0,98) | 0,62 (0,94) |
| Refus justes avec le seuil calibré (extractif) | — | 0,33 | — | **1,00** |
| Indexation des 3 505 morceaux | 43 s | 61 s | 82 s | 183 s |

## Ce qu'on retient pour le cours

1. **Le seuil dépend du modèle *et* du corpus (S3.2, CACE, à deux niveaux).** Sur Solvéo
   (15 morceaux), `bge-m3` demandait 0,46 ; sur Service-Public (3 505 morceaux), 0,65. Rien n'a
   changé dans le code ni dans le modèle : avec trois mille morceaux, il y a toujours un passage
   « assez proche » d'une question hors sujet, les scores montent et le seuil doit suivre. Un seuil
   est une donnée de calibration, à refaire à chaque changement de modèle **ou de corpus** —
   d'où l'intérêt d'un banc d'essai qui le propose (`suggested_threshold`) et d'un jeu de
   validation qui le vérifie.

2. **Le seuil calibré tient sur des questions jamais vues pour `bge-m3`, moins pour `nomic`.**
   Validation : `bge-m3` à 0,65 rejette les 3 questions hors corpus et n'en refuse qu'une
   répondable sur 13 (« Quelle juridiction règle les litiges… ? », dont la fiche est réservée à
   la direction et mal classée). `nomic` à 0,73 ne rejette qu'une question hors corpus sur trois :
   ses deux populations de scores se recouvrent (0,81 contre 0,74). C'est la limite annoncée
   dans `suggest_threshold` : un seuil calibré sur le jeu qu'il évalue est optimiste.

3. **`bge-m3` est nettement meilleur que `nomic` sur ce corpus français** : hit@1 0,97 contre 0,78
   en calibration, 0,77 contre 0,69 en validation ; 7 questions répondables ratées par `nomic`
   derrière `llama3.2:3b` (report des congés, démission pendant les congés, durée légale, âge de
   la retraite…), 3 par `bge-m3`. Prix : un modèle 4 fois plus lourd (1,2 Go) et une indexation
   2 à 3 fois plus longue. **Proposition à trancher** : faire de `bge-m3` le modèle d'embeddings
   par défaut du cours, et raconter le passage `nomic` → `bge-m3` comme une décision motivée par
   la mesure — qui oblige à réindexer (le second verdict de la problématique).

4. **Le générateur ne rattrape pas la recherche (S2.3).** `llama3.2:3b` cite la bonne source
   dans 84 % des cas derrière `nomic`, 93 % derrière `bge-m3`, avec le même prompt et la même
   configuration : la différence vient entièrement des passages qu'on lui donne.

5. **Les droits d'accès : 0 fuite sur 504 appels**, mais les 4 questions « accès refusé » sont
   toutes répondues à partir de fiches publiques voisines (par exemple l'abandon de poste est
   évoqué dans la fiche publique sur la rupture du CDI). Le filtrage avant le prompt garantit
   la confidentialité, pas le silence : c'est une nuance à enseigner en S4.1.

6. **Un vrai corpus change les ordres de grandeur.** Latence médiane de `llama3.2:3b` : 1,6 à
   1,7 s (contre 0,5 s sur Solvéo) parce que les passages sont plus longs ; indexation en minutes
   et non en secondes ; 229 à 329 ms par question avec le générateur extractif (appel d'embeddings
   compris ; la recherche exhaustive seule sur 3 505 vecteurs prend ≈ 93 ms, mesuré sans modèle) —
   encore acceptable, mais c'est la limite au-delà de laquelle l'index JSON devient un vrai
   sujet (S4.3).

7. **Ce que le banc ne mesure pas** : la justesse du contenu (les mots-clés attendus sont un
   indicateur grossier : 0,80 pour `llama`, 0,31 pour l'extractif qui recopie une phrase entière) ;
   les réponses partielles issues d'une fiche voisine ; l'actualité juridique (fiches datées du
   11/09/2026).
