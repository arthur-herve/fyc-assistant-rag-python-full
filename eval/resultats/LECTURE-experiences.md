# Lecture des expériences du 11 septembre 2026 — corpus Service-Public, vrais modèles

Configuration de base : `config/app-ollama.toml` (bge-m3 + llama3.2:3b, seuil 0,65, découpage 800 / 120,
prompt `v1+9cf8bd83`, température 0,2, sans graine), 42 questions de calibration. Chaque expérience
change **une seule chose** ; les instantanés avant/après sont dans le dossier de l'expérience.

## Vue d'ensemble

| Expérience | Ce qui change | Taux de dérive | Changements de statut | Ce que ça coûte |
|---|---|---|---|---|
| Rien (`stabilite`, 3 passages) | rien | **0,67 à 0,71** | 0 | — |
| Prompt v1 → v2 | le prompt | 0,79 | 3 | réponses de 338 à 52 caractères |
| Découpage 800 → 300 | la taille des morceaux | 0,79 | 3 | 3 505 → 10 466 morceaux, indexation ×2,4 |
| Embeddings bge-m3 → nomic | le modèle d'embeddings | 0,81 | 5 | réindexation obligatoire (refus explicite avant) |
| Générateur llama → qwen3 (budget 1 200) | le modèle de génération | 1,00 | 5 | latence ×24 ; 5 réponses vides sur 12 |
| Générateur llama → qwen3 (budget 3 000) | idem, budget de réflexion adapté | 1,00 | 0 | latence ×19 ; 0 réponse vide |

## Ce qu'on retient pour le cours

1. **La ligne de base, c'est 70 % de dérive à configuration constante (S3.1).** Trois passages
   identiques donnent 67 à 71 % de réponses reformulées, 5 à 6 changements de sources citées, 0
   changement de statut. Toute expérience se lit *par rapport* à ce bruit : un taux de dérive de
   79 % après avoir changé le prompt ne signifie pas « le prompt a tout changé », mais « le prompt
   a changé ce que le hasard ne changeait pas » — la longueur (338 → 52 caractères) et 3 statuts.
   Ce qu'un test statistique doit borner, ce sont les changements de **statut** et de **sources**,
   pas les reformulations.

2. **Le prompt change la qualité, pas seulement la forme (S3.3).** `answer-v2` demande une phrase
   commençant par la valeur ou par Oui / Non. Résultat : « Non [4]. » pour le report des congés
   (la fiche dit « pas nécessairement perdus, sous conditions »), « Oui [1, 2, 3] » avec toutes les
   sources citées en bloc, et une question qui passe de réponse à « non sourcé ». Le taux de dérive
   ne dit pas laquelle des deux versions est la meilleure : il faut lire les réponses, ou avoir des
   attendus (mots-clés, source attendue). Le prompt est une configuration **surveillée comme du
   métier** (ADR 0005) : sa version est dans chaque trace, et une modification se mesure avant
   d'être déployée.

3. **Le découpage est un paramètre de données qui change les réponses (S3.2, CACE).** Passer de
   800 à 300 caractères, sans toucher au seuil : 10 466 morceaux au lieu de 3 505, indexation de
   81 à 198 s, taux de réponse 1,00 → 0,91, bonne source 0,94 → 0,86, une question passe en
   refus (grève : la fiche réservée à la direction n'atteint plus le seuil). Le seuil, calibré pour
   des morceaux de 800, ne vaut plus pour des morceaux de 300 — même modèle, même corpus, même code.
   Comparer avec la même expérience hors-ligne sur Solvéo (refus justes 0,80 → 0,40) : le sens de
   l'effet dépend du corpus, seul le fait qu'il y ait un effet est prévisible.

4. **Changer d'embeddings : un refus, une réindexation, puis 81 % de dérive (S2.3, S3.2).**
   Interroger l'index bge-m3 avec nomic est refusé avant tout appel au générateur
   (`IndexModelMismatchError`, dimensions 1 024 contre 768). Après réindexation (43 s), même
   générateur, même prompt : 34 réponses sur 42 changent, dont 5 statuts et 9 jeux de sources ;
   bonne source 0,94 → 0,87, refus justes 0,90 → 0,70. Le générateur n'a rien fait de différent :
   il répond à partir de ce qu'on lui donne.

5. **Changer de générateur : le même index, et pourtant tout change (S2.3, S3.3, S4.1).** Sur
   12 questions, `qwen3:4b` à la place de `llama3.2:3b` : `index_id` identique, aucune
   réindexation, 100 % de dérive, latence 1,8 → 34 s. Avec le budget de réflexion initial
   (1 200 jetons), 5 réponses sur 12 reviennent **vides** : qwen3 réfléchit sur ~2 300 jetons pour
   un prompt de 3 300 caractères, épuise son budget, et le garde-fou de forme rejette la sortie
   vide deux fois (« non sourcé »). Avec 3 000 jetons : 12 réponses sur 12, même latence, 0 rejet.
   La correction tient dans `config/ai_service.toml` (`thinking_tokens`), côté service IA, sans
   toucher à l'application : c'est la couche anticorruption qui absorbe la particularité du
   modèle. « Une ligne de configuration », oui — mais laquelle, et de quel côté de la frontière ?

6. **Le banc qwen3 du corpus réel (dossier `2026-09-11-service-public-qwen/`) mesuré avant cette
   correction** montre l'état « budget 1 200 » à l'échelle des 42 questions : 38 % de réponses non
   sourcées derrière chaque modèle d'embeddings, 1,6 tentative par question, latence médiane 50 s
   (p90 60 s). Lu seul, ce rapport dirait « qwen3 est mauvais » ; lu avec l'expérience 5, il dit
   « le budget de réflexion était sous-dimensionné pour ce corpus ». Un chiffre sans sa
   configuration ne se lit pas.

7. **Une panne réseau passagère a été absorbée pendant l'indexation** : Ollama a répondu une fois
   « HTTP 400 … sufficient buffer space » (socket Windows épuisé) au milieu des 3 505 morceaux ;
   le décorateur de nouvelles tentatives a réessayé une fois et l'indexation a abouti sans
   intervention. Sans ce décorateur, 80 s d'indexation à refaire à la main. C'est ce que S4.1
   appelle un garde-fou : une seconde chance pour une erreur *transitoire* (5xx, injoignable),
   jamais pour une requête refusée (modèle inconnu).

## Ce que ces expériences ne disent pas

- Elles n'ont pas de vérité terrain complète : « bonne source » et « refus justes » sont calculés
  sur les attendus du jeu de questions, « reformulation » ne juge pas le contenu.
- Un seul passage par expérience (sauf `stabilite`) : un écart de quelques points entre deux
  colonnes est dans le bruit de la ligne de base.
- Une seule machine, avec GPU ; les latences sur processeur seul ne sont pas mesurées.
