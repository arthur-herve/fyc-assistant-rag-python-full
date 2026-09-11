# ADR 0004 — Le seuil de pertinence est calibré par modèle d'embeddings et par corpus

**Date** : 11/09/2026 · **Statut** : acceptée

## Contexte

Le refus « je n'ai trouvé aucun document » est une règle métier : sans passage au-dessus d'un
seuil de similarité, on n'appelle pas le modèle. Mais les scores ne se comparent pas d'un modèle
à l'autre (mesuré : hors corpus à 0,64 pour `nomic`, 0,41 pour `bge-m3` sur Solvéo), ni d'un
corpus à l'autre (`bge-m3` : 0,46 sur 15 morceaux, 0,65 sur 3 505).

## Décision

Le seuil est une table `[retrieval.min_score]` indexée par alias de modèle, dans le fichier de
configuration **du corpus** (`app.toml` pour Solvéo, `app-ollama.toml` pour Service-Public). Le
banc d'essai propose un seuil par modèle (`suggested_threshold`) sur un jeu de calibration, et
un jeu de validation distinct sert à le vérifier.

## Conséquences

- Changer de modèle d'embeddings ou de corpus oblige à recalibrer un paramètre métier : exemple
  concret du principe CACE (S3.2).
- Le seuil entre dans l'empreinte de configuration des instantanés ; un changement de seuil est
  visible à côté de la dérive qu'il provoque.
- Limite assumée : un seuil global par modèle est grossier ; un seuil par question ou un
  re-classement serait plus fin, mais hors sujet.

## Écarté

Un seuil unique dans le code (0,35) : c'est ce que la première version faisait, et c'est
exactement le « détail » qui a cassé au premier vrai modèle.
