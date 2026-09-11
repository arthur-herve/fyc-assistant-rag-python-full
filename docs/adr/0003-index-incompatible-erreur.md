# ADR 0003 — Un index construit avec un autre modèle est une erreur, pas un avertissement

**Date** : 11/09/2026 · **Statut** : acceptée

## Contexte

Les vecteurs d'un index vivent dans l'espace du modèle qui les a produits. Interroger un index
`bge-m3` avec des vecteurs `nomic` ne plante pas toujours : à dimension égale, la recherche rend
des scores plausibles et des résultats sans rapport (panne silencieuse). `AssistantQR` choisissait
d'avertir par défaut et d'échouer en mode strict, pour rendre la panne observable en cours.

## Décision

`AskQuestion` compare, **à chaque question**, l'identifiant et la dimension du modèle renvoyés par
le service au manifeste de l'index, et lève `IndexModelMismatchError`. Aucune option ne la
désactive. C'est la démonstration du verdict « changer le modèle d'embeddings impose de
réindexer ».

## Conséquences

- La panne silencieuse ne peut pas se produire dans l'application ; on la *raconte* (S3.2) au
  lieu de la subir.
- Une mise à jour des poids derrière le même alias (`ollama pull`) change l'empreinte, donc
  l'identifiant, donc déclenche l'erreur : c'est voulu.
- Un `--embedding-model` différent de celui de l'index oblige à réindexer avant de poser une
  question : la commande `status` le dit avant l'erreur.

## Écarté

Avertissement par défaut : plus pédagogique pour montrer la panne, mais un dépôt public
accessible cinq ans ne doit pas contenir un comportement dangereux « pour la démonstration ».
