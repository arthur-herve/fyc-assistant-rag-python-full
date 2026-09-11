# Exercice S4.1 — Ajouter un décorateur de validation de la sortie du modèle

Durée indicative : 45 minutes · Exercice non QCM, code à écrire · Corrigé en fin de document.

## Situation

Vous avez branché un nouveau modèle de génération, `qwen3:4b`. Sur certaines questions, il
« réfléchit » à voix haute avant de répondre, en anglais :

> Okay, let's see. The user is asking how many days of remote work per week. First, I need to
> check the provided passages. Passage [1] says two days per week…

Le cas d'usage `AskQuestion` a accepté ce texte comme une réponse : il contient bien une citation
`[1]`, et la vérification des citations (`domain/citations.py`) ne regarde que cela. Le banc
d'essai affichait 100 % de bonnes sources. Le texte a été montré à l'utilisateur.

## Ce qu'on vous demande

Ajouter une vérification **déterministe** de la forme de la sortie, **sans modifier**
`AskQuestion`, ni l'adaptateur HTTP `HttpGenerator`, ni le service IA.

1. Écrire la règle dans le domaine : `assistant/domain/output_rules.py`, fonction
   `check_output(text, max_chars) -> OutputCheck`, qui signale une réponse vide, trop longue,
   dans une autre langue que le français, ou qui contient un raisonnement déversé.
2. Écrire un **décorateur** du port `Generator` : `assistant/application/guards.py`, classe
   `OutputValidatingGenerator(inner, max_chars)`. Il appelle le générateur enveloppé, vérifie sa
   sortie et lève `ModelOutputRejectedError` (à ajouter dans `application/errors.py`) quand la
   forme est invalide.
3. Faire en sorte qu'une sortie rejetée compte comme une tentative ratée dans `AskQuestion`
   (c'est la seule modification autorisée du cas d'usage : attraper l'erreur, tracer le rejet,
   passer à la tentative suivante).
4. Brancher le décorateur **uniquement** dans `assistant/composition.py`, activé par une clé de
   configuration `[decorators] validate_output = true`.
5. Vérifier que le test d'architecture passe toujours : ni `application/`, ni `interface/`
   n'importent `infrastructure/`.

Les tests fournis (`tests/unit/test_output_rules.py`, `tests/unit/test_decorators.py`) doivent
passer. Point de départ : l'étiquette Git `s4.1-depart` (ces fichiers de test sont présents, le
code manque).

## Questions à se poser en chemin (elles seront reprises en correction)

- Pourquoi la règle « une réponse est en français et ne raisonne pas » est-elle dans le
  **domaine**, alors que le décorateur qui l'applique est dans l'**application** ?
- Pourquoi ne pas mettre cette vérification dans `ai_service/`, là où l'on neutralise déjà les
  balises `<think>` ?
- Que se passe-t-il si vous empilez le décorateur de validation **sous** le décorateur de
  nouvelles tentatives (`RetryingGenerator`) au lieu de le mettre au-dessus ?
- Cette vérification est une heuristique. Donnez une réponse française correcte qu'elle
  pourrait rejeter à tort, et une réponse fautive qu'elle laisserait passer.

## Corrigé

Le corrigé est le code du dépôt à l'étiquette `s4.1-solution` :

| Étape | Fichier | Ce qu'il fait |
|---|---|---|
| 1 | `assistant/domain/output_rules.py` | `check_output` : vide ; longueur ; marqueurs de raisonnement (`<think>`, « okay, let », « the user is asking »…) ; ratio de mots-outils anglais contre français sur les textes d'au moins 8 mots |
| 2 | `assistant/application/guards.py` | `OutputValidatingGenerator.generate` : appelle `inner.generate`, puis `check_output` ; lève `ModelOutputRejectedError(model, text, problems)` |
| 2 | `assistant/application/errors.py` | `ModelOutputRejectedError(ApplicationError)` avec `model`, `text`, `problems` |
| 3 | `assistant/application/ask_question.py` | dans la boucle des tentatives : `except ModelOutputRejectedError as rejected:` → `raw_outputs.append("<rejetée : …> " + texte)`, `continue` |
| 4 | `assistant/composition.py` | `decorate()` : `if options.get("validate_output", True): generator = OutputValidatingGenerator(generator, max_chars)` ; ordre : tentatives → journal → cache → validation |
| 5 | `tests/architecture/test_dependency_rule.py` | `test_only_the_composition_root_knows_the_infrastructure` |

Réponses aux questions :

- **Domaine / application.** La règle dit ce qu'est une réponse acceptable pour le métier
  (langue, longueur, pas de raisonnement) : elle survivrait à un changement complet de pile
  technique, donc elle est dans le domaine, pure et testable en quelques millisecondes. Le
  décorateur, lui, est un mécanisme : il sait qu'il existe un port `Generator` et une boucle de
  tentatives. C'est la même séparation que `check_citations` (domaine) et `AskQuestion`
  (application).
- **Pas dans le service IA.** Le service IA neutralise ce qui appartient au *modèle* (les balises
  `<think>` sont une convention de qwen3). Il ne sait pas ce qu'est une réponse acceptable pour
  *cette* application : une autre application pourrait vouloir des réponses en anglais ou longues.
  Mettre la règle dans le service, c'est faire fuir le métier vers l'infrastructure — l'inverse
  de ce qu'on cherche. Voir ADR 0008.
- **Ordre des décorateurs.** Validation sous les tentatives réseau : une sortie rejetée serait
  prise pour une panne du service et relancée avec le même prompt, en consommant les tentatives
  réseau ; et une vraie panne réseau serait masquée par une erreur de validation. La validation
  est une règle métier : au plus près du cas d'usage, donc à l'extérieur de la pile. (Le cache
  du dépôt ne concerne que les embeddings ; pour un cache de réponses, la même question se
  poserait : on ne mettrait en cache que ce qui a passé la validation.)
- **Limites de l'heuristique.** Rejet à tort possible : une réponse française qui cite un intitulé
  anglais long (« the General Data Protection Regulation… ») ; acceptation à tort : un raisonnement
  déversé *en français* sans les marqueurs listés. C'est la limite annoncée en S3.1 : un test
  déterministe attrape une classe d'erreurs, l'évaluation statistique (banc d'essai, instantanés)
  mesure le reste.

## Pour aller plus loin

Écrire un décorateur `CachedGenerator` et expliquer pourquoi il est plus dangereux qu'un
`CachedEmbedder` (indice : température, tentatives, traçabilité de la version du prompt).
