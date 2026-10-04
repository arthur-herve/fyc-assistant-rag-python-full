# Exercice S2.2 — Construire un cœur métier sans IA

Durée indicative : 45 minutes · Exercice guidé, code à écrire, tests fournis · Corrigé dans `solution/`.

## Ce que vous avez

Le dossier `depart/` est un mini-projet autonome : aucune IA, aucun réseau, aucune dépendance.

```
depart/
  coeur/
    model.py          les entités (Document, Chunk, User, Passage, Answer, AnswerTrace…)  — fourni
    ports.py          ce dont les cas d'usage ont besoin, sans dire comment (Embedder, Generator,
                      VectorIndex, PromptRepository…)                                        — fourni
    errors.py         les erreurs du domaine et de l'application                              — fourni
    index_corpus.py   le cas d'usage d'indexation, à lire comme exemple                       — fourni
    access.py         règle métier n° 1 : qui peut lire quoi                                  — À ÉCRIRE
    citations.py      règle métier n° 2 : toute réponse cite ses sources                      — À ÉCRIRE
    ask_question.py   le cas d'usage central : répondre à une question                        — À ÉCRIRE
  fakes.py            des doubles pour chaque port : embeddings « par mots-clés », générateur
                      scripté, index en mémoire, corpus en liste, prompts fixes               — fourni
  tests/              19 tests qui décrivent le comportement attendu                           — fournis
```

Lancer les tests depuis `depart/` :

```bash
python -m unittest discover -s tests -t .
```

Au départ, 19 tests échouent sur `NotImplementedError`. À la fin, tout est vert — **sans avoir
appelé un seul modèle**.

## Ce qu'on vous demande

1. `coeur/access.py` — `AccessPolicy.can_read` : un morceau marqué `tous` est lisible par tout le
   monde ; sinon il faut un groupe en commun. Tests : `tests/test_domain.py::AccessPolicyTest`.
2. `coeur/citations.py` — `check_citations(text, passage_count)` : reconnaît `[1]`, `[2, 3]`,
   `[2,3]` ; renvoie les numéros valides sans doublon et les numéros invalides ; un nombre trop
   grand (des milliers de chiffres, que `int()` refuse de convertir) est une citation invalide,
   pas une exception. Tests : `tests/test_domain.py::CitationsTest`.
3. `coeur/ask_question.py` — `format_passages` puis `AskQuestion.execute`, en suivant le déroulé
   décrit dans le fichier (question vide → index absent → modèle incompatible → recherche filtrée
   par les droits → seuil → prompt → tentatives → réponse sourcée ou refus).
   Tests : `tests/test_ask_question.py` (12 tests).

Ordre conseillé : 1, 2, puis 3 en faisant passer les tests un par un, dans l'ordre du fichier.

## Questions à se poser en chemin (reprises en correction)

- `AskQuestion` importe `AccessPolicy` et `check_citations`, mais jamais `fakes.py` ni une
  bibliothèque HTTP. Comment les tests arrivent-ils quand même à faire répondre le cas d'usage ?
- Pourquoi le test `test_restricted_passages_never_reach_the_prompt` inspecte-t-il le **prompt
  envoyé** au générateur, plutôt que la réponse ?
- Que se passerait-il si la vérification des citations était confiée au prompt (« cite tes
  sources ») au lieu d'être une fonction du domaine ?
- Les tests vérifient `IndexModelMismatchError`. Pourquoi le cas d'usage compare-t-il le modèle à
  **chaque** question, et pas seulement à l'indexation ?

## Corrigé

`solution/` contient les trois fichiers. Copiez-les dans `depart/coeur/` pour vérifier :
19 tests verts. Ils sont identiques, aux imports près, à `assistant/domain/access.py`,
`assistant/domain/citations.py` et `assistant/application/ask_question.py` du dépôt (le fil rouge
délègue en plus la recherche à un cas d'usage `SearchPassages`, réutilisé par le banc d'essai) :
ce que vous venez d'écrire est **le cœur réel** de l'assistant fil rouge, celui qui tourne en
séquence 2.3 derrière de vrais modèles.

Réponses aux questions :

- **Les doubles.** Les tests construisent `AskQuestion` avec des objets de `fakes.py` qui
  respectent les *ports* (`KeywordEmbedder`, `ScriptedGenerator`, `FakeIndex`, `StaticPrompts`).
  Le cas d'usage ne voit que les signatures de `ports.py` : un vrai adaptateur HTTP ou un double
  en mémoire, c'est pareil pour lui. C'est l'inversion des dépendances : le cœur définit ce dont
  il a besoin, l'extérieur s'y conforme.
- **Le prompt plutôt que la réponse.** La règle métier dit qu'un document interdit ne doit jamais
  *atteindre* le modèle, pas seulement ne jamais être cité. Un générateur scripté peut renvoyer
  n'importe quoi ; ce qui compte, c'est ce qu'il a reçu. Le test inspecte donc les requêtes
  enregistrées par `ScriptedGenerator`.
- **Citations par le prompt.** Le prompt est une consigne, pas une garantie : le modèle peut
  l'ignorer, inventer `[7]`, ou citer sans avoir lu. `check_citations` est déterministe, testable
  en microsecondes, et c'est elle qui décide du statut de la réponse. Le prompt tente d'obtenir
  une sortie acceptable ; le domaine vérifie qu'elle l'est.
- **À chaque question.** Le service IA peut changer de modèle entre deux appels (mise à jour des
  poids derrière le même alias, autre machine). L'index, lui, ne change pas. La seule vérité
  disponible est l'identifiant renvoyé *maintenant* par le service, comparé au manifeste de
  l'index construit *avant*. Voir ADR 0003.

## Pour aller plus loin

Écrire un test qui vérifie que la trace (`AnswerTrace`) d'un refus « aucune source pertinente »
contient bien les passages retrouvés et leurs scores : c'est ce qui permettra, en séquence 3.2,
de recalibrer le seuil sans rejouer les questions.
