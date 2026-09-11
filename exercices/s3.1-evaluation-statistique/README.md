# Exercice S3.1 — Un test déterministe et une évaluation statistique

Durée indicative : 45 minutes · Exercice non QCM, code à écrire · Corrigé en fin de document.

## Situation

Le générateur `extractive-bruite` du service IA imite un modèle imparfait : il oublie de citer
sa source 3 fois sur 10 et choisit au hasard entre les deux phrases les plus proches. Avec lui,
la même question ne donne pas toujours la même réponse. Un test qui affirme « la réponse est
*Deux jours par semaine [1]* » échoue au hasard.

Le cas d'usage `AskQuestion` fait deux tentatives au plus, puis renvoie une réponse « non
sourcée » (`AnswerStatus.UNSOURCED`) plutôt qu'un texte sans citation.

## Ce qu'on vous demande

Dans `tests/unit/`, écrire deux tests, sans IA ni réseau (doubles de `tests/fakes.py`) :

1. **Un test déterministe** de la règle métier : *quelle que soit la sortie du modèle, l'assistant
   n'affiche jamais une réponse sans citation valide*. Utiliser `ScriptedGenerator` avec des
   sorties choisies : sans citation, avec un numéro de passage inexistant (`[7]`), vide, avec
   une citation valide. Vérifier le statut et les sources de chaque réponse. Ce test doit passer
   100 fois sur 100.
2. **Une évaluation statistique** du comportement : avec un générateur qui oublie la citation
   avec une probabilité `FORGET_RATE = 0.3` (écrire ce double, aléatoire **non initialisé**),
   poser la même question `TRIALS = 50` fois et vérifier que le taux de réponses sourcées
   dépasse une tolérance `MIN_ANSWER_RATE` que **vous** choisissez.

   Documenter dans le test :
   - le taux attendu (calculé, pas deviné) ;
   - la tolérance retenue ;
   - la probabilité qu'un test échoue alors que le système est correct (« faux échec »).

3. Répondre par écrit : que teste chacun des deux tests, et que ne teste-t-il pas ? Lequel
   doit bloquer une livraison s'il échoue ?

## Indications

- Avec `max_attempts = 2`, une question est sourcée si l'une des deux tentatives cite : la
  probabilité est 1 − 0,3² = 0,91.
- Le nombre de succès sur 50 essais suit une loi binomiale B(50 ; 0,91). La probabilité que le
  taux tombe sous une tolérance *t* se calcule avec `math.comb` ; pas besoin d'autre bibliothèque.
- Un faux échec coûte cher (un développeur relance, doute du test, finit par l'ignorer) : viser
  moins de 0,1 %.

## Corrigé

Le corrigé est dans le dépôt : `tests/unit/test_ask_question.py` (déterministe, tests
`test_refuses_an_unsourced_answer_after_all_attempts`, `test_retries_when_the_model_forgets_to_cite`,
`test_answers_with_cited_sources`) et `tests/unit/test_statistical_evaluation.py` (statistique).

**Choix de la tolérance.** Taux attendu 0,91. Avec `MIN_ANSWER_RATE = 0.75`, le test échoue si
au plus 37 essais sur 50 sont sourcés : P(X ≤ 37) pour X ~ B(50 ; 0,91) vaut **0,00037**, soit
moins d'un faux échec pour 2 700 exécutions. Avec une tolérance à 0,80 (au plus 39 succès), la
probabilité monte à 0,0043 : un faux échec toutes les 230 exécutions, trop pour une intégration
continue. Sans nouvelle tentative (taux 0,70), la tolérance 0,75 échouerait 78 fois sur 100 :
la tolérance dépend donc du *mécanisme* testé (les tentatives), pas seulement du modèle.

**Ce que chaque test teste.**

| | Test déterministe | Évaluation statistique |
|---|---|---|
| Objet | la règle métier : jamais de réponse non sourcée affichée | la *qualité* du comportement : assez de réponses sourcées |
| Entrées | sorties choisies pour couvrir les cas limites | sorties tirées au hasard |
| Résultat | vrai ou faux, reproductible | une proportion, avec une tolérance et un risque de faux échec documentés |
| Ne teste pas | que le modèle réel cite souvent | que la règle tient dans tous les cas (il pourrait rater un cas rare) |
| S'il échoue | **bloque la livraison** : une garantie est cassée | alerte : le modèle, le prompt ou le seuil ont dérivé ; à lire avec les instantanés (`snapshot compare`) |

La frontière « propre » du cours passe ici : entre ce qui est **déterministe** (la règle, testée
par assertion) et ce qui relève de l'**incertitude** (le modèle, évalué par proportion). Le
garde-fou de forme de la séquence 4.1 (`domain/output_rules.py`) est du premier côté ; le banc
d'essai et `tools/experiences/stabilite.py` sont du second.

## Pour aller plus loin

Relancer `python tools/experiences/stabilite.py --runs 3` avec un vrai modèle : la dérive mesurée
à configuration constante est le non-déterminisme réel, celui que la tolérance d'un test
statistique doit absorber. Puis avec `--seed 42` : que devient-elle, et pourquoi ce n'est pas
une garantie ?
