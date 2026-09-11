# Expérience « stabilite » — 2026-09-11 19:29

Configuration `config/app-ollama.toml` · 42 questions de `eval/questions-service-public.json` · corpus `service-public`.

Rien ne change entre les passages : même index `ce6ac963f0e1`, même générateur `ollama:llama3.2:3b@a80c4f17acd5`, même prompt, même seuil, température 0.2, graine aucune.

## Chaque passage comparé au premier

| | passage 1 → 2 | passage 1 → 3 |
|---|---|---|
| questions comparées | 42 | 42 |
| réponses modifiées | 30 | 28 |
| taux de dérive | 0.71 | 0.67 |
| changements de statut | 0 | 0 |
| changements de sources | 6 | 5 |
| reformulations | 24 | 23 |

**Dérive moyenne à configuration constante : 0.691** (0 changement(s) de statut sur 2 comparaison(s)).

## Statuts par question

| Question | passage 1 | passage 2 | passage 3 |
|---|---|---|---|
| sp-conges-acquis | answered | answered | answered |
| sp-conges-report | answered | answered | answered |
| sp-conges-fermeture | answered | answered | answered |
| sp-conges-cdd | answered | answered | answered |
| sp-conges-demission | answered | answered | answered |
| sp-conges-sans-solde | answered | answered | answered |
| sp-conges-proche-aidant | answered | answered | answered |
| sp-temps-partiel-min | answered | answered | answered |
| sp-duree-legale | answered | answered | answered |
| sp-duree-max-jour | answered | answered | answered |
| sp-heures-sup | answered | answered | answered |
| sp-dimanche | answered | answered | answered |
| sp-teletravail-volontaire | answered | answered | answered |
| sp-paternite | answered | answered | answered |
| sp-naissance | answered | answered | answered |
| sp-maternite | answered | answered | answered |
| sp-grossesse | answered | answered | answered |
| sp-cpf-montant | answered | answered | answered |
| sp-stage-gratification | answered | answered | answered |
| sp-pmsmp | answered | answered | answered |
| sp-arret-maladie-sorties | answered | answered | answered |
| sp-droit-retrait | answered | answered | answered |
| sp-titres-restaurant | answered | answered | answered |
| sp-retraite-age | answered | answered | answered |
| sp-rh-age-minimum | answered | answered | answered |
| sp-rh-abandon-poste | answered | answered | answered |
| sp-rh-faute-simple | answered | answered | answered |
| sp-rh-promesse | answered | answered | answered |
| sp-rh-essai-renouvellement | answered | answered | answered |
| sp-dir-greve | answered | answered | answered |
| sp-dir-prudhommes | answered | answered | answered |
| sp-dir-inspection | answered | answered | answered |
| sp-acces-abandon-refuse | no_relevant_source | no_relevant_source | no_relevant_source |
| sp-acces-age-refuse | answered | answered | answered |
| sp-acces-greve-refuse | no_relevant_source | no_relevant_source | no_relevant_source |
| sp-acces-prudhommes-refuse | no_relevant_source | no_relevant_source | no_relevant_source |
| sp-hors-capitale | no_relevant_source | no_relevant_source | no_relevant_source |
| sp-hors-recette | no_relevant_source | no_relevant_source | no_relevant_source |
| sp-hors-permis | no_relevant_source | no_relevant_source | no_relevant_source |
| sp-hors-parking | no_relevant_source | no_relevant_source | no_relevant_source |
| sp-hors-difficile-fonction-publique | no_relevant_source | no_relevant_source | no_relevant_source |
| sp-hors-difficile-impots | no_relevant_source | no_relevant_source | no_relevant_source |

## Lecture

- C'est la mesure de base de la séquence 3.1 : un test par assertion exacte sur ces réponses échouerait au hasard. On teste donc une *proportion* (taux de réponses sourcées, de refus justes) avec une tolérance, et on documente la probabilité de faux échec.
- Les reformulations sont attendues ; les changements de statut (⚠) sont ce qu'un test statistique doit borner.
- Une graine réduit la variabilité pour un même modèle et un même moteur ; elle ne garantit rien d'un modèle à l'autre, ni d'une version d'Ollama à l'autre.
