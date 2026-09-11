# Expérience « changement-generateur » — 2026-09-11 19:33

Configuration `config/app-ollama.toml` · 12 questions de `eval/questions-service-public.json` · corpus `service-public`.

Une seule chose change : le modèle de génération, `llama3-2-3b` → `qwen3-4b`. L'index `ce6ac963f0e1` est construit une fois et partagé : **il n'est reconstruit à aucun moment**.

## Les deux générateurs

| | avant (`llama3-2-3b`) | après (`qwen3-4b`) |
|---|---|---|
| modèle servi | ollama:llama3.2:3b@a80c4f17acd5 | ollama:qwen3:4b@359d7dd4bcda |
| durée totale (s) | 16.60 | 400.50 |
| par question (s) | 1.40 | 33.40 |
| réponses non sourcées | 0 | 5 |
| répond (répondables) | 1.00 | 0.58 |
| bonne source | 0.92 | 0.86 |
| refus justes (hors corpus) | — | — |
| non sourcé | 0.00 | 0.42 |
| fuites d'accès | 0.00 | 0.00 |

## Dérive

| | avant → après |
|---|---|
| questions comparées | 12 |
| réponses modifiées | 12 |
| taux de dérive | 1.00 |
| changements de statut | 5 |
| changements de sources | 1 |
| reformulations | 6 |

## Détail

```
Comparaison : avant → apres

Différences de configuration
  - generation_model : llama3-2-3b → qwen3-4b
  - generation_model_id : ollama:llama3.2:3b@a80c4f17acd5 → ollama:qwen3:4b@359d7dd4bcda

Dérive
  questions comparées : 12
  réponses modifiées  : 12
  taux de dérive      : 100%

| Nature | Nombre | Lecture |
|---|---|---|
| statut modifié | 5 | changement de comportement : refus devenu réponse, ou l'inverse |
| sources modifiées | 1 | même décision, autres documents cités |
| texte modifié | 6 | mêmes sources, même décision : reformulation, la dérive la plus bénigne |
| identique | 0 | rien n'a bougé |
| absente d'un des deux | 0 | question présente d'un seul côté |

sp-conges-acquis [statut modifié]
  avant : answered ['F18260', 'F2931', 'F37482'] « Le salarié a droit à 2,5 jours ouvrables par mois de travail effectif, soit 30 jours ouvra »
  après : unsourced ['F18260', 'F2931', 'F37482'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
sp-conges-report [statut modifié]
  avant : answered ['F12410'] « Non, les congés payés non pris à la fin de la période de prise ne sont pas automatiquement »
  après : unsourced ['F12410', 'F2262', 'F37482'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
sp-conges-fermeture [texte modifié]
  avant : answered ['F1016'] « La durée de la fermeture annuelle de l'entreprise pour congés ne peut pas dépasser 24 jour »
  après : answered ['F1016'] « La durée de la fermeture annuelle de l'entreprise pour congés ne peut pas être supérieure  »
sp-conges-cdd [statut modifié]
  avant : answered ['F2931'] « Oui, le salarié en CDD a droit aux congés payés. Le salarié en CDD bénéficie des mêmes dro »
  après : unsourced ['F2931', 'F41'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
sp-conges-demission [sources modifiées]
  avant : answered ['F2265'] « Oui, vous pouvez démissionner pendant vos congés payés. Vous devez avertir votre employeur »
  après : answered ['F21050'] « Les passages fournis ne traitent pas explicitement de la possibilité de démissionner penda »
sp-conges-sans-solde [texte modifié]
  avant : answered ['F10431'] « Non, le congé sans solde n'est pas prévu par la législation. [4] 

L'employeur n'est pas o »
  après : answered ['F10431'] « Non, le congé sans solde n'est pas prévu par la loi [1]. »
sp-conges-proche-aidant [texte modifié]
  avant : answered ['F16920'] « Le congé de proche aidant ne peut pas dépasser 1 an sur l'ensemble de la carrière du salar »
  après : answered ['F16920'] « Le congé de proche aidant ne peut pas dépasser 1 an sur l'ensemble de la carrière du salar »
sp-temps-partiel-min [texte modifié]
  avant : answered ['F1915', 'F32428'] « La durée minimale hebdomadaire de travail d'un salarié à temps partiel est de 24 heures, e »
  après : answered ['F1915', 'F32428'] « La durée minimale hebdomadaire de travail d'un salarié à temps partiel est de 24 heures pa »
sp-duree-legale [texte modifié]
  avant : answered ['F1911'] « La durée légale du travail pour un temps complet du salarié du secteur privé est fixée à 3 »
  après : answered ['F1911'] « La durée légale du travail par semaine pour un temps plein est de 35 heures [1]. »
sp-duree-max-jour [statut modifié]
  avant : answered ['F1911', 'F316'] « La durée maximale quotidienne de travail est de 10 heures pour les salariés âgés d'au moin »
  après : unsourced ['F1911', 'F2216', 'F316'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
sp-heures-sup [texte modifié]
  avant : answered ['F2391'] « Les huit premières heures supplémentaires de la semaine sont majorées de 25 % [1]. 

Ces h »
  après : answered ['F2391'] « La majoration applicable aux huit premières heures supplémentaires de la semaine est de 25 »
sp-dimanche [statut modifié]
  avant : answered ['F13887'] « Non, on n'est pas payé double quand on travaille le dimanche. La loi n'impose pas le paiem »
  après : unsourced ['F13887'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
```

## Lecture

- Aucune donnée stockée n'a changé : même index, même manifeste. « Le générateur est un détail » est vrai pour les données.
- Mais les textes, la latence et le nombre de sorties rejetées par le garde-fou de forme changent : le prompt est réglé pour un modèle, chaque modèle a ses manies (longueur, langue, raisonnement). Une ligne de configuration, oui ; sans conséquence, non.
