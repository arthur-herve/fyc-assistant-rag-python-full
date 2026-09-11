# Expérience « cace-decoupage » — 2026-09-11 19:41

Configuration `config/app-ollama.toml` · 42 questions de `eval/questions-service-public.json` · corpus `service-public`.

Une seule chose change : la taille des morceaux. Même corpus, même modèle d'embeddings, même seuil, même prompt, même générateur.

## Les deux index

| | avant (800 / 120) | après (300 / 50) |
|---|---|---|
| morceaux | 3505 | 10466 |
| indexation (s) | 81.20 | 197.80 |
| identifiant de l'index | ce6ac963f0e1 | fd015388f51c |
| répond (répondables) | 1.00 | 0.91 |
| bonne source | 0.94 | 0.86 |
| refus justes (hors corpus) | 0.90 | 0.90 |
| non sourcé | 0.00 | 0.05 |
| fuites d'accès | 0.00 | 0.00 |

## Dérive

| | avant → après |
|---|---|
| questions comparées | 42 |
| réponses modifiées | 33 |
| taux de dérive | 0.79 |
| changements de statut | 3 |
| changements de sources | 10 |
| reformulations | 20 |

## Détail

```
Comparaison : avant → apres

Différences de configuration
  - index_id : ce6ac963f0e1 → fd015388f51c
  - splitter : {'type': 'paragraph', 'max_chars': 800, 'overlap_chars': 120, 'include_title': True} → {'type': 'paragraph', 'max_chars': 300, 'overlap_chars': 50, 'include_title': True}

Dérive
  questions comparées : 42
  réponses modifiées  : 33
  taux de dérive      : 79%

| Nature | Nombre | Lecture |
|---|---|---|
| statut modifié | 3 | changement de comportement : refus devenu réponse, ou l'inverse |
| sources modifiées | 10 | même décision, autres documents cités |
| texte modifié | 20 | mêmes sources, même décision : reformulation, la dérive la plus bénigne |
| identique | 9 | rien n'a bougé |
| absente d'un des deux | 0 | question présente d'un seul côté |

sp-conges-acquis [sources modifiées]
  avant : answered ['F18260', 'F37482'] « Le salarié a droit à 2,5 jours ouvrables par mois de travail effectif, soit 30 jours ouvra »
  après : answered ['F18260'] « Selon les passages, le salarié acquiert 24 jours ouvrables de congés payés par mois de tra »
sp-conges-report [sources modifiées]
  avant : answered ['F12410'] « Non, les congés payés non pris à la fin de la période ne sont pas nécessairement perdus, c »
  après : answered ['F2262'] « Non, selon l'article [4], les congés payés non pris pour cause de maladie ne sont pas perd »
sp-conges-fermeture [texte modifié]
  avant : answered ['F1016'] « La durée de la fermeture annuelle de l'entreprise pour congés ne peut pas dépasser 24 jour »
  après : answered ['F1016'] « La durée de la fermeture annuelle de l'entreprise pour congés peut durer un maximum de 24  »
sp-conges-cdd [sources modifiées]
  avant : answered ['F2931', 'F41'] « Oui, le salarié en CDD a droit aux congés payés, mais le calcul de ces congés est effectué »
  après : answered ['F2931'] « Oui, le salarié en CDD a droit aux congés payés, mais le calcul de ces congés est effectué »
sp-conges-demission [sources modifiées]
  avant : answered ['F2265'] « Oui, vous pouvez démissionner pendant votre congé de maternité. Vous devez avertir votre e »
  après : answered ['F21050', 'F2265'] « Non, vous ne pouvez pas démissionner pendant vos congés payés. Selon [1], le salarié peut  »
sp-conges-sans-solde [texte modifié]
  avant : answered ['F10431'] « Le congé sans solde n'est pas prévu par la législation, mais il peut être accordé par l'em »
  après : answered ['F10431'] « Le congé sans solde n'est pas prévu par la législation [1, 2]. Il n'existe pas de formalis »
sp-conges-proche-aidant [texte modifié]
  avant : answered ['F16920'] « Le congé de proche aidant ne peut pas dépasser 1 an sur l'ensemble de la carrière du salar »
  après : answered ['F16920'] « Le congé de proche aidant ne peut pas dépasser 1 an sur l'ensemble de la carrière du salar »
sp-temps-partiel-min [texte modifié]
  avant : answered ['F1915', 'F32428'] « La durée minimale hebdomadaire de travail d'un salarié à temps partiel est de 24 heures, s »
  après : answered ['F1915', 'F32428'] « La durée minimale hebdomadaire de travail d'un salarié à temps partiel est de 24 heures pa »
sp-duree-legale [texte modifié]
  avant : answered ['F1911'] « La durée légale du travail pour un temps complet du salarié du secteur privé est fixée à 3 »
  après : answered ['F1911'] « La durée légale du travail pour un temps complet du salarié du secteur privé est fixée à 3 »
sp-duree-max-jour [sources modifiées]
  avant : answered ['F1911', 'F316'] « La durée maximale de travail effectif par jour est de 10 heures pour les salariés âgés d'a »
  après : answered ['F1945'] « La durée de travail effectif ne doit pas dépasser 10 heures par jour.

Cette durée est fix »
sp-heures-sup [texte modifié]
  avant : answered ['F2391'] « Les huit premières heures supplémentaires de la semaine sont majorées de 25 %. [1] »
  après : answered ['F2391'] « La majoration s'applique sur les 8 premières heures supplémentaires travaillées dans la mê »
sp-dimanche [texte modifié]
  avant : answered ['F13887'] « Non, on n'est pas payé double quand on travaille le dimanche. La loi n'impose pas le paiem »
  après : answered ['F13887'] « Non, on ne travaille pas le dimanche pour gagner double. Selon les passages, le salarié do »
sp-teletravail-volontaire [texte modifié]
  avant : answered ['F13851'] « Non, l'employeur ne peut pas imposer le télétravail à un salarié sans son accord, à l'exce »
  après : answered ['F13851'] « Non, votre employeur ne peut pas vous imposer le télétravail sans votre accord. En effet,  »
sp-paternite [texte modifié]
  avant : answered ['F3156'] « Le congé de paternité et d'accueil de l'enfant dure 25 jours calendaires pour une naissanc »
  après : answered ['F3156'] « Le congé de paternité et d'accueil de l'enfant dure 25 jours calendaires. [3] »
sp-naissance [sources modifiées]
  avant : answered ['F2266'] « La durée du congé de naissance est fixée à 3 jours ouvrables pour chaque naissance survenu »
  après : answered ['F12647', 'F2266'] « La durée du congé de naissance est fixée à 3 jours ouvrables pour chaque naissance survenu »
sp-maternite [texte modifié]
  avant : answered ['F2265'] « La durée du congé de maternité pour un premier enfant est de 16 semaines, composée de 6 se »
  après : answered ['F2265'] « Pour un premier enfant, la durée du congé de maternité est de 16 semaines, composée de 6 s »
sp-grossesse [sources modifiées]
  avant : answered ['F1144', 'F2775'] « Une salariée enceinte n'a pas l'obligation d'informer son employeur de son état de grosses »
  après : answered ['F1144'] « Non, une salariée enceinte n'a pas l'obligation d'informer son employeur de son état de gr »
sp-cpf-montant [statut modifié]
  avant : answered ['F10705'] « Selon les passages fournis, le montant de votre compte personnel de formation (CPF) est al »
  après : unsourced ['F10705'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
sp-stage-gratification [texte modifié]
  avant : answered ['F16734'] « La gratification est obligatoire à partir de 2 mois consécutifs de stage, soit 44 jours à  »
  après : answered ['F16734'] « La gratification est obligatoire à partir de la 309e heure de stage, même si le stage est  »
sp-pmsmp [texte modifié]
  avant : answered ['F14102'] « Une période de mise en situation en milieu professionnel (PMSMP) vous permet de tester vos »
  après : answered ['F14102'] « Une période de mise en situation en milieu professionnel (PMSMP) vous permet de tester vos »
sp-arret-maladie-sorties [texte modifié]
  avant : answered ['F12415'] « Si votre arrêt de travail prévoit des sorties autorisées avec restriction d’horaires, vous »
  après : answered ['F12415'] « Selon les règles, vous devez être présent à votre domicile de 9 h à 11 h et de 14 h à 16 h »
sp-droit-retrait [texte modifié]
  avant : answered ['F1136'] « Oui, si un salarié pense avoir un motif raisonnable de croire à un danger possible, il peu »
  après : answered ['F1136'] « Non, le salarié qui exerce son droit de retrait de manière justifiée ne peut pas être sanc »
sp-titres-restaurant [texte modifié]
  avant : answered ['F21059'] « L'employeur doit financer entre 50 % et 60 % de la valeur du titre-restaurant, tandis que  »
  après : answered ['F21059'] « L'employeur doit financer entre 50 % et 60 % de la valeur du titre-restaurant. C'est-à-dir »
sp-retraite-age [sources modifiées]
  avant : answered ['F1761'] « Selon les conditions énoncées dans le passage [2], un salarié né en 1970 peut partir en re »
  après : answered ['F14043'] « Un salarié né en 1970 peut partir à la retraite à partir de 65 ans, car il n'est pas conce »
sp-rh-age-minimum [sources modifiées]
  avant : answered ['F1649'] « Il est possible de travailler dès l'âge de 16 ans, quel que soit le type de contrat (CDI,  »
  après : answered ['F32700'] « Il est possible d'embaucher un jeune dès l'âge de 16 ans, quel que soit le type de contrat »
sp-rh-abandon-poste [statut modifié]
  avant : answered ['F31209'] « Un salarié qui abandonne son poste dans le secteur privé doit reprendre son travail dans u »
  après : unsourced ['F31209'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
sp-rh-faute-simple [texte modifié]
  avant : answered ['F1137'] « Oui, le salarié perçoit une indemnité de licenciement s'il remplit les conditions d'ancien »
  après : answered ['F1137'] « Non, le salarié ne perçoit pas d'indemnité de licenciement s'il est licencié pour faute si »
sp-rh-promesse [texte modifié]
  avant : answered ['F10403'] « Si l'employeur ne respecte pas une promesse d'embauche unilatérale, il peut être considéré »
  après : answered ['F10403'] « L'employeur qui ne respecte pas une promesse d'embauche unilatérale peut être condamné à v »
sp-rh-essai-renouvellement [texte modifié]
  avant : answered ['F13919'] « Non, la période d'essai d'un CDI ne peut pas être renouvelée, car les passages [1], [2], [ »
  après : answered ['F13919'] « Non, la période d'essai d'un CDI n'est pas renouvelable. 

La durée maximale totale de la  »
sp-dir-greve [statut modifié]
  avant : answered ['F117'] « Une grève doit réunir les 3 conditions suivantes : arrêt total du travail, arrêt collectif »
  après : no_relevant_source [] « Je n'ai trouvé aucun document accessible qui réponde à cette question. »
sp-dir-prudhommes [sources modifiées]
  avant : answered ['F1052', 'F2360'] « Une affaire devant le conseil de prud'hommes commence par une requête auprès du conseil de »
  après : answered ['F2360'] « Une affaire devant le conseil de prud'hommes commence par une saisine du conseil de prud'h »
sp-dir-inspection [texte modifié]
  avant : answered ['F107'] « Les missions de l'inspection du travail sont les suivantes : contrôler, conseiller, concil »
  après : answered ['F107'] « Les missions de l'inspection du travail sont les suivantes : contrôler, conseiller et conc »
sp-acces-age-refuse [texte modifié]
  avant : answered ['F32700'] « Selon les conditions liées à l'âge, le CEJ est ouvert aux personnes de 16 à 25 ans inclus. »
  après : answered ['F32700'] « Selon les conditions du contrat d'engagement jeune, il est possible d'embaucher un jeune d »
```

## Lecture

- L'identifiant de l'index change : le découpage fait partie de ce qui définit un index (manifeste), au même titre que le corpus et le modèle.
- Le seuil de pertinence n'a pas été recalibré : des morceaux plus courts donnent des scores différents, donc des refus et des réponses qui bougent sans qu'aucune règle métier n'ait changé. C'est le principe CACE : *changing anything changes everything*.
- Ce que le taux de dérive ne dit pas : laquelle des deux versions répond le mieux. Pour cela, regarder « bonne source » et « refus justes » ci-dessus, question par question.
