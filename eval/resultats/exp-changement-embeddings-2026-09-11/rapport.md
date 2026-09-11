# Expérience « changement-embeddings » — 2026-09-11 19:22

Configuration `config/app-ollama.toml` · 42 questions de `eval/questions-service-public.json` · corpus `service-public`.

Une seule chose change : le modèle d'embeddings, `bge-m3` → `nomic`. Même corpus, même découpage, même prompt, même générateur ; le seuil est celui configuré pour chaque modèle.

## 1. Sans réindexer : l'application refuse

```
L'index a été construit avec « ollama:bge-m3@790764642607 » (1024 dim.) mais le modèle d'embeddings actuel est « ollama:nomic-embed-text@0a109f422b47 » (768 dim.). Il faut réindexer le corpus.
```

## 2. Après réindexation

| | avant (`bge-m3`) | après (`nomic`) |
|---|---|---|
| modèle servi | ollama:bge-m3@790764642607 | ollama:nomic-embed-text@0a109f422b47 |
| dimension | 1024 | 768 |
| morceaux | 3505 | 3505 |
| indexation (s) | 75.00 | 43.00 |
| seuil | 0.65 | 0.73 |
| répond (répondables) | 1.00 | 0.97 |
| bonne source | 0.94 | 0.87 |
| refus justes (hors corpus) | 0.90 | 0.70 |
| non sourcé | 0.02 | 0.00 |
| fuites d'accès | 0.00 | 0.00 |

## Dérive

| | avant → après |
|---|---|
| questions comparées | 42 |
| réponses modifiées | 34 |
| taux de dérive | 0.81 |
| changements de statut | 5 |
| changements de sources | 9 |
| reformulations | 20 |

## Détail

```
Comparaison : avant → apres

Différences de configuration
  - embedding_model : bge-m3 → nomic
  - embedding_model_id : ollama:bge-m3@790764642607 → ollama:nomic-embed-text@0a109f422b47
  - index_id : ce6ac963f0e1 → f084960ff355
  - min_score : 0.65 → 0.73

Dérive
  questions comparées : 42
  réponses modifiées  : 34
  taux de dérive      : 81%

| Nature | Nombre | Lecture |
|---|---|---|
| statut modifié | 5 | changement de comportement : refus devenu réponse, ou l'inverse |
| sources modifiées | 9 | même décision, autres documents cités |
| texte modifié | 20 | mêmes sources, même décision : reformulation, la dérive la plus bénigne |
| identique | 8 | rien n'a bougé |
| absente d'un des deux | 0 | question présente d'un seul côté |

sp-conges-acquis [sources modifiées]
  avant : answered ['F18260'] « Selon l'article [1], un salarié a droit à 2,5 jours ouvrables par mois de travail effectif »
  après : answered ['F2931'] « Le salarié acquiert un nombre de jours de congés payés arrondi au nombre immédiatement sup »
sp-conges-report [sources modifiées]
  avant : answered ['F12410'] « Non, les congés payés non pris à la fin de la période ne sont pas nécessairement perdus. E »
  après : answered ['F37482'] « Non, les congés payés non pris à la fin de la période ne sont pas nécessairement perdus. S »
sp-conges-fermeture [texte modifié]
  avant : answered ['F1016'] « La durée de la fermeture annuelle de l'entreprise pour congés ne peut pas dépasser 24 jour »
  après : answered ['F1016'] « La fermeture annuelle de l'entreprise pour congés peut durer au plus 24 jours ouvrables [1 »
sp-conges-cdd [sources modifiées]
  avant : answered ['F2931', 'F41'] « Oui, le salarié en CDD a droit aux congés payés, mais le calcul de ces congés est effectué »
  après : answered ['F2931'] « Oui, le salarié en CDD a droit aux congés payés, il bénéficie des mêmes droits à congés pa »
sp-conges-demission [sources modifiées]
  avant : answered ['F21050', 'F2265'] « Oui, vous pouvez démissionner pendant votre congé de maternité. Vous devez avertir votre e »
  après : answered ['F2265', 'F37482'] « Non, vous ne pouvez pas démissionner pendant vos congés payés. Vous devez avertir votre em »
sp-conges-sans-solde [texte modifié]
  avant : answered ['F10431'] « Non, le congé sans solde n'est pas prévu par la législation. [4] 

L'employeur n'est pas o »
  après : answered ['F10431'] « Le congé sans solde n'est pas prévu par la législation, mais il peut être rémunéré dans le »
sp-conges-proche-aidant [texte modifié]
  avant : answered ['F16920'] « Le congé de proche aidant ne peut pas dépasser 1 an sur l'ensemble de la carrière du salar »
  après : answered ['F16920'] « Le congé de proche aidant ne peut pas dépasser 1 an sur l'ensemble de la carrière du salar »
sp-temps-partiel-min [texte modifié]
  avant : answered ['F1915', 'F32428'] « La durée minimale hebdomadaire de travail d'un salarié à temps partiel est de 24 heures, s »
  après : answered ['F1915', 'F32428'] « La durée minimale hebdomadaire de travail d'un salarié à temps partiel est de 24 heures. [ »
sp-duree-legale [sources modifiées]
  avant : answered ['F1911'] « La durée légale du travail par semaine pour un temps plein est de 35 heures. [1] »
  après : answered ['F32428'] « La durée légale du travail par semaine pour un temps plein est de 35 heures. [1, 3] »
sp-duree-max-jour [texte modifié]
  avant : answered ['F1911', 'F2216'] « La durée maximale de travail effectif par jour est de 10 heures pour les salariés âgés d'a »
  après : answered ['F1911', 'F2216'] « La durée maximale quotidienne de travail est de 8 heures [1] et de 10 heures par jour dans »
sp-dimanche [texte modifié]
  avant : answered ['F13887'] « Non, on n'est pas payé double quand on travaille le dimanche. La loi n'impose pas le paiem »
  après : answered ['F13887'] « Non, il n'est pas payé double quand on travaille le dimanche, mais il peut bénéficier d'un »
sp-teletravail-volontaire [texte modifié]
  avant : answered ['F13851'] « Non, l'employeur ne peut pas imposer le télétravail à un salarié sans son accord. En effet »
  après : answered ['F13851'] « Non, le salarié peut refuser le passage en télétravail. Le refus pour le salarié d'accepte »
sp-paternite [texte modifié]
  avant : answered ['F3156'] « Le congé de paternité et d'accueil de l'enfant dure 25 jours calendaires, soit 2 périodes  »
  après : answered ['F3156'] « Le congé de paternité et d'accueil de l'enfant dure 25 jours calendaires, soit 32 jours en »
sp-naissance [texte modifié]
  avant : answered ['F2266'] « La durée du congé de naissance est de 3 jours ouvrables sauf si des dispositions conventio »
  après : answered ['F2266'] « Le congé de naissance dure 3 jours ouvrables pour chaque naissance survenue au foyer, sauf »
sp-grossesse [sources modifiées]
  avant : answered ['F1144', 'F2775'] « Une salariée enceinte n'a pas l'obligation d'informer son employeur de son état de grosses »
  après : answered ['F1144'] « Une salariée enceinte n'a pas l'obligation d'informer son employeur de son état de grosses »
sp-cpf-montant [texte modifié]
  avant : answered ['F10705'] « Le montant de votre compte personnel de formation (CPF) est alimenté de 800 € par année de »
  après : answered ['F10705'] « Selon les informations fournies, votre compte personnel de formation (CPF) est alimenté à  »
sp-stage-gratification [texte modifié]
  avant : answered ['F16734'] « La gratification est obligatoire à partir de 2 mois consécutifs de stage, soit 44 jours à  »
  après : answered ['F16734'] « La gratification est obligatoire à partir de la 309e heure de stage, même si le stage est  »
sp-pmsmp [texte modifié]
  avant : answered ['F14102'] « Une période de mise en situation en milieu professionnel (PMSMP) vous permet de tester vos »
  après : answered ['F14102'] « Une période de mise en situation en milieu professionnel (PMSMP) vous permet de tester vos »
sp-arret-maladie-sorties [texte modifié]
  avant : answered ['F12415'] « Vous devez être présent à votre domicile de 9 h à 11 h et de 14 h à 16 h, y compris les sa »
  après : answered ['F12415'] « Selon les conditions de votre arrêt de travail, vous devez être présent à votre domicile d »
sp-droit-retrait [texte modifié]
  avant : answered ['F1136'] « Oui, si tu exerce ton droit de retrait face à un danger, tu peux être sanctionné par ton e »
  après : answered ['F1136'] « Un salarié peut refuser de travailler s'il pense être exposé à une situation dangereuse si »
sp-titres-restaurant [texte modifié]
  avant : answered ['F21059'] « L'employeur doit financer entre 50 % et 60 % de la valeur du titre-restaurant, tandis que  »
  après : answered ['F21059'] « L'employeur doit financer entre 50 % et 60 % de la valeur du titre-restaurant, tandis que  »
sp-retraite-age [sources modifiées]
  avant : answered ['F1761'] « Selon le passage [2], un salarié né en 1970 peut partir en retraite à partir de 67 ans, ca »
  après : answered ['F21552'] « Un salarié né en 1970 peut partir à la retraite à 67 ans. [4] »
sp-rh-age-minimum [statut modifié]
  avant : answered ['F1649'] « Il est possible de travailler dès l'âge de 16 ans, quel que soit le type de contrat (CDI,  »
  après : no_relevant_source [] « Je n'ai trouvé aucun document accessible qui réponde à cette question. »
sp-rh-abandon-poste [texte modifié]
  avant : answered ['F31209'] « Un salarié qui abandonne son poste dans le secteur privé doit reprendre son travail dans u »
  après : answered ['F31209'] « Le salarié doit reprendre son poste dans un délai de 15 jours calendaires après la demande »
sp-rh-faute-simple [sources modifiées]
  avant : answered ['F1137'] « Oui, le salarié perçoit une indemnité de licenciement s'il remplit les conditions d'ancien »
  après : answered ['F1137', 'F31225'] « Non, le salarié ne perçoit pas d'indemnité de licenciement pour faute simple, car il est l »
sp-rh-promesse [texte modifié]
  avant : answered ['F10403'] « Si l'employeur ne respecte pas une promesse d'embauche unilatérale, il peut être considéré »
  après : answered ['F10403'] « L'employeur qui ne respecte pas une promesse d'embauche unilatérale peut être assimilé à u »
sp-rh-essai-renouvellement [texte modifié]
  avant : answered ['F13919'] « La période d'essai d'un CDI peut être renouvelée si les 3 conditions suivantes sont réunie »
  après : answered ['F13919'] « La période d'essai d'un CDI peut être renouvelée une seule fois, sous réserve des conditio »
sp-dir-greve [texte modifié]
  avant : answered ['F117'] « Une grève doit réunir les 3 conditions suivantes pour être licite : arrêt total du travail »
  après : answered ['F117'] « Une grève doit réunir les 3 conditions suivantes : arrêt total du travail, arrêt collectif »
sp-dir-prudhommes [sources modifiées]
  avant : answered ['F1052'] « Une affaire devant le conseil de prud'hommes commence par une requête auprès du conseil de »
  après : answered ['F1052', 'F2360'] « Une phase initiale de conciliation a lieu entre vous et votre employeur. 

En cas de conci »
sp-dir-inspection [texte modifié]
  avant : answered ['F107'] « Les missions de l'inspection du travail sont les suivantes : contrôler, conseiller, concil »
  après : answered ['F107'] « Les missions de l'inspection du travail sont les suivantes : contrôler, conseiller, concil »
sp-acces-abandon-refuse [statut modifié]
  avant : no_relevant_source [] « Je n'ai trouvé aucun document accessible qui réponde à cette question. »
  après : answered ['F32428'] « Je n'ai pas trouvé d'information spécifique sur le délai minimum pour reprendre son poste  »
sp-acces-age-refuse [statut modifié]
  avant : unsourced ['F32700'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
  après : no_relevant_source [] « Je n'ai trouvé aucun document accessible qui réponde à cette question. »
sp-hors-difficile-fonction-publique [statut modifié]
  avant : no_relevant_source [] « Je n'ai trouvé aucun document accessible qui réponde à cette question. »
  après : answered ['F2328'] « Un fonctionnaire territorial a 61 jours de congés annuels. [1] »
sp-hors-difficile-impots [statut modifié]
  avant : no_relevant_source [] « Je n'ai trouvé aucun document accessible qui réponde à cette question. »
  après : answered ['F2391'] « Vous devez déclarer vos impôts sur le revenu avant la fin du mois de mars de l'année suiva »
```

## Lecture

- Changer de modèle d'embeddings coûte une réindexation complète (voir la durée) : les vecteurs stockés vivent dans l'espace du modèle qui les a produits.
- Le refus est explicite parce que le service IA renvoie l'identifiant concret du modèle et que l'application le compare au manifeste à chaque question (ADR 0003). Sans cela, à dimension égale, l'index aurait répondu à côté sans rien signaler.
- Après réindexation, le générateur n'a pas changé et pourtant les réponses bougent : il ne répond qu'à partir de ce que la recherche lui donne.
