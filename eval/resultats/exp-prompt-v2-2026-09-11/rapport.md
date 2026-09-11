# Expérience « prompt-v2 » — 2026-09-11 19:26

Configuration `config/app-ollama.toml` · 42 questions de `eval/questions-service-public.json` · corpus `service-public`.

Une seule chose change : le prompt, `answer` → `answer-v2` (versions `v1+9cf8bd83` → `v2+67815ad6`). Même index `ce6ac963f0e1`, même générateur, même seuil.

## Les deux prompts

| | avant (`answer`) | après (`answer-v2`) |
|---|---|---|
| version tracée | v1+9cf8bd83 | v2+67815ad6 |
| longueur moyenne des réponses | 338 | 52 |
| répond (répondables) | 0.97 | 0.94 |
| bonne source | 0.97 | 0.97 |
| refus justes (hors corpus) | 0.90 | 0.90 |
| non sourcé | 0.02 | 0.05 |
| fuites d'accès | 0.00 | 0.00 |

## Dérive

| | avant → après |
|---|---|
| questions comparées | 42 |
| réponses modifiées | 33 |
| taux de dérive | 0.79 |
| changements de statut | 3 |
| changements de sources | 3 |
| reformulations | 27 |

## Détail

```
Comparaison : avant → apres

Différences de configuration
  - prompt : answer → answer-v2
  - prompt_version : v1+9cf8bd83 → v2+67815ad6

Dérive
  questions comparées : 42
  réponses modifiées  : 33
  taux de dérive      : 79%

| Nature | Nombre | Lecture |
|---|---|---|
| statut modifié | 3 | changement de comportement : refus devenu réponse, ou l'inverse |
| sources modifiées | 3 | même décision, autres documents cités |
| texte modifié | 27 | mêmes sources, même décision : reformulation, la dérive la plus bénigne |
| identique | 9 | rien n'a bougé |
| absente d'un des deux | 0 | question présente d'un seul côté |

sp-conges-acquis [texte modifié]
  avant : answered ['F18260'] « Un salarié a droit à 2,5 jours ouvrables par mois de travail effectif (soit 30 jours ouvra »
  après : answered ['F18260'] « 2,5 jours ouvrables par mois de travail effectif (soit 30 jours ouvrables de congés annuel »
sp-conges-report [texte modifié]
  avant : answered ['F12410'] « Non, les congés payés non pris à la fin de la période ne sont pas nécessairement perdus. E »
  après : answered ['F12410'] « Non [4]. »
sp-conges-fermeture [texte modifié]
  avant : answered ['F1016'] « La durée de la fermeture annuelle de l'entreprise pour congés ne peut pas dépasser 24 jour »
  après : answered ['F1016'] « 24 jours ouvrables [1, 2, 3, 4]. »
sp-conges-cdd [sources modifiées]
  avant : answered ['F2931', 'F41'] « Oui, le salarié en CDD a droit aux congés payés, qui sont calculés en fonction du nombre d »
  après : answered ['F2931'] « Oui [1, 2, 3] »
sp-conges-demission [texte modifié]
  avant : answered ['F2265'] « Oui, vous pouvez démissionner pendant votre congé de maternité. Vous devez avertir votre e »
  après : answered ['F2265'] « Oui [1]. »
sp-conges-sans-solde [texte modifié]
  avant : answered ['F10431'] « Le congé sans solde n'est pas prévu par la législation, mais il peut être prévu par une co »
  après : answered ['F10431'] « Non [1, 2, 3, 4]. »
sp-conges-proche-aidant [statut modifié]
  avant : answered ['F16920'] « Le congé de proche aidant ne peut pas dépasser 1 an sur l'ensemble de la carrière du salar »
  après : unsourced ['F16920'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
sp-temps-partiel-min [texte modifié]
  avant : answered ['F1915', 'F32428'] « La durée minimale hebdomadaire de travail d'un salarié à temps partiel est de 24 heures, s »
  après : answered ['F1915', 'F32428'] « 24 heures [1, 2, 3, 4] »
sp-duree-legale [texte modifié]
  avant : answered ['F1911'] « La durée légale de travail par semaine pour un temps plein est de 35 heures. [1, 2] »
  après : answered ['F1911'] « 35 heures [1, 2]. »
sp-duree-max-jour [texte modifié]
  avant : answered ['F1911'] « La durée maximale de travail effectif au maximum par jour est de 10 heures pour un salarié »
  après : answered ['F1911'] « 10 heures [1]. »
sp-heures-sup [texte modifié]
  avant : answered ['F2391'] « Les huit premières heures supplémentaires de la semaine sont majorées de 25 % [1]. 

Ces h »
  après : answered ['F2391'] « 25 % [1] »
sp-dimanche [texte modifié]
  avant : answered ['F13887'] « Non, on n'est pas payé double quand on travaille le dimanche. La loi n'impose pas le paiem »
  après : answered ['F13887'] « Non [1, 2, 3, 4] »
sp-teletravail-volontaire [texte modifié]
  avant : answered ['F13851'] « Non, l'employeur ne peut pas imposer le télétravail à un salarié sans son accord, à moins  »
  après : answered ['F13851'] « Non [1, 2, 3, 4]. »
sp-paternite [sources modifiées]
  avant : answered ['F3156'] « Le congé de paternité et d'accueil de l'enfant dure 25 jours calendaires, soit 2 périodes  »
  après : answered ['F12647', 'F3156'] « 25 jours calendaires. [1], [2], [3], [4] »
sp-naissance [texte modifié]
  avant : answered ['F2266'] « La durée du congé de naissance est de 3 jours ouvrables sauf si des dispositions conventio »
  après : answered ['F2266'] « 3 jours ouvrables. [1] »
sp-maternite [texte modifié]
  avant : answered ['F2265'] « La durée du congé de maternité pour un premier enfant est de 16 semaines, composée de 6 se »
  après : answered ['F2265'] « 16 semaines [1], [3]. »
sp-grossesse [texte modifié]
  avant : answered ['F1144', 'F2775'] « Une salariée enceinte n'a pas l'obligation d'informer son employeur de son état de grosses »
  après : answered ['F1144', 'F2775'] « Non [1, 2, 3, 4]. »
sp-cpf-montant [texte modifié]
  avant : answered ['F10705'] « Selon les informations fournies, votre compte personnel de formation (CPF) est alimenté de »
  après : answered ['F10705'] « 800 € par année, jusqu'à atteindre 8 000 € maximum, selon les conditions de travail et les »
sp-stage-gratification [texte modifié]
  avant : answered ['F16734'] « La gratification est obligatoire à partir de la durée de 2 mois consécutifs de stage, soit »
  après : answered ['F16734'] « La gratification est obligatoire à partir de 2 mois consécutifs de stage, soit 44 jours à  »
sp-pmsmp [texte modifié]
  avant : answered ['F14102'] « Une période de mise en situation en milieu professionnel (PMSMP) sert à tester vos choix d »
  après : answered ['F14102'] « La période de mise en situation en milieu professionnel (PMSMP) vous permet de tester vos  »
sp-arret-maladie-sorties [texte modifié]
  avant : answered ['F12415'] « Vous devez être présent à votre domicile de 9 h à 11 h et de 14 h à 16 h, y compris les sa »
  après : answered ['F12415'] « 9 h à 11 h et de 14 h à 16 h, y compris les samedis, dimanches et jours fériés [1, 2, 3]. »
sp-droit-retrait [texte modifié]
  avant : answered ['F1136'] « Oui, si un salarié pense avoir un motif raisonnable de croire à un danger possible, il peu »
  après : answered ['F1136'] « Non [1, 2, 3, 4]. »
sp-titres-restaurant [texte modifié]
  avant : answered ['F21059'] « L'employeur doit financer entre 50 % et 60 % de la valeur du titre-restaurant, tandis que  »
  après : answered ['F21059'] « Entre 50 % et 60 % de la valeur du titre-restaurant. [1] »
sp-retraite-age [sources modifiées]
  avant : answered ['F14043'] « Selon les informations fournies, un salarié né en 1970 peut partir en retraite à partir de »
  après : answered ['F14043', 'F1761'] « 64 ans [1, 2] »
sp-rh-age-minimum [texte modifié]
  avant : answered ['F1649'] « Il est possible de travailler dès l'âge de 16 ans, quel que soit le type de contrat (CDI,  »
  après : answered ['F1649'] « 16 ans [1, 3, 4]. »
sp-rh-abandon-poste [texte modifié]
  avant : answered ['F31209'] « Un salarié qui abandonne son poste dans le secteur privé doit reprendre son travail dans u »
  après : answered ['F31209'] « 15 jours calendaires [1] »
sp-rh-faute-simple [texte modifié]
  avant : answered ['F1137'] « Oui, le salarié perçoit une indemnité de licenciement s'il remplit les conditions d'ancien »
  après : answered ['F1137'] « Oui [1, 2] et [4]. »
sp-rh-promesse [texte modifié]
  avant : answered ['F10403'] « Si l'employeur ne respecte pas une promesse d'embauche unilatérale, il peut être condamné  »
  après : answered ['F10403'] « L'employeur qui ne respecte pas une promesse d'embauche unilatérale peut être condamné à v »
sp-rh-essai-renouvellement [texte modifié]
  avant : answered ['F13919'] « Non, la période d'essai d'un CDI ne peut pas être renouvelée, sauf si les 3 conditions sui »
  après : answered ['F13919'] « Non [1, 2, 3, 4]. »
sp-dir-greve [texte modifié]
  avant : answered ['F117'] « Une grève doit réunir les 3 conditions suivantes pour être licite : arrêt total du travail »
  après : answered ['F117'] « Une grève doit réunir les 3 conditions suivantes : arrêt total du travail, arrêt collectif »
sp-dir-prudhommes [statut modifié]
  avant : answered ['F1052', 'F2360'] « Une affaire devant le conseil de prud'hommes commence par une requête auprès du conseil de »
  après : unsourced ['F1052', 'F2360'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
sp-dir-inspection [statut modifié]
  avant : unsourced ['F107'] « Je n'ai pas pu produire de réponse correctement sourcée. Consultez directement les passage »
  après : answered ['F107'] « Les missions de l'inspection du travail sont les suivantes : contrôler, conseiller, concil »
sp-acces-age-refuse [texte modifié]
  avant : answered ['F32700'] « Selon les conditions liées à l'âge, le CEJ est ouvert aux personnes de 16 à 25 ans inclus. »
  après : answered ['F32700'] « 16 ans [1, 1] »
```

## Lecture

- La version du prompt (déclarée + empreinte du fichier) est dans chaque trace : la dérive est attribuable à cette seule modification.
- Ce que le prompt change (forme, longueur, ton) n'est pas ce que le domaine garantit (citations vérifiées, forme validée, droits filtrés) : c'est ce qui permet de le traiter comme une configuration *surveillée comme du métier* (ADR 0005).
- Avec `extractive` (hors-ligne), 0 % de dérive : ce générateur ignore les consignes. Un modèle qui ignore le prompt produit exactement cette signature.
