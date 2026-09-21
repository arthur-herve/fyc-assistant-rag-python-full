# Corpus documentaires

Deux corpus, choisis par `[corpus] directory` dans le fichier de configuration.

| Dossier | Contenu | Usage |
|---|---|---|
| `solveo/` | 9 documents **fictifs** (entreprise Solvéo : RH et informatique), dont 3 à accès restreint | tests automatisés, démarrage hors-ligne, exercices courts (`config/app.toml`) |
| `service-public/` | 322 fiches **réelles** de Service-Public.gouv.fr, thème « Travail - Formation », secteur privé | fil rouge avec de vrais modèles, banc d'essai, expériences (`config/app-ollama.toml`) |

Format commun : un fichier Markdown par document, avec un en-tête entre deux lignes `---`
lu par `assistant/infrastructure/markdown_corpus.py` :

```markdown
---
id: identifiant-unique
titre: Titre du document
groupes: tous
---
Texte du document…
```

`groupes` est obligatoire : `tous` pour un document public, sinon des groupes en minuscules
séparés par des virgules (`rh, direction`), sans commentaire sur la ligne. Une ligne oubliée,
vide ou mal formée (majuscule, espace, commentaire) arrête le chargement du corpus : elle ne rend
jamais un document public. Un groupe bien formé mais mal orthographié (`rhh`) passe, et rend le
document illisible par tout le monde : sûr, mais silencieux, à vérifier avec `ask --user`. Les autres
clés de l'en-tête (source, date, thème…) sont conservées à titre documentaire et ignorées par
l'application.

> Un dossier de corpus ne contient que des documents : tout fichier `.md` doit porter l'en-tête, c'est pourquoi l'attribution ci-dessous n'est pas dans `service-public/`.

## `service-public/` : fiches Service-Public.gouv.fr, thème « Travail - Formation », secteur privé

### Source et licence

- **Source** : « Fiches pratiques et ressources de Service-Public.gouv.fr — Particuliers »,
  Direction de l'information légale et administrative (DILA), publiées sur data.gouv.fr :
  <https://www.data.gouv.fr/datasets/fiches-pratiques-et-ressources-de-service-public-gouv-fr-particuliers>
- **Fichier** : `vosdroits-latest.zip` (flux XML, schéma 3.5), téléchargé le **11 septembre 2026** depuis
  <https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip>.
  Date de dernière modification des fiches retenues : 2026-09-11.
- **Licence** : Licence Ouverte / Open Licence 2.0 (Etalab). Mention obligatoire :
  *Service-Public.gouv.fr / DILA*. Chaque fiche porte dans son en-tête l'URL de la fiche
  d'origine (`source`) et sa date de modification (`date`).
- Les fiches sont converties en Markdown par `tools/import_service_public.py`
  (bibliothèque standard). La conversion conserve le texte, les titres, les listes, les tableaux
  et les encadrés ; elle écarte les contacts, services en ligne, références légales et renvois.
  **Le contenu n'est pas modifié.** Pour une version à jour : `python tools/import_service_public.py --download`.

### Périmètre

322 fiches du thème « Travail - Formation », limitées aux dossiers qui concernent un salarié du
secteur privé (les dossiers « fonction publique » et « particulier employeur » sont écartés).

### Droits d'accès : une simulation

Ces fiches sont publiques. Le fil rouge du cours joue un **intranet d'entreprise** où l'assistant
ne montre à chaque salarié que ce qu'il a le droit de lire : les droits ci-dessous sont donc une
règle pédagogique, attribuée par dossier dans `tools/import_service_public.py` (tableau `DOSSIERS`).

| Groupe | Dossiers | Fiches |
|---|---|---|
| `tous` | Conditions de travail · Maladie ou accident du travail · Handicap et emploi · Congés · Contrats de travail · Contrats d'insertion · Retraite · Formation des salariés · Formation des personnes handicapées · Stage en entreprise · Temps de travail · Représentation du personnel | 248 |
| `rh` | Recrutement · Licenciement pour motif personnel · Licenciement économique · Rupture du contrat de travail | 61 |
| `direction` | Conflits du travail | 13 |

Utilisateurs de démonstration (`config/app-ollama.toml`) : `alice` (tous), `bruno` (tous, rh),
`claire` (tous, direction).

### Ce que cette conversion ne fait pas

Les fiches renvoient souvent à d'autres fiches, à des simulateurs ou à des textes de loi : ces liens
sont retirés, une réponse de l'assistant peut donc paraître incomplète par rapport au site. Les
montants et délais sont ceux de la date de téléchargement ; ils changent. Pour toute décision
réelle, consulter <https://www.service-public.gouv.fr>.
