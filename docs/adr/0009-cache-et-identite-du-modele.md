# ADR 0009 — Le cache d'embeddings dérive de l'index ; l'identité d'un modèle Ollama est relue à chaque appel

**Date** : 21/09/2026 · **Statut** : acceptée · complète les ADR 0003 et 0008 · erratum du 01/10/2026 (en fin de document)

## Contexte

La revue de code du 21/09/2026 a trouvé deux trous dans la promesse de l'ADR 0003 (« un index
construit avec un autre modèle est une erreur ») :

- le cache d'embeddings (ADR 0008), actif dans les configurations livrées, était indexé par le seul
  texte et gardait aussi les vecteurs des documents. Dans un processus long (`serve`), une
  réindexation après un changement de modèle derrière l'alias relisait les anciens vecteurs :
  l'index restait sur l'ancien modèle, et l'application refusait toute question nouvelle jusqu'à
  son redémarrage ;
- le service IA lisait l'empreinte d'un modèle Ollama une seule fois, au premier appel : un
  `ollama pull` n'était vu qu'au redémarrage du service, et un échec de `/api/tags` donnait un
  identifiant sans empreinte.

## Décision

- Le cache d'embeddings ne sert que l'**index courant** : il reçoit le manifeste de l'index et se
  vide dès que ce manifeste change (un nouvel index est un nouvel objet manifeste). Il ne garde que
  des vecteurs du modèle et de la dimension de cet index, et ne met jamais les documents en cache.
- Le backend Ollama du service IA relit l'empreinte dans `/api/tags` **avant et après** chaque
  inférence. Pas d'empreinte : erreur 502 définitive (`retryable: false`, le modèle manque) ; une
  empreinte qui change pendant l'appel : erreur 502 passagère. Jamais un identifiant incertain. Les
  noms sont comparés comme Ollama les compare (casse, tag `latest` implicite, registre et espace
  `library` facultatifs).
- Les préfixes d'un alias (`query_prefix`, `document_prefix`) entrent dans l'identifiant renvoyé
  (`…+prefixes-<empreinte>`) : les changer change les vecteurs, donc l'identifiant, donc impose
  de réindexer, quel que soit le moteur.

## Conséquences

- Garanti : le cache ne mélange jamais deux index. Il se vide à chaque réindexation, même quand
  l'identifiant du modèle ne change pas (préfixes modifiés, moteur qui ne fournit pas d'empreinte) :
  après une réindexation, index et questions viennent du même modèle. Et une question contrôle le
  modèle sur l'index même qu'elle interroge, y compris quand un autre processus le reconstruit
  pendant qu'elle cherche (`SearchPassages` recommence alors).
- Pas garanti : avec le cache, une question déjà posée ne repart pas au service. Elle reste servie,
  de façon cohérente, avec les vecteurs du modèle de l'index ; c'est une question nouvelle, ou
  `status` (qui n'utilise pas le cache), qui révèle un changement de modèle servi. En `serve`, on
  surveille donc `/v1/status`.
- L'identifiant n'est aussi précis que ce que le moteur fournit : empreinte des poids pour Ollama,
  simple nom pour un serveur compatible OpenAI ou sentence-transformers (préfixes compris dans tous
  les cas). Avec un tel moteur, des poids changés sous le même nom ne se voient pas avant la
  réindexation : jusque-là, l'application compare sans le savoir des vecteurs de deux modèles.
  C'est la limite de la garantie ; le cache se vide à chaque réindexation pour ne pas l'aggraver.
- Deux appels à `/api/tags` par inférence Ollama : négligeable en local (quelques millisecondes).

## Écarté

- Une clé de cache par nom de modèle (première version du correctif) : insuffisante quand les
  vecteurs changent sans que le nom change.
- Une clé par modèle *servi* : il faudrait interroger le service à chaque question, ce que le cache
  veut précisément éviter.
- Supprimer le cache : c'est l'exemple de décorateur technique de la séquence 4.1, et la leçon
  « un cache est un artefact dérivé » vaut d'être montrée plutôt qu'évitée.

## Erratum (01/10/2026)

Deux faits énoncés ci-dessus sont faux.

- Contexte : un redémarrage ne suffisait pas. La réindexation faite dans `serve` reprenait les
  anciens vecteurs dans le cache et les écrivait sur le disque : l'index périmé était persisté.
  Après un redémarrage, toutes les questions, même celles déjà posées, restaient refusées (409),
  jusqu'à une réindexation faite par un processus au cache vide (`index` en ligne de commande, ou
  `serve` redémarré). L'artefact dérivé, une fois écrit, survivait au processus qui l'avait produit.
- Conséquences : la garantie « une question contrôle le modèle sur l'index même qu'elle
  interroge » ne tenait pas. `SearchPassages` relisait le manifeste après la recherche : un index
  remplacé puis rétabli pendant la recherche (A, puis B, puis A) passait inaperçu, et la réponse
  citait des passages de B sous la trace de A. Désormais, l'index vérifie son identifiant au moment
  même de la recherche (`search(…, index_id)`, sinon `IndexReplacedError`), et `SearchPassages`
  recommence alors une fois.
