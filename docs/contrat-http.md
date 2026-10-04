# Contrat HTTP du service IA — version 1

Le service IA ne connaît rien au métier. Il reçoit des textes, renvoie des vecteurs ou du texte, et **dit toujours quel modèle a réellement servi**. C'est ce champ qui permet à l'application de détecter qu'un index n'est plus compatible.

Version 1, figée le 11/09/2026. Deux routes seulement : le service produit des vecteurs et du texte,
l'index reste côté application (ADR 0002). L'exploration antérieure de l'équipe (`AssistantQR`)
hébergeait l'index dans le service (`/index/reset`, `/index/upsert`, `/index/search` avec filtre
d'accès) ; ce contrat-ci ne reprend pas ces routes, volontairement : la règle d'accès reste dans le
domaine et l'index reste une donnée que l'application sait décrire (`status`). Tout ajout de route
passe par une nouvelle version (`/v2/`).

## `GET /health` (ou `HEAD`)

```json
{"status": "ok", "contract_version": "1"}
```

## `GET /v1/models` (ou `HEAD`)

```json
{
  "embedding":  [{"alias": "nomic", "backend": "ollama", "description": "..."}],
  "generation": [{"alias": "qwen3-4b", "backend": "ollama", "description": "..."}]
}
```

## `POST /v1/embeddings`

Requête :

```json
{"model": "nomic", "input_type": "query", "inputs": ["Combien de jours de télétravail ?"]}
```

- `model` : alias déclaré dans `config/ai_service.toml`.
- `input_type` : `"query"` ou `"document"` (défaut). L'application déclare une **intention** ; le service applique les préfixes propres au modèle (`search_query: `, `query: `…).
- `inputs` : 1 à 256 textes.

Réponse :

```json
{
  "model": "ollama:nomic-embed-text@0a109f422b47+prefixes-03aa22a9",
  "alias": "nomic",
  "dimension": 768,
  "vectors": [[0.012, -0.034, ...]],
  "duration_ms": 41
}
```

`model` est l'identifiant concret : empreinte du modèle quand le moteur la fournit (Ollama : relue avant et après chaque inférence), et empreinte des préfixes quand l'alias en déclare (`+prefixes-…`), car ils changent les vecteurs (ADR 0009). Si l'équipe qui exploite le service met à jour le modèle derrière le même alias, `model` change, et l'application refuse de chercher dans l'ancien index avec des vecteurs du nouveau modèle.

## `POST /v1/generate`

Requête :

```json
{
  "model": "qwen3-4b",
  "system": "Tu es l'assistant documentaire…",
  "prompt": "Passages : … Question : …",
  "temperature": 0.2,
  "max_tokens": 400,
  "seed": null
}
```

Réponse :

```json
{"model": "ollama:qwen3:4b@a383baf4993b", "alias": "qwen3-4b", "text": "Deux jours par semaine [1].", "duration_ms": 5230}
```

Le prompt est construit **côté application** : il dépend du métier (numérotation des passages, obligation de citer). Le service ne fait que le transmettre.

`max_tokens` est le budget de la **réponse**. Pour un modèle à réflexion (`think = true` dans `config/ai_service.toml`), le service ajoute un budget de réflexion (`thinking_tokens`) que l'application ne voit pas : la réponse peut donc dépasser `max_tokens` si le modèle réfléchit peu. C'est la validation de forme côté application (`max_output_chars`) qui borne la longueur montrée à l'utilisateur.

## Erreurs

Toujours au format :

```json
{"error": {"code": "unknown_model", "message": "modèle de génération inconnu : gpt-9 (…)"}}
```

Pour un 502, le champ `retryable` de l'objet `error` dit si réessayer peut réussir : `true` pour un
moteur injoignable, surchargé ou coupé en pleine réponse, `false` pour un modèle absent ou une
réponse que le moteur renverra toujours pareille. Les clients ne réessaient que les 5xx qui ne
portent pas `"retryable": false`.

```json
{"error": {"code": "backend_error", "message": "empreinte de bge-m3 introuvable dans … (ollama pull bge-m3) ?", "retryable": false}}
```

| HTTP | `code` | Cause |
|---|---|---|
| 400 | `invalid_request` | champ manquant, type ou valeur invalide ; corps qui n'est pas un objet JSON strict (clé en double, chaîne avec un surrogate UTF-16 isolé, `NaN` ou `Infinity`, entier de plus de 4300 chiffres, plus de 64 niveaux d'imbrication) ; `Content-Length` invalide, ou absent d'un envoi en morceaux (`Transfer-Encoding`, que le service ne lit pas) ; ligne de requête illisible ; deux `Content-Length` différents (répété à l'identique, il est fondu en un) |
| 404 | `unknown_model` | alias absent de la configuration |
| 404 | `not_found` | route inconnue, quelle que soit la méthode |
| 405 | `method_not_allowed` | route connue, autre méthode que les siennes (GET et HEAD pour `/health` et `/v1/models`, POST pour `/v1/embeddings` et `/v1/generate`) : l'en-tête `Allow` les donne (`GET, HEAD` ou `POST`), le message aussi (« permises : … ») ; HEAD reçoit les en-têtes de GET, sans corps |
| 413 | `payload_too_large` | corps annoncé de plus de 16 Mio (16 777 216 octets) sur `/v1/embeddings` ou `/v1/generate` : refusé sans être lu |
| 414, 431, 505 | `invalid_request` | requête refusée avant d'être lue : ligne de requête trop longue (414), en-têtes trop longs ou trop nombreux (431), version HTTP non prise en charge (505) |
| 502 | `backend_error` | le moteur (Ollama, serveur OpenAI-compatible…) est injoignable ou a échoué : réponse coupée ou qui n'arrive plus, même au milieu d'une réponse d'erreur du moteur (`retryable: true`), réponse qui n'est pas du HTTP ou pas le JSON attendu (`retryable: false`), refus HTTP 4xx (`retryable: false`, sauf 408 et 429) ; avec Ollama, aussi : empreinte du modèle introuvable (`retryable: false`) ou changée pendant l'appel (`retryable: true`) |
| 500 | `internal_error` | erreur imprévue |

Le service répond en HTTP/1.0 : une connexion par requête, fermée après la réponse. Une
connexion où le client n'envoie plus rien pendant 30 s (requête ou corps attendu) est fermée sans
réponse. Sur un port déjà pris, `python -m ai_service` s'arrête sur « Erreur : impossible d'écouter
(port déjà pris, adresse inconnue ou non autorisée) — … », code 1.

Côté application, une erreur n'est lue que si son corps est du JSON strict et `error.message` un
texte ; sinon le corps est cité tel quel, et un 5xx reste passager. Une réponse 200 qui n'est pas du
JSON strict en UTF-8 (`NaN`, clé en double, entier de plus de 4300 chiffres, plus de 900 niveaux
d'imbrication) est une erreur non passagère : le client ne réessaie pas. C'est aussi le cas si un
champ manque ou n'a pas le type du contrat : `model` et `text` sont des textes Unicode, `dimension`
un entier de 32 bits, `vectors` autant de listes de nombres finis que de textes envoyés, de la
dimension annoncée. Une marque d'ordre des octets en tête de corps est acceptée, en succès comme en
erreur. Un délai dépassé ou une coupure au milieu du corps d'une réponse, d'erreur ou non, est une
erreur passagère (« Service IA injoignable … »), comme avant les en-têtes. Les deux versions de
l'application envoient le même JSON, en UTF-8 : les mêmes champs, avec les mêmes valeurs. Seule
l'écriture peut différer (espaces, accents échappés en `\u00e9` ou écrits tels quels, `1.0` ou `1`
pour un réel entier) : le service lit les valeurs, pas les octets. La réponse à GET /v1/models est
lue comme les autres : UTF-8 strict, marque d'ordre des octets acceptée, JSON strict.

## Évolution du contrat

Toute modification incompatible (champ renommé, sémantique changée) passe par un nouveau préfixe (`/v2/…`). Les tests `tests/contract/` vérifient ce que l'application envoie et attend ; `tests/ai_service/test_server.py` vérifie ce que le service accepte et renvoie.
