# Contrat HTTP du service IA — version 1

Le service IA ne connaît rien au métier. Il reçoit des textes, renvoie des vecteurs ou du texte, et **dit toujours quel modèle a réellement servi**. C'est ce champ qui permet à l'application de détecter qu'un index n'est plus compatible.

Version 1, figée le 11/09/2026. Deux routes seulement : le service produit des vecteurs et du texte,
l'index reste côté application (ADR 0002). L'exploration antérieure de l'équipe (`AssistantQR`)
hébergeait l'index dans le service (`/index/reset`, `/index/upsert`, `/index/search` avec filtre
d'accès) ; ce contrat-ci ne reprend pas ces routes, volontairement : la règle d'accès reste dans le
domaine et l'index reste une donnée que l'application sait décrire (`status`). Tout ajout de route
passe par une nouvelle version (`/v2/`).

## `GET /health`

```json
{"status": "ok", "contract_version": "1"}
```

## `GET /v1/models`

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

Pour un 502, `retryable` dit si réessayer peut réussir : `true` pour un moteur injoignable ou
surchargé, `false` pour un modèle absent ou une réponse que le moteur renverra toujours pareille.
Les clients ne réessaient que les 5xx qui ne portent pas `"retryable": false`.

| HTTP | `code` | Cause |
|---|---|---|
| 400 | `invalid_request` | champ manquant, type ou valeur invalide |
| 404 | `unknown_model` | alias absent de la configuration |
| 404 | `not_found` | route inconnue |
| 405 | `method_not_allowed` | méthode autre que GET (`/health`, `/v1/models`) ou POST (`/v1/embeddings`, `/v1/generate`) |
| 502 | `backend_error` | le moteur (Ollama, serveur OpenAI-compatible…) est injoignable ou a échoué ; avec Ollama, aussi : empreinte du modèle introuvable (`retryable: false`) ou changée pendant l'appel (`retryable: true`) |
| 500 | `internal_error` | erreur imprévue |

## Évolution du contrat

Toute modification incompatible (champ renommé, sémantique changée) passe par un nouveau préfixe (`/v2/…`). Les tests `tests/contract/` vérifient ce que l'application envoie et attend ; `tests/ai_service/test_server.py` vérifie ce que le service accepte et renvoie.
