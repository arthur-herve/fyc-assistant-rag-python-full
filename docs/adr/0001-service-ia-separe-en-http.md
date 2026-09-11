# ADR 0001 — Le service IA est un déployable séparé, joint en HTTP

**Date** : 11/09/2026 · **Statut** : acceptée

## Contexte

Le fil rouge doit montrer jusqu'où un modèle d'IA peut être traité comme un « détail
d'infrastructure ». En entreprise, les modèles tournent sur des machines de calcul (GPU) que
l'équipe IA administre, et l'application tourne sur des serveurs applicatifs ordinaires. Une
bibliothèque importée dans l'application aurait masqué cette réalité.

## Décision

Deux programmes : `assistant/` (application) et `ai_service/` (embeddings et génération). Ils ne
partagent aucun code (vérifié par `tests/architecture/`) et ne communiquent que par le contrat
`docs/contrat-http.md`. Le service IA ne connaît ni les documents, ni les droits, ni les citations.

## Conséquences

- Les particularités des modèles (préfixes `query:`/`passage:`, balises `<think>`, budget de
  réflexion) s'arrêtent dans `ai_service/` : l'application ne déclare qu'une intention.
- Le service **renvoie toujours l'identifiant concret du modèle** qui a servi (nom + empreinte
  des poids) : c'est ce qui permet à l'application de détecter un index incompatible (ADR 0003).
- Coûts assumés : un second processus à lancer, une latence réseau, des pannes de service à
  gérer (décorateur de nouvelles tentatives, ADR 0008), un contrat à versionner (`/v1/`).
- **Nuance à enseigner** : la frontière réseau isole la *technologie*, pas le *couplage*. L'index
  côté application dépend toujours du modèle servi côté IA.

## Écarté

Importer un client Ollama ou une bibliothèque d'embeddings dans l'application : plus simple à
installer, mais la dépendance au modèle aurait été invisible dans l'architecture.
