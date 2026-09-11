# ADR 0002 — L'index vectoriel vit côté application, pas côté service IA

**Date** : 11/09/2026 · **Statut** : acceptée

## Contexte

L'exploration antérieure d'un membre de l'équipe (`AssistantQR`) hébergeait l'index dans le
service Python (`/index/reset`, `/index/upsert`, `/index/search` avec un filtre d'accès). Il
fallait choisir.

## Décision

L'index (`data/index*.json`) est un adaptateur de l'application, derrière le port `VectorIndex`.
Le service IA ne fait que deux choses : des vecteurs et du texte.

## Conséquences

- La règle métier « filtrer par droits d'accès » reste dans le domaine (`AccessPolicy`) et
  s'applique par un prédicat passé à la recherche : elle est testable sans réseau (ADR 0006).
- Le manifeste de l'index (`IndexManifest`) est une donnée de l'application : empreinte du corpus,
  découpage, modèle concret, date. `status` peut le confronter à l'état courant (S4.2).
- Le service IA reste sans état : on peut le redémarrer, le remplacer, le partager entre
  applications.
- Coût : l'application doit envoyer tous les morceaux au service pour les vectoriser, et
  refaire une recherche exhaustive en Python (≈ 0,3 s pour 3 500 morceaux : suffisant ici,
  limite à nommer en S4.3).

## Écarté

Index côté service IA : moins de transferts, filtrage possible avant le top-k (pré-filtrage),
mais la règle d'accès migre dans un composant technique hors de portée des tests du domaine, et
l'index devient invisible pour l'application qui en dépend pourtant. Le dilemme pré/post-filtrage
reste une démonstration optionnelle.
