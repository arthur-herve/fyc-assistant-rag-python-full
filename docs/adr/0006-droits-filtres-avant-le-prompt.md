# ADR 0006 — Les droits d'accès sont filtrés avant le prompt, jamais confiés au modèle

**Date** : 11/09/2026 · **Statut** : acceptée

## Contexte

Un assistant sur documents internes ne doit révéler à un salarié que ce qu'il a le droit de
lire. Demander au modèle de « ne pas révéler » un document est une consigne, pas une garantie.

## Décision

`AccessPolicy.can_read(user, chunk)` (domaine) est appliqué comme prédicat de la recherche : un
morceau interdit n'entre ni dans le classement, ni dans le prompt, ni dans la trace. Les groupes
autorisés sont portés par chaque document et hérités par ses morceaux ; ils entrent dans
l'empreinte du corpus (un changement de droits = corpus modifié = réindexation).

## Conséquences

- Fuites mesurées : 0 sur 936 appels des bancs Solvéo et Service-Public du 11/09 (0 aussi sur les bancs qwen3 et de validation).
- La règle vit dans le domaine et se teste sans IA ni réseau.
- Nuance à enseigner : filtrer avant le prompt garantit la confidentialité, **pas le silence** :
  une fiche publique voisine peut fournir une réponse partielle à une question dont la vraie
  réponse est réservée.
- Coût : post-filtrage sur un top-k global, donc un top-k parfois appauvri (des places prises par
  des documents interdits puis retirés). Le pré-filtrage côté index (ADR 0002, écarté) éviterait
  cela au prix d'une règle métier dans un composant technique.

## Écarté

Compter sur le prompt (« ne cite pas les documents RH ») ou vérifier les citations après coup
seulement : le modèle aurait vu le contenu interdit.
