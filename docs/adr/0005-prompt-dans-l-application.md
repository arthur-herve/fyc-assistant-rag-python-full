# ADR 0005 — Le prompt est construit et versionné dans l'application

**Date** : 11/09/2026 · **Statut** : acceptée

## Contexte

Le prompt dit au modèle de citer ses sources entre crochets, en français, en trois phrases : il
encode des règles métier, mais dans un texte faillible, réglé pour un modèle donné. Est-ce de la
configuration, de l'infrastructure ou du métier ? (S3.3)

## Décision

Le prompt est un fichier `assistant/prompts/answer.toml`, chargé par un port `PromptRepository`,
**construit côté application** (numérotation des passages, question) et envoyé tel quel au
service IA, qui ne fait que le transmettre. Sa version déclarée et une empreinte de son contenu
sont inscrites dans chaque `AnswerTrace`.

## Conséquences

- Les règles métier ne dépendent pas du prompt : citations et forme de la sortie sont vérifiées
  après coup, de façon déterministe (`domain/citations.py`, `domain/output_rules.py`). Le prompt
  est une *tentative de persuasion* ; la garantie est ailleurs.
- Modifier une virgule change l'empreinte : une dérive de réponses devient attribuable.
- Le prompt est réglé pour `llama3.2:3b` ; un autre générateur peut demander une autre
  formulation (S3.3, variante v2 à mesurer).

## Écarté

Le prompt côté service IA (« c'est un détail du modèle ») : le service devrait alors connaître
la numérotation des passages et l'obligation de citer, c'est-à-dire le métier.
