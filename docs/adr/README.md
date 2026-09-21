# Décisions d'architecture (ADR)

Une décision par fichier : contexte, décision, conséquences, ce qu'on a écarté. Elles se lisent
dans l'ordre. Une ADR n'est jamais modifiée après coup : une décision qui change donne une
nouvelle ADR qui remplace l'ancienne. Seules exceptions, datées : la ligne de statut (« complétée
par l'ADR … ») et un erratum en fin de document quand un fait énoncé s'avère faux.

| N° | Décision | Séquences |
|---|---|---|
| [0001](0001-service-ia-separe-en-http.md) | Le service IA est un déployable séparé, joint en HTTP | S2.3, S4.1 |
| [0002](0002-index-cote-application.md) | L'index vectoriel vit côté application, pas côté service IA | S3.2, S4.2 |
| [0003](0003-index-incompatible-erreur.md) | Un index construit avec un autre modèle est une erreur, pas un avertissement | S2.3, S3.2 |
| [0004](0004-seuil-par-modele.md) | Le seuil de pertinence est calibré par modèle d'embeddings et par corpus | S3.2 |
| [0005](0005-prompt-dans-l-application.md) | Le prompt est construit et versionné dans l'application | S3.3 |
| [0006](0006-droits-filtres-avant-le-prompt.md) | Les droits d'accès sont filtrés avant le prompt, jamais confiés au modèle | S4.1 |
| [0007](0007-bibliotheque-standard.md) | Bibliothèque standard uniquement dans l'application | S1.1, S4.3 |
| [0008](0008-decorateurs-et-validation-de-sortie.md) | Les garde-fous sont des décorateurs de ports ; la forme de la sortie est une règle métier | S4.1 |
| [0009](0009-cache-et-identite-du-modele.md) | Le cache d'embeddings dérive de l'index ; l'identité d'un modèle Ollama est relue à chaque appel | S4.1, S4.2 |
