class ApplicationError(Exception):
    """Erreur levée par un cas d'usage."""


class IndexNotBuiltError(ApplicationError):
    def __init__(self) -> None:
        super().__init__(
            "Aucun index n'a été construit. Lancez d'abord l'indexation du corpus."
        )


class EmptyCorpusError(ApplicationError):
    def __init__(self) -> None:
        super().__init__("Le corpus ne contient aucun texte à indexer.")


class IndexModelMismatchError(ApplicationError):
    """Le modèle d'embeddings a changé depuis la construction de l'index.

    C'est le cœur de la problématique : l'index est une donnée persistée qui
    dépend du modèle. On ne peut pas « juste changer une ligne de config ».
    """

    def __init__(
        self,
        index_model: str,
        index_dimension: int,
        current_model: str,
        current_dimension: int,
    ) -> None:
        self.index_model = index_model
        self.current_model = current_model
        super().__init__(
            f"L'index a été construit avec « {index_model} » ({index_dimension} dim.) "
            f"mais le modèle d'embeddings actuel est « {current_model} » "
            f"({current_dimension} dim.). Il faut réindexer le corpus."
        )


class IndexReplacedError(ApplicationError):
    """L'index a été reconstruit (par un autre processus) entre le contrôle du modèle et la recherche.

    Fait partie du contrat du port `VectorIndex` : la recherche la lève au lieu de chercher dans un
    autre index que celui contrôlé. `SearchPassages` recommence une fois, puis la laisse passer.
    """

    def __init__(self) -> None:
        super().__init__("L'index a été reconstruit pendant la recherche. Reposez la question.")


class IndexWriteError(ApplicationError):
    """L'index n'a pas pu être écrit : fait partie du contrat du port `VectorIndex`.

    L'index en service reste le précédent, en mémoire comme sur le disque.
    """

    def __init__(self, location: str, detail: str) -> None:
        super().__init__(f"écriture impossible de l'index ({location}) : {detail}")


class InconsistentEmbeddingsError(ApplicationError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"Embeddings incohérents pendant l'indexation : {detail}")


class AIServiceError(ApplicationError):
    """Le service IA est injoignable ou a renvoyé une erreur.

    `transient` : vrai quand réessayer a un sens (injoignable, délai dépassé,
    erreur 5xx) ; faux pour une requête refusée (modèle inconnu, 4xx).
    """

    def __init__(self, message: str, transient: bool = True) -> None:
        super().__init__(message)
        self.transient = transient


class ModelOutputRejectedError(ApplicationError):
    """La sortie du modèle n'a pas la forme d'une réponse (règles de output_rules.py).

    Levée par le décorateur de validation ; le cas d'usage la compte comme une
    tentative ratée et réessaie, puis renvoie une réponse « non sourcée ».
    """

    def __init__(self, model: str, text: str, problems: tuple[str, ...]) -> None:
        self.model = model
        self.text = text
        self.problems = problems
        super().__init__("sortie du modèle rejetée : " + " ; ".join(problems))


class SnapshotNotFoundError(ApplicationError):
    """Aucun instantané de ce nom : fait partie du contrat du port `SnapshotStore`."""


class PromptNotFoundError(ApplicationError):
    """Aucun prompt de ce nom : fait partie du contrat du port `PromptRepository`.

    Le message nomme le prompt, l'endroit où il a été cherché et les prompts connus (« aucun » s'il n'y en a
    pas) : un nom mal tapé se corrige d'un coup d'œil.
    """

    def __init__(self, name: str, location: str, known: list[str]) -> None:
        super().__init__(f"prompt introuvable : {name} dans {location} (connus : {', '.join(known) or 'aucun'})")
