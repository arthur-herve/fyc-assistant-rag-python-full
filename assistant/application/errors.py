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


class InconsistentEmbeddingsError(ApplicationError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"Embeddings incohérents pendant l'indexation : {detail}")


class AIServiceError(ApplicationError):
    """Le service IA est injoignable ou a renvoyé une erreur."""


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
