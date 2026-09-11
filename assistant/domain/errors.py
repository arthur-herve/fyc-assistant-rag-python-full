class DomainError(Exception):
    """Erreur liée à une règle métier."""


class EmptyQuestionError(DomainError):
    def __init__(self) -> None:
        super().__init__("La question est vide.")
