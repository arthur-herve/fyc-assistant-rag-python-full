"""Outils partagés par les expériences (séquences 3.1, 3.2, 3.3).

Chaque expérience change UNE chose, enregistre un instantané avant et après,
et écrit un rapport Markdown reproductible dans eval/resultats/exp-<nom>-<date>/.
Les scripts s'exécutent depuis la racine du projet :

    python tools/experiences/<nom>.py --config config/app-ollama.toml --questions eval/questions-service-public.json
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from assistant.application.errors import ApplicationError  # noqa: E402
from assistant.application.ports import Snapshot  # noqa: E402
from assistant.application.snapshots import (  # noqa: E402
    IDENTICAL, MISSING, SOURCES_CHANGED, STATUS_CHANGED, TEXT_CHANGED, SnapshotComparison, SnapshotQuestion,
    compare_snapshots,
)
from assistant.composition import AppConfig, UnknownUserError, build, config_file  # noqa: E402
from assistant.domain.errors import DomainError  # noqa: E402
from assistant.domain.model import AnswerStatus  # noqa: E402
from assistant.interface.benchmark import (  # noqa: E402
    EvalQuestion, check_ai_service, check_served_model, create_new_dir, keyword_coverage, limit_questions,
    load_questions, results_dir,
)
from assistant.interface.presenter import comparison_to_text  # noqa: E402

# Noms des statuts dans les instantanés : ceux du domaine, pas des chaînes recopiées.
ANSWERED, NO_RELEVANT_SOURCE, UNSOURCED = (s.value for s in (
    AnswerStatus.ANSWERED, AnswerStatus.NO_RELEVANT_SOURCE, AnswerStatus.UNSOURCED))


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    # Sans --config : ASSISTANT_CONFIG, sinon config/app.toml, comme `python -m assistant`.
    p.add_argument("--config", help="fichier de configuration (défaut : ASSISTANT_CONFIG, sinon config/app.toml)")
    p.add_argument("--questions", default="eval/questions.json")
    p.add_argument("--limit", type=int, help="ne garder que les N premières questions")
    p.add_argument("--out", help="dossier de sortie (défaut : eval/resultats/exp-<nom>-<date>)")
    return p


def run(main: Callable[[], None]) -> None:
    """Lance une expérience : une option, une configuration ou un fichier invalide donne « Erreur : … »
    et le code 1, comme `python -m assistant`, pas une trace."""
    # Sortie en UTF-8 même redirigée vers un fichier (Windows encoderait en cp1252), comme `python -m assistant`.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        main()
    except (ApplicationError, DomainError, UnknownUserError, ValueError) as error:
        sys.exit(f"Erreur : {error}")
    except OSError as error:
        sys.exit(f"Erreur : fichier ou dossier inaccessible — {error}")


class Experiment:
    """Un dossier de sortie, une configuration, des questions, des instantanés.

    Les options sont toutes vérifiées à la construction, et là seulement : après la lecture de la
    configuration, avant tout travail. Les questions (--limit, utilisateurs), le prompt de la
    configuration, ce que l'expérience change (`changes` : un second prompt, un autre découpage),
    monté sans rien calculer, et le nombre de passages (`runs`) d'une expérience qui se répète :
    une faute de frappe ne coûte ni dossier, ni index, ni appel au service IA.
    Puis `start`, avant tout index : le service IA doit répondre et servir l'alias --other avec le
    bon type, et un seuil `default` est annoncé. Le dossier n'est créé qu'à la première indexation ;
    celui par défaut, daté à la seconde, n'est jamais un dossier qui existe déjà (« -2 », « -3 »…).
    """

    def __init__(self, name: str, args: argparse.Namespace, *, changes: dict[str, Any] | None = None,
                 runs: int | None = None) -> None:
        self.name = name
        # Sans --out : sous la racine du projet, d'où qu'on lance le script, créé par _create_out.
        self.out = Path(args.out) if args.out else results_dir(f"exp-{name}-", datetime.now())
        self._new_out_dir = not args.out
        source = config_file(args.config)
        base = AppConfig.load(source)
        # Instantanés et index de l'expérience restent dans son dossier.
        self.config = replace(base, snapshots_dir=self.out / "instantanes")
        # Les options, toutes ici : la configuration est lue, et rien n'est encore fait.
        self.questions: list[EvalQuestion] = limit_questions(load_questions(args.questions), args.limit)
        for q in self.questions:
            self.config.user(q.user)   # utilisateur inconnu : UnknownUserError avant tout travail
        self._check()
        if changes:
            self._check(**changes)
        if runs is not None and runs < 2:
            raise ValueError("--runs doit valoir au moins 2 : il faut deux passages pour mesurer une dérive")
        self._started = False   # voir start()
        self.lines: list[str] = [f"# Expérience « {name} » — {datetime.now():%Y-%m-%d %H:%M}", ""]
        # Les chemins tels qu'ils ont été donnés (sans --config : le fichier lu, ASSISTANT_CONFIG ou
        # config/app.toml en chemin complet), avec des « / » : le même rapport sous Windows et Linux.
        config_path, questions_path = (str(path).replace("\\", "/") for path in (args.config or source, args.questions))
        self.log(f"Configuration `{config_path}` · {len(self.questions)} questions de `{questions_path}` · "
                 f"corpus `{self.config.corpus_dir.name}`.")
        self.log("")

    # --- assemblage -----------------------------------------------------

    def _check(self, **overrides: Any) -> None:
        """Monte la configuration que l'expérience utilisera, sans rien calculer : un découpage hors bornes
        ou un prompt introuvable (celui de la configuration, ou le second de prompt-v2) est dit avant
        l'indexation et l'instantané « avant »."""
        container = build(self.config, **overrides)
        container.prompts.get(container.settings.prompt_name)

    def start(self, *, embedding: str | None = None, generation: str | None = None) -> None:
        """Les options vérifiées (à la construction), le service IA doit répondre, et servir l'alias --other
        (`embedding` ou `generation`) avec ce type (GET /v1/models). Puis le seuil `default` du modèle
        principal est annoncé : après les vérifications, jamais avant une erreur."""
        check_ai_service(self.config.ai_base_url)
        for kind, alias in (("embedding", embedding), ("generation", generation)):
            if alias is not None:
                check_served_model(self.config.ai_base_url, kind, alias)
        self._started = True
        self.warn_if_default_threshold(self.config.embedding_model)

    def warn_if_default_threshold(self, embedding_model: str) -> None:
        """Aucun seuil configuré pour cet alias : la valeur `default` s'applique, et on le dit (console
        et rapport), comme la ligne de commande et le banc d'essai (ADR 0004)."""
        if self.config.has_threshold_for(embedding_model):
            return
        message = (f"Attention : aucun seuil de pertinence configuré pour « {embedding_model} » : valeur "
                   f"`default` {self.config.min_score_for(embedding_model)} (ADR 0004 : lancer le banc d'essai)")
        print(message, file=sys.stderr)
        self.log(f"> {message}")
        self.log("")

    def container(self, index_name: str, **overrides: Any):
        return build(self.config, index_path=self.out / f"index-{index_name}.json", **overrides)

    def index(self, index_name: str, **overrides: Any):
        if not self._started:
            raise RuntimeError("Experiment.start() d'abord : options et service IA vérifiés avant tout index")
        self._create_out()   # le travail commence
        container = self.container(index_name, **overrides)
        start = time.perf_counter()
        manifest = container.index_corpus.execute()
        seconds = time.perf_counter() - start
        print(f"  index « {index_name} » : {manifest.chunk_count} morceaux, {manifest.embedding_model}, "
              f"{manifest.dimension} dim., {seconds:.1f} s")
        return container, manifest, seconds

    def record(self, snapshot_name: str, container) -> tuple[Snapshot, float]:
        questions = [SnapshotQuestion(q.id, self.config.user(q.user), q.question) for q in self.questions]
        start = time.perf_counter()
        snapshot = container.record_snapshot.execute(snapshot_name, questions)
        seconds = time.perf_counter() - start
        print(f"  instantané « {snapshot_name} » : {len(snapshot.entries)} réponses en {seconds:.1f} s")
        return snapshot, seconds

    def _create_out(self) -> None:
        """Crée le dossier de sortie, s'il ne l'est pas déjà (à chaque index, et avant le rapport). Celui par défaut,
        daté à la seconde, n'est jamais un dossier qui existe déjà : « -2 », « -3 »… (create_new_dir) ; index et
        instantanés le suivent. --out peut exister."""
        if not self._new_out_dir:
            self.out.mkdir(parents=True, exist_ok=True)
            return
        self.out = create_new_dir(self.out)
        self.config = replace(self.config, snapshots_dir=self.out / "instantanes")
        self._new_out_dir = False

    # --- mesures --------------------------------------------------------

    def stats(self, snapshot: Snapshot) -> dict[str, float | None]:
        by_id = {q.id: q for q in self.questions}
        answerable = [e for e in snapshot.entries if by_id[e.question_id].answerable]
        unanswerable = [e for e in snapshot.entries if not by_id[e.question_id].answerable]
        with_expected = [e for e in answerable if by_id[e.question_id].expected_documents]
        coverages = [keyword_coverage(e.text, by_id[e.question_id].expected_keywords)
                     for e in answerable if e.status == ANSWERED]
        coverages = [c for c in coverages if c is not None]
        return {
            "répond (répondables)": _rate([e.status == ANSWERED for e in answerable]),
            "bonne source": _rate([bool(set(e.cited_documents) & set(by_id[e.question_id].expected_documents))
                                   for e in with_expected if e.status == ANSWERED]),
            # Mesure (grossière) du contenu : le taux de dérive seul ne distingue pas une reformulation
            # d'une réponse inversée.
            "mots-clés (réponses données)": round(sum(coverages) / len(coverages), 2) if coverages else None,
            "refus justes (sans réponse accessible)": _rate([e.status == NO_RELEVANT_SOURCE for e in unanswerable]),
            "non sourcé": _rate([e.status == UNSOURCED for e in snapshot.entries]),
            "fuites d'accès": float(sum(bool(set(e.cited_documents) & set(by_id[e.question_id].forbidden_documents))
                                        for e in snapshot.entries)),
        }

    def compare(self, before: Snapshot, after: Snapshot) -> SnapshotComparison:
        return compare_snapshots(before, after)

    # --- rapport --------------------------------------------------------

    def log(self, line: str = "") -> None:
        self.lines.append(line)

    def table(self, rows: dict[str, dict[str, Any]]) -> None:
        """rows : {libellé de colonne: {métrique: valeur}} → tableau métriques × colonnes."""
        if not rows:
            self.log("(rien à comparer)")
            self.log("")
            return
        columns = list(rows)
        metrics = list(next(iter(rows.values())))
        self.log("| | " + " | ".join(columns) + " |")
        self.log("|---|" + "---|" * len(columns))
        for metric in metrics:
            self.log(f"| {metric} | " + " | ".join(_fmt(rows[c].get(metric)) for c in columns) + " |")
        self.log("")

    def comparison(self, comparison: SnapshotComparison, title: str) -> None:
        self.log(f"## {title}")
        self.log("")
        self.log("```")
        self.log(comparison_to_text(comparison))
        self.log("```")
        self.log("")

    def write(self) -> Path:
        self._create_out()
        path = self.out / "rapport.md"
        # newline="\n" et des « / » : sous Windows aussi, des fins de ligne et des chemins écrits comme sous Linux.
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8", newline="\n")
        print(f"\nRapport : {path.as_posix()}")
        return path


def drift_summary(comparison: SnapshotComparison) -> dict[str, Any]:
    return {
        "questions comparées": comparison.compared,
        "réponses modifiées": comparison.changed,
        "taux de dérive": comparison.drift_rate,
        "changements de statut": comparison.count(STATUS_CHANGED),
        "changements de sources": comparison.count(SOURCES_CHANGED),
        "textes modifiés (à relire)": comparison.count(TEXT_CHANGED),
    }


def _rate(values: Sequence[bool]) -> float | None:
    return None if not values else round(sum(values) / len(values), 2)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


# Tout ce que les scripts voisins importent d'ici (test_banc.py le vérifie).
__all__ = ["ANSWERED", "Experiment", "IDENTICAL", "MISSING", "NO_RELEVANT_SOURCE", "STATUS_CHANGED", "UNSOURCED",
           "drift_summary", "parser", "run"]
