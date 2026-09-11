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
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from assistant.application.ports import Snapshot  # noqa: E402
from assistant.application.snapshots import (  # noqa: E402
    IDENTICAL, MISSING, SnapshotComparison, SnapshotQuestion, compare_snapshots,
)
from assistant.composition import AppConfig, build  # noqa: E402
from assistant.interface.benchmark import EvalQuestion, check_ai_service, load_questions  # noqa: E402
from assistant.interface.presenter import comparison_to_text  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config/app.toml")
    p.add_argument("--questions", default="eval/questions.json")
    p.add_argument("--limit", type=int, help="ne garder que les N premières questions")
    p.add_argument("--out", help="dossier de sortie (défaut : eval/resultats/exp-<nom>-<date>)")
    return p


class Experiment:
    """Un dossier de sortie, une configuration, des questions, des instantanés."""

    def __init__(self, name: str, args: argparse.Namespace) -> None:
        self.name = name
        self.out = Path(args.out or f"eval/resultats/exp-{name}-{datetime.now():%Y%m%d-%H%M%S}")
        self.out.mkdir(parents=True, exist_ok=True)
        base = AppConfig.load(args.config)
        check_ai_service(base.ai_base_url)
        # Instantanés et index de l'expérience restent dans son dossier.
        self.config = replace(base, snapshots_dir=self.out / "instantanes")
        self.questions: list[EvalQuestion] = load_questions(args.questions)[: args.limit or None]
        self.lines: list[str] = [f"# Expérience « {name} » — {datetime.now():%Y-%m-%d %H:%M}", ""]
        self.log(f"Configuration `{args.config}` · {len(self.questions)} questions de `{args.questions}` · "
                 f"corpus `{self.config.corpus_dir.name}`.")
        self.log("")

    # --- assemblage -----------------------------------------------------

    def container(self, index_name: str, **overrides: Any):
        return build(self.config, index_path=self.out / f"index-{index_name}.json", **overrides)

    def index(self, index_name: str, **overrides: Any):
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

    # --- mesures --------------------------------------------------------

    def stats(self, snapshot: Snapshot) -> dict[str, float | None]:
        by_id = {q.id: q for q in self.questions}
        answerable = [e for e in snapshot.entries if by_id[e.question_id].answerable]
        unanswerable = [e for e in snapshot.entries if not by_id[e.question_id].answerable]
        with_expected = [e for e in answerable if by_id[e.question_id].expected_documents]
        return {
            "répond (répondables)": _rate([e.status == "answered" for e in answerable]),
            "bonne source": _rate([bool(set(e.cited_documents) & set(by_id[e.question_id].expected_documents))
                                   for e in with_expected if e.status == "answered"]),
            "refus justes (hors corpus)": _rate([e.status == "no_relevant_source" for e in unanswerable]),
            "non sourcé": _rate([e.status == "unsourced" for e in snapshot.entries]),
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
        path = self.out / "rapport.md"
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\nRapport : {path}")
        return path


def drift_summary(comparison: SnapshotComparison) -> dict[str, Any]:
    return {
        "questions comparées": comparison.compared,
        "réponses modifiées": comparison.changed,
        "taux de dérive": comparison.drift_rate,
        "changements de statut": comparison.count("statut modifié"),
        "changements de sources": comparison.count("sources modifiées"),
        "reformulations": comparison.count("texte modifié"),
    }


def _rate(values: Sequence[bool]) -> float | None:
    return None if not values else round(sum(values) / len(values), 2)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


__all__ = ["Experiment", "IDENTICAL", "MISSING", "drift_summary", "parser"]
