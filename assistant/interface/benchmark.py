"""Banc d'essai : compare des modèles d'embeddings et de génération.

Pour chaque modèle d'embeddings :
  1. réindexe le corpus (obligatoire : un index ne survit pas au changement) ;
  2. mesure la recherche seule, sans génération (déterministe) ;
  3. propose un seuil de pertinence propre à ce modèle.
Puis, pour chaque modèle de génération, pose chaque question `runs` fois et
mesure le résultat de façon statistique (séquence 3.1).

Les résultats bruts vont dans un CSV, la synthèse dans un rapport Markdown.
"""

from __future__ import annotations

import csv
import json
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from assistant.application.errors import ApplicationError
from assistant.composition import AppConfig, build
from assistant.domain.model import AnswerStatus


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    user: str
    answerable: bool
    expected_documents: tuple[str, ...]
    forbidden_documents: tuple[str, ...]
    expected_keywords: tuple[str, ...]


def load_questions(path: str | Path) -> list[EvalQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return [
            EvalQuestion(
                id=q["id"],
                question=q["question"],
                user=q.get("user", "alice"),
                answerable=bool(q.get("answerable", True)),
                expected_documents=tuple(q.get("expected_documents", [])),
                forbidden_documents=tuple(q.get("forbidden_documents", [])),
                expected_keywords=tuple(q.get("expected_keywords", [])),
            )
            for q in data["questions"]
        ]
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"{path} : jeu de questions mal formé ({type(error).__name__} : {error})") from error


def _plain(text: str) -> str:
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


def keyword_coverage(text: str, keywords: tuple[str, ...]) -> float | None:
    if not keywords:
        return None
    plain = _plain(text)
    return sum(1 for k in keywords if _plain(k) in plain) / len(keywords)


def suggest_threshold(answerable: list[float], unanswerable: list[float]) -> tuple[float | None, float | None]:
    """Seuil qui sépare le mieux les deux populations de scores (et sa justesse).

    Attention : calibré sur les questions d'évaluation elles-mêmes, il est
    optimiste. Il faudrait un jeu de questions distinct pour le valider.
    """
    if not answerable or not unanswerable:
        return None, None
    candidates = sorted(set(answerable + unanswerable))
    best, best_correct = None, -1
    for i, value in enumerate(candidates):
        upper = candidates[i + 1] if i + 1 < len(candidates) else value + 0.01
        threshold = (value + upper) / 2
        correct = sum(s >= threshold for s in answerable) + sum(s < threshold for s in unanswerable)
        if correct > best_correct:
            best, best_correct = threshold, correct
    low = min(candidates) - 0.01
    if sum(s >= low for s in answerable) > best_correct:
        best, best_correct = low, sum(s >= low for s in answerable)
    return round(best, 3), round(best_correct / (len(answerable) + len(unanswerable)), 3)


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 1)


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


@dataclass(frozen=True)
class RetrievalScore:
    question: EvalQuestion
    top1: float
    hit: bool | None
    hit1: bool | None


@dataclass(frozen=True)
class BenchmarkRow:
    """Une réponse mesurée par le banc : une ligne de resultats.csv (mêmes colonnes en C#)."""

    embedding: str
    generation: str
    run: int
    question_id: str
    answerable: bool
    status: str
    attempts: int | None = None   # colonnes dans l'ordre des résultats de référence (eval/resultats)
    latency_ms: int = 0
    generation_model_id: str = ""
    cited_documents: str = ""
    source_hit: bool | None = None
    forbidden_leak: bool = False
    keyword_coverage: float | None = None
    error: str | None = None
    text: str = ""

    ERROR = "error"

    @property
    def generated(self) -> bool:
        """Le modèle a-t-il été appelé ? Un refus sans passage pertinent n'en a pas besoin."""
        return bool(self.attempts)

    def csv_row(self) -> dict[str, Any]:
        return {key: "" if value is None else value for key, value in asdict(self).items()}


CSV_FIELDS = [f.name for f in fields(BenchmarkRow)]


def _retrieval_scores(container, questions: list[EvalQuestion]) -> list[RetrievalScore]:
    """Score top-1 et succès de la recherche pour chaque question (déterministe).

    Passe par le cas d'usage SearchPassages : mêmes droits, même contrôle du modèle
    que les vraies questions, jamais une recherche refaite à côté."""
    scores = []
    for q in questions:
        user = container.config.user(q.user)
        passages = container.search_passages.execute(user, q.question, container.settings.top_k).passages
        first = passages[0].chunk.document_id if passages else None
        found = {p.chunk.document_id for p in passages}
        scores.append(RetrievalScore(
            question=q,
            top1=passages[0].score if passages else 0.0,
            hit=bool(found & set(q.expected_documents)) if q.expected_documents else None,
            hit1=(first in q.expected_documents) if q.expected_documents else None,
        ))
    return scores


def validate_threshold(scores: list[RetrievalScore], threshold: float) -> dict[str, Any]:
    """Le seuil tient-il sur des questions qu'il n'a pas vues ? (séquence 3.1)"""
    answerable = [s for s in scores if s.question.answerable]
    unanswerable = [s for s in scores if not s.question.answerable]
    return {
        "questions": len(scores),
        "hit@1": _mean([1.0 if s.hit1 else 0.0 for s in answerable if s.hit1 is not None]),
        "kept_answerable": _mean([float(s.top1 >= threshold) for s in answerable]),
        "correct_refusals": _mean([float(s.top1 < threshold) for s in unanswerable]),
        "threshold": threshold,
    }


def _check_options(config: AppConfig, questions: list[EvalQuestion],
                   validation: list[EvalQuestion] | None, min_score_mode: str) -> float | None:
    """Tout ce qui peut être faux dans les options l'est dit avant la première indexation."""
    for q in questions + (validation or []):
        config.user(q.user)   # utilisateur inconnu : UnknownUserError tout de suite
    if min_score_mode in ("auto", "config"):
        return None
    try:
        return float(min_score_mode)
    except ValueError:
        raise ValueError(f"--min-score attend config, auto ou un nombre (ex. 0.6), pas « {min_score_mode} »") from None


@contextmanager
def _results_file(path: Path):
    """Chaque ligne est écrite dès qu'elle est connue : une coupure en fin de campagne ne perd rien."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        rows: list[BenchmarkRow] = []

        def record(row: BenchmarkRow) -> None:
            rows.append(row)
            writer.writerow(row.csv_row())
            handle.flush()

        yield record, rows


def run_benchmark(config: AppConfig, embedding_models: list[str], generation_models: list[str],
                  questions: list[EvalQuestion], runs: int, out_dir: Path,
                  min_score_mode: str = "config", splitter_overrides: dict[str, Any] | None = None,
                  validation: list[EvalQuestion] | None = None, log=print) -> dict[str, Any]:
    fixed_min_score = _check_options(config, questions, validation, min_score_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    retrieval_summary: list[dict[str, Any]] = []
    with _results_file(out_dir / "resultats.csv") as (record, rows):
        for emb in embedding_models:
            log(f"\n=== Embeddings : {emb} ===")
            index_path = out_dir / f"index-{emb}.json"
            container = build(config, embedding_model=emb, index_path=index_path,
                              splitter_overrides=splitter_overrides)
            start = time.perf_counter()
            try:
                manifest = container.index_corpus.execute()
            except ApplicationError as error:
                log(f"  ÉCHEC de l'indexation : {error}")
                retrieval_summary.append({"embedding": emb, "error": str(error)})
                continue
            summary = _measure_retrieval(container, emb, manifest, round(time.perf_counter() - start, 1),
                                         questions, validation, min_score_mode, fixed_min_score, log)
            retrieval_summary.append(summary)
            for gen in generation_models:
                log(f"  --- Génération : {gen} ({runs} passage(s)) ---")
                ask = build(config, embedding_model=emb, generation_model=gen, index_path=index_path,
                            min_score=summary["threshold_used"],
                            splitter_overrides=splitter_overrides).ask_question
                _measure_generation(ask, config, emb, gen, questions, runs, record, log)

    summary = _summarize(rows, retrieval_summary)
    (out_dir / "synthese.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_version = build(config).prompts.get(config.prompt_name).version
    splitter = {**config.splitter, **(splitter_overrides or {})}
    (out_dir / "rapport.md").write_text(
        _report(summary, runs, len(questions), prompt_version, splitter, config),
        encoding="utf-8")
    log(f"\nRapport : {out_dir / 'rapport.md'}")
    return summary


def _measure_retrieval(container, emb: str, manifest, index_seconds: float, questions: list[EvalQuestion],
                       validation: list[EvalQuestion] | None, min_score_mode: str,
                       fixed_min_score: float | None, log) -> dict[str, Any]:
    """La recherche seule, sans génération : hit@1, hit@k, seuil suggéré puis éprouvé."""
    log(f"  {manifest.chunk_count} morceaux indexés en {index_seconds} s "
        f"({manifest.embedding_model}, {manifest.dimension} dim.)")
    top_k = container.settings.top_k
    all_scores = _retrieval_scores(container, questions)
    scored = [s for s in all_scores if s.question.answerable]
    unanswered = [s for s in all_scores if not s.question.answerable]
    hits = [1.0 if s.hit else 0.0 for s in scored if s.hit is not None]
    hits_top1 = [1.0 if s.hit1 else 0.0 for s in scored if s.hit1 is not None]
    suggested, separation = suggest_threshold([s.top1 for s in scored], [s.top1 for s in unanswered])
    configured = container.config.min_score_for(emb)
    used = fixed_min_score if fixed_min_score is not None else (
        suggested if min_score_mode == "auto" and suggested is not None else configured)
    summary = {
        "embedding": emb,
        "model_id": manifest.embedding_model,
        "dimension": manifest.dimension,
        "chunks": manifest.chunk_count,
        "index_seconds": index_seconds,
        "hit@1": _mean(hits_top1),
        "hit@k": _mean(hits),
        "top_k": top_k,
        "top1_median_answerable": _median([s.top1 for s in scored]),
        "top1_median_unanswerable": _median([s.top1 for s in unanswered]),
        "configured_threshold": configured,
        "configured_threshold_is_default": not container.config.has_threshold_for(emb),
        "suggested_threshold": suggested,
        "separation_accuracy": separation,
        "threshold_used": used,
    }
    default = " (default : aucun seuil pour cet alias)" if summary["configured_threshold_is_default"] else ""
    log(f"  hit@1={_fmt(_mean(hits_top1))} · hit@{top_k}={_fmt(_mean(hits))} · seuil configuré={configured}{default} · "
        f"seuil suggéré={_fmt(suggested)} · seuil utilisé={used}")
    if validation:
        # Le seuil retenu, éprouvé sur des questions qu'il n'a pas vues.
        checked = validate_threshold(_retrieval_scores(container, validation), used)
        summary["validation"] = checked
        log(f"  validation ({checked['questions']} questions jamais vues, seuil {used}) : "
            f"hit@1={_fmt(checked['hit@1'])} · répondables retenues={_fmt(checked['kept_answerable'])} · "
            f"refus justes={_fmt(checked['correct_refusals'])}")
    return summary


def _measure_generation(ask, config: AppConfig, emb: str, gen: str, questions: list[EvalQuestion],
                        runs: int, record, log) -> None:
    """Chaque question `runs` fois : la génération se mesure en proportions (séquence 3.1)."""
    for run in range(1, runs + 1):
        for q in questions:
            start = time.perf_counter()
            try:
                answer = ask.execute(config.user(q.user), q.question)
            except ApplicationError as error:
                record(BenchmarkRow(emb, gen, run, q.id, q.answerable, BenchmarkRow.ERROR,
                                    latency_ms=round((time.perf_counter() - start) * 1000), error=str(error)))
                log(f"    {q.id} : erreur — {error}")
                continue
            cited = sorted({s.document_id for s in answer.sources})
            answered = answer.status is AnswerStatus.ANSWERED
            record(BenchmarkRow(
                emb, gen, run, q.id, q.answerable, answer.status.value,
                latency_ms=round((time.perf_counter() - start) * 1000),
                attempts=answer.trace.attempts,
                generation_model_id=answer.trace.generation_model or "",
                cited_documents="|".join(cited),
                source_hit=(bool(set(cited) & set(q.expected_documents))
                            if answered and q.expected_documents else None),
                forbidden_leak=bool(set(cited) & set(q.forbidden_documents)),
                keyword_coverage=keyword_coverage(answer.text, q.expected_keywords) if answered else None,
                text=answer.text.replace("\n", " "),
            ))
        log(f"    passage {run}/{runs} terminé")


def _summarize(rows: list[BenchmarkRow], retrieval: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        groups[(row.embedding, row.generation)].append(row)

    generation = []
    for (emb, gen), group in groups.items():
        answerable = [r for r in group if r.answerable]
        unanswerable = [r for r in group if not r.answerable]
        ok = [r for r in group if r.status != BenchmarkRow.ERROR]

        per_question: dict[str, list[tuple]] = defaultdict(list)
        for r in ok:
            per_question[r.question_id].append((r.status, r.cited_documents))
        stability = [
            Counter(outcomes).most_common(1)[0][1] / len(outcomes)
            for outcomes in per_question.values() if len(outcomes) > 1
        ]
        # Latence des seules réponses générées : un refus sans passage pertinent ne coûte presque
        # rien et tirerait la médiane vers le bas.
        generated = [float(r.latency_ms) for r in ok if r.generated]
        generation.append({
            "embedding": emb,
            "generation": gen,
            "calls": len(group),
            "errors": len(group) - len(ok),
            "answer_rate": _mean([float(r.status == AnswerStatus.ANSWERED.value) for r in answerable]),
            "source_hit_rate": _mean([float(r.source_hit) for r in answerable if r.source_hit is not None]),
            "keyword_coverage": _mean([r.keyword_coverage for r in answerable if r.keyword_coverage is not None]),
            "unsourced_rate": _mean([float(r.status == AnswerStatus.UNSOURCED.value) for r in group]),
            "correct_refusal_rate": _mean([float(r.status == AnswerStatus.NO_RELEVANT_SOURCE.value)
                                           for r in unanswerable]),
            "forbidden_leaks": sum(r.forbidden_leak for r in group),
            "stability": _mean(stability),
            "mean_attempts": _mean([float(r.attempts) for r in ok if r.generated]),
            "latency_median_ms": _median(generated),
            "latency_p90_ms": _p90(generated),
            "refusals_without_generation": sum(1 for r in ok if not r.generated),
        })
    return {"retrieval": retrieval, "generation": generation}


def _report(summary, runs, question_count, prompt_version, splitter, config) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Banc d'essai — {now}",
        "",
        f"{question_count} questions · {runs} passage(s) par question · prompt `{prompt_version}` · "
        f"découpage `{json.dumps(splitter, ensure_ascii=False)}` · top_k={config.top_k} · "
        f"température={config.temperature}",
        "",
        "## Recherche (sans génération)",
        "",
        "| Embeddings | Modèle servi | Dim. | Morceaux | Indexation (s) | Hit@1 | Hit@k | Score top-1 médian (répondables) | (hors corpus) | Seuil configuré | Seuil suggéré | Séparation | Seuil utilisé |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["retrieval"]:
        if "error" in r:
            lines.append(f"| {r['embedding']} | ÉCHEC : {r['error']} |" + " |" * 11)
            continue
        lines.append(
            f"| {r['embedding']} | `{r['model_id']}` | {r['dimension']} | {r['chunks']} | "
            f"{_fmt(r['index_seconds'])} | {_fmt(r['hit@1'])} | {_fmt(r['hit@k'])} | "
            f"{_fmt(r['top1_median_answerable'])} | "
            f"{_fmt(r['top1_median_unanswerable'])} | {_fmt(r['configured_threshold'])}"
            f"{' (default)' if r['configured_threshold_is_default'] else ''} | "
            f"{_fmt(r['suggested_threshold'])} | {_fmt(r['separation_accuracy'])} | "
            f"{_fmt(r['threshold_used'])} |"
        )
    if any("validation" in r for r in summary["retrieval"]):
        lines += [
            "",
            "## Validation du seuil sur des questions jamais vues",
            "",
            "| Embeddings | Seuil éprouvé | Questions | Hit@1 | Répondables retenues | Refus justes (sans réponse accessible) |",
            "|---|---|---|---|---|---|",
        ]
        for r in summary["retrieval"]:
            v = r.get("validation")
            if v:
                lines.append(f"| {r['embedding']} | {_fmt(v['threshold'])} | {v['questions']} | {_fmt(v['hit@1'])} | "
                             f"{_fmt(v['kept_answerable'])} | {_fmt(v['correct_refusals'])} |")
    lines += [
        "",
        "## Réponses (avec génération)",
        "",
        "| Embeddings | Génération | Répond (répondables) | Bonne source | Mots-clés | Non sourcé | Refus justes (sans réponse accessible) | Fuites d'accès | Stabilité | Tentatives | Latence médiane des réponses générées (ms) | p90 (ms) | Erreurs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in summary["generation"]:
        lines.append(
            f"| {g['embedding']} | {g['generation']} | {_fmt(g['answer_rate'])} | "
            f"{_fmt(g['source_hit_rate'])} | {_fmt(g['keyword_coverage'])} | "
            f"{_fmt(g['unsourced_rate'])} | {_fmt(g['correct_refusal_rate'])} | "
            f"{g['forbidden_leaks']} | {_fmt(g['stability'])} | {_fmt(g['mean_attempts'])} | "
            f"{_ms(g['latency_median_ms'])} | {_ms(g['latency_p90_ms'])} | {g['errors']} |"
        )
    lines += [
        "",
        "## Lecture",
        "",
        "- **Hit@1 / Hit@k** : part des questions répondables dont un document attendu arrive en tête / figure "
        "dans les k passages retrouvés. Sur un petit corpus, Hit@k est vite saturé : regarder Hit@1.",
        "- **Seuil suggéré** : sépare au mieux les questions répondables des questions hors corpus. "
        "Calibré sur ces mêmes questions, il est optimiste.",
        "- **Bonne source** : parmi les réponses données, part qui cite un document attendu.",
        "- **Répond** : statut « answered ». Une réponse qui dit « je ne sais pas » en citant une source "
        "compte comme une réponse : l'indicateur ne lit pas le texte.",
        "- **Mots-clés** : part des mots-clés attendus présents dans la réponse (indicateur grossier).",
        "- **Refus justes** : questions sans réponse dans un document accessible (hors corpus ou accès "
        "refusé) auxquelles l'assistant n'a pas répondu.",
        "- **Latence** : sur les seules réponses générées ; un refus sans passage pertinent est quasi immédiat.",
        "- **Non sourcé** : le modèle n'a pas cité correctement ses sources malgré les tentatives.",
        "- **Fuites d'accès** : doit toujours valoir 0, le filtrage est fait avant le modèle.",
        "- **Stabilité** : pour une même question, part des passages qui donnent le même statut "
        "et les mêmes documents cités (1 = parfaitement stable). Nécessite au moins 2 passages.",
        "",
    ]
    return "\n".join(lines)


def check_ai_service(base_url: str) -> None:
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=5) as response:
            json.loads(response.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            f"Service IA injoignable sur {base_url} ({error}).\n"
            "Lancez-le d'abord dans un autre terminal : python -m ai_service"
        ) from None


def main_benchmark(args) -> None:
    config = AppConfig.load(args.config)
    check_ai_service(config.ai_base_url)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    validation = load_questions(args.validate_with) if args.validate_with else None
    splitter_overrides = {}
    if args.max_chars is not None:
        splitter_overrides["max_chars"] = args.max_chars
    if args.overlap_chars is not None:
        splitter_overrides["overlap_chars"] = args.overlap_chars
    out_dir = Path(args.out or f"eval/resultats/{datetime.now():%Y%m%d-%H%M%S}")
    run_benchmark(
        config,
        embedding_models=args.embedding or [config.embedding_model],
        generation_models=args.generation or [config.generation_model],
        questions=questions,
        runs=args.runs,
        out_dir=out_dir,
        min_score_mode=args.min_score,
        splitter_overrides=splitter_overrides or None,
        validation=validation,
    )
