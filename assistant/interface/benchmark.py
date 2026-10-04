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
import itertools
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

from assistant.application.errors import AIServiceError, ApplicationError
from assistant.composition import PROJECT_ROOT, AppConfig, build, read_json, read_json_text
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


# Clés d'une question (eval/questions*.json, partagés avec la version C#). Une faute de frappe
# (« expected_document ») ne doit pas vider une attente en silence : une clé inconnue est refusée.
QUESTION_KEYS = ("id", "user", "question", "answerable", "expected_documents", "forbidden_documents",
                 "expected_keywords")


def _strings(item: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = item.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where} : « {key} » doit être une liste de textes")
    return tuple(value)


def _question(item: Any, number: int) -> EvalQuestion:
    """Une question relue, types vérifiés : "false" n'est pas un booléen, "teletravail"
    n'est pas une liste de documents, 12 n'est pas une question."""
    where = f"question n° {number}"
    if not isinstance(item, dict):
        raise ValueError(f"{where} : un objet JSON est attendu")
    # Une clé qui commence par « _ » est un commentaire : JSON n'en a pas d'autres.
    unknown = sorted(key for key in set(item) - set(QUESTION_KEYS) if not key.startswith("_"))
    if unknown:
        raise ValueError(f"{where} : clé(s) inconnue(s) [{', '.join(unknown)}] "
                         f"(connues : {', '.join(sorted(QUESTION_KEYS))})")
    for key in ("id", "question"):
        if not isinstance(item.get(key), str):
            raise ValueError(f"{where} : champ « {key} » manquant ou non textuel")
    user = item.get("user", "alice")
    if not isinstance(user, str):
        raise ValueError(f"{where} : « user » doit être un texte")
    answerable = item.get("answerable", True)
    if not isinstance(answerable, bool):
        raise ValueError(f"{where} : « answerable » doit valoir true ou false")
    return EvalQuestion(
        id=item["id"],
        question=item["question"],
        user=user,
        answerable=answerable,
        expected_documents=_strings(item, "expected_documents", where),
        forbidden_documents=_strings(item, "forbidden_documents", where),
        expected_keywords=_strings(item, "expected_keywords", where),
    )


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """Un jeu de questions d'évaluation : UTF-8 strict (BOM accepté), JSON strict (clé en double, NaN,
    entier de plus de 4300 chiffres, plus de 900 niveaux, chaîne qui n'est pas du texte), clés « _… »
    ignorées (des commentaires). Tout défaut est dit avec le fichier, puis la question (son numéro) et
    le champ quand il y en a un ; un défaut du JSON lui-même, repéré à sa lecture, est dit sans la
    question (avec la clé, pour une clé en double)."""
    try:
        # UTF-8 strict (BOM accepté), JSON strict : clé en double, NaN, entier de plus de 4300 chiffres, plus
        # de 900 niveaux, « \ud800 » isolé, même dans un commentaire (ValueError).
        data = read_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
            raise ValueError("un objet JSON avec une liste « questions » est attendu")
        questions = [_question(item, number) for number, item in enumerate(data["questions"], start=1)]
        seen: set[str] = set()
        for number, q in enumerate(questions, start=1):
            # Un identifiant sert de clé (stabilité, instantanés, indicateurs) : deux questions ne le partagent pas.
            if q.id in seen:
                raise ValueError(f"question n° {number} : identifiant « {q.id} » déjà utilisé")
            seen.add(q.id)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} : jeu de questions mal formé (JSON invalide : {error})") from error
    except ValueError as error:
        raise ValueError(f"{path} : jeu de questions mal formé ({error})") from error
    if not questions:
        raise ValueError(f"aucune question dans {path}")
    return questions


def check_limit(limit: int | None) -> None:
    """--limit N : au moins 1 (None : pas de limite), pour le banc, les expériences et snapshot record. 0 ou
    un nombre négatif est refusé : [:-1] retirerait la dernière question en silence. La ligne de commande
    le vérifie d'emblée, avant de lire la configuration."""
    if limit is not None and limit < 1:
        raise ValueError(f"--limit doit valoir au moins 1, pas {limit}")


def limit_questions(questions: list[EvalQuestion], limit: int | None) -> list[EvalQuestion]:
    """--limit N : les N premières questions (toutes sans --limit) ; voir check_limit."""
    check_limit(limit)
    return questions if limit is None else questions[:limit]


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
    """Une réponse mesurée par le banc : une ligne de resultats.csv."""

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


# --- synthèses : typées ici, écrites dans synthese.json sous des noms fixes (eval/resultats en a des exemples) ---


@dataclass(frozen=True)
class ThresholdValidation:
    """Le seuil retenu, éprouvé sur des questions qu'il n'a pas vues (séquence 3.1)."""

    questions: int
    hit_at_1: float | None
    kept_answerable: float | None
    correct_refusals: float | None
    threshold: float

    def to_json(self) -> dict[str, Any]:
        return {
            "questions": self.questions,
            "hit@1": self.hit_at_1,
            "kept_answerable": self.kept_answerable,
            "correct_refusals": self.correct_refusals,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class RetrievalSummary:
    """La recherche seule, pour un modèle d'embeddings : hit@1, hit@k, seuils configuré, suggéré, utilisé."""

    embedding: str
    model_id: str
    dimension: int
    chunks: int
    index_seconds: float
    hit_at_1: float | None
    hit_at_k: float | None
    top_k: int
    top1_median_answerable: float | None
    top1_median_unanswerable: float | None
    configured_threshold: float
    configured_threshold_is_default: bool
    suggested_threshold: float | None
    separation_accuracy: float | None
    threshold_used: float
    validation: ThresholdValidation | None = None

    def to_json(self) -> dict[str, Any]:
        values = {
            "embedding": self.embedding,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "chunks": self.chunks,
            "index_seconds": self.index_seconds,
            "hit@1": self.hit_at_1,
            "hit@k": self.hit_at_k,
            "top_k": self.top_k,
            "top1_median_answerable": self.top1_median_answerable,
            "top1_median_unanswerable": self.top1_median_unanswerable,
            "configured_threshold": self.configured_threshold,
            "configured_threshold_is_default": self.configured_threshold_is_default,
            "suggested_threshold": self.suggested_threshold,
            "separation_accuracy": self.separation_accuracy,
            "threshold_used": self.threshold_used,
        }
        if self.validation is not None:
            values["validation"] = self.validation.to_json()
        return values


@dataclass(frozen=True)
class IndexingFailure:
    """Un modèle d'embeddings dont l'indexation a échoué : le banc passe au suivant."""

    embedding: str
    error: str

    def to_json(self) -> dict[str, Any]:
        return {"embedding": self.embedding, "error": self.error}


@dataclass(frozen=True)
class GenerationSummary:
    """Les réponses d'un couple (embeddings, génération), mesurées en proportions (séquence 3.1)."""

    embedding: str
    generation: str
    calls: int
    errors: int
    answer_rate: float | None
    source_hit_rate: float | None
    keyword_coverage: float | None
    unsourced_rate: float | None
    correct_refusal_rate: float | None
    forbidden_leaks: int
    stability: float | None
    mean_attempts: float | None
    latency_median_ms: float | None
    latency_p90_ms: float | None
    refusals_without_generation: int

    def to_json(self) -> dict[str, Any]:
        return {
            "embedding": self.embedding,
            "generation": self.generation,
            "calls": self.calls,
            "errors": self.errors,
            "answer_rate": self.answer_rate,
            "source_hit_rate": self.source_hit_rate,
            "keyword_coverage": self.keyword_coverage,
            "unsourced_rate": self.unsourced_rate,
            "correct_refusal_rate": self.correct_refusal_rate,
            "forbidden_leaks": self.forbidden_leaks,
            "stability": self.stability,
            "mean_attempts": self.mean_attempts,
            "latency_median_ms": self.latency_median_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "refusals_without_generation": self.refusals_without_generation,
        }


@dataclass(frozen=True)
class BenchmarkSummary:
    """Le contenu de synthese.json : la recherche par modèle d'embeddings, les réponses par couple de modèles."""

    retrieval: tuple[RetrievalSummary | IndexingFailure, ...]
    generation: tuple[GenerationSummary, ...]

    def to_json(self) -> dict[str, Any]:
        return {"retrieval": [r.to_json() for r in self.retrieval],
                "generation": [g.to_json() for g in self.generation]}


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


def validate_threshold(scores: list[RetrievalScore], threshold: float) -> ThresholdValidation:
    """Le seuil tient-il sur des questions qu'il n'a pas vues ? (séquence 3.1)"""
    answerable = [s for s in scores if s.question.answerable]
    unanswerable = [s for s in scores if not s.question.answerable]
    return ThresholdValidation(
        questions=len(scores),
        hit_at_1=_mean([1.0 if s.hit1 else 0.0 for s in answerable if s.hit1 is not None]),
        kept_answerable=_mean([float(s.top1 >= threshold) for s in answerable]),
        correct_refusals=_mean([float(s.top1 < threshold) for s in unanswerable]),
        threshold=threshold,
    )


def fixed_min_score(min_score_mode: str) -> float | None:
    """--min-score : `config`, `auto` (None : pas de seuil imposé) ou un seuil de [-1, 1], comme ceux
    de la configuration. nan échoue à la comparaison : il est refusé comme 5 ou -2."""
    if min_score_mode in ("auto", "config"):
        return None
    try:
        value = float(min_score_mode)
    except ValueError:
        value = None
    if value is None or not -1 <= value <= 1:
        raise ValueError(f"--min-score attend config, auto ou un nombre de [-1, 1] (ex. 0.6), "
                         f"pas « {min_score_mode} »")
    return value


def _check_options(config: AppConfig, questions: list[EvalQuestion], validation: list[EvalQuestion] | None,
                   runs: int, min_score_mode: str,
                   splitter_overrides: dict[str, Any] | None) -> tuple[float | None, str]:
    """Tout ce qui peut être faux dans les options l'est dit avant la première indexation, et avant
    de créer le dossier de sortie : utilisateurs, passages, seuil, découpage, prompt.
    Renvoie le seuil imposé par --min-score (None : config ou auto) et la version du prompt."""
    for q in questions + (validation or []):
        config.user(q.user)   # utilisateur inconnu : UnknownUserError tout de suite
    if runs < 1:
        raise ValueError(f"--runs doit valoir au moins 1, pas {runs}")
    fixed = fixed_min_score(min_score_mode)
    # Monter la configuration ne calcule rien : un découpage hors bornes (ValueError) ou un prompt
    # introuvable se voient ici, pas après une indexation.
    container = build(config, splitter_overrides=splitter_overrides)
    return fixed, container.prompts.get(config.prompt_name).version


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
                  validation: list[EvalQuestion] | None = None, prompt_name: str | None = None,
                  new_out_dir: bool = False, log=print) -> BenchmarkSummary:
    # --prompt : toutes les compositions du banc s'en servent. Donné vide, il vaut celui de la configuration,
    # comme pour les autres commandes (build : « prompt_name or … »).
    if prompt_name:
        config = replace(config, prompt_name=prompt_name)
    imposed_min_score, prompt_version = _check_options(config, questions, validation, runs, min_score_mode,
                                                       splitter_overrides)
    # Les options d'abord, toutes : une faute de frappe se dit sans service IA.
    check_ai_service(config.ai_base_url)
    if new_out_dir:   # le dossier par défaut, daté à la seconde : jamais un dossier qui existe déjà (create_new_dir)
        out_dir = create_new_dir(out_dir)
    else:   # --out : un dossier existant convient
        out_dir.mkdir(parents=True, exist_ok=True)
    retrieval: list[RetrievalSummary | IndexingFailure] = []
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
                retrieval.append(IndexingFailure(emb, str(error)))
                continue
            measured = _measure_retrieval(container, emb, manifest, round(time.perf_counter() - start, 1),
                                          questions, validation, min_score_mode, imposed_min_score, log)
            retrieval.append(measured)
            for gen in generation_models:
                log(f"  --- Génération : {gen} ({runs} passage(s)) ---")
                ask = build(config, embedding_model=emb, generation_model=gen, index_path=index_path,
                            min_score=measured.threshold_used,
                            splitter_overrides=splitter_overrides).ask_question
                _measure_generation(ask, config, emb, gen, questions, runs, record, log)

    summary = BenchmarkSummary(tuple(retrieval), tuple(summarize(rows)))
    # newline="\n" : les mêmes octets sous Windows et Linux (Windows écrirait \r\n).
    (out_dir / "synthese.json").write_text(
        json.dumps(summary.to_json(), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    splitter = {**config.splitter, **(splitter_overrides or {})}
    (out_dir / "rapport.md").write_text(
        _report(summary, runs, len(questions), prompt_version, splitter, config),
        encoding="utf-8", newline="\n")
    log(f"\nRapport : {(out_dir / 'rapport.md').as_posix()}")   # des « / », sous Windows aussi
    return summary


def _measure_retrieval(container, emb: str, manifest, index_seconds: float, questions: list[EvalQuestion],
                       validation: list[EvalQuestion] | None, min_score_mode: str,
                       imposed_min_score: float | None, log) -> RetrievalSummary:
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
    is_default = not container.config.has_threshold_for(emb)
    used = imposed_min_score if imposed_min_score is not None else (
        suggested if min_score_mode == "auto" and suggested is not None else configured)
    default = " (default : aucun seuil pour cet alias)" if is_default else ""
    log(f"  hit@1={_fmt(_mean(hits_top1))} · hit@{top_k}={_fmt(_mean(hits))} · "
        f"seuil configuré={configured}{default} · seuil suggéré={_fmt(suggested)} · seuil utilisé={used}")
    checked = None
    if validation:
        # Le seuil retenu, éprouvé sur des questions qu'il n'a pas vues.
        checked = validate_threshold(_retrieval_scores(container, validation), used)
        log(f"  validation ({checked.questions} questions jamais vues, seuil {used}) : "
            f"hit@1={_fmt(checked.hit_at_1)} · répondables retenues={_fmt(checked.kept_answerable)} · "
            f"refus justes={_fmt(checked.correct_refusals)}")
    return RetrievalSummary(
        embedding=emb,
        model_id=manifest.embedding_model,
        dimension=manifest.dimension,
        chunks=manifest.chunk_count,
        index_seconds=index_seconds,
        hit_at_1=_mean(hits_top1),
        hit_at_k=_mean(hits),
        top_k=top_k,
        top1_median_answerable=_median([s.top1 for s in scored]),
        top1_median_unanswerable=_median([s.top1 for s in unanswered]),
        configured_threshold=configured,
        configured_threshold_is_default=is_default,
        suggested_threshold=suggested,
        separation_accuracy=separation,
        threshold_used=used,
        validation=checked,
    )


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


def summarize(rows: list[BenchmarkRow]) -> list[GenerationSummary]:
    """Une synthèse des réponses par couple (embeddings, génération), lue dans les lignes du CSV."""
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
        generation.append(GenerationSummary(
            embedding=emb,
            generation=gen,
            calls=len(group),
            errors=len(group) - len(ok),
            answer_rate=_mean([float(r.status == AnswerStatus.ANSWERED.value) for r in answerable]),
            source_hit_rate=_mean([float(r.source_hit) for r in answerable if r.source_hit is not None]),
            keyword_coverage=_mean([r.keyword_coverage for r in answerable if r.keyword_coverage is not None]),
            unsourced_rate=_mean([float(r.status == AnswerStatus.UNSOURCED.value) for r in group]),
            correct_refusal_rate=_mean([float(r.status == AnswerStatus.NO_RELEVANT_SOURCE.value)
                                        for r in unanswerable]),
            forbidden_leaks=sum(r.forbidden_leak for r in group),
            stability=_mean(stability),
            mean_attempts=_mean([float(r.attempts) for r in ok if r.generated]),
            latency_median_ms=_median(generated),
            latency_p90_ms=_p90(generated),
            refusals_without_generation=sum(1 for r in ok if not r.generated),
        ))
    return generation


def _report(summary: BenchmarkSummary, runs, question_count, prompt_version, splitter, config) -> str:
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
    measured = [r for r in summary.retrieval if isinstance(r, RetrievalSummary)]
    for r in summary.retrieval:
        if isinstance(r, IndexingFailure):
            lines.append(f"| {r.embedding} | ÉCHEC : {r.error} |" + " |" * 11)
            continue
        lines.append(
            f"| {r.embedding} | `{r.model_id}` | {r.dimension} | {r.chunks} | "
            f"{_fmt(r.index_seconds)} | {_fmt(r.hit_at_1)} | {_fmt(r.hit_at_k)} | "
            f"{_fmt(r.top1_median_answerable)} | "
            f"{_fmt(r.top1_median_unanswerable)} | {_fmt(r.configured_threshold)}"
            f"{' (default)' if r.configured_threshold_is_default else ''} | "
            f"{_fmt(r.suggested_threshold)} | {_fmt(r.separation_accuracy)} | "
            f"{_fmt(r.threshold_used)} |"
        )
    if any(r.validation is not None for r in measured):
        lines += [
            "",
            "## Validation du seuil sur des questions jamais vues",
            "",
            "| Embeddings | Seuil éprouvé | Questions | Hit@1 | Répondables retenues | Refus justes (sans réponse accessible) |",
            "|---|---|---|---|---|---|",
        ]
        for r in measured:
            v = r.validation
            if v is not None:
                lines.append(f"| {r.embedding} | {_fmt(v.threshold)} | {v.questions} | {_fmt(v.hit_at_1)} | "
                             f"{_fmt(v.kept_answerable)} | {_fmt(v.correct_refusals)} |")
    lines += [
        "",
        "## Réponses (avec génération)",
        "",
        "| Embeddings | Génération | Répond (répondables) | Bonne source | Mots-clés | Non sourcé | Refus justes (sans réponse accessible) | Fuites d'accès | Stabilité | Tentatives | Latence médiane des réponses générées (ms) | p90 (ms) | Erreurs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in summary.generation:
        lines.append(
            f"| {g.embedding} | {g.generation} | {_fmt(g.answer_rate)} | "
            f"{_fmt(g.source_hit_rate)} | {_fmt(g.keyword_coverage)} | "
            f"{_fmt(g.unsourced_rate)} | {_fmt(g.correct_refusal_rate)} | "
            f"{g.forbidden_leaks} | {_fmt(g.stability)} | {_fmt(g.mean_attempts)} | "
            f"{_ms(g.latency_median_ms)} | {_ms(g.latency_p90_ms)} | {g.errors} |"
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
    """Le service IA doit répondre (GET /health) avant un banc ou une expérience : on le vérifie d'abord, en une
    requête. Ses messages sont dits comme toute erreur : « Erreur : … », code 1.
    Le corps de la réponse est lu en entier, sans être interprété : coupé, ou muet au-delà du délai, celui
    d'un 200 comme celui d'une erreur, le service est dit injoignable."""
    import urllib.error
    import urllib.parse
    import urllib.request
    url = base_url.rstrip("/") + "/health"
    unreachable = f"service IA injoignable ({base_url}). Lancez-le d'abord : python -m ai_service"
    try:
        # HTTP seulement : urllib lirait aussi un fichier local (« file: ») ou un serveur FTP.
        if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            raise urllib.error.URLError(f"protocole autre que HTTP : {url}")
        try:
            response = urllib.request.urlopen(url, timeout=5)
        except urllib.error.HTTPError as error:   # joignable, mais en erreur : son corps se lit comme celui d'un 200
            response = error
        with response:
            response.read()   # coupé, ou muet au-delà du délai : injoignable
    except Exception:  # noqa: BLE001 — refus de connexion, délai, adresse invalide, autre protocole : le même message
        raise AIServiceError(unreachable, transient=False) from None
    if isinstance(response, urllib.error.HTTPError):
        raise AIServiceError(f"le service IA répond HTTP {response.code} sur {url}", transient=False)


# Les deux types de modèles de GET /v1/models, nommés comme dans les messages.
_MODEL_KINDS = {"embedding": "modèle d'embeddings", "generation": "modèle de génération"}


def check_served_model(base_url: str, kind: str, alias: str) -> None:
    """L'alias --other d'une expérience est-il servi, et du bon type (GET /v1/models) ? Vérifié avant tout
    index : sinon l'erreur ne se verrait qu'après l'index et l'instantané « avant »."""
    import urllib.request
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            # UTF-8 strict, marque d'ordre des octets acceptée, puis JSON strict, comme toute réponse du service IA
            # (docs/contrat-http.md) : une clé en double, même là où rien n'est lu, est hors contrat.
            models = read_json_text(response.read().decode("utf-8-sig"))
            served = [model["alias"] for model in models[kind]]
    except Exception:  # noqa: BLE001 — injoignable, statut d'erreur ou réponse hors contrat : même message
        served = None
    if served is None or not all(isinstance(name, str) for name in served):
        raise AIServiceError(f"le service IA ne donne pas la liste de ses modèles ({url})", transient=False)
    if alias not in served:
        raise ValueError(f"le service IA ne sert pas « {alias} » comme {_MODEL_KINDS[kind]} "
                         f"(servis : {', '.join(served) or 'aucun'})")


def results_dir(prefix: str, now: datetime) -> Path:
    """Le dossier de sortie par défaut du banc (`prefix` vide) ou d'une expérience (« exp-<nom>- »), daté
    AAAAMMJJ-HHMMSS (« 20261001-164152 ») : sous la racine du projet, d'où qu'on lance la commande, en
    chemin complet."""
    return PROJECT_ROOT / "eval" / "resultats" / f"{prefix}{now:%Y%m%d-%H%M%S}"


def create_new_dir(path: Path) -> Path:
    """Crée le dossier `path`, ou s'il existe déjà « path-2 », puis « path-3 »… : le premier nom libre, renvoyé. Pour le
    dossier de sortie par défaut, daté à la seconde (results_dir) : deux bancs lancés dans la même seconde (un script,
    deux terminaux) n'écrivent plus dans le même dossier, où le second écrasait le rapport du premier."""
    # Les dossiers parents d'abord, hors de la boucle : un parent qui existe sans être un dossier (eval/resultats, un
    # fichier) est une erreur, dite ici. Dans la boucle, son FileExistsError passerait pour un nom pris : « -2 »,
    # « -3 »… sans fin.
    path.parent.mkdir(parents=True, exist_ok=True)
    for number in itertools.count(1):
        candidate = path if number == 1 else path.with_name(f"{path.name}-{number}")
        try:
            candidate.mkdir()   # sans exist_ok : échoue s'il existe, même créé à l'instant ailleurs
        except FileExistsError:
            continue   # pris, peut-être par un autre processus lancé en même temps : le nom suivant
        return candidate


def main_benchmark(args) -> None:
    config = AppConfig.load(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    questions = limit_questions(load_questions(args.questions), args.limit)
    validation = load_questions(args.validate_with) if args.validate_with else None
    splitter_overrides = {}
    if args.max_chars is not None:
        splitter_overrides["max_chars"] = args.max_chars
    if args.overlap_chars is not None:
        splitter_overrides["overlap_chars"] = args.overlap_chars
    # Sans --out : sous la racine du projet, d'où qu'on lance la commande, et jamais un dossier qui existe déjà
    # (new_out_dir).
    out_dir = Path(args.out) if args.out else results_dir("", datetime.now())
    run_benchmark(   # vérifie toutes les options, puis le service IA, avant le moindre travail
        config,
        embedding_models=args.embedding or [config.embedding_model],
        generation_models=args.generation or [config.generation_model],
        questions=questions,
        runs=args.runs,
        out_dir=out_dir,
        min_score_mode=args.min_score,
        splitter_overrides=splitter_overrides or None,
        validation=validation,
        prompt_name=args.prompt,
        new_out_dir=not args.out,
    )
