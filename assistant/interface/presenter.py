"""Présentation des résultats des cas d'usage (JSON pour l'API, texte pour le terminal).

Un texte qui se lit : null et true plutôt que None et True, listes sans guillemets, clés du
découpage triées, scores à quatre décimales, « 33 % ».
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from assistant.application.ports import IndexManifest
from assistant.application.snapshots import (
    IDENTICAL, MISSING, SOURCES_CHANGED, STATUS_CHANGED, TEXT_CHANGED, SnapshotComparison, config_value_to_text,
)
from assistant.application.status import StatusReport, describe_splitter
from assistant.domain.model import Answer


def answer_to_dict(answer: Answer, include_raw: bool = False) -> dict[str, Any]:
    trace = asdict(answer.trace)
    if not include_raw:
        trace.pop("raw_outputs")
    trace["retrieved"] = [{"chunk_id": c, "score": s} for c, s in answer.trace.retrieved]
    return {
        "question": answer.question,
        "status": answer.status.value,
        "text": answer.text,
        "sources": [asdict(s) for s in answer.sources],
        "trace": trace,
    }


def answer_to_text(answer: Answer, verbose: bool = False) -> str:
    lines = [answer.text, ""]
    if answer.sources:
        lines.append("Sources :")
        lines += [f"  [{s.number}] {s.document_title} ({s.chunk_id})" for s in answer.sources]
    t = answer.trace
    lines.append("")
    lines.append(
        f"statut={answer.status.value} · embeddings={t.embedding_model} · "
        f"génération={t.generation_model or '—'} · prompt={t.prompt_version or '—'} · "
        f"index={t.index_id} · tentatives={t.attempts}"
    )
    if verbose:
        lines.append(f"seuil={round(t.min_score, 4)}")
        lines += [f"  retrouvé {chunk} score={round(score, 4)}" for chunk, score in t.retrieved]
        for i, raw in enumerate(t.raw_outputs, start=1):
            # Entre guillemets, sauts de ligne échappés : la sortie exacte du modèle, sur une ligne.
            lines.append(f"  sortie brute {i} : {json.dumps(raw, ensure_ascii=False)}")
    return "\n".join(lines)


def manifest_to_dict(manifest: IndexManifest) -> dict[str, Any]:
    # Clés du découpage triées : la même sortie, quel que soit l'ordre des clés dans le fichier d'index.
    return {**asdict(manifest), "splitter": dict(sorted(manifest.splitter.items()))}


def status_to_dict(report: StatusReport) -> dict[str, Any]:
    return {
        "up_to_date": report.up_to_date,
        "unverified": report.unverified,
        "issues": list(report.issues),
        "index": manifest_to_dict(report.index) if report.index else None,
        "corpus": {"documents": report.corpus_documents, "fingerprint": report.corpus_fingerprint},
        "splitter": dict(sorted(report.splitter.items())),
        "ai_service": {"embedding_model": report.embedding_model,
                       "dimension": report.embedding_dimension, "error": report.ai_service_error},
        "prompt_version": report.prompt_version,
    }


def status_to_text(report: StatusReport) -> str:
    lines = ["État de l'assistant", "==================="]
    if report.index is None:
        lines.append("Index      : aucun")
    else:
        m = report.index
        lines.append(f"Index      : {m.index_id} · {m.chunk_count} morceaux de {m.document_count} documents "
                     f"· construit le {m.created_at}")
        lines.append(f"             modèle d'embeddings {m.embedding_model} ({m.dimension} dim.) "
                     f"· découpage {describe_splitter(m.splitter)}")
        lines.append(f"             empreinte du corpus {m.corpus_fingerprint[:12]}…")
    lines.append(f"Corpus     : {report.corpus_documents} documents · empreinte {report.corpus_fingerprint[:12]}…")
    lines.append(f"Découpage  : {describe_splitter(report.splitter)}")
    if report.ai_service_error:
        lines.append(f"Service IA : en erreur ({report.ai_service_error})")
    else:
        lines.append(f"Service IA : sert {report.embedding_model} ({report.embedding_dimension} dim.)")
    lines.append(f"Prompt     : {report.prompt_version}")
    lines.append("")
    if report.up_to_date:
        lines.append("Verdict    : à jour, l'index est cohérent avec le corpus, le découpage et le modèle servi.")
    elif report.unverified:
        lines.append("Verdict    : NON VÉRIFIÉ (corpus et découpage cohérents, modèle servi inconnu)")
        lines += [f"  - {issue}" for issue in report.issues]
    else:
        lines.append("Verdict    : À REFAIRE")
        lines += [f"  - {issue}" for issue in report.issues]
    return "\n".join(lines)


def comparison_to_text(comparison: SnapshotComparison, show_changes: bool = True) -> str:
    lines = [f"Comparaison : {comparison.baseline} → {comparison.candidate}", ""]
    lines.append("Différences de configuration")
    if comparison.configuration_differences:
        lines += [f"  - {key} : {config_value_to_text(before)} → {config_value_to_text(after)}"
                  for key, before, after in comparison.configuration_differences]
    else:
        lines.append("  (aucune : même configuration des deux côtés)")
    lines += ["", "Dérive",
              f"  questions comparées : {comparison.compared}",
              f"  réponses modifiées  : {comparison.changed}",
              f"  taux de dérive      : "
              + ("—" if comparison.drift_rate is None else f"{comparison.drift_rate * 100:.0f} %"),
              ""]
    lines.append("| Nature | Nombre | Lecture |")
    lines.append("|---|---|---|")
    readings = {
        STATUS_CHANGED: "changement de comportement : refus devenu réponse, ou l'inverse",
        SOURCES_CHANGED: "même décision, autres documents cités",
        TEXT_CHANGED: "mêmes sources, même statut, texte différent : à relire, le sens a pu changer (Oui devenu Non…)",
        IDENTICAL: "rien n'a bougé",
        MISSING: "question présente d'un seul côté",
    }
    for kind, reading in readings.items():
        lines.append(f"| {kind} | {comparison.count(kind)} | {reading} |")
    if show_changes:
        changes = [d for d in comparison.differences if d.kind not in (IDENTICAL, MISSING)]
        if changes:
            lines.append("")
            for d in changes:
                lines.append(f"{d.question_id} [{d.kind}]")
                lines.append(f"  avant : {d.before.status} [{', '.join(d.before.cited_documents)}] "
                             f"« {d.before.text[:90]} »")
                lines.append(f"  après : {d.after.status} [{', '.join(d.after.cited_documents)}] "
                             f"« {d.after.text[:90]} »")
    return "\n".join(lines)
