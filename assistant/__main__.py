"""Ligne de commande de l'application.

    python -m assistant index
    python -m assistant ask "Combien de jours de télétravail ?" --user alice
    python -m assistant serve
    python -m assistant status
    python -m assistant snapshot record reference
    python -m assistant snapshot compare reference candidat
    python -m assistant benchmark --embedding hashing nomic --generation extractive qwen3-1b7
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from assistant.application.errors import ApplicationError
from assistant.composition import AppConfig, UnknownUserError, build
from assistant.domain.errors import DomainError
from assistant.interface.presenter import (
    answer_to_dict, answer_to_text, comparison_to_text, manifest_to_dict, status_to_dict,
    status_to_text,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="fichier de configuration (défaut : config/app.toml)")
    parser.add_argument("--embedding-model", help="alias du modèle d'embeddings (surcharge)")
    parser.add_argument("--generation-model", help="alias du modèle de génération (surcharge)")
    parser.add_argument("--prompt", help="nom du prompt dans assistant/prompts/ (défaut : answer)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assistant", description="Assistant documentaire RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="indexer le corpus")
    _common(p_index)

    p_ask = sub.add_parser("ask", help="poser une question")
    _common(p_ask)
    p_ask.add_argument("question")
    p_ask.add_argument("--user", default="alice", help="utilisateur déclaré dans la configuration (défaut : alice)")
    p_ask.add_argument("--json", action="store_true", help="sortie JSON")
    p_ask.add_argument("-v", "--verbose", action="store_true", help="afficher la trace complète")

    p_serve = sub.add_parser("serve", help="lancer l'API HTTP de l'application")
    _common(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_status = sub.add_parser("status", help="l'index est-il cohérent avec le corpus, le découpage et le modèle servi ?")
    _common(p_status)
    p_status.add_argument("--json", action="store_true", help="sortie JSON")

    p_snap = sub.add_parser("snapshot", help="figer et comparer le comportement (instantanés)")
    snap_sub = p_snap.add_subparsers(dest="snapshot_command", required=True)
    p_record = snap_sub.add_parser("record", help="enregistrer un instantané")
    _common(p_record)
    p_record.add_argument("name")
    p_record.add_argument("--questions", default="eval/questions.json")
    p_record.add_argument("--limit", type=int)
    p_compare = snap_sub.add_parser("compare", help="comparer deux instantanés")
    _common(p_compare)
    p_compare.add_argument("baseline")
    p_compare.add_argument("candidate")
    _common(snap_sub.add_parser("list", help="lister les instantanés"))

    p_bench = sub.add_parser("benchmark", help="comparer des modèles")
    p_bench.add_argument("--config")
    p_bench.add_argument("--embedding", nargs="+", help="alias des modèles d'embeddings")
    p_bench.add_argument("--generation", nargs="+", help="alias des modèles de génération")
    p_bench.add_argument("--questions", default="eval/questions.json")
    p_bench.add_argument("--validate-with", help="second jeu de questions, jamais vu, pour éprouver le seuil")
    p_bench.add_argument("--runs", type=int, default=3, help="passages par question (défaut 3)")
    p_bench.add_argument("--limit", type=int, help="ne garder que les N premières questions")
    p_bench.add_argument("--min-score", default="config",
                         help="'config' (défaut), 'auto' (seuil suggéré) ou une valeur")
    p_bench.add_argument("--max-chars", type=int, help="surcharge du découpage (expérience CACE)")
    p_bench.add_argument("--overlap-chars", type=int, help="surcharge du recouvrement")
    p_bench.add_argument("--seed", type=int, help="graine de génération")
    p_bench.add_argument("--out", help="dossier de sortie (défaut : eval/resultats/<date>)")

    args = parser.parse_args(argv)
    # Sortie en UTF-8 même redirigée vers un fichier (Windows encoderait en cp1252).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    # Le journal des décorateurs (composition.py) : INFO avec -v, sinon ASSISTANT_LOG.
    level = "INFO" if getattr(args, "verbose", False) else os.environ.get("ASSISTANT_LOG", "WARNING").upper()
    logging.basicConfig(level=level, format="[%(name)s] %(message)s", stream=sys.stderr)

    try:
        if args.command == "benchmark":
            from assistant.interface.benchmark import main_benchmark
            main_benchmark(args)
            return 0

        config = AppConfig.load(args.config)
        container = build(config, args.embedding_model, args.generation_model,
                          prompt_name=args.prompt)
        embedding_model = args.embedding_model or config.embedding_model
        if not config.has_threshold_for(embedding_model):
            print(f"Attention : aucun seuil de pertinence configuré pour « {embedding_model} » : valeur "
                  f"`default` {config.min_score_for(embedding_model)} (ADR 0004 : lancer le banc d'essai)",
                  file=sys.stderr)

        if args.command == "index":
            manifest = container.index_corpus.execute()
            print(json.dumps(manifest_to_dict(manifest), ensure_ascii=False, indent=2))
        elif args.command == "ask":
            answer = container.ask_question.execute(config.user(args.user), args.question)
            if args.json:
                print(json.dumps(answer_to_dict(answer, include_raw=args.verbose),
                                 ensure_ascii=False, indent=2))
            else:
                print(answer_to_text(answer, verbose=args.verbose))
        elif args.command == "status":
            report = container.check_status.execute()
            if args.json:
                print(json.dumps(status_to_dict(report), ensure_ascii=False, indent=2))
            else:
                print(status_to_text(report))
            if report.up_to_date:
                return 0
            return 3 if report.unverified else 2
        elif args.command == "snapshot":
            from assistant.application.snapshots import SnapshotQuestion, compare_snapshots
            from assistant.interface.benchmark import load_questions
            if args.snapshot_command == "record":
                questions = load_questions(_project_path(args.questions))[: args.limit or None]
                snapshot = container.record_snapshot.execute(
                    args.name,
                    [SnapshotQuestion(q.id, config.user(q.user), q.question) for q in questions],
                )
                print(f"Instantané « {snapshot.name} » : {len(snapshot.entries)} réponses, "
                      f"enregistré dans {config.snapshots_dir}")
                for key, value in snapshot.configuration.items():
                    print(f"  {key} = {value}")
            elif args.snapshot_command == "compare":
                store = container.snapshots
                print(comparison_to_text(
                    compare_snapshots(store.load(args.baseline), store.load(args.candidate))))
            else:
                names = container.snapshots.names()
                print("\n".join(names) if names else f"Aucun instantané dans {config.snapshots_dir}")
        elif args.command == "serve":
            from assistant.interface.http_api import create_server
            server = create_server(container, args.host, args.port)
            host, port = server.server_address[:2]   # --port 0 : le port réellement choisi
            print(f"Application sur http://{host}:{port} "
                  f"(service IA : {config.ai_base_url})")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nArrêt de l'application.")
            finally:
                server.server_close()
        return 0
    except (ApplicationError, DomainError, UnknownUserError, ValueError) as error:
        # ValueError : configuration, corpus, index ou questions mal formés (messages explicites).
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Erreur : fichier ou dossier inaccessible — {error}", file=sys.stderr)
        return 1


def _project_path(path: str) -> str:
    """Un chemin relatif qui n'existe pas depuis le dossier courant est cherché depuis la racine du projet."""
    from pathlib import Path

    from assistant.composition import PROJECT_ROOT
    candidate = Path(path)
    if not candidate.is_absolute() and not candidate.exists() and (PROJECT_ROOT / path).exists():
        return str(PROJECT_ROOT / path)
    return path


if __name__ == "__main__":
    sys.exit(main())
