"""Ligne de commande de l'application.

    python -m assistant index [--if-stale]
    python -m assistant ask "Combien de jours de télétravail ?" --user alice
    python -m assistant serve [--port 8000]          (--port 0 : un port libre, affiché au démarrage)
    python -m assistant status
    python -m assistant snapshot record reference
    python -m assistant snapshot compare reference candidat
    python -m assistant benchmark --embedding hashing nomic --generation extractive qwen3-1b7

Codes de retour : 0 ok · 1 erreur, saisie comprise · status : 2 à refaire, 3 non vérifié ·
index --if-stale : 3 non vérifié. La commande vient d'abord, puis ses options (argparse, sans
abréviation : « --us » n'est pas « --user »). Une option inconnue ou d'une autre commande, et un
argument en trop, sont refusés, jamais ignorés. « -- » termine les options : ce qui suit est un
argument (ask -- "-vingt degrés ?" : sans « -- », cette question qui commence par -v serait prise
pour une option).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, NoReturn

from assistant.application.errors import ApplicationError
from assistant.application.snapshots import config_value_to_text
from assistant.composition import AppConfig, UnknownUserError, build
from assistant.domain.errors import DomainError
from assistant.interface.presenter import (
    answer_to_dict, answer_to_text, comparison_to_text, manifest_to_dict, status_to_dict, status_to_text,
)

# Les commandes qui cherchent des passages, donc qui appliquent un seuil de pertinence.
_SEARCHING = ("ask", "serve", "snapshot record")
# Sans --config, pour toutes les commandes : ASSISTANT_CONFIG, sinon config/app.toml (composition.config_file).
_CONFIG_HELP = "fichier de configuration (défaut : ASSISTANT_CONFIG, sinon config/app.toml)"
_UNRECOGNIZED = "unrecognized arguments: "


class _Parser(argparse.ArgumentParser):
    """argparse, sans abréviation (« --us » n'est pas « --user ») et avec le code 1 pour une erreur
    de saisie : 2 veut dire « index à refaire » pour status."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        # Une option inconnue, d'une autre commande ou placée avant la commande, et un argument en trop :
        # argparse les dit ensemble, et l'aide de chaque commande donne ce qu'elle accepte.
        if message.startswith(_UNRECOGNIZED):
            message = (f"option(s) ou argument(s) non reconnu(s) : {message.removeprefix(_UNRECOGNIZED)} "
                       f"(voir assistant <commande> -h)")
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help=_CONFIG_HELP)
    parser.add_argument("--embedding-model", help="alias du modèle d'embeddings (surcharge)")
    parser.add_argument("--generation-model", help="alias du modèle de génération (surcharge)")
    parser.add_argument("--prompt", help="nom du prompt dans assistant/prompts/ (défaut : answer)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="journal des décorateurs (et, pour ask, trace complète)")


def main(argv: list[str] | None = None) -> int:
    # Sortie en UTF-8 même redirigée vers un fichier (Windows encoderait en cp1252).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = _Parser(prog="assistant", description="Assistant documentaire RAG",
                     epilog="La commande vient d'abord, puis ses options. « -- » termine les options : ce qui suit "
                            "est un argument (ask -- \"-vingt degrés ?\" : sans « -- », cette question qui commence "
                            "par -v serait prise pour une option).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="indexer le corpus")
    _common(p_index)
    p_index.add_argument("--if-stale", action="store_true",
                         help="seulement si status dit « à refaire » (code 3 : non vérifié)")

    p_ask = sub.add_parser("ask", help="poser une question")
    _common(p_ask)
    p_ask.add_argument("question")
    p_ask.add_argument("--user", default="alice", help="utilisateur déclaré dans la configuration (défaut : alice)")
    p_ask.add_argument("--json", action="store_true", help="sortie JSON")

    p_serve = sub.add_parser("serve", help="lancer l'API HTTP de l'application")
    _common(p_serve)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000, help="défaut 8000 ; 0 : un port libre, affiché au démarrage")
    p_serve.add_argument("--quiet", action="store_true", help="sans le journal des requêtes")

    p_status = sub.add_parser("status", help="l'index est-il cohérent avec le corpus, le découpage et le modèle servi ?")
    _common(p_status)
    p_status.add_argument("--json", action="store_true", help="sortie JSON")

    p_snap = sub.add_parser("snapshot", help="figer et comparer le comportement (instantanés)")
    snap_sub = p_snap.add_subparsers(dest="snapshot_command", required=True)
    p_record = snap_sub.add_parser("record", help="enregistrer un instantané")
    _common(p_record)
    p_record.add_argument("name")
    p_record.add_argument("--questions", default="eval/questions.json")
    p_record.add_argument("--limit", type=int, help="ne garder que les N premières questions (au moins 1)")
    p_compare = snap_sub.add_parser("compare", help="comparer deux instantanés")
    _common(p_compare)
    p_compare.add_argument("baseline")
    p_compare.add_argument("candidate")
    _common(snap_sub.add_parser("list", help="lister les instantanés"))

    # Pas de -v : au banc, le journal des décorateurs se demande par ASSISTANT_LOG.
    p_bench = sub.add_parser("benchmark", help="comparer des modèles")
    p_bench.add_argument("--config", help=_CONFIG_HELP)
    p_bench.add_argument("--embedding", nargs="+", help="alias des modèles d'embeddings")
    p_bench.add_argument("--generation", nargs="+", help="alias des modèles de génération")
    p_bench.add_argument("--embedding-model", help="un seul modèle d'embeddings, à défaut de --embedding")
    p_bench.add_argument("--generation-model", help="un seul modèle de génération, à défaut de --generation")
    p_bench.add_argument("--prompt", help="nom du prompt dans assistant/prompts/ (défaut : answer)")
    p_bench.add_argument("--questions", default="eval/questions.json")
    p_bench.add_argument("--validate-with", help="second jeu de questions, jamais vu, pour éprouver le seuil")
    p_bench.add_argument("--runs", type=int, default=3, help="passages par question (défaut 3)")
    p_bench.add_argument("--limit", type=int, help="ne garder que les N premières questions (au moins 1)")
    p_bench.add_argument("--min-score", default="config", metavar="config|auto|<n de [-1, 1]>",
                         help="'config' (défaut), 'auto' (seuil suggéré) ou une valeur")
    p_bench.add_argument("--max-chars", type=int, help="surcharge du découpage (expérience CACE)")
    p_bench.add_argument("--overlap-chars", type=int, help="surcharge du recouvrement")
    p_bench.add_argument("--seed", type=int, help="graine de génération")
    p_bench.add_argument("--out", help="dossier de sortie (défaut : eval/resultats/<date>)")

    args = parser.parse_args(argv)
    command = f"snapshot {args.snapshot_command}" if args.command == "snapshot" else args.command
    problem = _invalid_value(args)
    if problem:
        print(f"Erreur : {problem}", file=sys.stderr)
        return 1
    # Le journal des décorateurs (composition.py) : INFO avec -v, sinon le niveau de logging que nomme ASSISTANT_LOG,
    # sans tenir compte de la casse ; toute autre valeur vaut WARNING (passée telle quelle à logging.basicConfig, une
    # valeur inconnue finissait en pile d'erreurs).
    asked = os.environ.get("ASSISTANT_LOG", "").upper()
    level = asked if asked in logging.getLevelNamesMapping() else "WARNING"
    logging.basicConfig(level="INFO" if getattr(args, "verbose", False) else level,
                        format="[%(name)s] %(message)s", stream=sys.stderr)

    try:
        if args.command == "benchmark":
            from assistant.interface.benchmark import main_benchmark
            # --embedding-model / --generation-model : un seul modèle, comme pour les autres commandes.
            args.embedding = args.embedding or ([args.embedding_model] if args.embedding_model else None)
            args.generation = args.generation or ([args.generation_model] if args.generation_model else None)
            main_benchmark(args)
            return 0

        config = AppConfig.load(args.config)
        container = build(config, args.embedding_model, args.generation_model,
                          prompt_name=args.prompt)
        embedding_model = args.embedding_model or config.embedding_model
        if command in _SEARCHING and not config.has_threshold_for(embedding_model):
            print(f"Attention : aucun seuil de pertinence configuré pour « {embedding_model} » : valeur "
                  f"`default` {config.min_score_for(embedding_model)} (ADR 0004 : lancer le banc d'essai)",
                  file=sys.stderr)

        if args.command == "index":
            if args.if_stale:
                # Le cycle « réentraînement » d'un RAG : détecter que l'index ne
                # correspond plus au corpus, au découpage ou au modèle servi (status),
                # puis le reconstruire. Rien n'est réappris : on recalcule un dérivé.
                report = container.check_status.execute()
                if report.up_to_date:
                    print("Index à jour : rien à refaire.")
                    return 0
                if report.unverified:
                    print("Index non vérifié : le service IA n'a pas pu être interrogé, impossible de réindexer.",
                          file=sys.stderr)
                    return 3
                print("Index à refaire :")
                for issue in report.issues:
                    print(f"  - {issue}")
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
            from assistant.interface.benchmark import limit_questions, load_questions
            if args.snapshot_command == "record":
                questions = limit_questions(load_questions(_project_path(args.questions)), args.limit)
                snapshot = container.record_snapshot.execute(
                    args.name,
                    [SnapshotQuestion(q.id, config.user(q.user), q.question) for q in questions],
                )
                print(f"Instantané « {snapshot.name} » : {len(snapshot.entries)} réponses, "
                      f"enregistré dans {config.snapshots_dir.as_posix()}")
                for key, value in sorted(snapshot.configuration.items()):
                    print(f"  {key} = {config_value_to_text(value)}")
            elif args.snapshot_command == "compare":
                store = container.snapshots
                print(comparison_to_text(
                    compare_snapshots(store.load(args.baseline), store.load(args.candidate))))
            else:
                names = container.snapshots.names()
                print("\n".join(names) if names else f"Aucun instantané dans {config.snapshots_dir.as_posix()}")
        elif args.command == "serve":
            from assistant.interface.http_api import create_server
            try:
                server = create_server(container, args.host, args.port, quiet=args.quiet)
            except OSError as error:   # port déjà pris, adresse inconnue ou interdite
                print(f"Erreur : impossible d'écouter (port déjà pris, adresse inconnue ou non autorisée) — {error}",
                      file=sys.stderr)
                return 1
            port = server.server_address[1]   # --port 0 : le port réellement choisi
            # Une adresse à laquelle on peut se connecter : l'hôte demandé (server_address le donnerait résolu :
            # localhost → 127.0.0.1), sauf 0.0.0.0, qui n'en est pas une (toutes les interfaces) : 127.0.0.1, la
            # machine elle-même. flush : l'adresse doit se lire tout de suite, même sortie redirigée.
            everywhere = args.host == "0.0.0.0"
            address = f"http://{'127.0.0.1' if everywhere else args.host}:{port}"
            listening = ", à l'écoute sur toutes les interfaces" if everywhere else ""
            print(f"Application sur {address}{listening} (service IA : {config.ai_base_url}) — Ctrl+C pour arrêter",
                  flush=True)
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


def _invalid_value(args: argparse.Namespace) -> str | None:
    """Valeurs vérifiées avant tout travail."""
    from assistant.interface.benchmark import check_limit
    port = getattr(args, "port", None)
    if port is not None and not 0 <= port <= 65535:
        return f"l'option --port attend un port entre 0 et 65535, pas « {port} »"
    try:
        check_limit(getattr(args, "limit", None))   # au moins 1 : banc et snapshot record, même message
    except ValueError as error:
        return str(error)
    return None


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
