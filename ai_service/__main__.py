"""Lancement : python -m ai_service [--config config/ai_service.toml]"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib

from .registry import ConfigError, ModelRegistry
from .server import create_server


def _error(message: str) -> int:
    """« Erreur : … » sur la sortie d'erreur, sans pile, et le code 1 : comme « assistant »."""
    print(f"Erreur : {message}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Service IA : embeddings et génération en HTTP")
    parser.add_argument("--config", default="config/ai_service.toml")
    parser.add_argument("--host", help="adresse d'écoute (défaut : celle du fichier)")
    parser.add_argument("--port", type=int, help="port d'écoute (défaut : celui du fichier)")
    args = parser.parse_args()

    try:
        with open(args.config, "rb") as handle:
            config = tomllib.load(handle)
    except OSError as error:   # absent, dossier, droits
        return _error(f"fichier de configuration inaccessible — {error}")
    except ValueError as error:   # TOML invalide : TOMLDecodeError, ou UnicodeDecodeError (pas de l'UTF-8)
        return _error(f"{args.config} : TOML invalide — {error}")
    try:
        registry = ModelRegistry.from_dict(config)
    except ConfigError as error:   # moteur inconnu, champ « backend » absent
        return _error(f"{args.config} : {error}")
    server_config = config.get("server", {})
    host = args.host or server_config.get("host", "127.0.0.1")
    port = args.port if args.port is not None else server_config.get("port", 8100)
    # Sinon une pile à l'ouverture du port (OverflowError, TypeError), ou true (TOML) pris pour le port 1.
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        # Les messages d'« assistant » : celui de « serve » pour l'option, celui de sa configuration pour le fichier.
        if args.port is not None:
            return _error(f"l'option --port attend un port entre 0 et 65535, pas « {port} »")
        return _error(f"{args.config} [server] : « port » = {json.dumps(port)}, doit être un entier compris entre 0 et "
                      "65535")

    try:
        server = create_server(registry, host, port)
    except OSError as error:   # port déjà pris, adresse inconnue ou interdite : comme « assistant serve »
        return _error(f"impossible d'écouter (port déjà pris, adresse inconnue ou non autorisée) — {error}")
    models = registry.describe()
    # Une adresse où se connecter, comme « assistant serve » : l'hôte demandé, sauf 0.0.0.0, qui n'en est pas une
    # (toutes les interfaces) : 127.0.0.1, la machine elle-même, en le disant.
    everywhere = host == "0.0.0.0"
    address = f"http://{'127.0.0.1' if everywhere else host}:{server.server_address[1]}"
    listening = ", à l'écoute sur toutes les interfaces" if everywhere else ""
    # flush, comme « assistant serve » : l'adresse doit se lire tout de suite, même sortie redirigée (avec --port 0,
    # c'est le seul moyen de connaître le port). Sur la dernière ligne : les trois partent ensemble.
    print(f"Service IA sur {address}{listening}")
    print("  embeddings :", ", ".join(m["alias"] for m in models["embedding"]))
    print("  génération :", ", ".join(m["alias"] for m in models["generation"]), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du service IA.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
