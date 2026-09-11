"""Lancement : python -m ai_service [--config config/ai_service.toml]"""

from __future__ import annotations

import argparse
import tomllib

from .registry import ModelRegistry
from .server import create_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Service IA : embeddings et génération en HTTP")
    parser.add_argument("--config", default="config/ai_service.toml")
    parser.add_argument("--host", help="adresse d'écoute (défaut : celle du fichier)")
    parser.add_argument("--port", type=int, help="port d'écoute (défaut : celui du fichier)")
    args = parser.parse_args()

    with open(args.config, "rb") as handle:
        config = tomllib.load(handle)
    registry = ModelRegistry.from_dict(config)
    server_config = config.get("server", {})
    host = args.host or server_config.get("host", "127.0.0.1")
    port = args.port or server_config.get("port", 8100)

    server = create_server(registry, host, port)
    models = registry.describe()
    print(f"Service IA sur http://{host}:{port}")
    print("  embeddings :", ", ".join(m["alias"] for m in models["embedding"]))
    print("  génération :", ", ".join(m["alias"] for m in models["generation"]))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du service IA.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
