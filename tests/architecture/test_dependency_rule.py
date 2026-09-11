"""Règle de dépendance, vérifiée automatiquement (séquence 2.2).

- le domaine n'importe que la bibliothèque standard et lui-même ;
- l'application n'importe que le domaine et elle-même ;
- aucune des deux ne touche au réseau, aux fichiers de config ou aux modèles ;
- l'interface (CLI, API, banc d'essai) ne construit aucun adaptateur : seule la
  racine de composition connaît l'infrastructure (séquence 4.1) ;
- l'application et le service IA sont deux déployables qui ne partagent aucun code.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IN_CORE = {"urllib", "http", "socket", "tomllib", "requests", "httpx", "numpy",
                     "sentence_transformers", "torch", "openai", "ollama"}


def imports_of(package: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in package.rglob("*.py"):
        names: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module)
        found[str(path.relative_to(ROOT))] = names
    return found


def violations(package: str, allowed_internal: tuple[str, ...]) -> list[str]:
    problems = []
    for file, names in imports_of(ROOT / package).items():
        for name in names:
            top = name.split(".")[0]
            if top in FORBIDDEN_IN_CORE:
                problems.append(f"{file} importe {name}")
            if top in ("assistant", "ai_service") and not name.startswith(allowed_internal):
                problems.append(f"{file} importe {name}")
    return problems


class DependencyRuleTest(unittest.TestCase):
    def test_domain_depends_on_nothing(self):
        self.assertEqual(violations("assistant/domain", ("assistant.domain",)), [])

    def test_application_depends_only_on_domain(self):
        self.assertEqual(
            violations("assistant/application", ("assistant.domain", "assistant.application")), [])

    def test_infrastructure_does_not_depend_on_interface(self):
        problems = [f"{f} importe {n}" for f, names in imports_of(ROOT / "assistant/infrastructure").items()
                    for n in names if n.startswith(("assistant.interface", "assistant.composition"))]
        self.assertEqual(problems, [])

    def test_only_the_composition_root_knows_the_infrastructure(self):
        """Décorateurs, adaptateurs HTTP, fichiers : assemblés dans composition.py, nulle part ailleurs."""
        problems = [f"{f} importe {n}" for f, names in imports_of(ROOT / "assistant/interface").items()
                    for n in names if n.startswith("assistant.infrastructure")]
        problems += [f"{f} importe {n}" for f, names in imports_of(ROOT / "assistant/application").items()
                     for n in names if n.startswith("assistant.infrastructure")]
        self.assertEqual(problems, [])

    def test_application_and_ai_service_share_no_code(self):
        problems = [f"{f} importe {n}" for f, names in imports_of(ROOT / "assistant").items()
                    for n in names if n.startswith("ai_service")]
        problems += [f"{f} importe {n}" for f, names in imports_of(ROOT / "ai_service").items()
                     for n in names if n.startswith("assistant")]
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
