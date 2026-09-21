"""Règle de dépendance, vérifiée automatiquement (séquence 2.2).

- le domaine n'importe que la bibliothèque standard et lui-même ;
- l'application n'importe que le domaine et elle-même ;
- aucune des deux ne touche au réseau, aux fichiers de config ou aux modèles ;
- l'interface (CLI, API, banc d'essai) ne construit aucun adaptateur : seule la
  racine de composition connaît l'infrastructure (séquence 4.1) ;
- l'application et le service IA sont deux déployables qui ne partagent aucun code.

Les imports relatifs (`from ..infrastructure import …`) sont résolus : c'est le style
du dépôt, une règle qui les ignorerait se contournerait sans effort. Et tout module doit
appartenir à une couche : un nouveau fichier hors de la table est une erreur, pas un oubli.
"""

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IN_CORE = {"urllib", "http", "socket", "tomllib", "requests", "httpx", "numpy",
                     "sentence_transformers", "torch", "openai", "ollama"}

# Couche (module ou paquet) → couches qu'elle a le droit d'importer.
LAYERS = {
    "assistant.domain": ("assistant.domain",),
    "assistant.application": ("assistant.domain", "assistant.application"),
    "assistant.infrastructure": ("assistant.domain", "assistant.application", "assistant.infrastructure"),
    "assistant.interface": ("assistant.domain", "assistant.application", "assistant.interface",
                            "assistant.composition"),
    "assistant.__main__": ("assistant.domain", "assistant.application", "assistant.interface",
                           "assistant.composition"),
    "assistant.composition": ("assistant",),
}
ROOT_PACKAGE = "assistant"   # assistant/__init__.py : n'importe rien du projet


def _layer_of(module: str) -> tuple[str | None, tuple[str, ...]]:
    if module == ROOT_PACKAGE:
        return ROOT_PACKAGE, ()
    for layer, allowed in LAYERS.items():
        if module == layer or module.startswith(layer + "."):
            return layer, allowed
    return None, ()


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports_of(package: Path, root: Path = ROOT) -> dict[str, set[str]]:
    """Modules importés par chaque fichier, imports relatifs résolus en noms absolus."""
    found: dict[str, set[str]] = {}
    files = [package] if package.is_file() else package.rglob("*.py")
    for path in files:
        module = _module_name(path, root)
        is_package = path.name == "__init__.py"
        names: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    names.add(node.module or "")
                else:
                    anchor = module if is_package else module.rpartition(".")[0]
                    base = importlib.util.resolve_name("." * node.level + (node.module or ""), anchor)
                    if node.module:                      # from ..infrastructure.cache import X
                        names.add(base)
                    else:                                # from .. import infrastructure
                        names.update(f"{base}.{alias.name}" for alias in node.names)
        found[module] = names
    return found


def _from_imports(path: Path, root: Path) -> list[tuple[str, str]]:
    """(module d'origine résolu, nom importé) pour chaque `from X import nom` du fichier."""
    module = _module_name(path, root)
    anchor = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            source = node.module if node.level == 0 else importlib.util.resolve_name(
                "." * node.level + node.module, anchor)
            found += [(source, alias.name) for alias in node.names]
    return found


def reexport_violations(root: Path = ROOT) -> list[str]:
    """La racine de composition importe l'infrastructure : lui reprendre un de ces noms, c'est
    importer l'infrastructure par la porte de derrière."""
    composition = root / ROOT_PACKAGE / "composition.py"
    if not composition.is_file():
        return []
    adapters = {name for source, name in _from_imports(composition, root)
                if source.startswith("assistant.infrastructure")}
    problems = []
    for path in (root / ROOT_PACKAGE).rglob("*.py"):
        module = _module_name(path, root)
        for source, name in _from_imports(path, root):
            if source == "assistant.composition" and name in adapters:
                problems.append(f"{module} importe {name} (infrastructure) par assistant.composition")
    return problems


def layer_violations(root: Path = ROOT) -> list[str]:
    problems = reexport_violations(root)
    for module, names in imports_of(root / ROOT_PACKAGE, root).items():
        layer, allowed = _layer_of(module)
        if layer is None:
            problems.append(f"{module} n'appartient à aucune couche (à ajouter dans LAYERS)")
            continue
        for name in names:
            top = name.split(".")[0]
            if layer in ("assistant.domain", "assistant.application") and top in FORBIDDEN_IN_CORE:
                problems.append(f"{module} importe {name}")
            if top == ROOT_PACKAGE and not any(name == a or name.startswith(a + ".") for a in allowed):
                problems.append(f"{module} importe {name}")
    return problems


class DependencyRuleTest(unittest.TestCase):
    def test_every_layer_imports_only_what_it_may(self):
        self.assertEqual(layer_violations(), [])

    def test_application_and_ai_service_share_no_code(self):
        problems = [f"{f} importe {n}" for f, names in imports_of(ROOT / "assistant").items()
                    for n in names if n.startswith("ai_service")]
        problems += [f"{f} importe {n}" for f, names in imports_of(ROOT / "ai_service").items()
                     for n in names if n.startswith("assistant")]
        self.assertEqual(problems, [])

    def test_the_rule_catches_a_relative_import_and_a_module_outside_the_layers(self):
        """Test du test : un import interdit, écrit comme le reste du dépôt, doit être vu ; un
        module hors couche aussi, même s'il se contente d'importer l'infrastructure."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for package in ("assistant", "assistant/application", "assistant/infrastructure", "assistant/interface"):
                (root / package).mkdir()
                (root / package / "__init__.py").write_text("", encoding="utf-8")
            (root / "assistant/infrastructure/cache.py").write_text("X = 1\n", encoding="utf-8")
            (root / "assistant/application/use_case.py").write_text(
                "from ..infrastructure.cache import X\n", encoding="utf-8")
            (root / "assistant/interface/api.py").write_text(
                "from .. import infrastructure\n", encoding="utf-8")
            (root / "assistant/outils.py").write_text(
                "from .infrastructure.cache import X\n", encoding="utf-8")
            (root / "assistant/__init__.py").write_text(
                "from .infrastructure import cache\n", encoding="utf-8")
            (root / "assistant/composition.py").write_text(
                "from .infrastructure.cache import X\n", encoding="utf-8")
            (root / "assistant/interface/presenter.py").write_text(
                "from ..composition import X\n", encoding="utf-8")
            self.assertEqual(sorted(layer_violations(root)), [
                "assistant importe assistant.infrastructure",
                "assistant.application.use_case importe assistant.infrastructure.cache",
                "assistant.interface.api importe assistant.infrastructure",
                "assistant.interface.presenter importe X (infrastructure) par assistant.composition",
                "assistant.outils n'appartient à aucune couche (à ajouter dans LAYERS)",
            ])


if __name__ == "__main__":
    unittest.main()
