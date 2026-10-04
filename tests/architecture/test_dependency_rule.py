"""Règle de dépendance, vérifiée automatiquement (séquence 2.2).

- le domaine n'importe que la bibliothèque standard et lui-même ;
- l'application n'importe que le domaine et elle-même ;
- aucune des deux ne touche au réseau, aux fichiers de config ou aux modèles ;
- la bibliothèque standard leur reste permise, json compris (l'identité de l'index, dans index_corpus.py) ;
- l'interface (CLI, API, banc d'essai) ne construit aucun adaptateur : seule la
  racine de composition connaît l'infrastructure (séquence 4.1) ;
- l'application et le service IA sont deux déployables qui ne partagent aucun code.

Les imports relatifs (`from ..infrastructure import …`) sont résolus : c'est le style
du dépôt, une règle qui les ignorerait se contournerait sans effort. Et tout module doit
appartenir à une couche : un nouveau fichier hors de la table est une erreur, pas un oubli.

L'accès par attribut compte comme un import (`import assistant.composition as c` puis
`c.JsonVectorIndex`). Et la racine de composition ne sert pas de relais : on ne lui prend
que ce qu'elle lie en clair (import, affectation, def, class) sans toucher à l'infrastructure
au chargement. Ce que la règle ne pourrait pas suivre est refusé : `import *`, le module
passé comme une valeur (`getattr(composition, …)`), un nom qu'elle ne voit pas défini.

Limite : l'analyse lit les noms et leurs liaisons, pas ce que font les appels (un décorateur en
est un) ni l'introspection : le corps d'une fonction de la composition n'est pas lu (assembler
les adaptateurs est son rôle ; sa signature l'est), ni `registre.append(…)`, ni une écriture
dynamique (`globals()`, `setattr`) sur un nom déjà défini, ni `build.__globals__`, ni un import
dynamique (`importlib`, `__import__`). La portée des noms n'est pas suivie : un paramètre homonyme
d'un module importé compte comme lui. Le test attrape les erreurs, pas la malveillance.
"""

import ast
import importlib.util
import tempfile
import unittest
from collections.abc import Iterable, Iterator
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
COMPOSITION = "assistant.composition"
INFRASTRUCTURE = "assistant.infrastructure"


def _within(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def _layer_of(module: str) -> tuple[str | None, tuple[str, ...]]:
    if module == ROOT_PACKAGE:
        return ROOT_PACKAGE, ()
    for layer, allowed in LAYERS.items():
        if _within(module, layer):
            return layer, allowed
    return None, ()


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _parse(path: Path, root: Path) -> tuple[str, str, ast.Module]:
    """Nom du module, paquet d'où partent ses imports relatifs, et arbre syntaxique."""
    module = _module_name(path, root)
    anchor = module if path.name == "__init__.py" else module.rpartition(".")[0]
    return module, anchor, ast.parse(path.read_text(encoding="utf-8"))


def _imports(nodes: Iterable[ast.AST], anchor: str) -> Iterator[tuple[str, str, str]]:
    """(nom importé, nom lié dans le fichier, ce que désigne ce nom), imports relatifs résolus :
    `import a.b` importe a.b et lie a (→ a) ; `import a.b as c` lie c (→ a.b) ;
    `from ..x import y` importe et lie y (→ assistant.x.y) ; `from x import *` importe x.*."""
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                yield alias.name, alias.asname or top, alias.name if alias.asname else top
        elif isinstance(node, ast.ImportFrom):
            source = (node.module or "") if node.level == 0 else importlib.util.resolve_name(
                "." * node.level + (node.module or ""), anchor)
            for alias in node.names:
                yield f"{source}.{alias.name}", alias.asname or alias.name, f"{source}.{alias.name}"


def _dotted(node: ast.expr) -> str | None:
    """`c.JsonVectorIndex.load` → "c.JsonVectorIndex.load" ; None si la chaîne ne part pas d'un nom."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and (base := _dotted(node.value)):
        return f"{base}.{node.attr}"
    return None


def _read(path: Path, root: Path) -> tuple[str, set[str], set[str]]:
    """Le module, les noms complets qu'il importe, et ceux qu'il emploie par un nom importé :
    `import assistant.composition as c` puis `c.build()` emploie assistant.composition.build,
    et `m = c` le module lui-même. Seule la chaîne entière compte (c.X.y, pas aussi c.X)."""
    module, anchor, tree = _parse(path, root)
    imports = list(_imports(ast.walk(tree), anchor))
    bound = {local: target for _, local, target in imports}
    inner = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and id(node) not in inner and (chain := _dotted(node)):
            head, _, rest = chain.partition(".")
            if head in bound:
                used.add(f"{bound[head]}.{rest}" if rest else bound[head])
    return module, {name for name, _, _ in imports}, used


def imports_of(package: Path, root: Path = ROOT) -> dict[str, set[str]]:
    """Ce que chaque fichier prend ailleurs, en noms complets : ce qu'il importe (`from
    ..infrastructure.cache import X` donne assistant.infrastructure.cache.X) et ce qu'il
    emploie par un nom importé (`c.JsonVectorIndex` donne assistant.composition.JsonVectorIndex)."""
    found: dict[str, set[str]] = {}
    files = [package] if package.is_file() else package.rglob("*.py")
    for path in files:
        module, imported, used = _read(path, root)
        found[module] = imported | used
    return found


def _at_load(node: ast.AST) -> Iterator[ast.AST]:
    """Les nœuds d'une instruction de niveau module qui s'exécutent au chargement : tous, sauf
    le corps des fonctions (def), qui ne s'exécute qu'à l'appel."""
    yield node
    for field, value in ast.iter_fields(node):
        if field == "body" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in value if isinstance(value, list) else [value]:
            if isinstance(child, ast.AST):
                yield from _at_load(child)


def _binds(node: ast.AST) -> set[str]:
    """Les noms qu'un nœud lie ou modifie : `x = …` sous toutes ses formes (for, with, :=, del…),
    `x.a = …` et `x[k] = …` (qui modifient x), le nom d'un def, d'une class, d'un `except … as`
    ou d'une capture de `match`. Les imports sont lus à part."""
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        if isinstance(node.ctx, ast.Load):
            return set()
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return {node.id} if isinstance(node, ast.Name) else set()
    if isinstance(node, ast.alias):
        return set()
    return {value for value in (getattr(node, "name", None), getattr(node, "rest", None)) if isinstance(value, str)}


def _lent(composition: Path, root: Path) -> tuple[set[str], set[str], set[str]]:
    """Ce que la racine de composition peut prêter : (noms qu'elle lie en clair, ceux d'entre eux
    qui touchent à l'infrastructure, problèmes). Un nom y touche s'il en est importé (`import
    assistant.x` lie assistant), ou s'il est lié par une instruction de niveau module qui nomme
    un nom qui y touche, de proche en proche et dans n'importe quel ordre. L'instruction se prend
    entière : bloc if/try/for/with/match, classe avec son corps ; d'une fonction, l'en-tête seul."""
    _, anchor, tree = _parse(composition, root)
    units: list[tuple[set[str], set[str]]] = []
    defined: set[str] = set()
    touched: set[str] = set()
    problems: set[str] = set()
    for statement in tree.body:
        nodes = list(_at_load(statement))
        bound = {name for node in nodes for name in _binds(node)}
        for name, local, target in _imports(nodes, anchor):
            bound.add(local)
            if local == "*":
                problems.add(f"{COMPOSITION} importe {name} : on ne verrait plus ce qu'elle expose")
            elif _within(target, INFRASTRUCTURE) or _within(INFRASTRUCTURE, target):
                touched.add(local)
        units.append(({node.id for node in nodes if isinstance(node, ast.Name)}, bound))
        defined |= bound
    changed = True
    while changed:
        changed = False
        for named, bound in units:
            if named & touched and not bound <= touched:
                touched |= bound
                changed = True
    return defined, touched, problems


def reexport_violations(root: Path = ROOT) -> list[str]:
    """La racine de composition importe l'infrastructure : lui reprendre un nom qui y touche, c'est
    importer l'infrastructure par la porte de derrière. Toutes les écritures comptent :
    `from ..composition import X`, l'attribut (`composition.X`, `c.X`), et `import *` ou le module
    passé comme une valeur, refusés d'office (on ne verrait plus ce qui est pris)."""
    composition = root / ROOT_PACKAGE / "composition.py"
    if not composition.is_file():
        return []
    defined, touched, problems = _lent(composition, root)
    for path in (root / ROOT_PACKAGE).rglob("*.py"):
        module, imported, used = _read(path, root)
        if COMPOSITION in used:
            problems.add(f"{module} se sert du module {COMPOSITION} comme d'une valeur : on ne verrait plus ce qu'il y prend")
        for name in imported | used:
            if not name.startswith(COMPOSITION + "."):
                continue
            taken = name.removeprefix(COMPOSITION + ".").split(".")[0]
            if taken == "*":
                problems.add(f"{module} importe {name} : on ne verrait plus ce qu'il y prend")
            elif taken in touched:
                problems.add(f"{module} importe {taken} (infrastructure) par {COMPOSITION}")
            elif taken not in defined:
                problems.add(f"{module} importe {taken} par {COMPOSITION}, où la règle ne le voit pas défini")
    return sorted(problems)


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
            if top == ROOT_PACKAGE and not any(_within(name, a) for a in allowed):
                problems.append(f"{module} importe {name}")
    return problems


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


PACKAGES = {"assistant/__init__.py": "", "assistant/application/__init__.py": "",
            "assistant/infrastructure/__init__.py": "", "assistant/interface/__init__.py": "",
            "assistant/infrastructure/vector_index.py": "class JsonVectorIndex:\n    pass\n"}


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
            self.assertEqual(layer_violations(root), [])   # pas encore de racine de composition : rien à relayer
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
                "assistant importe assistant.infrastructure.cache",
                "assistant.application.use_case importe assistant.infrastructure.cache.X",
                "assistant.interface.api importe assistant.infrastructure",
                "assistant.interface.presenter importe X (infrastructure) par assistant.composition",
                "assistant.outils n'appartient à aucune couche (à ajouter dans LAYERS)",
            ])

    def test_the_composition_root_lends_nothing_that_touches_the_infrastructure(self):
        """Test du test : la racine de composition lie des noms de toutes les façons, et un module
        de l'interface les lui prend tous. Ce qui touche à l'infrastructure au chargement est refusé,
        ce qu'elle ne lie pas en clair aussi ; ce qu'elle définit sans y toucher reste permis."""
        composition = (
            "from __future__ import annotations\n"
            "import assistant.infrastructure.vector_index\n"            # lie assistant
            "from pathlib import Path\n"
            "from .infrastructure.vector_index import JsonVectorIndex\n"
            "from .infrastructure import vector_index as vi\n"
            "from .infrastructure import *\n"
            "try:\n    import rapide as Rapide\nexcept ImportError:\n"
            "    from .infrastructure.vector_index import JsonVectorIndex as Rapide\n"
            "RACINE = Path('.')\n"
            "Defaut = JsonVectorIndex(RACINE)\n"                      # RACINE, seulement lu, reste propre
            "Index = JsonVectorIndex\n"
            "Annote: type = Index\n"
            "for Boucle in (Index,):\n    pass\n"
            "match {'index': Index}:\n    case {**Reste}:\n        pass\n"
            "Registre = {'index': {}}\nRegistre['index']['classe'] = Index\n"
            "class Sous(vi.JsonVectorIndex):\n    pass\n"
            "class Boite:\n    index = JsonVectorIndex\n"
            "def ouvre(chemin) -> Adaptateur:\n    pass\n"              # nom défini plus bas : l'ordre ne compte pas
            "Adaptateur = JsonVectorIndex\n"
            "def fabrique():\n    pass\n"
            "fabrique.index = JsonVectorIndex\n"
            "globals()['Cache'] = JsonVectorIndex\n"
            "class Config:\n    racine: Path = RACINE\n"
            "def build(config: Config):\n    return JsonVectorIndex(config.racine)\n"
            "async def charge(config: Config):\n    return JsonVectorIndex(config.racine)\n")
        touching = ["assistant", "JsonVectorIndex", "vi", "Rapide", "Defaut", "Index", "Annote", "Boucle", "Reste",
                    "Registre", "Sous", "Boite", "ouvre", "Adaptateur", "fabrique"]
        unseen = ["vector_index", "Cache"]            # liés par `*` ou par globals() : la règle ne les voit pas
        permitted = ["Path", "RACINE", "Config", "build", "charge"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tree(root, {**PACKAGES, "assistant/composition.py": composition,
                               "assistant/interface/prend.py":
                                   f"from ..composition import ({', '.join(touching + unseen + permitted)})\n"})
            self.assertEqual(sorted(layer_violations(root)), sorted([
                "assistant.composition importe assistant.infrastructure.* : on ne verrait plus ce qu'elle expose",
                *(f"assistant.interface.prend importe {name} (infrastructure) par assistant.composition" for name in touching),
                *(f"assistant.interface.prend importe {name} par assistant.composition, où la règle ne le voit pas défini"
                  for name in unseen),
            ]))

    def test_the_rule_sees_every_way_of_taking_from_the_composition_root(self):
        """Test du test : chaque module de l'interface prend à la racine de composition d'une autre
        façon. L'adaptateur doit être vu quelle que soit l'écriture, et ce que la règle ne pourrait pas
        suivre (`import *`, le module passé comme une valeur) est refusé ; build reste permis partout."""
        leak = "importe JsonVectorIndex (infrastructure) par assistant.composition"
        unseen = "on ne verrait plus ce qu'il y prend"
        interface = {   # module de l'interface → (son code, ce que la règle doit en dire ; None : rien)
            "direct": ("from ..composition import JsonVectorIndex\n", leak),
            "attribut": ("from .. import composition\nindex = composition.JsonVectorIndex()\n", leak),
            "alias": ("import assistant.composition as c\nindex = c.JsonVectorIndex()\n", leak),
            "absolu": ("from assistant import composition\nindex = composition.JsonVectorIndex.load()\n", leak),
            "complet": ("import assistant.composition\nindex = assistant.composition.JsonVectorIndex()\n", leak),
            "etoile": ("from ..composition import *\n", f"importe assistant.composition.* : {unseen}"),
            "valeur": ("from .. import composition\nindex = getattr(composition, 'JsonVectorIndex')()\n",
                       f"se sert du module assistant.composition comme d'une valeur : {unseen}"),
            "racine": ("import assistant.application\nindex = assistant.infrastructure.vector_index.JsonVectorIndex()\n",
                       "importe assistant.infrastructure.vector_index.JsonVectorIndex"),
            "permis": ("from assistant import composition\nimport assistant.composition as c\nimport assistant.composition\n"
                       "from ..composition import build\n"
                       "containers = composition.build(), c.build(), assistant.composition.build(), build()\n", None),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_tree(root, {
                **PACKAGES,
                "assistant/composition.py": "from .infrastructure.vector_index import JsonVectorIndex\n"
                                            "def build():\n    return JsonVectorIndex()\n",
                **{f"assistant/interface/{name}.py": code for name, (code, _) in interface.items()},
            })
            self.assertEqual(sorted(layer_violations(root)), sorted(
                f"assistant.interface.{name} {said}" for name, (_, said) in interface.items() if said))


if __name__ == "__main__":
    unittest.main()
