"""La documentation ne cite que des tests qui existent.

Un nom de test cité par une page (README, docs/, exercices/, cas-pratique/…) est une promesse :
l'apprenant le cherche dans le code. Un test renommé ou supprimé sans que la page suive le laisse
chercher en vain ; c'est arrivé au corrigé S4.1, qui citait un test d'architecture disparu.

Un nom de test s'écrit comme du code : entre accents graves, ou dans un bloc de code. La prose n'est
pas lue. `test_x`, `Classe.test_x`, `tests.unit.test_cli.Classe.test_x` et `fichier.py::Classe::test_x`
y citent la fonction test_x ; un module ou un chemin (`tests.unit.test_cli`, `tests/unit/test_cli.py`)
n'est pas un nom de test.

À l'inverse, le tableau « Tests » du README nomme chaque dossier de tests/.
"""

import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIPPED = {".git", "__pycache__", ".venv", "data"}   # ni du dépôt, ni de la documentation
_CODE = re.compile(r"```.*?```|`[^`\n]+`", re.S)     # bloc de code, ou passage entre accents graves
_DOTTED = re.compile(r"[\w.:/]+")                    # un nom, pointé ou non, ou un chemin
_SEPARATOR = re.compile(r"(::|[.:/])")
_TEST = re.compile(r"test_\w+")
_DEFINED = re.compile(r"^\s*(?:async\s+)?def\s+(test_\w+)\s*\(", re.M)


def _files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.rglob(pattern) if not SKIPPED & set(path.relative_to(root).parts))


def cited_tests(markdown: str) -> set[str]:
    """Les noms de tests que cite une page : dans ce qu'elle écrit comme du code, un `test_…` qui finit
    le nom, seul, après « :: » ou après une classe (« Classe.test_x ») ; pas un module ni un fichier."""
    names = set()
    for dotted in _DOTTED.findall(" ".join(_CODE.findall(markdown))):
        parts = _SEPARATOR.split(dotted)   # nom, séparateur, nom… : « a.B.c » donne a . B . c
        last = parts[-1]
        before = parts[-2] if len(parts) > 1 else ""
        owner = parts[-3] if len(parts) > 2 else ""
        if _TEST.fullmatch(last) and (before in ("", "::") or before == "." and owner[:1].isupper()):
            names.add(last)
    return names


def missing_tests(root: Path) -> list[str]:
    """Chaque test cité par un .md et défini nulle part dans les .py du dépôt (kits d'exercices compris)."""
    defined = {name for path in _files(root, "*.py") for name in _DEFINED.findall(path.read_text(encoding="utf-8"))}
    return [f"{page.relative_to(root).as_posix()} cite {name}, qu'aucun fichier .py ne définit"
            for page in _files(root, "*.md")
            for name in sorted(cited_tests(page.read_text(encoding="utf-8")) - defined)]


class CitedTestsTest(unittest.TestCase):
    def test_every_test_cited_by_the_documentation_exists(self):
        self.assertEqual(missing_tests(ROOT), [])

    def test_the_check_reads_what_a_page_writes_as_code(self):
        """Test du test : le cas du corrigé S4.1 (un test d'architecture remplacé, la page pas mise à jour),
        et un test renommé cité dans un bloc de code. Un test d'un kit d'exercices compte ; un module, un
        fichier ou la prose ne sont pas des noms de tests."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kit/tests").mkdir(parents=True)
            (root / "kit/tests/test_regle.py").write_text(
                "class RegleTest:\n    def test_every_layer_imports_only_what_it_may(self):\n        pass\n",
                encoding="utf-8")
            (root / "README.md").write_text(
                "| 5 | `kit/tests/test_regle.py` | `test_only_the_composition_root_knows_the_infrastructure` |\n"
                "Lancer `python -m unittest tests.test_regle` (`tests/test_regle.py::RegleTest`) ; "
                "test_cite_en_prose n'est pas lu.\n"
                "```bash\npython -m unittest tests.test_regle.RegleTest.test_every_layer_imports_only_what_it_may\n"
                "python -m unittest tests.test_regle.RegleTest.test_renomme_depuis\n```\n",
                encoding="utf-8")
            self.assertEqual(missing_tests(root), [
                f"README.md cite {name}, qu'aucun fichier .py ne définit"
                for name in ("test_only_the_composition_root_knows_the_infrastructure", "test_renomme_depuis")
            ])

    def test_the_readme_lists_every_test_folder(self):
        """L'inverse, pour les dossiers : le tableau « Tests » du README en nomme chacun (tests/tools/ y manquait).
        Un dossier de tests est un paquet (avec un __init__.py), le seul que unittest parcourt : un cache d'outil
        (.pytest_cache, .mypy_cache, __pycache__) n'en est pas un."""
        folders = sorted(path.name for path in (ROOT / "tests").iterdir() if (path / "__init__.py").is_file())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertEqual([name for name in folders if f"| `tests/{name}/` |" not in readme], [])


if __name__ == "__main__":
    unittest.main()
