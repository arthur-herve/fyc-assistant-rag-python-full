"""Importe des fiches pratiques Service-Public.gouv.fr dans le format du corpus.

Source : « Fiches pratiques et ressources de Service-Public.gouv.fr — Particuliers »,
Direction de l'information légale et administrative (DILA), data.gouv.fr,
Licence Ouverte / Open Licence 2.0. Archive XML : voir ARCHIVE_URL.

Chaque fiche XML devient un fichier Markdown avec l'en-tête attendu par
`assistant/infrastructure/markdown_corpus.py` (id, titre, groupes), complété
par la source, la date et le thème pour l'attribution exigée par la licence.

Les droits d'accès sont une SIMULATION pédagogique : les fiches sont publiques,
mais le fil rouge joue un intranet d'entreprise où certains dossiers (recrutement,
licenciement, rupture, conflits du travail) sont réservés aux RH ou à la direction.
Le tableau DOSSIERS ci-dessous est la seule règle ; il est documenté dans
corpus/README.md.

Usage, depuis la racine du projet :

    python tools/import_service_public.py --download            # télécharge puis importe
    python tools/import_service_public.py --archive vosdroits-latest.zip
    python tools/import_service_public.py --archive ... --limit 50 --out corpus/essai

Bibliothèque standard uniquement.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ARCHIVE_URL = "https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip"
DATASET_URL = (
    "https://www.data.gouv.fr/datasets/"
    "fiches-pratiques-et-ressources-de-service-public-gouv-fr-particuliers"
)
THEME = "Travail - Formation"

# Dossiers retenus (secteur privé uniquement : le fil rouge est un intranet
# d'entreprise) et groupes autorisés à les lire. « tous » = tout salarié.
DOSSIERS: dict[str, tuple[str, ...]] = {
    "Conditions de travail dans le secteur privé": ("tous",),
    "Maladie ou accident du travail dans le secteur privé": ("tous",),
    "Handicap et emploi dans le secteur privé": ("tous",),
    "Congés dans le secteur privé": ("tous",),
    "Contrats de travail dans le secteur privé": ("tous",),
    "Contrats d'insertion": ("tous",),
    "Retraite d'un salarié du secteur privé": ("tous",),
    "Formation des salariés du secteur privé": ("tous",),
    "Formation des personnes handicapées": ("tous",),
    "Stage en entreprise": ("tous",),
    "Temps de travail dans le secteur privé": ("tous",),
    "Représentation du personnel dans l'entreprise": ("tous",),
    "Recrutement dans le secteur privé": ("rh",),
    "Licenciement d'un salarié du secteur privé pour motif personnel": ("rh",),
    "Licenciement économique": ("rh",),
    "Rupture du contrat de travail dans le secteur privé": ("rh",),
    "Conflits du travail dans le secteur privé": ("direction",),
}

DC = "{http://purl.org/dc/elements/1.1/}"

# Éléments de bloc ignorés : contacts, services en ligne, références légales,
# définitions et renvois. Ils n'apportent pas de contenu à interroger.
SKIPPED = {
    "OuSAdresser", "ServiceEnLigne", "ServiceEnLigneAnnexe", "Complement", "Reference",
    "Definition", "Abreviation", "QuestionReponse", "VoirAussi", "PourEnSavoirPlus",
    "TitreRiche", "Titre", "Theme", "FilDAriane", "SousThemePere", "DossierPere",
    "SurTitre", "Audience", "Canal", "Colonne", "LienExterne", "LienWeb", "RessourceWeb",
    "Video", "Infographie", "Image", "Questionnaire", "QuiPeutMAider", "PivotLocal",
    "NoticeLiee", "RechercheGuidee", "RechercheGuideePere", "FicheConnexe", "Aide",
    "Problematiques", "Connexion", "CommentFaireSi", "Dossier", "FranceConnect",
}
CALLOUTS = {"ANoter": "À noter", "ASavoir": "À savoir", "Attention": "Attention",
            "Rappel": "Rappel", "Exemple": "Exemple", "Avertissement": "Avertissement"}


@dataclass(frozen=True)
class Fiche:
    id: str
    title: str
    theme: str
    dossier: str
    url: str
    modified: str
    groups: tuple[str, ...]
    body: str


def _text(node: ET.Element) -> str:
    """Texte d'un élément, espaces normalisés (l'espace insécable compris)."""
    raw = "".join(node.itertext())
    return re.sub(r"\s+", " ", raw.replace("\xa0", " ")).strip()


def _title(node: ET.Element) -> str:
    title = node.find("Titre")
    return _text(title) if title is not None else ""


def _render(node: ET.Element, level: int, out: list[str]) -> None:
    """Parcourt un bloc XML et ajoute des blocs Markdown (séparés par des lignes vides)."""
    for child in node:
        tag = child.tag
        if tag in SKIPPED:
            continue
        if tag == "Paragraphe":
            text = _text(child)
            if text:
                out.append(text)
        elif tag in ("Situation", "Cas", "Chapitre", "SousChapitre"):
            title = _title(child)
            if title:
                out.append(f"{'#' * min(level, 6)} {title}")
            _render(child, level + 1, out)
        elif tag == "Liste":
            out.append(_list(child, 0))
        elif tag == "Tableau":
            out.append(_table(child))
        elif tag in CALLOUTS:
            body = " ".join(_text(p) for p in child.iter("Paragraphe"))
            if body:
                out.append(f"{CALLOUTS[tag]} : {body}")
        elif tag == "TitreFlottant":
            text = _text(child)
            if text:
                out.append(f"**{text}**")
        else:
            # Texte, Introduction, ListeSituations, BlocCas… : conteneurs.
            _render(child, level, out)


def _list(node: ET.Element, indent: int) -> str:
    lines: list[str] = []
    for item in node.findall("Item"):
        parts = [_text(p) for p in item.findall("Paragraphe")]
        lines.append("  " * indent + "- " + " ".join(p for p in parts if p))
        for sub in item.findall("Liste"):
            lines.append(_list(sub, indent + 1))
    return "\n".join(lines)


def _table(node: ET.Element) -> str:
    lines: list[str] = []
    caption = node.find("TitreRiche")
    if caption is not None and _text(caption):
        lines.append(f"*{_text(caption)}*")
        lines.append("")
    rows = node.findall("Rangée")
    for i, row in enumerate(rows):
        cells = [_text(c) for c in row.findall("Cellule")]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "---|" * len(cells))
    return "\n".join(lines)


def parse_fiche(xml: bytes) -> Fiche | None:
    """Convertit une fiche XML ; None si elle n'est pas dans le périmètre."""
    root = ET.fromstring(xml)
    theme = root.findtext(f"{DC}subject") or ""
    dossier = root.findtext("DossierPere/Titre") or ""
    if theme != THEME or dossier not in DOSSIERS:
        return None
    blocks: list[str] = []
    _render(root, 2, blocks)
    body = "\n\n".join(b for b in blocks if b.strip())
    if not body:
        return None
    modified = (root.findtext(f"{DC}date") or "").replace("modified ", "")[:10]
    return Fiche(
        id=root.get("ID") or root.findtext(f"{DC}identifier") or "",
        title=_text(root.find(f"{DC}title")),
        theme=theme,
        dossier=dossier,
        url=root.get("spUrl") or "",
        modified=modified,
        groups=DOSSIERS[dossier],
        body=body,
    )


def to_markdown(fiche: Fiche) -> str:
    header = "\n".join([
        "---",
        f"id: {fiche.id}",
        f"titre: {fiche.title}",
        f"groupes: {', '.join(fiche.groups)}",
        f"theme: {fiche.theme}",
        f"dossier: {fiche.dossier}",
        f"source: {fiche.url}",
        f"date: {fiche.modified}",
        "licence: Licence Ouverte 2.0 — Service-Public.gouv.fr / DILA",
        "---",
    ])
    return f"{header}\n# {fiche.title}\n\n{fiche.body}\n"


def import_archive(archive: Path, out_dir: Path, limit: int | None = None) -> list[Fiche]:
    fiches: list[Fiche] = []
    with zipfile.ZipFile(archive) as zf:
        names = sorted(n for n in zf.namelist() if n.startswith("F") and n.endswith(".xml"))
        for name in names:
            fiche = parse_fiche(zf.read(name))
            if fiche is not None:
                fiches.append(fiche)
    fiches.sort(key=lambda f: (f.dossier, f.id))
    if limit:
        fiches = fiches[:limit]
    if out_dir.exists():
        others = [p.name for p in out_dir.iterdir() if p.suffix != ".md" or not p.name.startswith("F")]
        if others:
            raise SystemExit(f"{out_dir} contient autre chose que des fiches importées ({others[:3]}…) : "
                             "choisir un dossier dédié avec --out")
        for old in out_dir.glob("F*.md"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fiche in fiches:
        (out_dir / f"{fiche.id}.md").write_text(to_markdown(fiche), encoding="utf-8", newline="\n")
    return fiches


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", default="vosdroits-latest.zip",
                        help="archive XML de la DILA (défaut : vosdroits-latest.zip)")
    parser.add_argument("--download", action="store_true",
                        help=f"télécharger l'archive depuis {ARCHIVE_URL}")
    parser.add_argument("--out", default="corpus/service-public", help="dossier de sortie")
    parser.add_argument("--limit", type=int, help="ne garder que les N premières fiches")
    args = parser.parse_args(argv)

    archive = Path(args.archive)
    if args.download:
        print(f"Téléchargement de {ARCHIVE_URL} …")
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
    if not archive.exists():
        print(f"Archive introuvable : {archive} (utiliser --download)", file=sys.stderr)
        return 1

    fiches = import_archive(archive, Path(args.out), args.limit)
    by_dossier: dict[str, int] = {}
    for fiche in fiches:
        by_dossier[fiche.dossier] = by_dossier.get(fiche.dossier, 0) + 1
    print(f"{len(fiches)} fiches écrites dans {args.out} (téléchargement : {date.today().isoformat()}, "
          f"dernière modification : {max((f.modified for f in fiches), default='')})")
    for dossier, count in sorted(by_dossier.items()):
        print(f"  {count:4d}  {dossier}  → {', '.join(DOSSIERS[dossier])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
