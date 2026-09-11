"""L'importateur Service-Public : une fiche XML devient un document du corpus."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "tools"))

from import_service_public import parse_fiche, to_markdown  # noqa: E402

from assistant.infrastructure.markdown_corpus import parse_markdown_document  # noqa: E402

FICHE = """<?xml version="1.0" encoding="UTF-8"?>
<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F9999" type="Fiche d'information"
             spUrl="https://www.service-public.gouv.fr/particuliers/vosdroits/F9999">
  <dc:title>Congé pour événement familial</dc:title>
  <dc:subject>Travail - Formation</dc:subject>
  <dc:date>modified 2026-03-02</dc:date>
  <dc:identifier>F9999</dc:identifier>
  <SousThemePere ID="N1">Congés</SousThemePere>
  <DossierPere ID="N2"><Titre>{dossier}</Titre></DossierPere>
  <Introduction><Texte><Paragraphe>Le salarié a droit à un congé.</Paragraphe></Texte></Introduction>
  <Texte>
    <Chapitre><Titre><Paragraphe>Durée</Paragraphe></Titre>
      <Paragraphe>Elle dépend de l'événement&#160;:</Paragraphe>
      <Liste type="puce"><Item><Paragraphe>mariage : 4 jours</Paragraphe></Item>
        <Item><Paragraphe>naissance : 3 jours</Paragraphe></Item></Liste>
      <SousChapitre><Titre><Paragraphe>Cas du 2<Exposant>e</Exposant> mariage</Paragraphe></Titre>
        <Paragraphe>Mêmes droits.</Paragraphe></SousChapitre>
    </Chapitre>
    <ANoter><Titre>À noter</Titre><Paragraphe>Le congé est rémunéré.</Paragraphe></ANoter>
    <Tableau><Rangée type="header"><Cellule><Paragraphe>Événement</Paragraphe></Cellule><Cellule><Paragraphe>Jours</Paragraphe></Cellule></Rangée>
      <Rangée><Cellule><Paragraphe>Décès</Paragraphe></Cellule><Cellule><Paragraphe>3</Paragraphe></Cellule></Rangée></Tableau>
    <OuSAdresser ID="R1"><Titre>Contact</Titre><Paragraphe>Ne doit pas apparaître.</Paragraphe></OuSAdresser>
  </Texte>
  <Reference type="Texte de référence" URL="https://www.legifrance.gouv.fr/x"><Titre>Code du travail</Titre></Reference>
</Publication>
"""


class ImportServicePublicTest(unittest.TestCase):
    def _fiche(self, dossier="Congés dans le secteur privé"):
        return parse_fiche(FICHE.format(dossier=dossier).encode("utf-8"))

    def test_fiche_is_converted_with_headings_lists_tables_and_callouts(self):
        fiche = self._fiche()
        self.assertEqual(fiche.id, "F9999")
        self.assertEqual(fiche.modified, "2026-03-02")
        self.assertEqual(fiche.groups, ("tous",))
        body = fiche.body
        self.assertIn("Le salarié a droit à un congé.", body)
        self.assertIn("## Durée", body)
        self.assertIn("### Cas du 2e mariage", body)
        self.assertIn("- mariage : 4 jours\n- naissance : 3 jours", body)
        self.assertIn("À noter : Le congé est rémunéré.", body)
        self.assertIn("| Événement | Jours |\n|---|---|\n| Décès | 3 |", body)
        self.assertIn("Elle dépend de l'événement :", body)  # espace insécable normalisée

    def test_contacts_and_legal_references_are_dropped(self):
        body = self._fiche().body
        self.assertNotIn("Ne doit pas apparaître", body)
        self.assertNotIn("Code du travail", body)

    def test_restricted_dossiers_get_simulated_groups(self):
        self.assertEqual(self._fiche("Licenciement économique").groups, ("rh",))
        self.assertEqual(self._fiche("Conflits du travail dans le secteur privé").groups, ("direction",))

    def test_out_of_scope_fiche_is_skipped(self):
        self.assertIsNone(self._fiche("Congés dans la fonction publique"))

    def test_markdown_is_readable_by_the_corpus_adapter(self):
        document = parse_markdown_document(to_markdown(self._fiche("Licenciement économique")))
        self.assertEqual(document.id, "F9999")
        self.assertEqual(document.title, "Congé pour événement familial")
        self.assertEqual(document.allowed_groups, frozenset({"rh"}))
        self.assertTrue(document.text.startswith("# Congé pour événement familial"))


if __name__ == "__main__":
    unittest.main()
