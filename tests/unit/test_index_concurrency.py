"""Index persisté partagé entre processus : relu quand il change, jamais mélangé (S4.2).

Les « autres processus » sont joués par d'autres instances de JsonVectorIndex sur le même
fichier, comme `serve` et la ligne de commande sur data/index.json.
"""

import errno
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

from assistant.application.errors import IndexNotBuiltError, IndexWriteError
from assistant.application.ports import IndexManifest
from assistant.application.search_passages import SearchPassages
from assistant.domain.model import Chunk, User
from assistant.infrastructure.decorators import CachedEmbedder
from assistant.infrastructure.vector_index import JsonVectorIndex
from tests.fakes import VOCABULARY, KeywordEmbedder

ALICE = User("alice", frozenset({"tous"}))
QUERY = [1.0] + [0.0] * (len(VOCABULARY) - 1)   # « télétravail »


def content(index_id: str, doc_id: str, groups=("tous",), model: str = "fake-keywords",
            dimension: int = len(VOCABULARY)):
    """Un index d'un seul morceau, qui parle de télétravail."""
    manifest = IndexManifest(index_id, model, dimension, "empreinte", {"type": "whole"}, 1, 1,
                             "2026-10-01T12:00:00+00:00")
    chunk = Chunk(f"{doc_id}#0", doc_id, doc_id.capitalize(), f"Télétravail selon {doc_id}.", 0, frozenset(groups))
    return manifest, [chunk], [[1.0] + [0.0] * (dimension - 1)]


class Meanwhile:
    """Embedder qui laisse un autre processus agir pendant son premier appel."""

    def __init__(self, inner, action) -> None:
        self._inner, self._action = inner, action

    def embed_query(self, text):
        action, self._action = self._action, None
        if action is not None:
            action()
        return self._inner.embed_query(text)

    def embed_documents(self, texts):
        return self._inner.embed_documents(texts)


class FailingWrite(JsonVectorIndex):
    def __init__(self, path) -> None:
        super().__init__(path, sleep=lambda delay: None)

    def _move_into_place(self, tmp):
        # Fichier en lecture seule, par exemple : les nouvelles tentatives n'y changent rien.
        raise PermissionError(errno.EACCES, "accès refusé")


class OtherWriterAround(JsonVectorIndex):
    """Un autre processus réécrit l'index juste avant, ou juste après, notre renommage."""

    def __init__(self, path, other, after: bool) -> None:
        super().__init__(path)
        self._other, self._after = other, after

    def _move_into_place(self, tmp):
        other, self._other = self._other, None
        if other is not None and not self._after:
            other()
        super()._move_into_place(tmp)
        if other is not None and self._after:
            other()


class ReloadedIndexTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "index.json"

    def write(self, *index):
        """`index` en ligne de commande : une nouvelle instance à chaque fois."""
        JsonVectorIndex(self.path).replace(*index)

    def test_a_question_searches_the_index_whose_model_it_checked(self):
        """A → B → A pendant une question : deux réindexations par la ligne de commande, et une autre
        requête de `serve` qui recharge B entre-temps. Les passages viennent de A, l'index annoncé."""
        self.write(*content("index-a", "a"))
        server = JsonVectorIndex(self.path)

        def meanwhile():
            self.write(*content("index-b", "b"))
            server.manifest()
            self.write(*content("index-a", "a"))

        retrieval = SearchPassages(Meanwhile(KeywordEmbedder(), meanwhile), server).execute(ALICE, "télétravail", 4)
        self.assertEqual(retrieval.manifest.index_id, "index-a")
        self.assertEqual([p.chunk.document_id for p in retrieval.passages], ["a"])

    def test_an_index_of_another_dimension_in_between_is_not_a_crash(self):
        """A → C → A, C d'une autre dimension : on recommence (A), au lieu d'une erreur 500."""
        self.write(*content("index-a", "a"))
        server = JsonVectorIndex(self.path)

        def meanwhile():
            self.write(*content("index-c", "c", model="autre-modele", dimension=2 * len(VOCABULARY)))
            server.manifest()
            self.write(*content("index-a", "a"))

        retrieval = SearchPassages(Meanwhile(KeywordEmbedder(), meanwhile), server).execute(ALICE, "télétravail", 4)
        self.assertEqual([p.chunk.document_id for p in retrieval.passages], ["a"])

    def test_the_cache_reloading_the_index_does_not_mix_two_indexes(self):
        """Le cache relit le manifeste (donc le fichier) au milieu de la question : B est chargé là,
        puis A revient pendant l'appel au service. La recherche ne porte toujours que sur A."""
        self.write(*content("index-a", "a"))
        server = JsonVectorIndex(self.path)
        first = [True]

        def current_index():
            if first[0]:
                first[0] = False
                self.write(*content("index-b", "b"))
            return server.manifest()

        embedder = CachedEmbedder(Meanwhile(KeywordEmbedder(), lambda: self.write(*content("index-a", "a"))),
                                  current_index=current_index)
        retrieval = SearchPassages(embedder, server).execute(ALICE, "télétravail", 4)
        self.assertEqual(retrieval.manifest.index_id, "index-a")
        self.assertEqual([p.chunk.document_id for p in retrieval.passages], ["a"])

    def test_a_deleted_index_file_means_no_index(self):
        """Plus de fichier, plus d'index : comme le voient la ligne de commande et un `serve` redémarré."""
        self.write(*content("index-a", "a"))
        server = JsonVectorIndex(self.path)
        self.assertEqual(server.manifest().index_id, "index-a")
        self.path.unlink()
        self.assertIsNone(server.manifest())
        self.assertIsNone(JsonVectorIndex(self.path).manifest())
        with self.assertRaises(IndexNotBuiltError):
            SearchPassages(KeywordEmbedder(), server).execute(ALICE, "télétravail", 4)
        self.write(*content("index-b", "b"))
        self.assertEqual(server.manifest().index_id, "index-b")

    def test_no_index_at_all_is_answered_without_waiting(self):
        """Pas de fichier, pas de dossier, ou un fichier supprimé : « aucun index » tout de suite. Seules les
        collisions passagères (fichier ouvert par un autre processus) méritent une nouvelle tentative."""
        slept = []
        self.assertIsNone(JsonVectorIndex(self.path, sleep=slept.append).manifest())
        self.assertIsNone(JsonVectorIndex(self.path.parent / "absent" / "index.json", sleep=slept.append).manifest())
        self.write(*content("index-a", "a"))
        server = JsonVectorIndex(self.path, sleep=slept.append)
        self.assertEqual(server.manifest().index_id, "index-a")
        self.path.unlink()
        self.assertIsNone(server.manifest())
        self.assertEqual(slept, [])

    def test_temporary_files_abandoned_for_more_than_an_hour_are_deleted_by_the_next_write(self):
        """Un rédacteur tué au milieu d'une écriture laisse son fichier temporaire : la prochaine écriture réussie le
        supprime s'il a plus d'une heure. Jamais celui d'un rédacteur actif, qui est récent ; et l'ancien
        index que met de côté le File.Replace de la version C# (….old.tmp) n'est pas touché : il est à sa charge. Un nom
        qui ne se supprime pas, ou un dossier qui ne se liste pas, ne fait jamais échouer une écriture réussie."""
        def temporary(name, age, folder=False):
            path = self.path.with_name(name)
            if folder:
                path.mkdir()
            else:
                path.write_text("{}", encoding="utf-8")
            # Vieilli par sa date de modification, que regarde le ménage ; la date d'accès reste celle d'aujourd'hui.
            os.utime(path, (time.time(), time.time() - age))
            return name

        def names():
            return sorted(p.name for p in self.path.parent.iterdir())

        killed, active, beside_a_folder, hours = uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex, 2 * 3600
        temporary(f"index.json.{killed}.tmp", hours)   # rédacteur tué
        temporary(f"index.json.{uuid.uuid4().hex}.tmp", 61 * 60)   # abandonné depuis un peu plus d'une heure
        kept = [
            temporary(f"index.json.{uuid.uuid4().hex}.tmp", 59 * 60),   # abandonné depuis un peu moins d'une heure
            temporary(f"index.json.{active}.tmp", 0),   # rédacteur actif
            temporary(f"index.json.{killed}.old.tmp", hours),   # mis de côté par la version C# : à elle de le supprimer
            temporary("index.json.sauvegarde.tmp", hours),   # pas des fichiers temporaires de l'index
            temporary(f"index.json.{uuid.uuid4().hex}.tmp.bak", hours),
            temporary(f"index.json.{uuid.uuid4().hex.upper()}.tmp", hours),   # majuscules : jamais écrit ainsi
            temporary(f"index.json.{uuid.uuid4().hex[:31]}.tmp", hours),   # 31 chiffres, puis 33 : jamais écrits ainsi
            temporary(f"index.json.{uuid.uuid4().hex}0.tmp", hours),
            temporary(f"indexXjson.{uuid.uuid4().hex}.tmp", hours),   # le « . » du nom de l'index, pris à la lettre
            temporary(f"index.json.{beside_a_folder}.tmp", hours, folder=True),   # dossier : ne se supprime pas
        ]
        before = names()
        with self.assertRaises(IndexWriteError):   # une écriture qui échoue ne fait pas le ménage
            FailingWrite(self.path).replace(*content("index-a", "a"))
        self.assertEqual(names(), before)
        # Dossier qui ne se liste pas un instant : l'écriture a réussi, le ménage attend la suivante.
        with mock.patch.object(Path, "iterdir", side_effect=PermissionError(errno.EACCES, "accès refusé")):
            self.write(*content("index-a", "a"))
        self.assertEqual(JsonVectorIndex(self.path).manifest().index_id, "index-a")
        self.assertEqual(names(), sorted(before + ["index.json"]))
        self.write(*content("index-b", "b"))
        self.assertEqual(names(), sorted(kept + ["index.json"]))

    def test_a_failed_write_leaves_the_index_in_service_unchanged(self):
        """Le fichier d'abord, la mémoire ensuite : `serve` ne sert jamais un index absent du disque."""
        self.write(*content("index-a", "a"))
        server = FailingWrite(self.path)
        self.assertEqual(server.manifest().index_id, "index-a")
        with self.assertRaises(IndexWriteError) as caught:
            server.replace(*content("index-b", "b"))
        self.assertIn(f"écriture impossible de l'index ({self.path})", str(caught.exception))
        self.assertEqual(server.manifest().index_id, "index-a")
        self.assertEqual([p.chunk.document_id for p in server.search(QUERY, 4, lambda c: True)], ["a"])
        self.assertEqual(JsonVectorIndex(self.path).manifest().index_id, "index-a")
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])   # le fichier temporaire ne traîne pas

    def test_an_invalid_index_is_refused_before_anything_is_written(self):
        """Des vecteurs de la mauvaise dimension sont refusés avant d'écrire : le fichier n'est jamais
        remplacé par un index illisible, et l'index en service reste le précédent."""
        self.write(*content("index-a", "a"))
        server = JsonVectorIndex(self.path)
        self.assertEqual(server.manifest().index_id, "index-a")
        manifest, chunks, vectors = content("index-b", "b")
        with self.assertRaises(ValueError):
            server.replace(manifest, chunks, [vectors[0] + [0.0]])
        self.assertEqual(JsonVectorIndex(self.path).manifest().index_id, "index-a")
        self.assertEqual(server.manifest().index_id, "index-a")
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_the_version_recorded_is_that_of_the_file_this_process_wrote(self):
        """Un autre processus réécrit l'index juste après notre renommage : on le relira, droits compris,
        au lieu de garder pour toujours en mémoire un index qui n'est plus sur le disque."""
        server = OtherWriterAround(self.path, lambda: self.write(*content("index-b", "b", groups=("rh",))), after=True)
        server.replace(*content("index-a", "a"))
        self.assertEqual(server.manifest().index_id, "index-b")
        self.assertEqual(server.search(QUERY, 4, lambda c: True)[0].chunk.allowed_groups, frozenset({"rh"}))

    def test_two_simultaneous_reindexations_do_not_share_a_temporary_file(self):
        """Une autre réindexation complète pendant la nôtre : chacune son fichier temporaire, la dernière
        installée gagne, et tout le monde la voit."""
        other = JsonVectorIndex(self.path)
        server = OtherWriterAround(self.path, lambda: other.replace(*content("index-b", "b")), after=False)
        server.replace(*content("index-a", "a"))
        self.assertEqual(JsonVectorIndex(self.path).manifest().index_id, "index-a")
        self.assertEqual(server.manifest().index_id, "index-a")
        self.assertEqual(other.manifest().index_id, "index-a")
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_a_reader_holding_the_file_does_not_make_the_reindexation_fail(self):
        """Sous Windows, un fichier ouvert (par `ask`, `status`, un autre `serve`…) ne se remplace pas :
        on réessaie brièvement. Ailleurs, le remplacement réussit du premier coup."""
        self.write(*content("index-a", "a"))
        reader = open(self.path, "rb")
        self.addCleanup(reader.close)
        writer = JsonVectorIndex(self.path, sleep=lambda delay: reader.close())   # le lecteur a fini entre-temps
        writer.replace(*content("index-b", "b"))
        self.assertEqual(JsonVectorIndex(self.path).manifest().index_id, "index-b")

    @unittest.skipUnless(os.name == "nt", "verrou d'octets propre à Windows (msvcrt)")
    def test_a_file_that_cannot_be_read_for_a_moment_is_read_again(self):
        """Pendant qu'un autre processus le remplace, le fichier ne se lit pas un court instant."""
        import msvcrt

        self.write(*content("index-a", "a"))
        holder = open(self.path, "rb")
        self.addCleanup(holder.close)
        msvcrt.locking(holder.fileno(), msvcrt.LK_NBLCK, 1)

        def release(delay):
            msvcrt.locking(holder.fileno(), msvcrt.LK_UNLCK, 1)

        self.assertEqual(JsonVectorIndex(self.path, sleep=release).manifest().index_id, "index-a")


if __name__ == "__main__":
    unittest.main()
