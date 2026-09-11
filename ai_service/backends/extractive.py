"""Génération hors-ligne : recopie la phrase la plus proche de la question.

Double de test assumé : il connaît le format des passages envoyés par
l'application (« [n] Titre » puis le texte). C'est un couplage volontaire,
à signaler aux apprenants.

Avec `citation_failure_rate` > 0 et `randomize`, il simule les défauts d'un
vrai modèle (réponses variables, citations oubliées) sans aucun GPU : utile
pour travailler la séquence 3.1 hors-ligne.
"""

from __future__ import annotations

import random
import re
import unicodedata

_MARKER = re.compile(r"^\[(\d+)\]\s", re.MULTILINE)
_QUESTION = re.compile(r"Question\s*:\s*(.+)", re.IGNORECASE)


def _words(text: str) -> set[str]:
    ascii_text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return {w[:6] for w in re.findall(r"[a-z0-9]+", ascii_text) if len(w) > 3}


class ExtractiveGenerationBackend:
    def __init__(self, citation_failure_rate: float = 0.0, randomize: bool = False,
                 model_id: str = "extractive") -> None:
        self._failure_rate = citation_failure_rate
        self._randomize = randomize
        self._model_id = model_id

    def generate(self, system, prompt, temperature, max_tokens, seed):
        rng = random.Random(seed)  # seed=None : aléatoire à chaque appel
        match = _QUESTION.search(prompt)
        question_words = _words(match.group(1)) if match else set()
        body = prompt[: match.start()] if match else prompt

        candidates: list[tuple[int, int, str]] = []
        markers = list(_MARKER.finditer(body))
        for i, marker in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
            number = int(marker.group(1))
            lines = body[marker.end():end].strip().splitlines()[1:]  # 1re ligne : titre
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # on ignore les titres Markdown
                for sentence in re.split(r"(?<=[.!?])\s+", line):
                    sentence = sentence.strip(" -*")
                    if len(sentence) > 20:
                        overlap = len(question_words & _words(sentence))
                        candidates.append((overlap, number, sentence))

        if not candidates:
            return self._model_id, "Les documents fournis ne permettent pas de répondre."

        candidates.sort(key=lambda c: c[0], reverse=True)
        pool = candidates[:2] if self._randomize and len(candidates) > 1 else candidates[:1]
        _, number, sentence = rng.choice(pool)
        if rng.random() < self._failure_rate:
            return self._model_id, sentence
        return self._model_id, f"{sentence} [{number}]"
