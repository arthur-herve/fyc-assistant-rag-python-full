"""Adaptateur : JSON lu par l'application (index, instantanés, jeux de questions, réponses du service IA ;
corps des requêtes HTTP et GET /v1/models, par la racine de composition).

Lu strictement : json.loads garderait sans rien dire la dernière valeur d'une clé en double, accepterait
NaN et Infinity (ce n'est pas du JSON), refuserait un entier de plus de 4300 chiffres avec un message de
CPython, en anglais (int() ne le convertit pas), et lèverait RecursionError, une trace d'erreur, vers un
millier de niveaux d'imbrication. Ici, tout cela est une ValueError, avec un message en français. Sur
demande (`strings` : index, instantanés, jeux de questions, corps des requêtes HTTP, GET /v1/models), une
chaîne qui n'est pas du texte (« \\ud800 » isolé : du JSON valide, que json.loads garde, mais qui ne
s'écrit pas en UTF-8) est refusée aussi.
Au-delà de 900 niveaux d'imbrication, le texte est refusé d'emblée, avant json.loads, quoi qu'il
contienne d'autre : json.loads est récursif, et abandonne plus ou moins loin selon la version de Python
et la pile d'appels déjà là (en 3.11, à 1000 niveaux moins la profondeur de la pile : vers 990 en tête de
programme, 982 dans le serveur HTTP ; vers 3 000 en 3.13). Avec 900, il lit tout ce que la pré-lecture
laisse passer, sous toute version, depuis une pile de moins de 90 appels environ.
"""

from __future__ import annotations

import json
import re
from typing import Any, NoReturn

READ_DEPTH = 900   # niveaux d'imbrication : au-delà, refusé avant json.loads (pré-lecture)
MAX_DIGITS = 4300   # chiffres d'un entier : au-delà, int() refuse de le convertir
TOO_DEEP = f"JSON trop imbriqué : plus de {READ_DEPTH} niveaux"
NOT_TEXT = "chaîne qui n'est pas du texte : surrogate UTF-16 isolé (\\ud800 à \\udfff sans sa paire)"
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")   # moitié de paire UTF-16 : pas du texte
# Ce qui seul peut en donner une : un échappement « \ud800 » à « \udfff », en majuscules ou en minuscules. Le texte
# vient toujours d'un décodage UTF-8 strict (decode_utf8, read_utf8, corps HTTP, réponse du service IA), qui refuse
# une moitié de paire écrite telle quelle. Une recherche rapide (un préfixe littéral : 15 ms sur 40 Mo).
_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")
# Ce que la pré-lecture saute : une chaîne, même non fermée (un antislash garde le caractère qui le suit),
# puis tout ce qui n'est ni crochet ni accolade. « *+ » : répétition possessive (Python 3.11 et plus). Elle trouve
# les mêmes chaînes que « * » (« "? » réussit toujours : rien n'est à reprendre), sans garder de quoi revenir en
# arrière sur chaque morceau de la chaîne : avec « * », plus de 100 octets par échappement, soit 1,7 Gio pour un
# corps HTTP de 16 Mio plein de « \n ».
_STRING = re.compile(r'"(?:[^"\\]+|\\.)*+"?', re.DOTALL)
_NOT_BRACKET = re.compile(r"[^\[\]{}]+")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Une clé en double est refusée. Appelé à la fermeture de chaque objet : un objet imbriqué est
    vérifié avant celui qui le contient."""
    values: dict[str, Any] = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"clé « {key} » en double")
        values[key] = value
    return values


def _not_json(constant: str) -> NoReturn:
    """NaN, Infinity et -Infinity : json.loads les accepte, mais ce n'est pas du JSON (RFC 8259)."""
    raise ValueError(f"« {constant} » n'est pas du JSON")


def _integer(literal: str) -> int:
    """Un entier lu en JSON. Au-delà de 4300 chiffres (signe non compté), int() refuse de le convertir,
    avec un message de CPython, en anglais : refusé ici avec un message en français."""
    if len(literal.lstrip("-")) > MAX_DIGITS:
        raise ValueError(f"nombre entier de plus de {MAX_DIGITS} chiffres")
    return int(literal)


def _check_read_depth(text: str) -> None:
    """Plus de 900 niveaux d'imbrication (READ_DEPTH) : refusé d'emblée, avant json.loads, même si un autre défaut
    vient avant dans le texte. Les crochets et les accolades sont comptés hors des chaînes, sans lire la syntaxe."""
    brackets = _NOT_BRACKET.sub("", _STRING.sub("", text))
    if brackets.count("[") + brackets.count("{") <= READ_DEPTH:
        return   # pas plus de 900 ouvrants : pas plus de 900 niveaux
    depth = 0
    for bracket in brackets:
        depth += 1 if bracket in "[{" else -1
        if depth > READ_DEPTH:
            raise ValueError(TOO_DEEP)


def _check_text(value: Any) -> None:
    """Avec `strings` : une chaîne (clé comprise) qui n'est pas du texte est refusée. Un parcours sans
    récursion : les 900 niveaux que laisse passer la pré-lecture ne butent pas sur la pile d'appels."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if _LONE_SURROGATE.search(item):
                raise ValueError(NOT_TEXT)
        elif isinstance(item, dict):
            pending.extend(item)   # les clés
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def parse(text: str, *, strings: bool = False) -> Any:
    """La valeur JSON de `text` ; ValueError si ce n'est pas du JSON strict : syntaxe (JSONDecodeError),
    clé en double, NaN ou Infinity, entier de plus de 4300 chiffres, plus de 900 niveaux et, avec `strings`,
    chaîne qui n'est pas du texte. Le message ne nomme pas le fichier : l'appelant le préfixe. `text` vient d'un
    décodage UTF-8 strict : une moitié de paire n'y est jamais écrite telle quelle (voir _SURROGATE_ESCAPE)."""
    _check_read_depth(text)
    try:
        value = json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_not_json, parse_int=_integer)
    except RecursionError:   # filet : sous Python 3.11, une pile d'appels de plus de 90 appels environ
        raise ValueError(TOO_DEEP) from None
    # Les chaînes ne sont parcourues que si l'une d'elles peut ne pas être du texte : un index de 3 500 morceaux
    # de 1 024 dimensions, ce sont 3,6 millions de réels à parcourir pour rien.
    if strings and _SURROGATE_ESCAPE.search(text):
        _check_text(value)
    return value
