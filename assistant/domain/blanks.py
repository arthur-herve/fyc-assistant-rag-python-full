"""Les blancs de la seule règle de sortie de l'exercice S4.1 (output_rules.py) : la réponse en est rognée, et ils
séparent les mots d'un marqueur de raisonnement. Ce sont les blancs d'Unicode (propriété White_Space), ceux de
char.IsWhiteSpace en .NET, que nomme l'énoncé. Définis une seule fois, ici."""

# str.strip(), str.isspace() et le \s de Python y ajoutent les séparateurs \x1c à \x1f, des caractères de contrôle
# qu'Unicode ne compte pas parmi les blancs : avec eux, « \x1c » suivi de 1 500 lettres ne serait pas une réponse
# trop longue, et « ok,\x1clet » serait un raisonnement déversé.
WHITESPACE = ("\t\n\v\f\r \x85\xa0\u1680" + "".join(map(chr, range(0x2000, 0x200B)))
              + "\u2028\u2029\u202f\u205f\u3000")   # pour str.strip()
BLANK = r"[^\S\x1c-\x1f]"   # un blanc de WHITESPACE, dans une expression régulière
