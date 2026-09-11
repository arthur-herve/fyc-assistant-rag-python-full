"""Expérience (séquences 2.3 et 3.2) : changer le modèle d'embeddings.

Second verdict de la problématique : l'index ne survit pas au changement.
On montre l'erreur, on réindexe, puis on mesure ce qui a bougé.

    python tools/experiences/changement_embeddings.py --config config/app-ollama.toml \
        --questions eval/questions-service-public.json --other nomic
"""

from __future__ import annotations

from _commun import Experiment, drift_summary, parser
from assistant.application.errors import AIServiceError, IndexModelMismatchError


def main() -> None:
    p = parser(__doc__.split("\n")[0])
    p.add_argument("--other", required=True, help="alias du second modèle d'embeddings (ex. nomic)")
    args = p.parse_args()
    exp = Experiment("changement-embeddings", args)
    first = exp.config.embedding_model

    print(f"Avant : {first}")
    container_a, manifest_a, seconds_a = exp.index("avant")
    snapshot_a, _ = exp.record("avant", container_a)

    # 1. Le même index interrogé avec l'autre modèle : refus explicite.
    mismatch = None
    try:
        exp.container("avant", embedding_model=args.other).ask_question.execute(
            exp.config.user(exp.questions[0].user), exp.questions[0].question)
    except IndexModelMismatchError as error:
        mismatch = str(error)
    except AIServiceError as error:
        raise SystemExit(f"le service IA ne sert pas « {args.other} » : {error}")
    print("  sans réindexer :", mismatch or "AUCUNE ERREUR (inattendu)")

    # 2. Réindexation, puis mêmes questions.
    print(f"Après : {args.other}")
    container_b, manifest_b, seconds_b = exp.index("apres", embedding_model=args.other)
    snapshot_b, _ = exp.record("apres", container_b)

    comparison = exp.compare(snapshot_a, snapshot_b)
    exp.log(f"Une seule chose change : le modèle d'embeddings, `{first}` → `{args.other}`. "
            "Même corpus, même découpage, même prompt, même générateur ; le seuil est celui configuré "
            "pour chaque modèle.")
    exp.log("")
    exp.log("## 1. Sans réindexer : l'application refuse")
    exp.log("")
    exp.log("```")
    exp.log(mismatch or "aucune erreur levée")
    exp.log("```")
    exp.log("")
    exp.log("## 2. Après réindexation")
    exp.log("")
    exp.table({
        f"avant (`{first}`)": {"modèle servi": manifest_a.embedding_model, "dimension": manifest_a.dimension,
                               "morceaux": manifest_a.chunk_count, "indexation (s)": round(seconds_a, 1),
                               "seuil": container_a.settings.min_score, **exp.stats(snapshot_a)},
        f"après (`{args.other}`)": {"modèle servi": manifest_b.embedding_model, "dimension": manifest_b.dimension,
                                    "morceaux": manifest_b.chunk_count, "indexation (s)": round(seconds_b, 1),
                                    "seuil": container_b.settings.min_score, **exp.stats(snapshot_b)},
    })
    exp.log("## Dérive")
    exp.log("")
    exp.table({"avant → après": drift_summary(comparison)})
    exp.comparison(comparison, "Détail")
    exp.log("## Lecture")
    exp.log("")
    exp.log("- Changer de modèle d'embeddings coûte une réindexation complète (voir la durée) : les vecteurs "
            "stockés vivent dans l'espace du modèle qui les a produits.")
    exp.log("- Le refus est explicite parce que le service IA renvoie l'identifiant concret du modèle et que "
            "l'application le compare au manifeste à chaque question (ADR 0003). Sans cela, à dimension égale, "
            "l'index aurait répondu à côté sans rien signaler.")
    exp.log("- Après réindexation, le générateur n'a pas changé et pourtant les réponses bougent : il ne répond "
            "qu'à partir de ce que la recherche lui donne.")
    exp.write()


if __name__ == "__main__":
    main()
