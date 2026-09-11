"""Expérience (séquences 2.3 et 3.3) : changer le modèle de génération.

Premier verdict de la problématique : le même index sert aux deux, rien n'est
réindexé. Et sa nuance : les réponses, la latence et les rejets changent.

    python tools/experiences/changement_generateur.py --config config/app-ollama.toml \
        --questions eval/questions-service-public.json --other qwen3-4b --limit 12
"""

from __future__ import annotations

from _commun import Experiment, drift_summary, parser


def main() -> None:
    p = parser(__doc__.split("\n")[0])
    p.add_argument("--other", required=True, help="alias du second modèle de génération (ex. qwen3-4b)")
    args = p.parse_args()
    exp = Experiment("changement-generateur", args)
    first = exp.config.generation_model

    container_a, manifest, _ = exp.index("partage")
    print(f"Avant : {first}")
    snapshot_a, seconds_a = exp.record("avant", container_a)
    print(f"Après : {args.other}")
    container_b = exp.container("partage", generation_model=args.other)
    snapshot_b, seconds_b = exp.record("apres", container_b)
    same_index = snapshot_a.configuration.get("index_id") == snapshot_b.configuration.get("index_id")

    def rejected(snapshot):
        return sum(1 for e in snapshot.entries if e.status == "unsourced")

    comparison = exp.compare(snapshot_a, snapshot_b)
    exp.log(f"Une seule chose change : le modèle de génération, `{first}` → `{args.other}`. "
            f"L'index `{manifest.index_id}` est construit une fois et partagé : "
            + ("**il n'est reconstruit à aucun moment**." if same_index else "ATTENTION, index différent."))
    exp.log("")
    exp.log("## Les deux générateurs")
    exp.log("")
    exp.table({
        f"avant (`{first}`)": {"modèle servi": snapshot_a.configuration.get("generation_model_id"),
                               "durée totale (s)": round(seconds_a, 1),
                               "par question (s)": round(seconds_a / max(1, len(exp.questions)), 1),
                               "réponses non sourcées": rejected(snapshot_a), **exp.stats(snapshot_a)},
        f"après (`{args.other}`)": {"modèle servi": snapshot_b.configuration.get("generation_model_id"),
                                    "durée totale (s)": round(seconds_b, 1),
                                    "par question (s)": round(seconds_b / max(1, len(exp.questions)), 1),
                                    "réponses non sourcées": rejected(snapshot_b), **exp.stats(snapshot_b)},
    })
    exp.log("## Dérive")
    exp.log("")
    exp.table({"avant → après": drift_summary(comparison)})
    exp.comparison(comparison, "Détail")
    exp.log("## Lecture")
    exp.log("")
    exp.log("- Aucune donnée stockée n'a changé : même index, même manifeste. « Le générateur est un détail » "
            "est vrai pour les données.")
    exp.log("- Mais les textes, la latence et le nombre de sorties rejetées par le garde-fou de forme changent : "
            "le prompt est réglé pour un modèle, chaque modèle a ses manies (longueur, langue, raisonnement). "
            "Une ligne de configuration, oui ; sans conséquence, non.")
    exp.write()


if __name__ == "__main__":
    main()
