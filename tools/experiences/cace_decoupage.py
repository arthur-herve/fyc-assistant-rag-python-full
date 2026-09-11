"""Expérience CACE (séquence 3.2) : changer le découpage, et rien d'autre.

Le découpage n'est pas une règle métier — le domaine ignore jusqu'à l'existence
des morceaux — et pourtant il change les réponses, les scores et le seuil.

    python tools/experiences/cace_decoupage.py --config config/app-ollama.toml \
        --questions eval/questions-service-public.json --max-chars 300 --overlap-chars 50
"""

from __future__ import annotations

from _commun import Experiment, drift_summary, parser


def main() -> None:
    p = parser(__doc__.split("\n")[0])
    p.add_argument("--max-chars", type=int, default=300, help="taille des morceaux « après » (défaut 300)")
    p.add_argument("--overlap-chars", type=int, default=50, help="recouvrement « après » (défaut 50)")
    args = p.parse_args()
    exp = Experiment("cace-decoupage", args)
    before_splitter = exp.config.splitter
    after_splitter = {**before_splitter, "max_chars": args.max_chars, "overlap_chars": args.overlap_chars}

    print("Avant :", before_splitter)
    container_a, manifest_a, seconds_a = exp.index("avant")
    snapshot_a, _ = exp.record("avant", container_a)

    print("Après :", after_splitter)
    container_b, manifest_b, seconds_b = exp.index("apres", splitter_overrides=after_splitter)
    snapshot_b, _ = exp.record("apres", container_b)

    comparison = exp.compare(snapshot_a, snapshot_b)
    exp.log("Une seule chose change : la taille des morceaux. Même corpus, même modèle d'embeddings, "
            "même seuil, même prompt, même générateur.")
    exp.log("")
    exp.log("## Les deux index")
    exp.log("")
    exp.table({
        f"avant ({before_splitter['max_chars']} / {before_splitter['overlap_chars']})": {
            "morceaux": manifest_a.chunk_count, "indexation (s)": round(seconds_a, 1),
            "identifiant de l'index": manifest_a.index_id, **exp.stats(snapshot_a)},
        f"après ({after_splitter['max_chars']} / {after_splitter['overlap_chars']})": {
            "morceaux": manifest_b.chunk_count, "indexation (s)": round(seconds_b, 1),
            "identifiant de l'index": manifest_b.index_id, **exp.stats(snapshot_b)},
    })
    exp.log("## Dérive")
    exp.log("")
    exp.table({"avant → après": drift_summary(comparison)})
    exp.comparison(comparison, "Détail")
    exp.log("## Lecture")
    exp.log("")
    exp.log("- L'identifiant de l'index change : le découpage fait partie de ce qui définit un index "
            "(manifeste), au même titre que le corpus et le modèle.")
    exp.log("- Le seuil de pertinence n'a pas été recalibré : des morceaux plus courts donnent des scores "
            "différents, donc des refus et des réponses qui bougent sans qu'aucune règle métier n'ait changé. "
            "C'est le principe CACE : *changing anything changes everything*.")
    exp.log("- Ce que le taux de dérive ne dit pas : laquelle des deux versions répond le mieux. Pour cela, "
            "regarder « bonne source » et « refus justes » ci-dessus, question par question.")
    exp.write()


if __name__ == "__main__":
    main()
