"""Expérience (séquence 3.1) : rien ne change, et pourtant les réponses bougent.

On enregistre N instantanés avec exactement la même configuration et on
compare chaque passage au premier : la dérive mesurée est le non-déterminisme
du générateur. Avec --seed, on montre qu'une graine le réduit sans l'annuler.

    python tools/experiences/stabilite.py --config config/app-ollama.toml \
        --questions eval/questions-service-public.json --runs 3 [--seed 42]
"""

from __future__ import annotations

from dataclasses import replace

from _commun import STATUS_CHANGED, Experiment, drift_summary, parser


def main() -> None:
    p = parser(__doc__.split("\n")[0])
    p.add_argument("--runs", type=int, default=3, help="nombre de passages, au moins 2 (défaut 3)")
    p.add_argument("--seed", type=int, help="graine de génération (défaut : aucune)")
    args = p.parse_args()
    if args.runs < 2:
        raise SystemExit("--runs doit valoir au moins 2 : il faut deux passages pour mesurer une dérive")
    exp = Experiment("stabilite", args)
    if args.seed is not None:
        exp.config = replace(exp.config, seed=args.seed)

    container, manifest, _ = exp.index("partage")
    snapshots = [exp.record(f"passage-{i}", container)[0] for i in range(1, args.runs + 1)]

    exp.log(f"Rien ne change entre les passages : même index `{manifest.index_id}`, même générateur "
            f"`{snapshots[0].configuration.get('generation_model_id')}`, même prompt, même seuil, "
            f"température {exp.config.temperature}, graine {exp.config.seed if exp.config.seed is not None else 'aucune'}.")
    exp.log("")
    exp.log("## Chaque passage comparé au premier")
    exp.log("")
    comparisons = {f"passage 1 → {i}": exp.compare(snapshots[0], snapshot)
                   for i, snapshot in enumerate(snapshots[1:], start=2)}
    exp.table({label: drift_summary(comparison) for label, comparison in comparisons.items()})
    drifts = [c.drift_rate for c in comparisons.values() if c.drift_rate is not None]
    mean_drift = round(sum(drifts) / len(drifts), 3) if drifts else None
    statuses = sum(c.count(STATUS_CHANGED) for c in comparisons.values())
    exp.log(f"**Dérive moyenne à configuration constante : {mean_drift if mean_drift is not None else '—'}** "
            f"({statuses} changement(s) de statut sur {len(comparisons)} comparaison(s)).")
    exp.log("")
    exp.log("## Indicateurs par passage")
    exp.log("")
    exp.table({f"passage {i}": exp.stats(snapshot) for i, snapshot in enumerate(snapshots, start=1)})
    exp.log("## Statuts par question")
    exp.log("")
    exp.log("| Question | " + " | ".join(f"passage {i}" for i in range(1, args.runs + 1)) + " |")
    exp.log("|---|" + "---|" * args.runs)
    for q in exp.questions:
        cells = []
        for snapshot in snapshots:
            entry = next((e for e in snapshot.entries if e.question_id == q.id), None)
            cells.append(entry.status if entry else "—")
        marker = " ⚠" if len(set(cells)) > 1 else ""
        exp.log(f"| {q.id}{marker} | " + " | ".join(cells) + " |")
    exp.log("")
    exp.log("## Lecture")
    exp.log("")
    exp.log("- C'est la mesure de base de la séquence 3.1 : un test par assertion exacte sur ces réponses "
            "échouerait au hasard. On teste donc une *proportion* (taux de réponses sourcées, de refus justes) "
            "avec une tolérance, et on documente la probabilité de faux échec.")
    exp.log("- Le texte change presque toujours d'un passage à l'autre, et un « texte modifié » peut inverser la réponse (Oui devenu Non) : il se relit, la couverture des mots-clés le mesure grossièrement. Les changements de statut (⚠) existent aussi à configuration constante : un test statistique doit les borner avec une tolérance, pas exiger zéro.")
    exp.log("- Une graine réduit la variabilité pour un même modèle et un même moteur ; elle ne garantit rien "
            "d'un modèle à l'autre, ni d'une version d'Ollama à l'autre.")
    exp.write()


if __name__ == "__main__":
    main()
