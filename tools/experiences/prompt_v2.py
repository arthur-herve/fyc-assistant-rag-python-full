"""Expérience (séquence 3.3) : changer le prompt, et rien d'autre.

Le prompt est-il de la configuration ou de la logique métier ? On mesure ce
qu'une reformulation change (answer → answer-v2), à index, modèles et seuil
identiques. Avec le générateur hors-ligne `extractive`, qui ignore les
consignes, la dérive attendue est 0 : il faut un vrai modèle pour voir l'effet.

    python tools/experiences/prompt_v2.py --config config/app-ollama.toml \
        --questions eval/questions-service-public.json
"""

from __future__ import annotations

from _commun import ANSWERED, Experiment, drift_summary, parser, run


def mean_length(snapshot) -> int | None:
    """La longueur moyenne des réponses données, en caractères (len : un emoji compte pour un), arrondie à l'entier
    (round : à égalité, vers le pair) ; None sans réponse donnée."""
    answered = [len(e.text) for e in snapshot.entries if e.status == ANSWERED]
    return round(sum(answered) / len(answered)) if answered else None


def main() -> None:
    p = parser(__doc__.split("\n")[0])
    p.add_argument("--other", default="answer-v2", help="nom du second prompt (défaut answer-v2)")
    args = p.parse_args()
    # Le second prompt, vérifié avec les autres options : introuvable, il est dit avant l'index et l'instantané « avant ».
    exp = Experiment("prompt-v2", args, changes={"prompt_name": args.other})
    exp.start()
    first = exp.config.prompt_name

    container_a, manifest, _ = exp.index("partage")
    print(f"Avant : prompt {first}")
    snapshot_a, _ = exp.record("avant", container_a)
    print(f"Après : prompt {args.other}")
    container_b = exp.container("partage", prompt_name=args.other)
    snapshot_b, _ = exp.record("apres", container_b)

    comparison = exp.compare(snapshot_a, snapshot_b)
    exp.log(f"Une seule chose change : le prompt, `{first}` → `{args.other}` "
            f"(versions `{snapshot_a.configuration.get('prompt_version')}` → "
            f"`{snapshot_b.configuration.get('prompt_version')}`). Même index `{manifest.index_id}`, "
            "même générateur, même seuil.")
    exp.log("")
    exp.log("## Les deux prompts")
    exp.log("")
    exp.table({
        f"avant (`{first}`)": {"version tracée": snapshot_a.configuration.get("prompt_version"),
                               "longueur moyenne des réponses": mean_length(snapshot_a), **exp.stats(snapshot_a)},
        f"après (`{args.other}`)": {"version tracée": snapshot_b.configuration.get("prompt_version"),
                                    "longueur moyenne des réponses": mean_length(snapshot_b), **exp.stats(snapshot_b)},
    })
    exp.log("## Dérive")
    exp.log("")
    exp.table({"avant → après": drift_summary(comparison)})
    exp.comparison(comparison, "Détail")
    exp.log("## Lecture")
    exp.log("")
    exp.log("- La version du prompt (déclarée + empreinte du contenu) est dans chaque trace : la dérive est "
            "attribuable à cette seule modification.")
    exp.log("- Ce que le prompt change (forme, longueur, ton) n'est pas ce que le domaine garantit (citations "
            "vérifiées, forme validée, droits filtrés) : c'est ce qui permet de le traiter comme une "
            "configuration *surveillée comme du métier* (ADR 0005).")
    exp.log("- Avec `extractive` (hors-ligne), 0 % de dérive : ce générateur ignore les consignes. Un modèle "
            "qui ignore le prompt produit exactement cette signature.")
    exp.write()


if __name__ == "__main__":
    run(main)
