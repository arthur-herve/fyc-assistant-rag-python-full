# Guide d'installation pas à pas

Durée réaliste : **15 minutes** pour tout ce qui tourne hors-ligne (séquences 1 à 2.2), **30 à
60 minutes de plus** pour les vrais modèles (téléchargements compris, selon la connexion).
Rien n'exige de carte graphique ; un portable de 8 Go de RAM suffit pour le hors-ligne, 16 Go
sont confortables pour les vrais modèles.

Si une étape résiste, tout le cours jusqu'à la séquence 2.2 fonctionne en **mode hors-ligne**
(étape 3) : avancez, revenez à l'installation d'Ollama plus tard.

## 1. Python 3.11 ou plus (5 min)

| Système | Comment | Vérification |
|---|---|---|
| Windows 10/11 | <https://www.python.org/downloads/windows/> → « Windows installer (64-bit) ». Cocher **« Add python.exe to PATH »**. Ou `winget install Python.Python.3.12`. | `py --version` (ou `python --version`) |
| macOS | <https://www.python.org/downloads/macos/> ou `brew install python@3.12` | `python3 --version` |
| Linux (Debian/Ubuntu) | `sudo apt install python3 python3-venv` (3.11+ sur Ubuntu 24.04, Debian 12) | `python3 --version` |

Dans la suite, `python` désigne `py` sous Windows, `python3` sous macOS et Linux.
**Aucune bibliothèque à installer** : le projet n'utilise que la bibliothèque standard.

## 2. Git et le dépôt (3 min)

| Système | Comment |
|---|---|
| Windows | <https://git-scm.com/download/win> ou `winget install Git.Git` |
| macOS | `xcode-select --install` ou `brew install git` |
| Linux | `sudo apt install git` |

```bash
git clone https://github.com/arthur-herve/fyc-assistant-rag.git
cd fyc-assistant-rag
python -m unittest discover -s tests -t .
```

Attendu : `Ran 133 tests … OK` en moins de 15 secondes. Si c'est vert, votre poste est prêt
pour les séquences 1 à 2.2.

## 3. Le mode hors-ligne (2 min)

Deux terminaux, depuis la racine du dépôt.

Terminal 1 — le service IA (il reste ouvert) :

```bash
python -m ai_service
```

Terminal 2 — l'application :

```bash
python -m assistant index
python -m assistant ask "Combien de jours de télétravail par semaine ?"
python -m assistant status
```

Attendu : une réponse citée `[1]` avec la source `teletravail`, puis un verdict « à jour ». Les
modèles `hashing` (embeddings hachés) et `extractive` (recopie de la phrase la plus proche) sont
déterministes et sans réseau : ce sont ceux des tests.

Sous Windows, si les accents s'affichent mal : `set PYTHONIOENCODING=utf-8` (cmd) ou
`$env:PYTHONIOENCODING = "utf-8"` (PowerShell) avant de lancer les commandes.

## 4. Ollama et les modèles (10 min + téléchargements)

Ollama exécute les modèles ouverts en local et les expose sur `http://127.0.0.1:11434`.

| Système | Comment |
|---|---|
| Windows | <https://ollama.com/download/windows> ou `winget install Ollama.Ollama` — un service démarre en arrière-plan |
| macOS | <https://ollama.com/download/mac> ou `brew install ollama` puis `ollama serve` |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

Vérification : `ollama --version`, puis `ollama list` (vide au début).

Modèles du cours, à télécharger dans cet ordre (6 Go en tout ; les quatre ont pris une dizaine de
minutes le 11/09/2026 sur une connexion fibre, comptez plus en ADSL) :

```bash
ollama pull bge-m3              # embeddings par défaut, 1,2 Go
ollama pull llama3.2:3b         # génération par défaut, 2,0 Go
ollama pull nomic-embed-text    # embeddings « de rupture », 274 Mo
ollama pull qwen3:4b            # génération « de rupture », 2,5 Go (séquences 3.3 et 4.1)
```

Les deux premiers suffisent pour suivre le cours ; les deux autres servent aux expériences de
changement de modèle. Si le disque ou la connexion manquent, `gemma3:1b` (815 Mo) remplace
`llama3.2:3b` avec une qualité moindre.

## 5. Premier vrai passage (5 min)

Relancer le service IA (terminal 1, Ctrl+C puis `python -m ai_service`), puis :

```bash
python -m assistant index --config config/app-ollama.toml
python -m assistant ask "Combien de jours dure le congé de paternité ?" --config config/app-ollama.toml -v
```

Attendu : l'indexation des 322 fiches prend **1 à 3 minutes** avec `bge-m3` sur la machine de
référence (carte graphique) — elle ne se fait qu'une fois ; la réponse arrive ensuite en 1 à 3
secondes et cite une fiche sur le congé de paternité (`F3156` ou `F12647`). Sur processeur seul,
comptez un ordre de grandeur de plus (non mesuré : à relever sur vos machines et à nous signaler).

## Ce qui peut coincer

| Symptôme | Cause probable | Remède |
|---|---|---|
| `Service IA injoignable (http://127.0.0.1:8100)` | le terminal 1 n'est pas lancé | `python -m ai_service` |
| `HTTP 502 — … Ollama est-il lancé sur http://127.0.0.1:11434 ?` | Ollama arrêté ou modèle non téléchargé | `ollama serve` / `ollama pull <modèle>` |
| `L'index a été construit avec « … » mais le modèle d'embeddings actuel est « … »` | vous avez changé de modèle d'embeddings | c'est voulu (séquence 2.3) : `python -m assistant index …` avec ce modèle |
| `python : commande introuvable` sous Windows | PATH non mis à jour à l'installation | utiliser `py`, ou réinstaller Python en cochant « Add to PATH » |
| Réponses très lentes (> 1 min) avec `qwen3:4b` | mode réflexion, budget de jetons | normal sur CPU ; utiliser `llama3.2:3b` hors expériences |
| `MemoryError` ou Ollama qui se ferme | modèle trop gros pour la RAM | modèle plus petit (`gemma3:1b`, `nomic-embed-text`) |

## Machine de référence du cours

Les durées et les rapports de `eval/resultats/` ont été mesurés sur : Windows 11, AMD Ryzen 7
5800H, 15,4 Go de RAM, NVIDIA RTX 3070 Laptop 8 Go, Ollama 0.34, Python 3.13. Les temps sans carte
graphique n'ont pas encore été mesurés ; le cours reste suivable, le mode hors-ligne couvre tout
ce qui ne demande pas un vrai modèle.
