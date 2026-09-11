"""Assistant documentaire RAG pour l'intranet — version 0.3

Répond aux questions des salariés à partir des fiches Markdown du dossier corpus/,
en citant les fiches utilisées. Les fiches RH et direction sont réservées.

Usage :
    python assistant_rag.py index
    python assistant_rag.py ask "Combien de jours de congés ?" --user alice
    python assistant_rag.py ask "..." --user bruno --model qwen3-4b

Nécessite le service IA (python -m ai_service, port 8100) et Ollama.
TODO: ajouter des tests, gérer le cas où le service est down, passer à une vraie base vectorielle.
"""

import argparse
import json
import math
import os
import pickle
import re
import sys
import urllib.request

AI_URL = "http://127.0.0.1:8100"
EMBED_MODEL = "bge-m3"
GEN_MODEL = "llama3-2-3b"
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
INDEX_FILE = os.path.join(os.path.dirname(__file__), "index.pkl")
CHUNK_SIZE = 800
TOP_K = 4
THRESHOLD = 0.65   # ajusté à la main sur quelques questions, marche bien avec bge-m3
MAX_TOKENS = 400

USERS = {
    "alice": ["tous"],
    "bruno": ["tous", "rh"],
    "claire": ["tous", "direction"],
}

PROMPT = """Tu es l'assistant documentaire interne de l'entreprise.
Tu réponds en français, uniquement à partir des passages fournis, en trois phrases maximum.
Après chaque affirmation, indique entre crochets le numéro du passage, par exemple [1].
Si un passage est marqué (réservé), ne le cite pas et ne l'utilise pas si l'utilisateur n'y a pas droit.
Si les passages ne permettent pas de répondre, dis-le.

Passages :
{passages}

Question : {question}
"""

INDEX = None  # chargé à la demande


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_docs():
    docs = []
    for name in sorted(os.listdir(CORPUS_DIR)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(CORPUS_DIR, name), encoding="utf-8") as f:
            content = f.read()
        m = re.match(r"---\n(.*?)\n---\n(.*)", content, re.DOTALL)
        if not m:
            print("fichier ignoré (pas d'en-tête):", name)
            continue
        header, body = m.group(1), m.group(2)
        meta = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
        meta = {k.strip(): v.strip() for k, v in meta.items()}
        groups = [g.strip() for g in meta.get("groupes", "tous").split(",")]
        docs.append({"id": meta.get("id", name[:-3]), "title": meta.get("titre", name),
                     "text": body.strip(), "groups": groups})
    return docs


def split(text):
    chunks = []
    current = ""
    for para in text.split("\n\n"):
        if len(current) + len(para) > CHUNK_SIZE and current:
            chunks.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ---------------------------------------------------------------------------
# Service IA
# ---------------------------------------------------------------------------

def call(path, payload):
    req = urllib.request.Request(AI_URL + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed(texts, kind="document"):
    vectors = []
    for i in range(0, len(texts), 32):
        r = call("/v1/embeddings", {"model": EMBED_MODEL, "input_type": kind, "inputs": texts[i:i + 32]})
        vectors.extend(r["vectors"])
    return vectors


def generate(prompt, seed=None):
    r = call("/v1/generate", {"model": GEN_MODEL, "system": "", "prompt": prompt,
                              "temperature": 0.2, "max_tokens": MAX_TOKENS, "seed": seed})
    text = r["text"]
    # qwen3 met parfois son raisonnement entre <think>…</think>, on l'enlève
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def build_index():
    docs = load_docs()
    entries = []
    for d in docs:
        for n, chunk in enumerate(split(d["text"])):
            entries.append({"doc": d["id"], "title": d["title"], "groups": d["groups"],
                            "n": n, "text": d["title"] + "\n" + chunk})
    print(f"{len(entries)} morceaux pour {len(docs)} documents, vectorisation...")
    vectors = embed([e["text"] for e in entries])
    for e, v in zip(entries, vectors):
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        e["vector"] = [x / norm for x in v]
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(entries, f)
    print("index écrit :", INDEX_FILE)


def get_index():
    global INDEX
    if INDEX is None:
        if not os.path.exists(INDEX_FILE):
            print("Pas d'index, lancez d'abord : python assistant_rag.py index")
            sys.exit(1)
        with open(INDEX_FILE, "rb") as f:
            INDEX = pickle.load(f)
    return INDEX


def search(question, user):
    q = embed([question], "query")[0]
    norm = math.sqrt(sum(x * x for x in q)) or 1.0
    q = [x / norm for x in q]
    scored = []
    for e in get_index():
        score = sum(a * b for a, b in zip(q, e["vector"]))
        scored.append((score, e))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:TOP_K]
    # on retire ce que l'utilisateur n'a pas le droit de lire
    allowed = []
    for score, e in top:
        if "tous" in e["groups"] or set(e["groups"]) & set(USERS[user]):
            allowed.append((score, e))
        else:
            print(f"  (passage réservé ignoré : {e['doc']}#{e['n']}, score {score:.2f})")
    return [(s, e) for s, e in allowed if s >= THRESHOLD]


# ---------------------------------------------------------------------------
# Réponse
# ---------------------------------------------------------------------------

def ask(question, user):
    if user not in USERS:
        print("utilisateur inconnu :", user)
        sys.exit(1)
    passages = search(question, user)
    if not passages:
        return "Je n'ai trouvé aucun document qui réponde à cette question.", []
    blocks = []
    for i, (score, e) in enumerate(passages, start=1):
        tag = "" if "tous" in e["groups"] else " (réservé)"
        blocks.append(f"[{i}] {e['title']}{tag}\n{e['text']}")
    prompt = PROMPT.format(passages="\n\n".join(blocks), question=question)
    text = generate(prompt)
    if "[" not in text:
        # le modèle a oublié de citer, on lui redemande
        text = generate(prompt + "\nN'oublie pas les numéros de passage entre crochets.", seed=1)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    sources = [passages[n - 1][1]["doc"] for n in cited if n - 1 < len(passages)]
    return text, sources


def main():
    global EMBED_MODEL, GEN_MODEL
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index")
    pi.add_argument("--embed", help="modèle d'embeddings (ex. nomic)")
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--user", default="alice")
    p.add_argument("--model", help="modèle de génération (ex. qwen3-4b)")
    p.add_argument("--embed", help="modèle d'embeddings (ex. nomic)")
    args = parser.parse_args()
    if getattr(args, "model", None):
        GEN_MODEL = args.model
    if getattr(args, "embed", None):
        EMBED_MODEL = args.embed
    if args.cmd == "index":
        build_index()
    else:
        text, sources = ask(args.question, args.user)
        print(text)
        if sources:
            print("Sources :", ", ".join(sources))


if __name__ == "__main__":
    main()
