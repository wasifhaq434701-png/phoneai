"""Experiment 4 — per-user memory: chunk store + salience gate + hybrid selection into a fixed token budget.

Reader: Qwen3-4B-4bit. Benchmarks: LongMemEval-S cleaned (500 Qs, ~105K-token history each, MIT) and LoCoMo
(10 conversations, categories 1-4 = 1,540 Qs; category 5 adversarial dropped as in mem0 and later work; CC BY-NC).
Tuning happens ONLY on LongMemEval dev (every 5th question, 100); reported numbers are LME test (400) + all LoCoMo.

Store: SQLite rows (one per ≤1,000-char piece of a turn) + an fp16 memmap of
MiniLM embeddings + edges (same-session adjacency for expansion, cross-session similarity for recurrence).
Gate: rank a user's chunks by the mean percentile of four signals and keep the top `keep`
fraction — self (first-person statement by the memory's owner; stands in for an explicit "remember this", which the
benchmarks don't have), novelty (1 − max cosine to anything earlier), recurrence (other sessions holding a similar
chunk), surprise (mean token NLL under a small LM, `surprise` subcommand).

Budgets are memory-context tokens; KV MB = tokens × 144 KB (Qwen3-4B fp16 K+V). The whole LME history (~105K tokens,
~15 GB KV) cannot run on a 12 GB phone, which is the point of selecting.

Methods (name,key=value,… e.g. "hybrid,keep=0.5,edges=0"):
  rag      top cosine chunks            session  whole sessions ranked by best chunk     bm25  top BM25 chunks
  hybrid   gate → hybrid (cosine+BM25 reciprocal-rank fusion) → cluster-diversity rule (1.5-cluster threshold, prior work) → edge expansion
  full     whole history (LoCoMo only; prefix-cached)              oracle  evidence sessions only (upper bound)
Evidence recall is counted by turn IDs (LME has_answer / LoCoMo evidence), never by substring (substring recall under-counts).

Usage:
  python experiments/user_memory.py ingest | embed | check   (embed resumes an interrupted ingest)
  python experiments/user_memory.py surprise [model]           (resumable; writes chunks.nll)
  python experiments/user_memory.py recall [methods…]          (no LLM: gate recall curve + retrieval recall, LME dev)
  python experiments/user_memory.py run <method> <budget…> [--set lme-dev|lme-test|test|all]
  python experiments/user_memory.py summarise [n] | sumembed     (Exp 5.4 session notes; n = speed bench only)
  python experiments/user_memory.py judge                      (Qwen3-30B-A3B, LongMemEval's official prompts)
  python experiments/user_memory.py label <config> [name n]    (hand-label 2n answers (default 50), blind to the judge)
  python experiments/user_memory.py latency retrieval|ttft   (ttft needs the GPU to itself)
  python experiments/user_memory.py report
"""
import json
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

import routing_quality as rq
import task_adapters as ta

MAIN = "hybrid,div=0,edges=0"  # frozen on LME dev 2026-10-14: diversity and edges did not help
READERS = {"4b": ta.MODEL, "8b": "mlx-community/Qwen3-8B-4bit"}  # Exp 5.3: `rdr=8b` opt; same KV/token (36 L × 8 KV heads × 128)
READERS |= {"llama3b": "mlx-community/Llama-3.2-3B-Instruct-4bit", "gemma3": "mlx-community/gemma-3-text-4b-it-4bit"}  # Exp 5.9: non-Qwen families
READERS |= {f"mp{k}": str(Path(__file__).resolve().parent.parent / "data" / "exp5" / "mp" / k) for k in ("u4", "w", "m", "r", "u8", "x6w", "x12w", "x6r", "x12r", "x12s", "u3", "x3s", "x3r")}  # Exp 5.16 / 5.16b / 5.16d
MODEL, JUDGE, EMB = ta.MODEL, "mlx-community/Qwen3-30B-A3B-4bit", "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parent.parent
LME_M = os.environ.get("MEM_STORE") == "lme_m"  # Exp 5.5: LongMemEval-M in its own store + results dir, no LoCoMo
D, OUT = (ROOT / "data" / "exp5" / "lme_m", ROOT / "results" / "exp5" / "lme_m") if LME_M else (ROOT / "data" / "exp4", ROOT / "results" / "exp4")
DB, EMBF, QS, QEMB = D / "store.sqlite", D / "emb.f16", D / "questions.json", D / "qemb.npy"
LOCOMO = ROOT / "data" / "exp4" / "raw" / "locomo10.json"
D5 = ROOT / "data" / "exp5"
SUMDB, SUMEMB, SUM_MAX = D5 / "summaries.sqlite", D5 / "sumemb.f16", 120  # Exp 5.4 session notes (gitignored)
CHUNK_CHARS, DIM, SIM_EDGE, HEADER_TOK, MAX_NEW = 1000, 384, 0.6, 20, 200
KV_BYTES_PER_TOK = 36 * 2 * 8 * 128 * 2  # layers × (K,V) × kv heads × head dim × fp16
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}  # mem0's reading of the ids
FIRST = re.compile(r"\b(i|i'm|i've|i'd|i'll|my|me|mine|myself|we|our|us)\b", re.I)
STOP = set("the a an and or of to in on for is are was were be been it this that with as at by from i you me my your "
           "we our he she they them his her their what when where which who how do did does have has had not no".split())
PROMPT = ("Here are excerpts from past conversations, oldest first; each session is headed by its date.\n{ctx}\n\n"
          "Current date: {date}\nQuestion: {question}\n"
          "Answer using the excerpts. Think briefly if needed, then give the final answer on the last line as "
          "'Answer: <answer>'. If the question asks for advice or a recommendation, give it, tailored to what the excerpts "
          "reveal about the user. If it asks for a fact the excerpts do not contain, say that you don't know.")


SUM_PROMPT = ("Below is one past conversation session ({date}).\n\n{text}\n\n"
              "Write memory notes for this session in at most 80 words: facts about the people in it (names, events, "
              "relationships), their preferences and plans, dated events, changes to earlier facts, and any recommendations "
              "or answers given. Plain sentences, no preamble.")


# ---------- ingest: benchmarks → one SQLite store + fp16 embedding memmap ----------

def lme_file(name="longmemeval_m_cleaned.json" if LME_M else "longmemeval_s_cleaned.json"):
    return hf_hub_download("xiaowu0162/longmemeval-cleaned", name, repo_type="dataset")


def pieces(text, n=CHUNK_CHARS):
    out = []
    while len(text) > n:
        cut = text.rfind(" ", 0, n)
        cut = cut if cut > n // 2 else n
        out.append(text[:cut])
        text = text[cut:].lstrip()
    return out + [text]


def load_benchmarks():
    """sessions: key → (date, [(role, owner, text)]); users: user → [session keys, chronological]; questions."""
    sessions, users, qs, clashes = {}, {}, [], 0
    s_split = {q["qid"]: q["split"] for q in json.loads((ROOT / "data" / "exp4" / "questions.json").read_text())} if LME_M else {}

    def add(sid, date, turns, user):
        nonlocal clashes
        if sid in sessions and sessions[sid][1] != turns:  # same id, different text: keep both apart
            clashes, sid = clashes + 1, f"{sid}@{user}"
        sessions.setdefault(sid, (date, turns))
        return sid

    for i, x in enumerate(json.load(open(lme_file()))):
        u, keys, ev = x["question_id"], {}, []
        for sid, date, s in sorted(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]), key=lambda r: r[1]):
            keys[sid] = add(sid, date, [(t["role"], t["role"] == "user", t["content"]) for t in s], u)
            ev += [[keys[sid], j] for j, t in enumerate(s) if t.get("has_answer")]
        users[u] = list(keys.values())
        qs.append({"qid": u, "bench": "lme", "user": u, "type": x["question_type"], "question": x["question"],
                   "date": x["question_date"], "answer": str(x["answer"]), "abs": u.endswith("_abs"),
                   "ev_turns": ev, "ev_sessions": [keys[s] for s in x["answer_session_ids"] if s in keys],
                   "split": s_split[u] if LME_M else "dev" if i % 5 == 0 else "test"})  # M: same dev/test qids as S

    for x in [] if LME_M else json.load(open(LOCOMO)):
        u, c, dia = x["sample_id"], x["conversation"], {}
        n = sorted(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k))
        for s in n:
            date = datetime.strptime(c[f"session_{s}_date_time"], "%I:%M %p on %d %B, %Y").strftime("%Y/%m/%d (%a) %H:%M")
            turns = [(t["speaker"], True, t["text"] + (f" [shares a photo: {t['blip_caption']}]" if t.get("blip_caption") else ""))
                     for t in c[f"session_{s}"]]
            key = add(f"{u}_s{s}", date, turns, u)
            dia.update({t["dia_id"]: (key, j) for j, t in enumerate(c[f"session_{s}"])})
        users[u] = [f"{u}_s{s}" for s in n]
        for j, q in enumerate(x["qa"]):
            if q["category"] == 5:
                continue
            ev = [dia[f"D{a}:{b}"] for a, b in re.findall(r"D(\d+):(\d+)", " ".join(q.get("evidence", []))) if f"D{a}:{b}" in dia]
            qs.append({"qid": f"{u}_q{j}", "bench": "locomo", "user": u, "type": LOCOMO_CATS[q["category"]],
                       "question": q["question"], "date": sessions[users[u][-1]][0], "answer": str(q["answer"]),
                       "abs": False, "ev_turns": [list(e) for e in ev], "ev_sessions": sorted({k for k, _ in ev}),
                       "split": "test"})
    print(f"{len(sessions)} sessions, {len(users)} users, {len(qs)} questions, {clashes} session-id clashes")
    return sessions, users, qs


def ingest():
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    sessions, users, qs = load_benchmarks()
    D.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)
    db = sqlite3.connect(DB)
    db.executescript("""
        CREATE TABLE sessions(sid TEXT PRIMARY KEY, date TEXT);
        CREATE TABLE haystack(user TEXT, pos INT, sid TEXT);
        CREATE TABLE chunks(id INTEGER PRIMARY KEY, sid TEXT, turn INT, part INT, role TEXT, owner INT, text TEXT,
                            ntok INT, nll REAL);
        CREATE INDEX chunks_sid ON chunks(sid); CREATE INDEX haystack_user ON haystack(user);""")
    db.executemany("INSERT INTO sessions VALUES (?,?)", [(k, d) for k, (d, _) in sessions.items()])
    db.executemany("INSERT INTO haystack VALUES (?,?,?)", [(u, p, s) for u, ss in users.items() for p, s in enumerate(ss)])
    rows = [(sid, t, p, role, int(owner), piece) for sid, (_, turns) in sessions.items()
            for t, (role, owner, text) in enumerate(turns) for p, piece in enumerate(pieces(text))]
    tok = AutoTokenizer.from_pretrained(snapshot_download(MODEL))
    texts = [f"{r[3]}: {r[5]}" for r in rows]
    ntok = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    db.executemany("INSERT INTO chunks(id, sid, turn, part, role, owner, text, ntok) VALUES (?,?,?,?,?,?,?,?)",
                   [(i, *r, n) for i, (r, n) in enumerate(zip(rows, ntok))])
    db.commit()
    print(f"{len(rows)} chunks, {sum(ntok) / 1e6:.1f}M tokens", flush=True)
    QS.write_text(json.dumps(qs))
    EMBF.unlink(missing_ok=True)
    embed()


def embed(block=10_000):
    """Resumable: blocks are written in order and a block is never half-written past `done`.
    Clears the MPS cache per block — PyTorch's allocator otherwise grew to 46 GB over 50K variable-length texts."""
    import torch
    from sentence_transformers import SentenceTransformer
    db = sqlite3.connect(DB)
    rows = db.execute("SELECT role, text FROM chunks ORDER BY id").fetchall()
    done_f = D / "emb.done"
    done = int(done_f.read_text()) if EMBF.exists() and done_f.exists() else 0
    E = np.memmap(EMBF, np.float16, "r+" if done else "w+", shape=(len(rows), DIM))
    emb = SentenceTransformer(EMB)
    for i in range(done, len(rows), block):
        E[i:i + block] = emb.encode([f"{r}: {t}" for r, t in rows[i:i + block]], batch_size=128, normalize_embeddings=True)
        E.flush()
        done_f.write_text(str(min(i + block, len(rows))))
        torch.mps.empty_cache()
        if (i // block) % 5 == 0:
            print(f"  embedded {min(i + block, len(rows))}/{len(rows)}", flush=True)
    qs = json.loads(QS.read_text())
    np.save(QEMB, emb.encode([q["question"] for q in qs], normalize_embeddings=True).astype(np.float16))
    print("embedding done", flush=True)


def store():
    db = sqlite3.connect(DB)
    n = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return db, np.memmap(EMBF, np.float16, "r", shape=(n, DIM))


def load_user(db, E, user):
    """One user's chunk store, chronological. Arrays indexed by local chunk position."""
    rows = db.execute("""SELECT c.id, c.sid, c.turn, c.role, c.owner, c.text, c.ntok, c.nll, s.date FROM haystack h
                         JOIN chunks c ON c.sid = h.sid JOIN sessions s ON s.sid = h.sid
                         WHERE h.user = ? ORDER BY h.pos, c.turn, c.part""", (user,)).fetchall()
    ids, sid, turn, role, owner, text, ntok, nll, date = map(list, zip(*rows))
    sess = np.unique(sid, return_inverse=True)[1]
    m = {"id": np.array(ids), "sid": sid, "turn": np.array(turn), "role": role, "owner": np.array(owner, bool),
         "text": text, "ntok": np.array(ntok), "nll": np.array([np.nan if v is None else v for v in nll]),
         "date": date, "sess": sess, "E": np.asarray(E[ids], np.float32)}
    m["key"] = {(s, t) for s, t in zip(sid, turn)}
    return m


# ---------- gate ----------

def pct(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(len(x) - 1, 1)


def signals(m):
    E, sess = m["E"], m["sess"]
    S = E @ E.T
    earlier = np.where(np.tril(np.ones_like(S, bool), -1), S, -1)
    novelty = 1 - np.maximum(earlier.max(1), 0)
    similar = (S >= SIM_EDGE) & (sess[:, None] != sess[None, :])
    onehot = np.eye(sess.max() + 1, dtype=np.float32)[sess]
    recurrence = ((similar.astype(np.float32) @ onehot) > 0).sum(1)
    self_ = m["owner"] * np.array([len(FIRST.findall(t)) for t in m["text"]])
    out = {"self": self_, "novelty": novelty, "recurrence": recurrence}
    if not np.isnan(m["nll"]).any():
        out["surprise"] = m["nll"]
    return out


def gate(m, keep, which="all"):
    """Boolean admit mask: top `keep` fraction by mean percentile rank of the chosen signals."""
    if keep >= 1 or which == "none":
        return np.ones(len(m["id"]), bool)
    sig = m.setdefault("_sig", signals(m))
    names = list(sig) if which == "all" else which.split("+")
    score = np.mean([pct(sig[s]) for s in names], 0)
    return score >= np.quantile(score, 1 - keep)


# ---------- selection ----------

def words(t):
    return [w for w in re.findall(r"\w+", t.lower()) if w not in STOP]


def bm25(m, q, k1=1.2, b=0.75):
    if "_bm" not in m:
        docs = [Counter(words(t)) for t in m["text"]]
        df = Counter(w for d in docs for w in d)
        m["_bm"] = (docs, df, np.array([sum(d.values()) for d in docs], float))
    docs, df, dl = m["_bm"]
    n, avg = len(docs), dl.mean() or 1
    s = np.zeros(n)
    for w in set(words(q)):
        if w in df:
            tf = np.array([d[w] for d in docs], float)
            s += np.log(1 + (n - df[w] + .5) / (df[w] + .5)) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
    return s


def rrf(*scores, k=60):
    return sum(1 / (k + np.argsort(np.argsort(-s, kind="stable"), kind="stable")) for s in scores)


def fill(m, order, budget, expand=False):
    """Greedy: take chunks in `order` while they fit (chunk tokens + a date header per new session).
    expand: after each pick, also take its same-session neighbours (end of the previous turn, start of the next)."""
    picked, sessions, used = set(), set(), 0

    def take(i):
        nonlocal used
        cost = m["ntok"][i] + (HEADER_TOK if m["sess"][i] not in sessions else 0)
        if i in picked or used + cost > budget:
            return False
        picked.add(i)
        sessions.add(m["sess"][i])
        used += cost
        return True

    for i in order:
        if take(i) and expand:
            for d in (-1, 1):
                nb = [j for j in (i + d * np.arange(1, 4)) if 0 <= j < len(m["id"]) and m["sess"][j] == m["sess"][i]
                      and m["turn"][j] == m["turn"][i] + d]
                if nb:
                    take(nb[0])
        if used >= budget - 8:
            break
    return sorted(picked), used


def select(m, q, qe, budget, method, opt):
    cos = m["E"] @ qe
    if method == "full":
        return list(range(len(cos))), int(m["ntok"].sum())
    if method == "oracle":
        ev = set(q["ev_sessions"])
        return fill(m, [i for i, s in enumerate(m["sid"]) if s in ev], budget)
    if method == "rag":
        return fill(m, np.argsort(-cos), budget)
    if method == "bm25":
        return fill(m, np.argsort(-bm25(m, q["question"])), budget)
    if method == "session":
        best = np.full(m["sess"].max() + 1, -2.0)
        np.maximum.at(best, m["sess"], cos)
        return fill(m, sorted(range(len(cos)), key=lambda i: (-best[m["sess"][i]], i)), budget)
    assert method == "hybrid", method
    admit = gate(m, float(opt.get("keep", 1)), opt.get("gate", "all"))
    if int(opt.get("asst", 0)):  # Exp 5.5b: never gate out non-owner (assistant) turns; `keep` then applies to the owner's side
        admit |= ~m["owner"]
    score = rrf(cos, bm25(m, q["question"])) if int(opt.get("bm25", 1)) else cos
    order = [i for i in np.argsort(-score) if admit[i]]
    if int(opt.get("div", 1)):
        order = diversify(m, order, admit)
    idx, used = fill(m, order, adapt(m, cos, opt, budget), expand=bool(int(opt.get("edges", 1))))
    if opt.get("order") == "rel":  # most relevant last (Exp 5.1 distraction probe)
        rank = {i: r for r, i in enumerate(order)}
        idx = sorted(idx, key=lambda i: -rank.get(i, len(order)))
    return idx, used


def adapt(m, cos, opt, cap, floor=512):
    """Exp 5.1: this query's token budget from its sorted cosine curve; `cap` (the run's budget) is the ceiling.
    cut=α: chunks with cos ≥ α·top; mass=p: softmax(cos) mass ≥ p; knee=N: Kneedle elbow of the top-N curve."""
    o = np.argsort(-cos)
    s = cos[o]
    if "cut" in opt:
        n = int((s >= float(opt["cut"]) * s[0]).sum())
    elif "mass" in opt:
        w = np.exp((s - s[0]) / 0.05)  # shortcut: temperature fixed at 0.05 (spreads dev budgets ~0.2–9K), sweep it if mass wins
        n = int(np.searchsorted(np.cumsum(w) / w.sum(), float(opt["mass"]))) + 1
    elif "knee" in opt:
        y = s[:int(opt["knee"])]
        y = (y - y[-1]) / max(y[0] - y[-1], 1e-9)
        n = max(int(np.argmax(np.linspace(1, 0, len(y)) - y)), 1)  # keep chunks above the knee (farthest below the chord)
    else:
        return cap
    tok = m["ntok"][o[:n]].sum() + HEADER_TOK * len(set(m["sess"][o[:n]]))
    return int(np.clip(tok, floor, cap))


def kmeans(X, k, iters=30, seed=42):
    """k-means++ init + Lloyd. (sklearn's KMeans next to MLX segfaults at interpreter exit, losing buffered logs.)"""
    rng = np.random.default_rng(seed)
    C = X[[rng.integers(len(X))]]
    for _ in range(1, k):
        d = ((X[:, None] - C[None]) ** 2).sum(-1).min(1)
        C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum()) if d.sum() > 0 else rng.integers(len(X))]])
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


def diversify(m, order, admit, k=6, top=3):
    """Diversity rule: if flat top-3 sits inside one k-means cluster (< 1.5 clusters), lead with the best chunk of each
    cluster (clusters best-first), then continue in rank order; otherwise keep the flat order."""
    if "_km" not in m:
        idx = np.flatnonzero(admit)
        lab = np.full(len(admit), -1)
        lab[idx] = kmeans(m["E"][idx], min(k, len(idx)))
        m["_km"] = lab
    lab = m["_km"]
    if len({lab[i] for i in order[:top]}) >= 1.5:
        return order
    lead, seen = [], set()
    for i in order:
        if lab[i] not in seen:
            seen.add(lab[i])
            lead.append(i)
    return lead + [i for i in order if i not in set(lead)]


def render(m, idx):
    out, last = [], None
    for i in idx:
        if m["sess"][i] != last:
            out.append(f"\n### Session ({m['date'][i]})")
            last = m["sess"][i]
        out.append(f"{m['role'][i]}: {m['text'][i]}")
    return "\n".join(out)


def ev_recall(m, q, idx):
    ev = {tuple(e) for e in q["ev_turns"]}
    return len(ev & {(m["sid"][i], m["turn"][i]) for i in idx}) / len(ev) if ev else None


def parse(cfg):
    name, *kv = cfg.split(",")
    return name, dict(x.split("=") for x in kv)


def questions(split):
    qs, qe = json.loads(QS.read_text()), np.load(QEMB).astype(np.float32)
    sub = {"lme+loc500": lambda: stratified(qs, "locomo", 500), "loc+lme100": lambda: stratified(qs, "lme", 100)}.get(split, set)()
    keep = {"lme-dev": lambda q: q["bench"] == "lme" and q["split"] == "dev", "test": lambda q: q["split"] == "test",
            "lme-test": lambda q: q["bench"] == "lme" and q["split"] == "test",  # ablations: LME test only (time)
            "loc-test": lambda q: q["bench"] == "locomo" and q["split"] == "test",  # Exp 5.8: LME first, LoCoMo after
            "lme+loc500": lambda q: q["split"] == "test" and (q["bench"] == "lme" or q["qid"] in sub),  # Exp 5.3 (time)
            "loc+lme100": lambda q: q["split"] == "test" and (q["bench"] == "locomo" or q["qid"] in sub),  # Exp 5.4
            "all": lambda q: True}[split]
    return [(q, e) for q, e in zip(qs, qe) if keep(q)]


def stratified(qs, bench, n):
    """Fixed test subset of one benchmark, stratified by type (largest-remainder allocation, seed 0)."""
    import random
    rng, out = random.Random(0), set()
    pool = [q for q in qs if q["bench"] == bench and q["split"] == "test"]
    types = sorted({q["type"] for q in pool})
    share = {t: n * sum(q["type"] == t for q in pool) / len(pool) for t in types}
    k = {t: int(v) for t, v in share.items()}
    for t in sorted(types, key=lambda t: k[t] - share[t])[:n - sum(k.values())]:
        k[t] += 1
    for t in types:
        out |= set(rng.sample(sorted(q["qid"] for q in pool if q["type"] == t), k[t]))
    return out


def contexts(cfg, budget, split):
    """[(q, prompt text, memory tokens, evidence recall)] — users loaded once each."""
    name, opt = parse(cfg)
    prompt = PROMPT.replace("oldest first", "most relevant last") if opt.get("order") == "rel" else PROMPT
    notes = name in ("sumall", "sum") or "sum" in opt  # Exp 5.4
    if notes:
        prompt = prompt.replace("excerpts", "notes and excerpts" if name == "hybrid" else "notes")
    db, E = store()
    out, m = [], None
    for q, qe in sorted(questions(split), key=lambda r: r[0]["user"]):
        if m is None or m["_user"] != q["user"]:
            m = load_user(db, E, q["user"])
            m["_user"] = q["user"]
        if notes:
            ms = m.setdefault("_ms", summaries(m))
            sidx, sused = note_select(ms, q, qe, budget, name, opt)
            idx, used = select(m, q, qe, budget - sused, name, opt) if name == "hybrid" else ([], 0)
            ctx, used = render_mix(m, idx, ms, sidx), used + sused
        else:
            idx, used = select(m, q, qe, budget, name, opt)
            ctx = render(m, idx)
        if "snap" in opt:  # Exp 5.8 diagnostic: evidence lines to locate in the prompt + share of them a direct small retrieval keeps
            ev = {tuple(e) for e in q["ev_turns"]}
            evi = [i for i in idx if (m["sid"][i], m["turn"][i]) in ev]
            small = set(select(m, q, qe, int(opt["snap"]), name, opt)[0])
            q = {**q, "_ev": [f"{m['role'][i]}: {m['text'][i]}" for i in evi],
                 "_ev_small": float(sum(m["ntok"][i] for i in evi if i in small) / sum(m["ntok"][i] for i in evi)) if evi else None}
        out.append((q, prompt.format(ctx=ctx, date=q["date"], question=q["question"]), int(used),
                    ev_recall(m, q, idx) if not notes or name == "hybrid" else None))
    return out


# ---------- runs ----------

def gen_prefixed(model, tok, items):
    """Full-context LoCoMo: prefill each conversation once, then answer its questions from a trimmed copy of the KV."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    outs, cache, prefix = [], None, None
    for _, text, _, _ in items:
        full = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
        cut = full.index("\n\nCurrent date:")
        if full[:cut] != prefix:
            prefix, cache = full[:cut], make_prompt_cache(model)
            ptoks = mx.array(tok.encode(prefix, add_special_tokens=False))[None]
            for i in range(0, ptoks.shape[1], 2048):
                model(ptoks[:, i:i + 2048], cache=cache)
                mx.eval([c.state for c in cache])
            base = cache[0].offset
        suffix = tok.encode(full[cut:], add_special_tokens=False)
        new = []
        for t, _ in generate_step(mx.array(suffix), model, max_tokens=MAX_NEW, prompt_cache=cache):
            t = t.item() if hasattr(t, "item") else t
            if t in tok.eos_token_ids:
                break
            new.append(t)
        outs.append(tok.decode(new))
        trim_prompt_cache(cache, cache[0].offset - base)
        if len(outs) % 100 == 0:
            print(f"  full {len(outs)}/{len(items)}", flush=True)
    return outs


SNAP = {"on": False, "scores": []}  # Exp 5.8: window→prefix attention mass per layer, filled by the hooked SDPA
QUEST = {"on": False, "P": 0, "budget": 0, "layer": 0, "n": 0, "bounds": {}}  # Exp 5.10: Quest page selection at decode
PAGE = 16  # Quest's page size


def quest_bound(q, kmin, kmax):
    """Quest's per-page upper bound on q·k: Σ_d max(q_d·min_d, q_d·max_d). q (…, D), kmin/kmax (…, pages, D)."""
    a, b = q[..., None, :] * kmin, q[..., None, :] * kmax
    return ((a + b + abs(a - b)) / 2).sum(-1)  # elementwise max, written so numpy and mlx arrays both work


def snap_keep(s, k, pool):
    """SnapKV selection: s = (kv heads, prefix) attention mass from the window → sorted top-k prefix positions per head.
    Pooling = F.avg_pool1d(kernel, padding kernel//2), zero padded, as in the SnapKV reference code."""
    if pool > 1:
        s = np.lib.stride_tricks.sliding_window_view(np.pad(s, ((0, 0), (pool // 2, pool // 2))), pool, axis=1).mean(-1)
    k = min(k, s.shape[1])
    return np.sort(np.argpartition(-s, k - 1, axis=1)[:, :k], axis=1)


def snap_patch():
    import mlx.core as mx
    from mlx_lm.models import qwen3
    sdpa = qwen3.scaled_dot_product_attention
    if getattr(sdpa, "_snap", False):
        return

    def hooked(queries, keys, values, cache=None, scale=1.0, mask=None, **kw):
        if SNAP["on"]:  # queries = the observation window (last W prompt tokens), keys = every cached token
            B, H, T, W = 1, keys.shape[1], keys.shape[2], queries.shape[2]
            q = queries.reshape(B, H, queries.shape[1] // H, W, -1).astype(mx.float32)
            a = (q * scale) @ keys[:, :, None].astype(mx.float32).swapaxes(-1, -2)  # (1, H, G, W, T)
            a = mx.where(mx.arange(T)[None] <= (T - W + mx.arange(W))[:, None], a, -mx.inf)
            a = mx.softmax(a, axis=-1)[..., :T - W].sum(axis=3).mean(axis=2)[0]  # sum over window, mean over GQA group
            SNAP["scores"].append(np.array(a))
        if QUEST["on"] and queries.shape[2] == 1:  # decode step: attend to the top context pages + everything after them
            l, QUEST["layer"] = QUEST["layer"], (QUEST["layer"] + 1) % QUEST["n"]
            npg = QUEST["P"] // PAGE
            kp = min(QUEST["budget"] // PAGE, npg)
            if l >= 2 and kp > 0:  # Quest keeps the first two layers dense
                H, G, T = keys.shape[1], queries.shape[1] // keys.shape[1], keys.shape[2]
                if l not in QUEST["bounds"]:
                    pg = keys[:, :, :npg * PAGE].reshape(1, H, npg, PAGE, -1).astype(mx.float32)
                    QUEST["bounds"][l] = (pg.min(axis=3)[:, :, None], pg.max(axis=3)[:, :, None])  # (1, H, 1, pages, D)
                kmin, kmax = QUEST["bounds"][l]
                ub = quest_bound(queries.reshape(1, H, G, -1).astype(mx.float32), kmin, kmax)  # (1, H, G, pages)
                top = mx.argpartition(-ub, kp - 1, axis=-1)[..., :kp]
                pages = mx.put_along_axis(mx.zeros(ub.shape, mx.bool_), top, mx.array(True), axis=-1)
                keep = mx.concatenate([mx.repeat(pages, PAGE, axis=-1), mx.ones((1, H, G, T - npg * PAGE), mx.bool_)], axis=-1)
                mask = keep.reshape(1, H * G, 1, T)  # query head h uses kv head h // G, as in MLX's GQA
        return sdpa(queries, keys, values, cache, scale=scale, mask=mask, **kw)

    hooked._snap = True
    qwen3.scaled_dot_product_attention = hooked


def gen_snap(model, tok, items, opt, name):
    """SnapKV (Li et al. 2024) on one prompt at a time: prefill everything, score prefix tokens by attention from the
    observation window, keep top-`snap` per KV head per layer (+ the window), decode from the kept KV only.
    win=q (default): window = the whole question suffix; win=N: last N prompt tokens. pool=7: avg-pool kernel.
    requery=1: drop the window's KV and re-feed it over the kept prefix (question no longer sees evicted tokens).
    Returns outputs + per-item extras (kv_ev = evidence tokens kept, mean over layers × heads). Resumable jsonl."""
    import hashlib
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    class SnapCache(KVCache):
        """Kept KV rows in preallocated storage; `offset` stays the true position so RoPE of new tokens is exact."""
        def __init__(self, k, v, pos, room):
            pad = lambda a: mx.concatenate([a, mx.zeros((*a.shape[:2], room, a.shape[3]), a.dtype)], axis=2)
            self.keys, self.values, self.n, self.offset = pad(k), pad(v), k.shape[2], pos

        def update_and_fetch(self, k, v):
            s = k.shape[2]
            assert self.n + s <= self.keys.shape[2], "SnapCache room exhausted"
            self.keys[..., self.n:self.n + s, :] = k
            self.values[..., self.n:self.n + s, :] = v
            self.n, self.offset = self.n + s, self.offset + s
            return self.keys[..., :self.n, :], self.values[..., :self.n, :]

        @property
        def state(self):
            return self.keys[..., :self.n, :], self.values[..., :self.n, :]

        def size(self):
            return self.n

    snap_patch()
    keep, pool, requery, win = int(opt["snap"]), int(opt.get("pool", 7)), int(opt.get("requery", 0)), opt.get("win", "q")
    path = OUT.parent / "exp3_gen" / f"{name}-{hashlib.md5(json.dumps([t for _, t, _, _ in items]).encode()).hexdigest()[:8]}.jsonl"
    done = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    with open(path, "a") as f:
        for n, (q, text, _, _) in enumerate(items[len(done):], len(done)):
            full = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False) + q.get("_prefix", "")
            cut = full.rfind(q["_cut"]) if "_cut" in q else full.index("\n\nCurrent date:")  # Exp 5.14 RULER: own question marker + answer prefix
            enc = tok._tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)  # same ids as ta.gen's baselines
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            T = len(ids) - 1  # the last prompt token goes to generate_step
            W = T - sum(b <= cut for _, b in offs) if win == "q" else int(win)  # q: every token from the question on
            P = T - W
            cache = make_prompt_cache(model)
            for i in range(0, P, 2048):
                model(mx.array(ids[i:min(i + 2048, P)])[None], cache=cache)
                mx.eval([c.state for c in cache])
            SNAP["on"], SNAP["scores"] = True, []
            model(mx.array(ids[P:T])[None], cache=cache)
            SNAP["on"] = False
            kept = []
            for l, c in enumerate(cache):
                K, V = c.state
                idx = snap_keep(SNAP["scores"][l], keep, pool)
                kept.append(idx)
                if not requery:
                    idx = np.concatenate([idx, np.broadcast_to(np.arange(P, T), (len(idx), W))], axis=1)
                ix = mx.broadcast_to(mx.array(idx)[None, :, :, None], (1, *idx.shape, K.shape[3]))
                cache[l] = SnapCache(mx.take_along_axis(K, ix, axis=2), mx.take_along_axis(V, ix, axis=2),
                                     T if not requery else P, W + MAX_NEW + 16)
            mx.eval([c.state for c in cache])
            new = []
            for t, _ in generate_step(mx.array(ids[T:] if not requery else ids[P:]), model, max_tokens=q.get("_max", MAX_NEW), prompt_cache=cache):
                t = t.item() if hasattr(t, "item") else t
                if t in tok.eos_token_ids:
                    break
                new.append(t)
            ev = np.zeros(P, bool)  # prompt positions inside evidence lines
            spans, pos = [], 0
            for line in q.get("_ev", []):
                pos = full.index(line, pos)
                spans.append((pos, pos + len(line)))
                pos += len(line)
            for j, (a, b) in enumerate(offs[:P]):
                ev[j] = any(a < e and b > s for s, e in spans)
            if int(opt.get("ov", 0)):  # Exp 5.12: could one layer's kept indices serve another layer?
                M = np.zeros((len(kept), kept[0].shape[0], P), np.float32)
                for l, idx in enumerate(kept):
                    np.put_along_axis(M[l], idx, 1.0, axis=1)
                S = np.stack(SNAP["scores"])  # (layers, kv heads, prefix) window attention mass
                inter, k = np.einsum("lhp,mhp->lmh", M, M), M.sum(2)
                cover = np.einsum("mhp,lhp->lmh", S, M) / (S * M).sum(2)[None]  # [l, m]: m's mass kept by l's set / by its own
                ov = {"jaccard": (inter / (k[:, None] + k[None] - inter)).mean(2).round(3).tolist(), "cover": cover.mean(2).round(3).tolist()}
            row = {"out": tok.decode(new), "kv_tok": int(min(keep, P)), **({"ov": ov} if int(opt.get("ov", 0)) else {}),
                   "kv_ev": float(np.mean([ev[idx].sum(1).mean() / ev.sum() for idx in kept])) if ev.any() else None,
                   "ev_small": q.get("_ev_small")}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done.append(row)
            mx.clear_cache()
            if n % 50 == 0:
                print(f"  snap {name} {n + 1}/{len(items)}  peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    return done


def gen_quest(model, tok, items, opt, name):
    """Quest (Tang et al. 2024) emulated for accuracy: full prefill and full KV kept; at every decode step each query head
    attends to its top `quest`/16 context pages by the min/max-key bound, plus the question and generated tokens.
    Dense ops, no speed-up (Quest's CUDA kernels give that). Layers 0–1 dense, as in the paper. Resumable jsonl."""
    import hashlib
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    snap_patch()
    QUEST.update(budget=int(opt["quest"]), n=len(model.layers))
    path = OUT.parent / "exp3_gen" / f"{name}-{hashlib.md5(json.dumps([t for _, t, _, _ in items]).encode()).hexdigest()[:8]}.jsonl"
    done = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    with open(path, "a") as f:
        for n, (q, text, _, _) in enumerate(items[len(done):], len(done)):
            full = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False)
            cut = full.index("\n\nCurrent date:")
            enc = tok._tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            cache = make_prompt_cache(model)
            for i in range(0, len(ids) - 1, 2048):
                model(mx.array(ids[i:min(i + 2048, len(ids) - 1)])[None], cache=cache)
                mx.eval([c.state for c in cache])
            QUEST.update(on=True, P=sum(b <= cut for _, b in enc["offset_mapping"]), layer=0, bounds={})
            new = []
            try:
                for t, _ in generate_step(mx.array(ids[-1:]), model, max_tokens=MAX_NEW, prompt_cache=cache):
                    t = t.item() if hasattr(t, "item") else t
                    if t in tok.eos_token_ids:
                        break
                    new.append(t)
            finally:
                QUEST["on"] = False
            row = {"out": tok.decode(new), "kv_tok": int(min(QUEST["budget"], QUEST["P"]))}  # active KV per step; all 8K stay resident
            f.write(json.dumps(row) + "\n")
            f.flush()
            done.append(row)
            mx.clear_cache()
            if n % 50 == 0:
                print(f"  quest {name} {n + 1}/{len(items)}  peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    return done


def quest_selfcheck(n=3):
    """Quest selecting every page must reproduce plain greedy decoding; the bound must never undercut a true page max."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    rng = np.random.default_rng(0)
    qv, k = rng.normal(size=(4, 8)), rng.normal(size=(4, 5, PAGE, 8))
    ub = quest_bound(mx.array(qv), mx.array(k.min(2)), mx.array(k.max(2)))
    assert np.all(np.array(ub) >= np.einsum("hd,hptd->hpt", qv, k).max(-1) - 1e-5)
    qq, kk, vv = mx.zeros((1, 4, 1, 8)), mx.zeros((1, 2, 3, 8)), mx.array(rng.normal(size=(1, 2, 3, 8)), mx.float32)
    m = mx.array([[[[True, False, False]]]] * 1).astype(mx.bool_)
    m = mx.concatenate([mx.ones((1, 1, 1, 3), mx.bool_), m, mx.ones((1, 2, 1, 3), mx.bool_)], axis=1)  # head 1: key 0 only
    o = mx.fast.scaled_dot_product_attention(qq, kk, vv, scale=1.0, mask=m)
    assert np.allclose(np.array(o[0, 1, 0]), np.array(vv[0, 0, 0]), atol=1e-4), "GQA head order is not h // G"
    model, tok = load(MODEL)
    items = contexts(MAIN, 8000, "lme-dev")[:n]
    for budget in ("100000", "2000"):  # all pages (must match), then the real budget (must run)
        got = gen_quest(model, tok, items, {"quest": budget}, f"questcheck-{budget}")
        for (q, text, _, _), g in zip(items, got):
            full = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False)
            ref = []
            for t, _ in generate_step(mx.array(tok.encode(full, add_special_tokens=False)), model, max_tokens=MAX_NEW):
                t = t.item() if hasattr(t, "item") else t
                if t in tok.eos_token_ids:
                    break
                ref.append(t)
            a, b = g["out"], tok.decode(ref)
            same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"quest selfcheck {budget}: {q['qid']} identical for {same}/{len(b)} chars", flush=True)
            if budget == "100000":
                assert same >= min(100, len(b)), (a[:300], b[:300])
    for p in (OUT.parent / "exp3_gen").glob("questcheck-*"):
        p.unlink()
    print("quest selfcheck pass")


def gen_kvq(model, tok, items, opt, name):
    """Exp 5.15: B=1 greedy decoding with the KV cache quantized to `kvb` bits (mlx-lm QuantizedKVCache, group 64,
    from the first prefill chunk on); kvb=16 = the unquantized B=1 control. Same joint prompt ids as ta.gen. Resumable jsonl."""
    import hashlib
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    bits = int(opt["kvb"])
    path = OUT.parent / "exp3_gen" / f"{name}-{hashlib.md5(json.dumps([t for _, t, _, _ in items]).encode()).hexdigest()[:8]}.jsonl"
    done = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    with open(path, "a") as f:
        for n, (q, text, _, _) in enumerate(items[len(done):], len(done)):
            ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                     enable_thinking=False, tokenize=False), add_special_tokens=False)
            new = []
            for t, _ in generate_step(mx.array(ids), model, max_tokens=MAX_NEW, kv_bits=None if bits >= 16 else bits,
                                      kv_group_size=64, quantized_kv_start=0):
                t = t.item() if hasattr(t, "item") else t
                if t in tok.eos_token_ids:
                    break
                new.append(t)
            row = {"out": tok.decode(new)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            done.append(row)
            mx.clear_cache()
            if n % 50 == 0:
                print(f"  kvq {name} {n + 1}/{len(items)}  peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    return done


def snap_selfcheck(n=3):
    """Keep-everything SnapKV must reproduce plain greedy decoding (tests the gather, RoPE offset and requery paths)."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    model, tok = load(MODEL)
    items = contexts(MAIN + ",snap=2000", 2000, "lme-dev")[:n]
    for opt in ({"snap": "100000"}, {"snap": "100000", "requery": "1"}, {"snap": "100000", "win": "32"}):
        got = gen_snap(model, tok, items, opt, f"snapcheck-{len(opt)}-{opt.get('win', 'q')}")
        for (q, text, _, _), g in zip(items, got):
            full = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False)
            ids = tok.encode(full, add_special_tokens=False)
            ref = []
            for t, _ in generate_step(mx.array(ids), model, max_tokens=MAX_NEW):
                t = t.item() if hasattr(t, "item") else t
                if t in tok.eos_token_ids:
                    break
                ref.append(t)
            a, b = g["out"], tok.decode(ref)
            same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"selfcheck {opt}: {q['qid']} identical for {same}/{len(b)} chars, kv_ev {g['kv_ev']}", flush=True)
            assert same >= min(100, len(b)), (a[:300], b[:300])  # fp16 chunking noise may flip a late greedy token
            assert g["kv_ev"] in (None, 1.0), g["kv_ev"]
    for p in (OUT.parent / "exp3_gen").glob("snapcheck-*"):
        p.unlink()
    print("snap selfcheck pass")


def run(cfg, budgets, split):
    from mlx_lm import load
    model, tok = load(READERS[parse(cfg)[1].get("rdr", "4b")])
    OUT.mkdir(parents=True, exist_ok=True)
    for budget in budgets:
        key = f"{cfg}@{budget}" + ("" if split == "test" else f"~{split}")
        path = OUT / f"{key}.json"
        if path.exists():
            print(f"{key}: cached")
            continue
        items = contexts(cfg, budget, split)
        if cfg == "full":
            items = [it for it in items if it[0]["bench"] == "locomo"]  # LME histories (~105K tokens) exceed 40K context
            outs = gen_prefixed(model, tok, items)
        elif "snap" in parse(cfg)[1] or "quest" in parse(cfg)[1] or "kvb" in parse(cfg)[1]:
            o = parse(cfg)[1]
            extra = (gen_snap if "snap" in o else gen_quest if "quest" in o else gen_kvq)(model, tok, items, o, f"exp4_{key}")
            outs = [x.pop("out") for x in extra]
        else:
            outs = ta.gen(model, tok, [t for _, t, _, _ in items], [MAX_NEW] * len(items), f"exp4_{key}")
        extra = extra if any(k in parse(cfg)[1] for k in ("snap", "quest", "kvb")) else [{}] * len(items)
        path.write_text(json.dumps({q["qid"]: {"out": o, "mem_tok": u, "recall": r, **x}  # snap: mem_tok = retrieved, kv_tok = kept
                                    for (q, _, u, r), o, x in zip(items, outs, extra)}))
        print(f"{key}: {len(items)} answers, mean memory tokens {np.mean([u for _, _, u, _ in items]):.0f}", flush=True)


# ---------- judge (prompts from LongMemEval src/evaluation/evaluate_qa.py, MIT) ----------

_ASK = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
        "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
        "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
        "only contains a subset of the information required by the answer, answer no. ")
_END = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE_PROMPTS = {
    "default": _ASK + _END,
    "temporal-reasoning": _ASK + "In addition, do not penalize off-by-one errors for the number of days. If the "
    "question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting "
    "19 days when the answer is 18), the model's response is still correct. " + _END,
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes "
    "if the response contains the correct answer. Otherwise, answer no. If the response contains some previous "
    "information along with an updated answer, the response should be considered as correct as long as the updated "
    "answer is the required answer." + _END,
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
    "The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please "
    "answer yes if the model correctly identifies the question as unanswerable. The model could say that the "
    "information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}"
    "\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only.",
}


def judge_prompt(q, response):
    kind = "abstention" if q["abs"] else q["type"] if q["bench"] == "lme" and q["type"] in JUDGE_PROMPTS else "default"
    return JUDGE_PROMPTS[kind].format(q["question"], q["answer"], response)


def judge():
    from mlx_lm import load
    qs = {q["qid"]: q for q in json.loads(QS.read_text())}
    model = tok = None
    for path in sorted(OUT.glob("*@*.json")):
        res = json.loads(path.read_text())
        todo = [k for k, v in res.items() if "judge" not in v]
        if not todo:
            continue
        if model is None:
            model, tok = load(JUDGE)
        outs = ta.gen(model, tok, [judge_prompt(qs[k], res[k]["out"]) for k in todo], [8] * len(todo), f"exp4_judge_{path.stem}")
        for k, o in zip(todo, outs):
            res[k]["judge"] = int("yes" in o.lower())
        path.write_text(json.dumps(res))
        print(f"judged {path.stem}: {np.mean([res[k]['judge'] for k in todo]):.1%}", flush=True)


def check_items(cfg, name="judge_check", per_bench=50):
    """A fixed judge-check set: per_bench LME test answers stratified by type + per_bench LoCoMo by category
    (judge_check = 4B, 100 items; judge_check_8b = Exp 5.3, 40 items).
    Raters: `human` (the author, via `label`) and `claude` (second annotator); both label blind to the judge."""
    import random
    path = OUT / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    qs = {q["qid"]: q for q in json.loads(QS.read_text())}
    res = json.loads((OUT / f"{cfg}.json").read_text())
    rng, picked = random.Random(0), []
    for bench in ("lme", "locomo"):
        pool = sorted(k for k in res if qs[k]["bench"] == bench)
        types = sorted({qs[k]["type"] for k in pool})
        for j in range(per_bench):
            t = types[j % len(types)]
            picked.append(rng.choice([k for k in pool if qs[k]["type"] == t and k not in picked]))
    items = [{"qid": k, "config": cfg, "human": None, "claude": None} for k in picked]
    path.write_text(json.dumps(items, indent=1))
    return items


def label(cfg, name="judge_check", per_bench=50):
    qs = {q["qid"]: q for q in json.loads(QS.read_text())}
    res = json.loads((OUT / f"{cfg}.json").read_text())
    path, items = OUT / f"{name}.json", check_items(cfg, name, per_bench)
    for it in items:
        if it["human"] is not None:
            continue
        q = qs[it["qid"]]
        print(f"\n{'=' * 80}\n[{q['bench']} / {q['type']}{' / UNANSWERABLE' if q['abs'] else ''}]  {q['question']}\n"
              f"GOLD: {q['answer']}\n--- response ---\n{res[it['qid']]['out']}\n")
        a = input("correct? [y/n, q=quit] ").strip().lower()
        if a == "q":
            break
        it["human"] = int(a == "y")
        path.write_text(json.dumps(items, indent=1))
    print(f"{sum(i['human'] is not None for i in items)}/{len(items)} labelled → {path}")


# ---------- surprise (gate signal) ----------

def surprise(model_name="mlx-community/Qwen3-0.6B-4bit", tok_budget=4096):
    """Mean token NLL per chunk, context-free. Token budget bounds the (batch × len × 152K-vocab) logits."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    model, tok = load(model_name)
    db = sqlite3.connect(DB)
    rows = db.execute("SELECT id, role, text FROM chunks WHERE nll IS NULL ORDER BY ntok").fetchall()
    print(f"{len(rows)} chunks to score", flush=True)
    i = steps = 0
    while i < len(rows):
        ids = [tok.encode(f"{r[1]}: {r[2]}") for r in rows[i:i + 512]]
        n = max(2, min(len(ids), tok_budget // max(len(ids[-1]), 1)))
        ids = ids[:n]
        L = max(len(x) for x in ids)
        x = mx.array([t + [0] * (L - len(t)) for t in ids])
        mask = mx.array([[1] * (len(t) - 1) + [0] * (L - len(t)) for t in ids], mx.float32)
        ce = nn.losses.cross_entropy(model(x[:, :-1]), x[:, 1:]) * mask
        nll = (ce.sum(1) / mx.maximum(mask.sum(1), 1)).tolist()
        db.executemany("UPDATE chunks SET nll = ? WHERE id = ?", [(v, r[0]) for v, r in zip(nll, rows[i:i + n])])
        db.commit()
        i, steps = i + n, steps + 1
        if steps % 500 == 0:
            print(f"  {i}/{len(rows)}", flush=True)


# ---------- offline recall (no LLM) ----------

def recall(methods):
    db, E = store()
    qs = questions("lme-dev")
    keeps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
    curves = {}
    for q, _ in qs:
        m = load_user(db, E, q["user"])
        if not q["ev_turns"]:
            continue
        ev = {tuple(e) for e in q["ev_turns"]}
        names = list(signals(m)) + ["all"]
        for s in names:
            for k in keeps:
                a = gate(m, k, s)
                curves.setdefault(s, {}).setdefault(k, []).append(
                    len(ev & {(m["sid"][i], m["turn"][i]) for i in np.flatnonzero(a)}) / len(ev))
    print("gate: evidence-turn recall vs kept fraction (LME dev)")
    print("signal      " + "".join(f"{k:>7}" for k in keeps))
    for s, c in curves.items():
        print(f"{s:12}" + "".join(f"{np.mean(c[k]):7.2f}" for k in keeps))
    for cfg in methods:
        for b in (1000, 2000, 4000, 8000):
            r = [x[3] for x in contexts(cfg, b, "lme-dev") if x[3] is not None]
            print(f"retrieval {cfg:28} @{b:5}: evidence recall {np.mean(r):.3f}")


def tier(keeps, taus, split):
    """Exp 5.2: the gate picks the RAM-resident hot set, nothing is deleted. Answer from the hot set unless its best
    cosine < tau, else fall back to the full store. Composes the hot-only run (MAIN,keep=k,gate=self) with the full
    MAIN run @2K — the fallback context is exactly MAIN's, so no new generation. One tau → writes tier,… result files.
    shortcut: BM25 IDF comes from the whole store, a real hot tier would use hot-set stats; matters only if 5.5 shows drift."""
    sfx = "" if split == "test" else f"~{split}"
    full = json.loads((OUT / f"{MAIN}@2000{sfx}.json").read_text())
    db, E = store()
    rows, m = [], None
    for q, qe in sorted(questions(split), key=lambda r: r[0]["user"]):
        if m is None or m["_user"] != q["user"]:
            m = load_user(db, E, q["user"])
            m["_user"] = q["user"]
            size = np.array([len(t.encode()) for t in m["text"]]) + DIM * 2  # text + fp16 embedding per chunk
        cos = m["E"] @ qe
        for k in keeps:
            hot = gate(m, k, "self")
            rows.append((q["qid"], k, float(cos[hot].max()), size[hot].sum() / 2 ** 20, size[~hot].sum() / 2 ** 10))
    print(f"{'keep':>5} {'tau':>5} {'acc':>6} {'fallback':>8} {'hot MB/user':>11} {'flash KB/query':>14}")
    for k in keeps:
        hot = json.loads((OUT / f"{MAIN},keep={k},gate=self@2000{sfx}.json").read_text())
        for tau in taus:
            res = {}
            for qid, kk, top, hot_mb, cold_kb in rows:
                if kk == k:
                    fb = top < tau
                    res[qid] = {**(full if fb else hot)[qid], "fallback": int(fb), "hot_mb": hot_mb, "flash_kb": cold_kb * fb}
            v = list(res.values())
            print(f"{k:5} {tau:5} {np.mean([r['judge'] for r in v]):6.1%} {np.mean([r['fallback'] for r in v]):8.1%} "
                  f"{np.mean([r['hot_mb'] for r in v]):11.2f} {np.mean([r['flash_kb'] for r in v]):14.0f}")
            if len(taus) == 1:
                (OUT / f"tier,keep={k},tau={tau}@2000{sfx}.json").write_text(json.dumps(res))


# ---------- Exp 5.4: session notes (summary + embedding per session) ----------

def summary_prompts(db):
    """One prompt per session of the LME dev + LoCoMo + LME test-100 users (dev first, so `summarise N` times dev)."""
    users = dict.fromkeys(q["user"] for q, _ in questions("lme-dev") + questions("loc+lme100"))
    sids = list(dict.fromkeys(s for u in users for (s,) in db.execute("SELECT sid FROM haystack WHERE user = ? ORDER BY pos", (u,))))
    prompts = []
    for s in sids:
        turns = {}
        for t, role, text in db.execute("SELECT turn, role, text FROM chunks WHERE sid = ? ORDER BY turn, part", (s,)):
            turns.setdefault(t, [role, []])[1].append(text)
        date = db.execute("SELECT date FROM sessions WHERE sid = ?", (s,)).fetchone()[0]
        prompts.append(SUM_PROMPT.format(date=date, text="\n".join(f"{r}: {' '.join(x)}" for r, x in turns.values())))
    return sids, prompts


def summarise(n_bench=0):
    """4B notes for every session (resumable via ta.gen's cache) → data/exp5/summaries.sqlite.
    n_bench > 0: time the 4B and the 0.6B on the first n LME dev sessions instead; writes no store.
    (A re-run reads the bench cache: delete results/exp3_gen/exp5_sumbench_* to re-time.)"""
    import time
    from mlx_lm import load
    sids, prompts = summary_prompts(store()[0])
    print(f"{len(sids)} sessions to summarise", flush=True)
    if n_bench:
        for name in (READERS["4b"], "mlx-community/Qwen3-0.6B-4bit"):
            model, tok = load(name)
            t = time.perf_counter()
            outs = ta.gen(model, tok, prompts[:n_bench], [SUM_MAX] * n_bench, f"exp5_sumbench_{name.split('/')[-1]}")
            s = (time.perf_counter() - t) / n_bench
            print(f"{name}: {s:.2f} s/session → all {len(sids)} ≈ {s * len(sids) / 3600:.1f} h; "
                  f"mean {np.mean([len(tok.encode(o)) for o in outs]):.0f} tok; e.g. {outs[0][:400]!r}", flush=True)
            del model
        return
    model, tok = load(READERS["4b"])
    outs = [o.strip() for o in ta.gen(model, tok, prompts, [SUM_MAX] * len(prompts), "exp5_summaries")]
    D5.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(SUMDB)
    db.execute("CREATE TABLE IF NOT EXISTS summaries(sid TEXT PRIMARY KEY, row INT, text TEXT, ntok INT)")
    db.executemany("INSERT OR REPLACE INTO summaries VALUES (?,?,?,?)",
                   [(s, i, o, len(tok.encode(f"notes: {o}", add_special_tokens=False))) for i, (s, o) in enumerate(zip(sids, outs))])
    db.commit()
    print(f"{len(outs)} notes, mean {np.mean([r[0] for r in db.execute('SELECT ntok FROM summaries')]):.0f} tokens", flush=True)


def sumembed():
    """MiniLM embeddings of the notes, row = summaries.row. Own process: torch next to MLX risks the exit segfault."""
    from sentence_transformers import SentenceTransformer
    rows = sqlite3.connect(SUMDB).execute("SELECT text FROM summaries ORDER BY row").fetchall()
    emb = SentenceTransformer(EMB, device="cpu")
    np.asarray(emb.encode([r[0] for r in rows], batch_size=128, normalize_embeddings=True), np.float16).tofile(SUMEMB)
    print(f"embedded {len(rows)} notes", flush=True)


def summaries(m):
    """This user's notes, one per session, chronological; same fields as a user store, so fill/bm25/render work on it."""
    sids = list(dict.fromkeys(m["sid"]))
    rows = {r[0]: r[1:] for r in sqlite3.connect(SUMDB).execute(
        f"SELECT sid, row, text, ntok FROM summaries WHERE sid IN ({','.join('?' * len(sids))})", sids)}
    assert len(rows) == len(sids), "notes missing for this user: run summarise"
    first = {s: m["sid"].index(s) for s in sids}
    row, text, ntok = map(list, zip(*(rows[s] for s in sids)))
    S = np.memmap(SUMEMB, np.float16, "r").reshape(-1, DIM)
    return {"id": np.array(row), "sid": sids, "turn": np.full(len(sids), -1), "role": ["notes"] * len(sids), "text": text,
            "ntok": np.array(ntok), "date": [m["date"][first[s]] for s in sids], "sess": np.arange(len(sids)),
            "E": np.asarray(S[row], np.float32), "first": [first[s] for s in sids]}


def note_select(ms, q, qe, budget, name, opt):
    """sumall: newest notes that fit (all of them at 8K); sum: hybrid over notes; hybrid,sum=f: hybrid notes get f of
    the budget, MAIN fills the rest with raw chunks. Headers are costed in both fills, so mixes slightly under-fill."""
    if name == "sumall":
        return fill(ms, np.arange(len(ms["id"]))[::-1], budget)
    order = np.argsort(-rrf(ms["E"] @ qe, bm25(ms, q["question"])))
    return fill(ms, order, budget if name == "sum" else int(float(opt["sum"]) * budget))


def render_mix(m, idx, ms, sidx):
    """Chronological; a session's note sits right under its date header, before that session's excerpts."""
    items = sorted([(ms["first"][j] - .5, ms["sid"][j], ms["date"][j], f"notes: {ms['text'][j]}") for j in sidx] +
                   [(i, m["sid"][i], m["date"][i], f"{m['role'][i]}: {m['text'][i]}") for i in idx])
    out, last = [], None
    for _, sid, date, line in items:
        if sid != last:
            out.append(f"\n### Session ({date})")
            last = sid
        out.append(line)
    return "\n".join(out)


# ---------- latency (M4 Max; the iPhone numbers come later, on the device) ----------

LAT = OUT / "latency.json" if LME_M else ROOT / "results" / "exp4_latency.json"


def ttft_prompts(lme, loc):
    """5 LME test prompts per hybrid budget + 5 whole-history LoCoMo prompts (one per conversation, ~20K tokens)."""
    want = {q["qid"] for q in lme[:5]}
    prompts = {f"hybrid@{b}": [t for q, t, _, _ in contexts(MAIN, b, "lme-test") if q["qid"] in want] for b in (1000, 2000, 4000, 8000)}
    one_per_conv = {q["user"]: q["qid"] for q in loc}
    want = set(list(one_per_conv.values())[:5])
    prompts["full@locomo"] = [t for q, t, _, _ in contexts("full", 0, "test") if q["qid"] in want]
    return prompts


def latency(part):
    """retrieval (CPU): per-query store load, query embedding, selection — the BM25 index build is timed apart
    (it is an ingest-time cost on the phone). ttft (GPU, nothing else may run on it): time to first token and decode
    speed for real prompts at each budget, fresh KV cache every time (no prefix-cache reuse: it skews latency),
    plus peak memory. Results merge into results/exp4_latency.json."""
    import time
    out = json.loads(LAT.read_text()) if LAT.exists() else {}
    ms = lambda xs: {"median_ms": float(np.median(xs) * 1e3), "p90_ms": float(np.percentile(xs, 90) * 1e3), "n": len(xs)}
    rng = np.random.default_rng(0)
    qs = [q for q, _ in questions("test")]
    lme = [qs[i] for i in rng.choice([i for i, q in enumerate(qs) if q["bench"] == "lme"], 100, replace=False)]
    loc = [qs[i] for i in rng.choice([i for i, q in enumerate(qs) if q["bench"] == "locomo"], 100, replace=False)]
    if part == "retrieval":
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(1)  # one core, closer to a phone's budget than the M4 Max's 14
        emb = SentenceTransformer(EMB, device="cpu")
        emb.encode(["warm-up"])
        db, E = store()
        res = {}
        for bench, sample in (("lme", lme), ("locomo", loc)):
            t = {"load": [], "embed": [], "index": [], "select": []}
            for q in sample:
                t0 = time.perf_counter()
                m = load_user(db, E, q["user"])
                t1 = time.perf_counter()
                qe = emb.encode([q["question"]], normalize_embeddings=True)[0]
                t2 = time.perf_counter()
                bm25(m, "warm")  # builds the per-user BM25 index
                t3 = time.perf_counter()
                for b in (1000, 2000, 4000, 8000):
                    s0 = time.perf_counter()
                    select(m, q, qe, b, *parse(MAIN))
                    t["select"].append(time.perf_counter() - s0)
                t["load"].append(t1 - t0); t["embed"].append(t2 - t1); t["index"].append(t3 - t2)
            res[bench] = {k: ms(v) for k, v in t.items()} | {"chunks_per_user": float(np.mean([len(load_user(db, E, q["user"])["id"]) for q in sample[:10]]))}
        out["retrieval"] = res | {"note": "1 CPU thread, M4 Max; select = hybrid at 1K/2K/4K/8K, index build excluded"}
    else:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        model, tok = load(MODEL)
        prompts = ttft_prompts(lme, loc)
        res = {}
        for name, texts in prompts.items():
            rows = []
            for k, text in enumerate([texts[0]] + texts):  # first run of each size is a discarded warm-up
                ids = mx.array(rq.chat(tok, text))
                mx.clear_cache(); mx.reset_peak_memory()
                cache = make_prompt_cache(model)
                t0 = time.perf_counter()
                gen = generate_step(ids, model, max_tokens=65, prompt_cache=cache)
                next(gen)
                t1 = time.perf_counter()
                n = sum(1 for _ in gen)
                t2 = time.perf_counter()
                if k:
                    rows.append((len(ids), t1 - t0, n / (t2 - t1), mx.get_peak_memory() / 2 ** 30))
            L, ttft, dec, peak = map(np.array, zip(*rows))
            res[name] = {"prompt_tokens": float(L.mean()), "ttft_s": float(np.median(ttft)), "ttft_s_max": float(ttft.max()),
                         "prefill_tok_s": float(np.median(L / ttft)), "decode_tok_s": float(np.median(dec)),
                         "peak_gb": float(peak.max()), "n": len(rows)}
            print(name, res[name], flush=True)
        out["ttft"] = res | {"note": "M4 Max GPU, Qwen3-4B-4bit, fresh cache per prompt, median of 5 real prompts"}
    LAT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out[part if part == "retrieval" else "ttft"], indent=1))


# ---------- report ----------

def report():
    qs = {q["qid"]: q for q in json.loads(QS.read_text())}
    rows, per_item = {}, {}
    for path in sorted(OUT.glob("*@*.json")):
        res = json.loads(path.read_text())
        if not all("judge" in v for v in res.values()):
            continue
        per_item[path.stem] = {k: v["judge"] for k, v in res.items()}
        r = {}
        for bench in ("lme", "locomo"):
            ks = [k for k in res if qs[k]["bench"] == bench]
            if not ks:
                continue
            r[bench] = float(np.mean([res[k]["judge"] for k in ks]))
            r[f"{bench}_types"] = {t: float(np.mean([res[k]["judge"] for k in ks if qs[k]["type"] == t]))
                                   for t in sorted({qs[k]["type"] for k in ks})}
            if bench == "locomo":
                r["locomo_f1"] = float(np.mean([ta.em_f1(ta.final_answer(res[k]["out"]), [qs[k]["answer"]])[1] for k in ks]))
            r[f"{bench}_mem_tok"] = float(np.mean([res[k]["mem_tok"] for k in ks]))
            r[f"{bench}_kv_mb"] = float(np.mean([res[k].get("kv_tok", res[k]["mem_tok"]) for k in ks])) * KV_BYTES_PER_TOK / 2 ** 20
            for x in ("kv_ev", "ev_small"):  # Exp 5.8: evidence tokens kept by SnapKV vs by direct small retrieval
                v = [res[k][x] for k in ks if res[k].get(x) is not None]
                if v:
                    r[f"{bench}_{x}"] = float(np.mean(v))
            rec = [res[k]["recall"] for k in ks if res[k]["recall"] is not None]
            r[f"{bench}_recall"] = float(np.mean(rec)) if rec else None
        rows[path.stem] = r
    agree = {}
    for check in sorted(OUT.glob("judge_check*.json")):
        items = json.loads(check.read_text())
        res = {c: json.loads((OUT / f"{c}.json").read_text()) for c in {i["config"] for i in items}}
        for i in items:
            i["judge"] = res[i["config"]][i["qid"]].get("judge")
        for a, b in [("human", "judge"), ("claude", "judge"), ("human", "claude")]:
            pairs = [(i[a], i[b]) for i in items if i.get(a) is not None and i.get(b) is not None]
            if pairs:
                x, y = np.array(pairs, float).T
                po, pe = (x == y).mean(), x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
                agree[f"{check.stem}: {a} vs {b}"] = {"n": len(pairs), "agreement": float(po),
                                        "kappa": float((po - pe) / (1 - pe)) if pe < 1 else None,
                                        f"{b}_minus_{a}": float(y.mean() - x.mean())}
    rng = np.random.default_rng(0)
    cis = {}
    pairs = []
    for cfg, ref in [(c, r) for c in per_item for r in ("rag", MAIN)]:  # paired bootstrap vs RAG and vs the main config
        budget = cfg.split("@")[1]  # same budget; a "~lme-test" ablation pairs with the full-test run on shared items
        base = next((b for b in (f"{ref}@{budget}", f"{ref}@{budget.split('~')[0]}") if b in per_item and b != cfg), None)
        if base is not None and not cfg.startswith(f"{ref}@"):
            pairs.append((cfg, base))
    pairs += [(c, f"{MAIN}@{parse(c.split('@')[0])[1]['snap']}") for c in per_item  # Exp 5.8: SnapKV vs retrieving that many tokens
              if ",snap=" in c and f"{MAIN}@{parse(c.split('@')[0])[1]['snap']}" in per_item]
    for cfg, base in pairs:
        for bench in ("lme", "locomo"):
            ks = [k for k in per_item[cfg] if qs[k]["bench"] == bench and k in per_item[base]]
            if ks:
                d = np.array([per_item[cfg][k] - per_item[base][k] for k in ks], float)
                bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                cis[f"{cfg} - {base} [{bench}]"] = [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
    (OUT / "summary.json" if LME_M else ROOT / "results" / "exp4.json").write_text(json.dumps({"table": rows, "judge_check": agree, "paired_ci": cis}, indent=1))
    print("| config | LME acc | LoCoMo J | LoCoMo F1 | memory tok (LME/LoCoMo) | KV MB (LME) | evidence recall (LME/LoCoMo) |")
    print("|---|---|---|---|---|---|---|")
    f = lambda v, p=True: "–" if v is None else (f"{v:.1%}" if p else f"{v:.0f}")
    for c, r in rows.items():
        print(f"| {c} | {f(r.get('lme'))} | {f(r.get('locomo'))} | {f(r.get('locomo_f1'))} | "
              f"{f(r.get('lme_mem_tok'), 0)}/{f(r.get('locomo_mem_tok'), 0)} | {f(r.get('lme_kv_mb'), 0)} | "
              f"{f(r.get('lme_recall'))}/{f(r.get('locomo_recall'))} |")
    for k, v in agree.items():
        print(f"{k}: " + ", ".join(f"{m} {x:.2f}" if isinstance(x, float) else f"{m} {x}" for m, x in v.items()))
    for k, (mu, lo, hi) in cis.items():
        print(f"{k:45} {mu * 100:+.1f} [{lo * 100:+.1f},{hi * 100:+.1f}]{'*' if lo > 0 or hi < 0 else ''}")


def _check():
    ps = pieces("ab " * 900)
    assert all(len(x) <= CHUNK_CHARS for x in ps) and " ".join(ps).split() == ["ab"] * 900
    assert pieces("x" * 2500) == ["x" * 1000, "x" * 1000, "x" * 500] and pieces("") == [""]
    E = np.eye(4, DIM, dtype=np.float32)
    m = {"id": np.arange(4), "sid": ["s0", "s0", "s1", "s1"], "turn": np.array([0, 1, 0, 1]), "role": ["user"] * 4,
         "owner": np.array([1, 0, 1, 0], bool), "text": ["I love jazz", "ok", "my cat Tom", "fine"],
         "ntok": np.array([10, 10, 10, 10]), "nll": np.full(4, np.nan), "date": ["d0", "d0", "d1", "d1"],
         "sess": np.array([0, 0, 1, 1]), "E": E}
    assert fill(m, [2, 0, 1, 3], 2 * (10 + HEADER_TOK))[0] == [0, 2]  # two chunks + two headers fit, third doesn't
    assert fill(m, [2], 100, expand=True)[0] == [2, 3]  # neighbour turn pulled in
    assert gate(m, 0.5, "self").tolist() == [True, False, True, False]
    assert select(m, {"question": "ok fine"}, E[1], 100, "hybrid", {"div": "0", "edges": "0", "keep": "0.5", "gate": "self", "asst": "1"})[0] == [0, 1, 2, 3]
    assert bm25(m, "jazz").argmax() == 0
    cos = np.array([.9, .1, .8, .2])
    assert adapt(m, cos, {"cut": "0.85"}, 100, floor=0) == 2 * (10 + HEADER_TOK)  # chunks 0, 2: two sessions
    assert adapt(m, cos, {"cut": "0"}, 50, floor=0) == 50 and adapt(m, cos, {}, 77) == 77  # ceiling; no rule = fixed
    assert adapt(m, cos, {"knee": "4"}, 100, floor=0) == 2 * (10 + HEADER_TOK)  # elbow after .9, .8
    assert adapt(m, cos, {"mass": "0.5"}, 100, floor=0) == 10 + HEADER_TOK
    assert select(m, {"question": "jazz cat"}, E[0] + E[2] * .5, 100, "hybrid", {"div": "0", "edges": "0", "order": "rel"})[0][-1] == 0
    assert ev_recall(m, {"ev_turns": [["s1", 0], ["s0", 1]]}, [2]) == 0.5
    assert judge_prompt({"abs": True, "bench": "lme", "type": "multi-session", "question": "q", "answer": "a"}, "r").startswith("I will give you an unanswerable")
    ms = {"id": np.arange(2), "sid": ["s0", "s1"], "turn": np.full(2, -1), "role": ["notes"] * 2, "date": ["d0", "d1"],
          "text": ["likes jazz", "cat named Tom"], "ntok": np.array([5, 5]), "sess": np.arange(2), "E": E[[0, 2]], "first": [0, 2]}
    assert note_select(ms, {"question": "x"}, E[0], 5 + HEADER_TOK, "sumall", {})[0] == [1]  # newest note wins
    assert note_select(ms, {"question": "cat"}, E[2], 100, "hybrid", {"sum": "0.25"})[0] == [1]  # 25 tok: one note
    assert [l for l in render_mix(m, [3, 1], ms, [1]).split("\n") if l] == ["### Session (d0)", "user: ok", "### Session (d1)",
                                                                "notes: cat named Tom", "user: fine"]
    sq = [{"qid": f"{t}{i}", "bench": "b", "split": "test", "type": t} for t, n in (("x", 5), ("y", 3), ("z", 2)) for i in range(n)]
    assert [sum(k.startswith(t) for k in stratified(sq, "b", 5)) for t in "xyz"] == [3, 1, 1]  # 2.5/1.5/1 → largest remainders
    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 10])
    lab = kmeans(X, 2)
    assert len(set(lab[:5])) == len(set(lab[5:])) == 1 and lab[0] != lab[5]
    s = np.array([[0, 5, 0, 0, 1, 0], [3, 0, 0, 0, 0, 2.]])
    assert snap_keep(s, 2, 1).tolist() == [[1, 4], [0, 5]]  # per head, sorted positions
    assert snap_keep(np.array([[0, 5, 0, 0, 4, 4, 4, 0.]]), 1, 3).tolist() == [[5]]  # pooled: a run of 4s beats the lone 5
    assert snap_keep(s, 99, 7).shape == (2, 6)
    qv, kp = np.array([[1., -2.]]), np.array([[[[0, 1], [2, -1]]]], float)  # one head, one page of two keys
    assert np.isclose(quest_bound(qv, kp.min(2), kp.max(2)), 1 * 2 + (-2) * (-1)).all()  # ≥ true max (2+2=4 vs 4)
    print("checks pass")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "tier":  # tier <keep,…> <tau,…> [--set lme-dev|test]
        tier([float(x) for x in args[0].split(",")], [float(x) for x in args[1].split(",")],
             args[args.index("--set") + 1] if "--set" in args else "test")
    elif cmd == "run":
        split = args[args.index("--set") + 1] if "--set" in args else "test"
        args = [a for a in args if a not in ("--set", split)]
        run(args[0], [int(b) for b in args[1:]] or [0], split)
    else:
        {"ingest": ingest, "embed": embed, "check": _check, "surprise": lambda: surprise(*args), "recall": lambda: recall(args),
         "judge": judge, "snapcheck": snap_selfcheck, "questcheck": quest_selfcheck, "summarise": lambda: summarise(*map(int, args)), "sumembed": sumembed, "latency": lambda: latency(args[0]), "label": lambda: label(args[0], *args[1:2], *map(int, args[2:3])), "report": report}[cmd]()
