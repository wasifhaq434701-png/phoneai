"""Exp 5.14: does "retrieval beats attention-based KV selection" hold on RULER (NVIDIA, Apache 2.0)?

Five RULER tasks at 8K (50 samples each, generated with RULER's own prepare.py, Qwen3-4B tokenizer, into data/exp5/ruler/):
niah_single_1 (one needle in noise), niah_multikey_3 (uuid needle among uuid distractors), vt (4-hop variable chains),
cwe (10 most common words: the answer is spread over the whole context), qa_1 (SQuAD over distractor documents).
Configs, all Qwen3-4B-4bit, B=1, through user_memory.gen_snap (keep-all = plain greedy decoding, self-checked in Exp 5.8):
  full   the whole ~8K context
  rag2k  hybrid retrieval (MiniLM cosine + BM25, RRF) of ≤1,000-char chunks to 2,000 tokens, kept in document order
  snap2k SnapKV keeping 2,000 KV per head/layer of the full prompt, question re-fed (the Exp 5.8 dev pick)
Scored with RULER's metrics (string_match_all; string_match_part for QA). Usage: python ruler_probe.py check | run | score
"""
import json
import sys

import numpy as np

import user_memory as um

D = um.D5 / "ruler"
OUTF = um.ROOT / "results" / "exp5" / "ruler.json"
BUDGET = 2000
# task: (end of the instruction before the context, start of the question after it, RULER tokens_to_generate)
TASKS = {"niah_single_1": ("afterwards.\n", "\nWhat is the special magic", 128),
         "niah_multikey_3": ("afterwards.\n", "\nWhat is the special magic", 128),
         "vt": ("following text.\n\n", "\nQuestion: Find all variables", 30),
         "cwe": ("most often.\n", "\nQuestion: What are the 10 most common", 120),
         "qa_1": ("The following are given documents.\n\n", "\n\nAnswer the question based on the given documents.", 32)}
CFGS = {"full": {"snap": "1000000"}, "rag2k": {"snap": "1000000"}, "snap2k": {"snap": str(BUDGET), "requery": "1"}}


def split(text, start, end):
    a, b = text.index(start) + len(start), text.rfind(end)
    assert a <= b, "context markers out of order"
    return text[:a], text[a:b], text[b:]


def chunks(ctx, n=um.CHUNK_CHARS):
    """Consecutive lines merged into ≤ n-char chunks (like conversation turns); longer lines cut by um.pieces."""
    out, cur = [], ""
    for line in ctx.split("\n"):
        for p in (um.pieces(line, n) if len(line) > n else [line]):
            if cur and len(cur) + 1 + len(p) > n:
                out.append(cur)
                cur = p
            else:
                cur = f"{cur}\n{p}" if cur else p
    return out + ([cur] if cur else [])


def retrieve(cs, ntok, cos, question, budget=BUDGET):
    """Hybrid rank (cosine + BM25 by reciprocal rank fusion) → greedy fill to `budget` tokens → original order."""
    score = um.rrf(cos, um.bm25({"text": cs}, question))
    keep, used = [], 0
    for i in np.argsort(-score, kind="stable"):
        if used + ntok[i] <= budget:
            keep.append(i)
            used += ntok[i]
    return sorted(keep), used


def metric(task, pred, refs):
    p = pred.lower()
    hits = [r.lower() in p for r in refs]
    return float(any(hits)) if task == "qa_1" else float(np.mean(hits))


def items():
    """{cfg: [(q, prompt, tokens, None)]} plus per-sample metadata; retrieval runs on CPU."""
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    tok, emb = AutoTokenizer.from_pretrained(um.MODEL), SentenceTransformer(um.EMB, device="cpu")
    out, meta = {c: [] for c in CFGS}, {}
    for task, (start, end, max_new) in TASKS.items():
        for n, r in enumerate(json.loads(l) for l in open(D / task / "validation.jsonl")):  # RULER's "index" repeats
            pre, ctx, post = split(r["input"], start, end)
            cs = chunks(ctx)
            ntok = np.array([len(tok.encode(c, add_special_tokens=False)) for c in cs])
            cos = emb.encode(cs, normalize_embeddings=True) @ emb.encode([post.strip()], normalize_embeddings=True)[0]
            keep, used = retrieve(cs, ntok, cos, post)
            small = pre + "\n".join(cs[i] for i in keep) + post
            qid = f"{task}-{n}"
            ev = [ln for ln in ctx.split("\n") if task in ("niah_single_1", "niah_multikey_3", "vt") and any(o in ln for o in r["outputs"])]
            q = {"qid": qid, "_cut": end, "_prefix": r["answer_prefix"], "_max": max_new, "_ev": ev}
            meta[qid] = {"task": task, "refs": r["outputs"], "ctx_tok": int(ntok.sum()), "rag_tok": int(used),
                         "rag_has_answer": float(np.mean([o in small for o in r["outputs"]]))}
            out["full"].append((q, r["input"], int(ntok.sum()), None))
            out["snap2k"].append((q, r["input"], int(ntok.sum()), None))
            out["rag2k"].append(({**q, "_ev": []}, small, int(used), None))
    return out, meta


def run():
    from mlx_lm import load
    model, tok = load(um.MODEL)
    its, meta = items()
    res = {}
    for cfg, opt in CFGS.items():
        rows = um.gen_snap(model, tok, its[cfg], opt, f"exp5_ruler_{cfg}")
        res[cfg] = {q["qid"]: {**row, "score": metric(meta[q["qid"]]["task"], row["out"], meta[q["qid"]]["refs"])}
                    for (q, _, _, _), row in zip(its[cfg], rows)}
        print(f"{cfg}: mean {np.mean([v['score'] for v in res[cfg].values()]):.3f}", flush=True)
    (um.ROOT / "results" / "exp5" / "ruler_outputs.json").write_text(json.dumps({"meta": meta, "res": res}))
    score()


def boot(d, n=10000):
    bs = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]


def score():
    raw = json.loads((um.ROOT / "results" / "exp5" / "ruler_outputs.json").read_text())
    meta, res = raw["meta"], raw["res"]
    qids = sorted(meta)
    out = {}
    for task in [*TASKS, "all"]:
        ks = [k for k in qids if task == "all" or meta[k]["task"] == task]
        s = {c: np.array([res[c][k]["score"] for k in ks]) for c in CFGS}
        out[task] = {"n": len(ks), **{c: float(v.mean()) for c, v in s.items()},
                     "rag2k-full": boot(s["rag2k"] - s["full"]), "snap2k-full": boot(s["snap2k"] - s["full"]),
                     "rag2k-snap2k": boot(s["rag2k"] - s["snap2k"]),
                     "rag_has_answer": float(np.mean([meta[k]["rag_has_answer"] for k in ks])),
                     "snap_kv_ev": (lambda v: float(np.mean(v)) if v else None)([res["snap2k"][k]["kv_ev"] for k in ks if res["snap2k"][k].get("kv_ev") is not None]),
                     "ctx_tok": float(np.mean([meta[k]["ctx_tok"] for k in ks]))}
        r = out[task]
        f = lambda x: f"{x[0] * 100:+.1f} [{x[1] * 100:+.1f},{x[2] * 100:+.1f}]"
        print(f"{task:16} full {r['full']:.1%}  rag2k {r['rag2k']:.1%}  snap2k {r['snap2k']:.1%}  | rag-full {f(r['rag2k-full'])}  "
              f"snap-full {f(r['snap2k-full'])}  rag-snap {f(r['rag2k-snap2k'])}  | rag keeps answer {r['rag_has_answer']:.0%}", flush=True)
    OUTF.write_text(json.dumps(out, indent=1))


def check():
    t = "Intro afterwards.\nA\nB needle 42\nC\nWhat is the special magic number? Q"
    pre, ctx, post = split(t, "afterwards.\n", "\nWhat is the special magic")
    assert pre + ctx + post == t and ctx == "A\nB needle 42\nC" and post.startswith("\nWhat")
    cs = chunks("\n".join(["x" * 400] * 5))
    assert [len(c) for c in cs] == [801, 801, 400] and all(len(c) <= um.CHUNK_CHARS for c in cs)
    cs = ["the sky is blue", "magic number for jittery-hospital is 42", "the number grass is green"]
    keep, used = retrieve(cs, np.array([5, 9, 5]), np.array([0.1, 0.9, 0.2]), "magic number jittery-hospital", budget=14)
    assert keep == [1, 2] and used == 14  # best chunk first, then the next that fits, back in document order
    assert metric("vt", "AB, CD and EF", ["ab", "cd", "zz"]) == 2 / 3 and metric("qa_1", "it is France", ["france", "x"]) == 1.0
    print("checks pass")


if __name__ == "__main__":
    {"check": check, "run": run, "score": score}[sys.argv[1]]()
