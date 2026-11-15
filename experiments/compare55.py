"""Exp 5.5: LongMemEval-M vs LongMemEval-S on the same 400 test questions (paired bootstrap, 10,000 resamples)."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent
M, S = R / "exp5" / "lme_m", R / "exp4"
H = "hybrid,div=0,edges=0"
qs = {q["qid"]: q for q in json.loads((R.parent / "data" / "exp4" / "questions.json").read_text())}
load = lambda p: json.loads(p.read_text())
runs = {"M hybrid@2K": M / f"{H}@2000~lme-test.json", "M hybrid@4K": M / f"{H}@4000~lme-test.json",
        "M rag@2K": M / "rag@2000~lme-test.json", "M gate0.5@2K": M / f"{H},keep=0.5,gate=self@2000~lme-test.json",
        "M tier0.5,tau0.35": M / "tier,keep=0.5,tau=0.35@2000~lme-test.json",
        "S hybrid@2K": S / f"{H}@2000.json", "S hybrid@4K": S / f"{H}@4000.json", "S rag@2K": S / "rag@2000.json",
        "S gate0.5@2K": S / f"{H},keep=0.5,gate=self@2000.json", "S tier0.5,tau0.35": S / "tier,keep=0.5,tau=0.35@2000.json"}
res = {k: load(p) for k, p in runs.items() if p.exists()}
ks = sorted(res["M hybrid@2K"])
rng = np.random.default_rng(0)


def ci(a, b):
    d = np.array([res[a][k]["judge"] - res[b][k]["judge"] for k in ks], float)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{d.mean() * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]{'*' if lo > 0 or hi < 0 else ''}"


acc = lambda r, sub=ks: np.mean([r[k]["judge"] for k in sub]) * 100
rec = lambda r: np.mean([r[k]["recall"] for k in ks if r[k].get("recall") is not None]) * 100
print(f"n = {len(ks)} LME test questions")
for name, r in res.items():
    extra = f", fallback {np.mean([r[k].get('fallback', 0) for k in ks]) * 100:.1f}%" if "tier" in name else ""
    rc = f", evidence recall {rec(r):.1f}%" if any(r[k].get("recall") is not None for k in ks) else ""
    print(f"{name:20} {acc(r):5.1f}%{rc}{extra}")
pairs = [("M hybrid@2K", "S hybrid@2K"), ("M hybrid@4K", "S hybrid@4K"), ("M rag@2K", "S rag@2K"),
         ("M hybrid@4K", "M hybrid@2K"), ("M hybrid@2K", "M rag@2K"), ("M gate0.5@2K", "M hybrid@2K"),
         ("M tier0.5,tau0.35", "M hybrid@2K"), ("S gate0.5@2K", "S hybrid@2K"), ("S tier0.5,tau0.35", "S hybrid@2K"), ("M gate0.5@2K", "S gate0.5@2K")]
for a, b in pairs:
    if a in res and b in res:
        print(f"{a} - {b}: {ci(a, b)}")
types = sorted({("abstention" if qs[k]["abs"] else qs[k]["type"]) for k in ks})
print("\nper type (n): " + " | ".join(n for n in res))
for t in types:
    sub = [k for k in ks if ("abstention" if qs[k]["abs"] else qs[k]["type"]) == t]
    print(f"{t:28} ({len(sub):3}) " + " ".join(f"{acc(r, sub):5.1f}" for r in res.values()))
