"""Exp 5.11: predict the retrieval budget per question instead of fixing it.

Labels = strict-judge correctness of the Qwen3-4B reader with hybrid retrieval at 1K/2K/4K/8K tokens (existing outputs,
no new generation), on LME test 400 + LoCoMo 1,540. Predictors of P(correct | question, budget):
  (1) ModernBERT-base fine-tuned on the question text (external classifier),
  (2) logistic regression and a small MLP on Qwen3-4B layer-18 hidden states of the question (internal probe).
Policy: the budget with the highest predicted P(correct), ties -> the smaller budget. 5-fold CV grouped by user
(LoCoMo: conversation), so no user is in train and test. Baselines: each fixed budget, the best fixed budget per
benchmark learned on the training folds, and the per-question oracle (upper bound incl. generation noise).

Usage (from experiments/): python budget_predictor.py check | feats | train
`feats` uses MLX (GPU); `train` uses torch (MPS) + sklearn — separate processes (sklearn + MLX segfaults at exit).
"""
import json
import sys

import numpy as np

import user_memory as um

BUDGETS = (1000, 2000, 4000, 8000)
LAYER = 18  # middle of Qwen3-4B's 36 layers, fixed a priori (no tuning on test)
HID = um.D5 / "qhid_l18.npz"
OUTF = um.ROOT / "results" / "exp5" / "budget_pred.json"
BERT = "answerdotai/ModernBERT-base"


def labels():
    """Test questions with all four budgets judged → (qs, Y correct (n, 4), T memory tokens (n, 4))."""
    res = {}
    for b in BUDGETS:
        keys = [f"{um.MAIN}@{b}"] if b < 8000 else [f"{um.MAIN}@8000~lme-test", f"{um.MAIN}@8000~loc-test"]
        for k in keys:
            for qid, v in json.loads((um.OUT / f"{k}.json").read_text()).items():
                res.setdefault(qid, {})[b] = (v["judge"], v["mem_tok"])
    qs = [q for q in json.loads(um.QS.read_text()) if q["split"] == "test" and len(res.get(q["qid"], {})) == len(BUDGETS)]
    Y = np.array([[res[q["qid"]][b][0] for b in BUDGETS] for q in qs], float)
    T = np.array([[res[q["qid"]][b][1] for b in BUDGETS] for q in qs], float)
    return qs, Y, T


def folds(groups, k=5):
    """Group k-fold: every group (user) lands in exactly one test fold; deterministic."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    of = {g: i % k for i, g in enumerate(uniq)}
    f = np.array([of[g] for g in groups])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def choose(P):
    return np.argmax(P, axis=1)  # first max = smallest budget on ties


def score(c, Y, T):
    i = np.arange(len(c))
    return Y[i, c], T[i, c]


def boot(a, b, n=10000):
    d = a - b
    bs = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]


def bench_best(Ytr, bench_tr, bench_te):
    """Per-benchmark best fixed budget on the training fold, applied to the test fold (as one-hot 'probabilities')."""
    P = np.zeros((len(bench_te), len(BUDGETS)))
    for b in set(bench_te):
        P[bench_te == b, Ytr[bench_tr == b].mean(0).argmax()] = 1
    return P


def probe(X, Y, tr, te, kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler().fit(X[tr])
    if kind == "logreg":
        return np.stack([LogisticRegression(C=0.1, max_iter=3000).fit(s.transform(X[tr]), Y[tr, j]).predict_proba(s.transform(X[te]))[:, 1]
                         for j in range(Y.shape[1])], 1)
    m = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=300, early_stopping=True, random_state=0)
    return m.fit(s.transform(X[tr]), Y[tr]).predict_proba(s.transform(X[te]))


def bert(texts, Y, tr, te, epochs=3, bs=16, lr=5e-5):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BERT)
    model = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=Y.shape[1],
                                                               problem_type="multi_label_classification").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    enc = lambda idx: {k: v.to(dev) for k, v in tok([texts[i] for i in idx], padding=True, truncation=True, max_length=128,
                                                    return_tensors="pt").items()}
    rng = np.random.default_rng(0)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            loss = model(**enc(idx), labels=torch.tensor(Y[idx], dtype=torch.float32, device=dev)).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(**enc(te[i:i + 64])).logits).float().cpu().numpy() for i in range(0, len(te), 64)])


def feats():
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(um.MODEL)
    qs, _, _ = labels()
    X = []
    for q in qs:
        h = model.model.embed_tokens(mx.array(tok.encode(q["question"], add_special_tokens=False))[None])
        for layer in model.model.layers[:LAYER]:
            h = layer(h, "causal" if h.shape[1] > 1 else None)
        X.append(np.array(h[0].astype(mx.float32).mean(0)))
    um.D5.mkdir(exist_ok=True)
    np.savez(HID, X=np.stack(X), qids=np.array([q["qid"] for q in qs]))
    print(f"feats: {len(X)} questions, dim {X[0].shape[0]} -> {HID}", flush=True)


def train():
    qs, Y, T = labels()
    hid = np.load(HID)
    assert list(hid["qids"]) == [q["qid"] for q in qs], "features out of date: rerun feats"
    X, texts = hid["X"], [q["question"] for q in qs]
    bench, groups = np.array([q["bench"] for q in qs]), [q["user"] for q in qs]
    P = {m: np.zeros_like(Y) for m in ("bench-best", "probe-logreg", "probe-mlp", "modernbert")}
    for n, (tr, te) in enumerate(folds(groups)):
        assert not {groups[i] for i in tr} & {groups[i] for i in te}
        P["bench-best"][te] = bench_best(Y[tr], bench[tr], bench[te])
        P["probe-logreg"][te] = probe(X, Y, tr, te, "logreg")
        P["probe-mlp"][te] = probe(X, Y, tr, te, "mlp")
        P["modernbert"][te] = bert(texts, Y, tr, te)
        print(f"  fold {n + 1}/5 done", flush=True)
    rows = {f"fixed {b // 1000}K": np.full(len(Y), j) for j, b in enumerate(BUDGETS)}
    rows |= {m: choose(p) for m, p in P.items()}
    rows["oracle"] = choose(Y)
    ref2k, refbb = score(rows["fixed 2K"], Y, T)[0], score(rows["bench-best"], Y, T)[0]
    out = {}
    for name, c in rows.items():
        acc, tok = score(c, Y, T)
        r = {}
        for b in ("lme", "locomo", "all"):
            k = bench == b if b != "all" else np.ones(len(Y), bool)
            r[b] = {"acc": float(acc[k].mean()), "mean_tok": float(tok[k].mean()), "vs_fixed2K": boot(acc[k], ref2k[k]),
                    "vs_bench_best": boot(acc[k], refbb[k]), "budget_share": np.bincount(c[k], minlength=4).tolist()}
        out[name] = r
        print(f"{name:13} " + "  ".join(f"{b} {r[b]['acc']:.1%} @{r[b]['mean_tok']:.0f} "
                                       f"(vs 2K {r[b]['vs_fixed2K'][0] * 100:+.1f} [{r[b]['vs_fixed2K'][1] * 100:+.1f},{r[b]['vs_fixed2K'][2] * 100:+.1f}])"
                                       for b in ("lme", "locomo")), flush=True)
    OUTF.write_text(json.dumps(out, indent=1))
    np.savez(OUTF.with_name("budget_pred_probs.npz"), Y=Y, T=T, qids=np.array([q["qid"] for q in qs]), bench=bench, **P)  # for post-hoc calibration

def stop_rule(P, lam):
    """Smallest budget whose predicted P(correct) >= lam; the largest budget (8K) if none (lam > 1 -> always 8K)."""
    ok = P >= lam
    return np.where(ok.any(1), ok.argmax(1), P.shape[1] - 1)


def ltt(P, Y, alpha, eps, grid):
    """Learn then Test, fixed-sequence (Wu et al. 2025 / Angelopoulos et al. 2021): risk = lost an answer the max budget
    gets right; walk the grid from safest (high lam) down, reject H_j: risk > alpha while the binomial-tail p-value
    P(Binom(n, alpha) <= n * risk_hat) <= eps; return the last rejected lam (> 1 = never stop early)."""
    from scipy.stats import binom
    best = 1.01
    for lam in grid:
        c = stop_rule(P, lam)
        risk = (Y[:, -1] * (1 - Y[np.arange(len(Y)), c])).sum()
        if binom.cdf(risk, len(Y), alpha) > eps:
            break
        best = lam
    return best


def calibrate(alphas=(0.02, 0.05, 0.10), eps=0.1, splits=200, method="probe-mlp"):
    """Post-hoc on the saved out-of-fold probabilities: half the users calibrate lam, the other half test; repeated."""
    z = np.load(OUTF.with_name("budget_pred_probs.npz"))
    P, Y, T, bench = z[method], z["Y"], z["T"], z["bench"]
    user = {q["qid"]: q["user"] for q in json.loads(um.QS.read_text())}
    groups = np.array([user[q] for q in z["qids"]])
    grid = np.round(np.arange(1.0, -0.001, -0.01), 2)
    rng, out = np.random.default_rng(0), {}
    for a in alphas:
        rows = []
        for _ in range(splits):
            cal_users = set(rng.permutation(sorted(set(groups)))[:len(set(groups)) // 2])
            cal = np.array([g in cal_users for g in groups])
            lam = ltt(P[cal], Y[cal], a, eps, grid)
            c = stop_rule(P[~cal], lam)
            acc, tok = score(c, Y[~cal], T[~cal])
            risk = (Y[~cal][:, -1] * (1 - acc)).mean()
            r = {"lam": lam, "risk": risk, "acc": acc.mean(), "tok": tok.mean()}
            for b in ("lme", "locomo"):
                k = bench[~cal] == b
                r[f"{b}_acc"], r[f"{b}_tok"] = acc[k].mean(), tok[k].mean()
                r[f"{b}_8k"] = Y[~cal][k, -1].mean()
            rows.append(r)
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        m["risk_violations"] = float(np.mean([r["risk"] > a for r in rows]))  # should be <= eps
        out[f"alpha={a}"] = m
        print(f"alpha {a:.2f}: lam {m['lam']:.2f}  test risk {m['risk']:.3f} (violated in {m['risk_violations']:.0%} of splits)  "
              f"LME {m['lme_acc']:.1%} @{m['lme_tok']:.0f}  LoCoMo {m['locomo_acc']:.1%} @{m['locomo_tok']:.0f}  "
              f"(8K: {m['lme_8k']:.1%} / {m['locomo_8k']:.1%} @~8000)", flush=True)
    (OUTF.with_name("budget_calib.json")).write_text(json.dumps({"method": method, "eps": eps, "splits": splits, **out}, indent=1))


def check():
    Y = np.array([[0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1.]])
    T = np.array([[1, 2, 4, 8]] * 3, float) * 1000
    assert choose(Y).tolist() == [1, 0, 3]  # ties go to the smaller budget
    acc, tok = score(choose(Y), Y, T)
    assert acc.tolist() == [1, 1, 1] and tok.tolist() == [2000, 1000, 8000]
    g = [f"u{i % 7}" for i in range(50)]
    fs = folds(g)
    assert sorted(np.concatenate([te for _, te in fs]).tolist()) == list(range(50))
    assert all(not {g[i] for i in tr} & {g[i] for i in te} for tr, te in fs)
    P = bench_best(np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.]]), np.array(["a", "a", "b"]), np.array(["b", "a"]))
    assert P.argmax(1).tolist() == [2, 1]
    assert boot(np.ones(10), np.zeros(10))[0] == 1.0
    P2 = np.array([[0.9, 0.95, 0.5, 0.5], [0.2, 0.3, 0.4, 0.5]])
    assert stop_rule(P2, 0.85).tolist() == [0, 3] and stop_rule(P2, 1.01).tolist() == [3, 3]
    Yc = np.array([[1, 1, 1, 1]] * 50 + [[0, 1, 1, 1]] * 50, float)  # half the questions are lost at 1K
    Pc = np.array([[0.9, 0.9, 0.9, 0.9]] * 100)
    assert ltt(Pc, Yc, 0.1, 0.1, [1.0, 0.95, 0.85]) == 0.95  # stopping at 1K (lam 0.85) loses 50% > alpha → rejected
    print("checks pass")


if __name__ == "__main__":
    {"check": check, "feats": feats, "train": train, "calib": calibrate}[sys.argv[1]]()
