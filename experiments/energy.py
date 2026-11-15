"""Exp 5.13 (Fortuna et al.'s energy model, arXiv 2609.18283, Eqs. 4-6): calibrate per-token energy of the Qwen3-4B reader on
the M4 Max and turn every retrieval budget into energy per correct answer.

  calib     run a grid of B=1 calls (prompt tokens p_in x forced output tokens p_out) and log their wall-clock windows.
            Needs `sudo powermetrics` sampling at the same time (see README, "Mac energy").
  fit <pm>  align the powermetrics log with the call log, integrate power per call, fit
            Eq. 4:  E = e0 + c_pre p_in + c_dec p_out            and
            Eq. 6:  E = e0 + c_pre p_in + (c_dec0 + c_kv c) p_out, c = p_in + p_out / 2 (mean context during decode),
            then apply the fit to every hybrid budget (real p_in, p_out per question) → energy per correct answer.
  check     self-test of parsing, alignment and integration on synthetic data (CPU).
powermetrics reports CPU + GPU + ANE power only (no DRAM/display): energies are SoC compute energy, a lower bound.
"""
import json
import re
import sys
import time
from datetime import datetime

import numpy as np

import user_memory as um

P_IN = (250, 1000, 2000, 4000, 8000)  # retrieval budgets → prompt ≈ budget + ~140 tokens of header/question/template
P_OUT = (16, 64, 256)
WINDOW, GAP, IDLE = 10.0, 3.0, 15.0  # s: min busy window per cell, idle gap between cells, idle at start/end
CALLS = um.ROOT / "results" / "exp5" / "energy_calls.json"
OUTF = um.ROOT / "results" / "exp5" / "energy.json"
WEIGHT_BYTES, B_DATASHEET = 2_263_022_529, 410e9  # Qwen3-4B-4bit safetensors; M4 Max 14C/32C GPU memory bandwidth


def calib(out=CALLS, reps=3):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    model, tok = load(um.MODEL)
    pools = {b: [rq for _, rq, _, _ in um.contexts(um.MAIN, b, "lme-dev")] for b in P_IN}  # 100 distinct real prompts each
    use = {b: 0 for b in P_IN}

    def call(b, n_out):
        text = pools[b][use[b] % len(pools[b])]
        use[b] += 1  # a fresh prompt every call, fresh KV cache (generate_step makes one): no prefix-cache reuse
        ids = mx.array(tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                          enable_thinking=False, tokenize=False), add_special_tokens=False))
        t0 = time.time()
        for _ in generate_step(ids, model, max_tokens=n_out):  # EOS ignored: exactly n_out tokens
            pass
        mx.eval(mx.array(0))
        return int(ids.size), t0, time.time()

    for b in P_IN:  # warm-up (not logged)
        call(b, 16)
    log = {"idle": [], "cells": [], "start": time.time()}
    t = time.time()
    time.sleep(IDLE)
    log["idle"].append([t, time.time()])
    rng = np.random.default_rng(0)
    grid = [(b, o) for b in P_IN for o in P_OUT]
    for rep in range(reps):
        for i in rng.permutation(len(grid)):
            b, o = grid[i]
            calls = []
            while not calls or (calls[-1][2] - calls[0][1] < WINDOW and len(calls) < 50) or len(calls) < 2:
                calls.append(call(b, o))
            log["cells"].append({"budget": b, "p_out": o, "rep": rep, "calls": calls})
            print(f"  cell {b}/{o} rep {rep}: {len(calls)} calls, mean p_in {np.mean([c[0] for c in calls]):.0f}, "
                  f"{(calls[-1][2] - calls[0][1]) / len(calls):.2f} s/call", flush=True)
            mx.clear_cache()
            time.sleep(GAP)
    t = time.time()
    time.sleep(IDLE)
    log["idle"].append([t, time.time()])
    out.write_text(json.dumps(log))
    print(f"calib: {len(log['cells'])} cells -> {out}", flush=True)


HEAD = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\) \*\*\*")


def parse(text):
    """powermetrics text → (sample end times s, durations s, power W), using the header clock + summed elapsed times."""
    blocks = HEAD.split(text)[1:]
    t0 = None
    ends, durs, watts = [], [], []
    acc = 0.0
    for i in range(0, len(blocks), 3):
        stamp, ms, body = blocks[i], float(blocks[i + 1]), blocks[i + 2]
        m = re.search(r"Combined Power \(CPU \+ GPU \+ ANE\): (\d+) mW", body)
        if m is None:
            m_cpu, m_gpu = re.search(r"CPU Power: (\d+) mW", body), re.search(r"GPU Power: (\d+) mW", body)
            if not (m_cpu and m_gpu):
                continue
            w = (int(m_cpu[1]) + int(m_gpu[1])) / 1000
        else:
            w = int(m[1]) / 1000
        if t0 is None:
            t0 = datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y %z").timestamp()  # second resolution → refined by align()
        acc += ms / 1000
        ends.append(t0 + acc)
        durs.append(ms / 1000)
        watts.append(w)
    return np.array(ends), np.array(durs), np.array(watts)


def energy(ends, durs, watts, a, b):
    """Joules between wall times a and b: each sample's power times its overlap with [a, b]."""
    starts = ends - durs
    ov = np.clip(np.minimum(ends, b) - np.maximum(starts, a), 0, None)
    return float((watts * ov).sum())


def align(ends, durs, watts, windows, span=2.0, step=0.02):
    """Clock offset (s) maximising the correlation of power with 'a call is running' (the header clock has 1 s steps)."""
    mid = ends - durs / 2
    ds, rs = np.arange(-span, span + step / 2, step), []
    for d in ds:
        busy = np.zeros(len(mid))
        for a, b in windows:
            busy[(mid + d >= a) & (mid + d < b)] = 1
        rs.append(np.corrcoef(busy, watts)[0, 1] if busy.std() > 0 else -2.0)
    rs = np.array(rs)
    return float(ds[rs >= rs.max() - 1e-9].mean()), float(rs.max())  # centre of the best plateau (samples are 100 ms)


def lsq(A, y):
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return coef, float(r2)


def fit(pm_path, calls=CALLS):
    log = json.loads(calls.read_text())
    ends, durs, watts = parse(open(pm_path).read())
    windows = [(c[1], c[2]) for cell in log["cells"] for c in cell["calls"]]
    off, r = align(ends, durs, watts, windows)
    ends = ends + off
    idle_w = np.mean([energy(ends, durs, watts, a, b) / (b - a) for a, b in log["idle"]])
    rows = []
    for cell in log["cells"]:
        c = cell["calls"]
        a, b = c[0][1], c[-1][2]
        e = energy(ends, durs, watts, a, b) / len(c)
        rows.append({**{k: cell[k] for k in ("budget", "p_out", "rep")}, "p_in": float(np.mean([x[0] for x in c])),
                     "s_per_call": (b - a) / len(c), "J_per_call": e, "J_dyn_per_call": e - idle_w * (b - a) / len(c)})
    out_cells = rows
    rows = [{k: float(np.median([x[k] for x in rows if (x["budget"], x["p_out"]) == key])) for k in ("p_in", "p_out", "s_per_call", "J_per_call", "J_dyn_per_call")}
            for key in sorted({(x["budget"], x["p_out"]) for x in out_cells})]
    pin, pout = np.array([x["p_in"] for x in rows]), np.array([x["p_out"] for x in rows], float)
    ctx = pin + pout / 2
    out = {"align_offset_s": off, "align_corr": r, "idle_W": idle_w, "cells": out_cells, "cell_medians": rows}
    for y in ("J_per_call", "J_dyn_per_call", "s_per_call"):
        v = np.array([x[y] for x in rows])
        c4, r4 = lsq(np.stack([np.ones_like(pin), pin, pout], 1), v)
        c6, r6 = lsq(np.stack([np.ones_like(pin), pin, pout, ctx * pout], 1), v)
        out[y] = {"eq4": {"e0": c4[0], "c_pre": c4[1], "c_dec": c4[2], "r2": r4},
                  "eq6": {"e0": c6[0], "c_pre": c6[1], "c_dec0": c6[2], "c_kv": c6[3], "r2": r6}}
    t = out["s_per_call"]["eq6"]
    out["achieved"] = {"prefill_tok_s": 1 / t["c_pre"], "decode_tok_s_at_2k": 1 / (t["c_dec0"] + t["c_kv"] * 2100),
                       "bandwidth_GBs_at_2k": (WEIGHT_BYTES + um.KV_BYTES_PER_TOK * 2100) / (t["c_dec0"] + t["c_kv"] * 2100) / 1e9,
                       "bandwidth_share_of_datasheet": (WEIGHT_BYTES + um.KV_BYTES_PER_TOK * 2100) / (t["c_dec0"] + t["c_kv"] * 2100) / B_DATASHEET}
    e = out["J_per_call"]["eq6"]
    out["prefill_dominates_when_pin_over_pout_gt"] = (e["c_dec0"] + e["c_kv"] * 2100) / e["c_pre"]
    out["per_correct"] = apply(e)
    OUTF.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("align_offset_s", "align_corr", "idle_W", "achieved", "prefill_dominates_when_pin_over_pout_gt")}, indent=1))
    for y in ("J_per_call", "J_dyn_per_call", "s_per_call"):
        print(y, json.dumps(out[y]))
    for k, v in out["per_correct"].items():
        print(f"{k:40} " + "  ".join(f"{b}: {x['acc']:.1%} {x['J_query']:.1f} J/query {x['J_per_correct']:.1f} J/correct "
                                     f"(prefill {x['prefill_share']:.0%})" for b, x in v.items()))


def apply(e):
    """Energy per query and per correct answer for each hybrid budget, from real prompt and answer lengths."""
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from transformers import AutoTokenizer
    tok = TokenizerWrapper(AutoTokenizer.from_pretrained(um.MODEL))
    res = {}
    cfgs = [(f"{um.MAIN}@{b}", b, None) for b in (1000, 2000, 4000)] + [(f"{um.MAIN}@8000", 8000, None),
            (f"{um.MAIN},snap=2000,requery=1@8000", 8000, 2000), (f"{um.MAIN},quest=2000@8000", 8000, 2000)]
    for name, budget, kv in cfgs:
        for bench, split in (("lme", "lme-test"), ("locomo", "loc-test")):
            keys = [f"{name}"] if budget < 8000 else [f"{name}~{split}"]
            path = um.OUT / f"{keys[0]}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            items = {q["qid"]: t for q, t, _, _ in um.contexts(name.split("@")[0], budget, split)}
            qids = [k for k in r if k in items]
            pin = np.array([len(tok.apply_chat_template([{"role": "user", "content": items[k]}], add_generation_prompt=True,
                                                        enable_thinking=False)) for k in qids], float)
            pout = np.array([len(tok.encode(r[k]["out"], add_special_tokens=False)) + 1 for k in qids], float)
            ctx = (kv + 120 if kv else pin) + pout / 2  # SnapKV/Quest decode over ~2K kept KV + question window
            Eq = e["e0"] + e["c_pre"] * pin + (e["c_dec0"] + e["c_kv"] * ctx) * pout
            acc = float(np.mean([r[k]["judge"] for k in qids]))
            res.setdefault(name, {})[bench] = {"n": len(qids), "acc": acc, "p_in": float(pin.mean()), "p_out": float(pout.mean()),
                                               "J_query": float(Eq.mean()), "J_per_correct": float(Eq.mean() / acc),
                                               "prefill_share": float((e["c_pre"] * pin).sum() / Eq.sum())}
    return res


def check():
    hdr = "*** Sampled system activity (Tue Oct 20 21:30:00 2026 +0530) ({} elapsed) ***\n"
    text = "".join(hdr.format("100.00ms") + f"Combined Power (CPU + GPU + ANE): {w} mW\n\n" for w in [1000] * 20 + [11000] * 30 + [1000] * 20)
    ends, durs, watts = parse(text)
    assert len(ends) == 70 and np.isclose(durs.sum(), 7.0)
    t0 = ends[0] - durs[0]
    off, r = align(ends, durs, watts, [(t0 + 2.3, t0 + 5.3)])  # true busy window starts 0.3 s after the power step
    assert abs(off - 0.3) < 0.03 and r > 0.99, (off, r)
    assert np.isclose(energy(ends + off, durs, watts, t0 + 2.3, t0 + 5.3), 33.0)
    coef, r2 = lsq(np.stack([np.ones(4), [1, 2, 3, 4.]], 1), np.array([3, 5, 7, 9.]))
    assert np.allclose(coef, [1, 2]) and r2 > 0.999
    print("checks pass")


if __name__ == "__main__":
    {"calib": lambda: calib(*([um.ROOT / sys.argv[2]] if len(sys.argv) > 2 else [])), "fit": lambda: fit(sys.argv[2], *([um.ROOT / sys.argv[3]] if len(sys.argv) > 3 else [])),
     "check": check}[sys.argv[1]]()
