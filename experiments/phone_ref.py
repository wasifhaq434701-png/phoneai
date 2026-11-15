"""Phase 0 phone benchmark: Mac reference and phone-vs-Mac comparison.

  ref                 run the 80 items of prompts/prompts.json on the Mac exactly as the phone app does (feed `ids`,
                      greedy, fresh KV per call, EOS ignored, 128 new tokens, prefill step 2048) → results/exp5/phone/mac_ref.json
  judge <phone.json…>  strict-judge the 2K answers of the Mac reference and the phone runs (same questions).
  compare <phone.json> per call: position of the first token where the phone's greedy output differs from the Mac's,
                      and phone/Mac speed ratios.
"""
import json
import sys
import time

import numpy as np

import user_memory as um

PROMPTS = um.ROOT / "prompts" / "prompts.json"
REF = um.ROOT / "results" / "exp5" / "phone" / "mac_ref.json"
N_NEW = 128


def ref():
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    meta = json.loads(PROMPTS.read_text())
    model, _ = load(meta["model"])

    def call(ids):
        t0 = time.perf_counter()
        out, ts = [], []
        for tok, _ in generate_step(mx.array(ids), model, max_tokens=N_NEW, prefill_step_size=2048):  # greedy, EOS ignored
            out.append(int(tok))
            ts.append(time.perf_counter())
        return {"ttft_s": ts[0] - t0, "prefill_tok_s": len(ids) / (ts[0] - t0),
                "decode_tok_s": (N_NEW - 1) / (ts[-1] - ts[0]), "out_ids": out}

    call(meta["items"][0]["ids"])  # warm-up, discarded (same as the phone)
    calls = []
    for it in sorted(meta["items"], key=lambda i: i["budget"]):
        r = call(it["ids"])
        calls.append({"qid": it["qid"], "budget": it["budget"], "prompt_tok": it["prompt_tok"], **r})
        print(f"{it['budget']:5d} {it['qid']} ttft {r['ttft_s']:.2f}s decode {r['decode_tok_s']:.1f} tok/s", flush=True)
    REF.parent.mkdir(parents=True, exist_ok=True)
    REF.write_text(json.dumps({"model": meta["model"], "mlx_lm": mlx_lm.__version__, "mlx": mx.__version__,
                               "prefill_step": 2048, "new_tokens": N_NEW, "calls": calls}))
    print("wrote", REF)


def first_diff(a, b):
    """Index of the first differing token, or None if the common prefix is identical and lengths match."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(path):
    mac = {(c["qid"], c["budget"]): c for c in json.loads(REF.read_text())["calls"]}
    phone = json.loads(open(path).read())
    rows = [(c, mac[(c["qid"], c["budget"])]) for c in phone["calls"] if c["phase"] == "item"]
    print(f"{phone['device']['machine']} {phone['model']['id']} kv_bits={phone['runtime'].get('kv_bits')}: {len(rows)} calls")
    for b in sorted({c["budget"] for c, _ in rows}):
        sub = [(c, m) for c, m in rows if c["budget"] == b]
        d = [first_diff(c["out_ids"], m["out_ids"]) for c, m in sub]
        same = sum(x is None for x in d)
        r = lambda k: np.median([c[k] / m[k] for c, m in sub])
        print(f"  {b:5d}: identical {same}/{len(sub)}, first diff at {sorted(x for x in d if x is not None)}; "
              f"phone/Mac prefill {r('prefill_tok_s'):.3f}, decode {r('decode_tok_s'):.3f}")


def answer_text():
    """ids → answer text cut at the first EOS (the bench apps ignore EOS unless natural answers are on)."""
    from mlx_lm import load
    _, tok = load(json.loads(PROMPTS.read_text())["model"])
    eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}
    return lambda ids: tok.decode(ids[:next((i for i, t in enumerate(ids) if t in eos), len(ids))])


def merge_items(parts):
    """Item calls of one run split over several files (Start at question N): first measurement of each call wins."""
    calls = {}
    for p in parts:
        for c in json.loads(open(p).read())["calls"]:
            if c["phase"] == "item":
                calls.setdefault((c["qid"], c["budget"]), c)
    return calls


def judge_all(arg):
    from mlx_lm import load
    import task_adapters as ta
    parts = arg.split("+")
    calls = merge_items(parts)
    qs = {q["qid"]: q for q in json.loads(um.QS.read_text())}
    text = answer_text()
    keys = sorted(calls)
    ans = [text(calls[k]["out_ids"]) for k in keys]
    model, jtok = load(um.JUDGE)
    res = ta.gen(model, jtok, [um.judge_prompt(qs[q], a) for (q, _), a in zip(keys, ans)], [8] * len(keys),
                 "phone_judge_all_" + parts[0].split("/")[-1].removesuffix(".json"))
    out = {"parts": [p.split("/")[-1] for p in parts], "budgets": {}}
    for b in sorted({b for _, b in keys}):
        mac = json.loads((um.OUT / f"{um.MAIN}@{b}~lme-dev.json").read_text())
        rows = {q: {"out": a, "judge": int("yes" in r.lower()), "mac_judge": mac[q]["judge"],
                    "n_tokens": len(calls[(q, bb)]["out_ids"])}
                for (q, bb), a, r in zip(keys, ans, res) if bb == b}
        ph, mc = [v["judge"] for v in rows.values()], [v["mac_judge"] for v in rows.values()]
        out["budgets"][b] = {"n": len(rows), "phone": float(np.mean(ph)), "mac": float(np.mean(mc)),
                             "only_phone": sum(p and not m for p, m in zip(ph, mc)),
                             "only_mac": sum(m and not p for p, m in zip(ph, mc)), "answers": rows}
        print(f"{b}: phone {np.mean(ph):.0%} vs Mac {np.mean(mc):.0%} of {len(rows)}; only phone "
              f"{out['budgets'][b]['only_phone']}, only Mac {out['budgets'][b]['only_mac']}", flush=True)
    path = parts[0].removesuffix(".json") + "-judge.json"
    open(path, "w").write(json.dumps(out, indent=1))
    print("→", path)


def judge(paths):
    """Strict-judge the 2K answers (128-token cap, cut at the first EOS) of the Mac reference and each phone run,
    on the same questions → results/exp5/phone/judge2k.json. Also the original uncapped Mac run (MAIN@2000) for context."""
    from mlx_lm import load
    import task_adapters as ta
    qs = {q["qid"]: q for q in json.loads(um.QS.read_text())}
    text = answer_text()

    runs = {"mac_ref": json.loads(REF.read_text())["calls"]}
    for p in paths:
        runs[p.split("/")[-1].removesuffix(".json")] = [c for c in json.loads(open(p).read())["calls"] if c["phase"] == "item"]
    answers = {name: {c["qid"]: text(c["out_ids"]) for c in calls if c["budget"] == 2000} for name, calls in runs.items()}
    full = json.loads((um.OUT / f"{um.MAIN}@2000~lme-dev.json").read_text())
    answers["mac_main_uncapped"] = {k: full[k]["out"] for k in answers["mac_ref"]}
    model, jtok = load(um.JUDGE)
    out = {}
    for name, ans in answers.items():
        keys = sorted(ans)
        res = ta.gen(model, jtok, [um.judge_prompt(qs[k], ans[k]) for k in keys], [8] * len(keys), f"phone_judge_{name}")
        out[name] = {k: {"out": ans[k], "judge": int("yes" in r.lower())} for k, r in zip(keys, res)}
        print(f"{name}: {np.mean([v['judge'] for v in out[name].values()]):.0%} of {len(keys)}", flush=True)
    (REF.parent / "judge2k.json").write_text(json.dumps(out, indent=1))


def judge_apple():
    """Strict-judge Apple's on-device model on the same 2K questions (results/exp5/phone/apple_fm_2k.json, written by
    experiments/apple_fm.swift) and add it to judge2k.json as "apple_fm", next to Qwen3-4B's "mac_main_uncapped"."""
    from mlx_lm import load
    import task_adapters as ta
    qs = {q["qid"]: q for q in json.loads(um.QS.read_text())}
    ans = {k: v["out"] for k, v in json.loads((REF.parent / "apple_fm_2k.json").read_text())["answers"].items()}
    model, jtok = load(um.JUDGE)
    keys = sorted(ans)
    res = ta.gen(model, jtok, [um.judge_prompt(qs[k], ans[k]) for k in keys], [8] * len(keys), "phone_judge_apple_fm")
    path = REF.parent / "judge2k.json"
    out = json.loads(path.read_text())
    out["apple_fm"] = {k: {"out": ans[k], "judge": int("yes" in r.lower())} for k, r in zip(keys, res)}
    path.write_text(json.dumps(out, indent=1))
    a, q = out["apple_fm"], out["mac_main_uncapped"]
    print(f"apple_fm {np.mean([a[k]['judge'] for k in keys]):.0%} vs Qwen3-4B (512 tok) {np.mean([q[k]['judge'] for k in keys]):.0%} "
          f"on {len(keys)}; both right {sum(a[k]['judge'] and q[k]['judge'] for k in keys)}, only Apple "
          f"{sum(a[k]['judge'] and not q[k]['judge'] for k in keys)}, only Qwen {sum(q[k]['judge'] and not a[k]['judge'] for k in keys)}")


def check():
    assert first_diff([1, 2, 3], [1, 2, 3]) is None
    assert first_diff([1, 2, 3], [1, 9, 3]) == 1
    assert first_diff([1, 2], [1, 2, 3]) == 2
    import tempfile
    with tempfile.TemporaryDirectory() as d:  # a stopped run's two parts: the call measured twice keeps the first
        a, b = f"{d}/a.json", f"{d}/b.json"
        open(a, "w").write(json.dumps({"calls": [{"phase": "warmup", "qid": "q1", "budget": 1000},
                                                {"phase": "item", "qid": "q1", "budget": 1000, "out_ids": [1]}]}))
        open(b, "w").write(json.dumps({"calls": [{"phase": "item", "qid": "q1", "budget": 1000, "out_ids": [2]},
                                                {"phase": "item", "qid": "q2", "budget": 1000, "out_ids": [3]}]}))
        m = merge_items([a, b])
        assert sorted(m) == [("q1", 1000), ("q2", 1000)] and m[("q1", 1000)]["out_ids"] == [1]
    print("ok")


if __name__ == "__main__":
    {"ref": ref, "compare": lambda: compare(sys.argv[2]), "judge": lambda: judge(sys.argv[2:]),
     "judge_all": lambda: judge_all(sys.argv[2]),
     "judge_apple": judge_apple, "check": check}[sys.argv[1]]()
