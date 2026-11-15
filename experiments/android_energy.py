"""Energy per memory query on an Android phone, from the bench app's energy grid (battery current x voltage, whole phone).

    python3 experiments/android_energy.py <grid.json> <idle_W | idle.json> [out.json]

idle.json = the app's "Idle baseline" results file (mean W over its window).

Per query at the reader's real LongMemEval lengths (results/exp5/energy.json): prefill = median over passes of the
measured prefill energy of the budget's 64-token cells (same prompt, so prefill doesn't depend on n_new), plus decode =
median J per token of the budget's 256-token cells x the real answer length. Idle power x time is subtracted (idle_W = 0
gives whole-phone energy). Measured cells are used instead of the linear Fortuna fit because on llama.cpp's CPU backend
prefill slows with the prompt (55 -> 25 tok/s from 1K to 8K on the S25 Ultra), which the fit's constant per-token cost
can't follow (negative intercept); the Eq. 4/6 fits are saved alongside for comparison with the iPhone and the Mac.
"""
import json
import statistics as st
import sys

from iphone_power import fit

BUDGETS = (1000, 2000, 4000, 8000)


def per_query(calls, idle_w, lengths):
    def med(b, n, f):
        return st.median(f(x["energy"]) for x in calls if x["budget"] == b and x["n_new"] == n)
    out = {}
    for b in BUDGETS:
        pre = med(b, 64, lambda e: e["prefill_j"] - idle_w * e["prefill_s"])
        pre_s = med(b, 64, lambda e: e["prefill_s"])
        dec_tok = med(b, 256, lambda e: (e["decode_j"] - idle_w * e["decode_s"]) / 256)
        dec_tok_s = med(b, 256, lambda e: e["decode_s"] / 256)
        r = lengths[b]
        j = pre + dec_tok * r["p_out"]
        out[b] = {"p_in": r["p_in"], "p_out": r["p_out"], "acc": r["acc"], "prefill_J": pre, "decode_J_per_token": dec_tok,
                  "query_J": j, "J_per_correct": j / r["acc"], "prefill_share": pre / j, "query_s": pre_s + dec_tok_s * r["p_out"]}
    return out


def main(grid_path, idle_w, out=None, energy_json="results/exp5/energy.json"):
    g = json.load(open(grid_path))
    calls = [x for x in g["calls"] if x["phase"] == "grid"]
    assert g["status"] == "complete" and g["energy"]["energy_run"], "not a finished energy run"
    passes = sorted({x["pass"] for x in calls})
    assert all(sum(x["pass"] == p for x in calls) == 3 * len(BUDGETS) for p in passes), "incomplete pass"
    e = json.load(open(energy_json))
    lengths = {b: e["per_correct"][f"hybrid,div=0,edges=0@{b}"]["lme"] for b in BUDGETS}
    pq = per_query(calls, idle_w, lengths)
    f = fit([{"prompt_tok": x["prompt_tok"], "n_new": x["n_new"],
              "J": x["energy"]["prefill_j"] + x["energy"]["decode_j"] - idle_w * (x["energy"]["prefill_s"] + x["energy"]["decode_s"])}
             for x in calls])
    print(f"{len(calls)} calls, passes {passes}, idle {idle_w:.2f} W")
    for b, r in pq.items():
        print(f"{b}: {r['query_J']:.0f} J/query ({r['query_s']:.0f} s), {r['J_per_correct']:.0f} J/correct, "
              f"prefill {r['prefill_share']:.0%}, decode {r['decode_J_per_token']:.2f} J/token")
    for k in ("eq4", "eq6"):
        print(k, {a: round(v, 6) for a, v in f[k].items()})
    if out:
        json.dump({"idle_W": idle_w, "per_query_lme": pq, "fit": f, "grid": grid_path}, open(out, "w"), indent=1)


def idle_watts(arg):
    if not arg.endswith(".json"):
        return float(arg)
    d = json.load(open(arg))
    e = d["idle"]["energy"]
    assert d["status"] == "complete" and "stopped" not in d and e["decode_s"] >= 0.9 * d["idle"]["target_s"], "idle run incomplete"
    return e["decode_j"] / e["decode_s"]


def prefill_only(paths, idle_w, out=None):
    """Prefill energy and time per budget, median over every call of the given grid files (for a phone whose decode is
    not usable, e.g. the Pixel 8's collapse to ~0.35 tok/s). Each budget's prompt is the same in every cell."""
    calls = [x for p in paths for x in json.load(open(p))["calls"] if x["phase"] == "grid"]
    res = {}
    for b in BUDGETS:
        cs = [x for x in calls if x["budget"] == b]
        res[b] = {"n": len(cs), "prompt_tok": cs[0]["prompt_tok"],
                  "prefill_tok_s": st.median(x["prefill_tok_s"] for x in cs),
                  "ttft_s": st.median(x["ttft_s"] for x in cs),
                  "prefill_J": st.median(x["energy"]["prefill_j"] - idle_w * x["energy"]["prefill_s"] for x in cs),
                  "decode_tok_s_median": st.median(x["decode_tok_s"] for x in cs)}
        print(b, {k: round(v, 2) for k, v in res[b].items()})
    if out:
        json.dump({"idle_W": idle_w, "prefill": res, "grids": paths}, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "prefill":  # prefill <grid.json,grid.json,...> <idle_W | idle.json> [out.json]
        prefill_only(sys.argv[2].split(","), idle_watts(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        main(sys.argv[1], idle_watts(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
