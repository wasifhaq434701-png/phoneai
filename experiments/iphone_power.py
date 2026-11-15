"""Record an iPhone's whole-device power from the Mac, through Apple's diagnostics service (pymobiledevice3).

The phone's battery gauge publishes PowerTelemetryData every ~20 s: SystemLoad (mW, the phone's own consumption =
SystemPowerIn from the charger - BatteryPower into the battery) and AccumulatedSystemLoad, a running sum of 1 Hz samples
(mW*s = mJ) with its sample counter. Energy between two updates = delta AccumulatedSystemLoad (mJ); mean power = that /
delta count. Works with the phone plugged in over USB (no battery window needed). Undocumented IORegistry keys: validated
only by the in = battery + load identity and an idle-vs-load test (see Result notes).

Needs the venv in data/exp5/iphone_energy/venv (pymobiledevice3 11.49.1):
    data/exp5/iphone_energy/venv/bin/python -I experiments/iphone_power.py <out.jsonl> <seconds>
    python3 experiments/iphone_power.py blocks <bench.json> <power.jsonl> <idle_W>   (any Python, no device)
    python3 experiments/iphone_power.py grid <grid.json> <power.jsonl> <idle_W> [out.json]   (energy grid → Eq. 4 fit)
    python3 experiments/iphone_power.py check                                            (self-check on synthetic data)
Each line: Mac wall time of the read + the phone's UpdateTime and the telemetry fields; only changed updates are written.
"""
import asyncio
import datetime as dt
import json
import sys
import time

KEYS = ["SystemLoad", "AccumulatedSystemLoad", "SystemLoadAccumulatorCount", "SystemPowerIn", "BatteryPower"]


async def record(path, seconds):
    from pymobiledevice3.lockdown import create_using_usbmux
    from pymobiledevice3.services.diagnostics import DiagnosticsService
    diag = DiagnosticsService(await create_using_usbmux())
    last, t_end = None, time.time() + seconds
    with open(path, "a") as f:
        while time.time() < t_end:
            t0 = time.time()
            b = await diag.get_battery()
            t = b["PowerTelemetryData"]
            row = {"mac_t": round(t0, 3), "update_t": b.get("UpdateTime"), **{k: t[k] for k in KEYS},
                   "instant_mA": b["InstantAmperage"], "voltage_mV": b["Voltage"], "charging": b["IsCharging"],
                   "level": b["CurrentCapacity"]}
            key = (row["update_t"], row["SystemLoadAccumulatorCount"])
            if key != last:
                last = key
                f.write(json.dumps(row) + "\n")
                f.flush()
            await asyncio.sleep(0.5)


def windows(rows):
    """(start, end, mean W) per telemetry update, on the phone's clock (= the bench app's started_at clock)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        n = b["SystemLoadAccumulatorCount"] - a["SystemLoadAccumulatorCount"]
        if n > 0:
            out.append((b["update_t"] - n, b["update_t"], (b["AccumulatedSystemLoad"] - a["AccumulatedSystemLoad"]) / n / 1000))
    return out


def blocks(bench_path, power_path, idle_w):
    """Energy per call for each contiguous block of one budget (smallest-first runs) and the sustained loop:
    mean power over the telemetry windows that fall entirely inside the block x block duration / calls."""
    ts = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    d = json.load(open(bench_path))
    w = windows([json.loads(l) for l in open(power_path)])
    starts = {}
    for c in d["calls"]:
        if c["phase"] == "item":
            starts.setdefault(c["budget"], []).append(ts(c["started_at"]))
    sus = [ts(x["started_at"]) for x in d["sustained"]["iterations"]]
    order = sorted(starts)
    spans = [(b, min(starts[b]), min(starts[order[i + 1]]) if i + 1 < len(order) else (sus[0] if sus else w[-1][1]))
             for i, b in enumerate(order)]
    if sus:
        spans.append(("sustained", sus[0], min(w[-1][1], sus[-1])))
    out = {}
    for b, s, e in spans:
        inside = [x for x in w if x[0] >= s and x[1] <= e]
        n = len(starts[b]) if b != "sustained" else sum(s <= t < e for t in sus)
        if not inside or not n:
            continue
        p = sum((x[1] - x[0]) * x[2] for x in inside) / sum(x[1] - x[0] for x in inside)
        out[str(b)] = {"calls": n, "block_s": e - s, "covered_s": sum(x[1] - x[0] for x in inside), "mean_W": round(p, 2),
                       "J_per_call": round(p * (e - s) / n, 1), "J_per_call_above_idle": round((p - idle_w) * (e - s) / n, 1)}
        print(b, out[str(b)])
    return out


def cell_energy(cells, w, idle_w):
    """Per grid cell: mean power over the telemetry windows inside it; J per call above idle. Cells less than half
    covered by whole windows are dropped (a 100 s cell normally holds 4-5 of the 20 s windows)."""
    out = []
    for c in cells:
        dur = c["end_t"] - c["start_t"]
        inside = [x for x in w if x[0] >= c["start_t"] and x[1] <= c["end_t"]]
        cov = sum(x[1] - x[0] for x in inside)
        if cov < dur / 2:
            continue
        p = sum((x[1] - x[0]) * x[2] for x in inside) / cov
        out.append({**{k: c[k] for k in ("pass", "budget", "prompt_tok", "n_new", "reps")}, "mean_W": p,
                    "J": (p - idle_w) * dur / c["reps"]})
    return out


def fit(rows):
    """Fortuna et al. on the per-(budget, n_new) medians, least squares. Eq. 4: E = e0 + c_pre*p_in + c_dec*p_out.
    Eq. 6 adds decode's KV reads: + c_kv*p_in*p_out (per-token decode cost grows with the context)."""
    import numpy as np
    med = {}
    for r in rows:
        med.setdefault((r["prompt_tok"], r["n_new"]), []).append(r["J"])
    keys = sorted(med)
    y = np.array([float(np.median(med[k])) for k in keys])
    out = {"cells": len(keys)}
    for name, X in (("eq4", [[1, p, n] for p, n in keys]), ("eq6", [[1, p, n, p * n] for p, n in keys])):
        X = np.array(X, float)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ c) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[name] = dict(zip(["e0_J", "c_pre_J", "c_dec_J", "c_kv_J"], map(float, c)), r2=float(r2))
    return out


def per_query(f6, mac4, lengths):
    """Energy per query at the reader's real mean prompt/answer lengths: phone (Eq. 6 fit) vs Mac (Eq. 4, Result 14)."""
    out = {}
    for b, r in lengths.items():
        pre = f6["c_pre_J"] * r["p_in"]
        dec = (f6["c_dec_J"] + f6["c_kv_J"] * r["p_in"]) * r["p_out"]
        ph = f6["e0_J"] + pre + dec
        out[b] = {"p_in": r["p_in"], "p_out": r["p_out"], "acc": r["acc"], "phone_J": ph, "phone_J_per_correct": ph / r["acc"],
                  "phone_prefill_share": pre / (pre + dec),
                  "mac_J": mac4["e0"] + mac4["c_pre"] * r["p_in"] + mac4["c_dec"] * r["p_out"]}
    return out


def grid(grid_path, power_path, idle_w, out=None, energy_json="results/exp5/energy.json"):
    g = json.load(open(grid_path))
    rows = cell_energy(g["cells"], windows([json.loads(l) for l in open(power_path)]), idle_w)
    f = fit(rows)
    print(f"{len(rows)}/{len(g['cells'])} cells usable, idle {idle_w:.2f} W")
    for k in ("eq4", "eq6"):
        print(k, {a: round(v, 6) for a, v in f[k].items()})
    e = json.load(open(energy_json))
    lengths = {b: e["per_correct"][f"hybrid,div=0,edges=0@{b}"]["lme"] for b in (1000, 2000, 4000, 8000)}
    pq = per_query(f["eq6"], e["J_per_call"]["eq4"], lengths)
    for b, r in pq.items():
        print(f"{b}: phone {r['phone_J']:.1f} J/query, {r['phone_J_per_correct']:.1f} J/correct, prefill "
              f"{r['phone_prefill_share']:.0%}; Mac SoC {r['mac_J']:.1f} J/query")
    if out:
        json.dump({"idle_W": idle_w, "fit": f, "per_query_lme": pq, "cells": rows, "grid": grid_path, "power": power_path},
                  open(out, "w"), indent=1)
    return f


def check():
    # synthetic phone: 0.5 W idle; a call costs 2 + 0.015*p_in + 0.2*p_out J; telemetry every 20 s with 1 Hz samples
    true = (2.0, 0.015, 0.2)
    cells, t, load = [], 1_000_000.0, {}
    for i, (p, n) in enumerate([(p, n) for p in (1100, 2150, 4170, 8200) for n in (16, 64, 256)]):
        reps, dur = 10, 120.0
        cells.append({"pass": 1, "budget": p, "prompt_tok": p, "n_new": n, "reps": reps, "start_t": t, "end_t": t + dur})
        watts = 0.5 + (true[0] + true[1] * p + true[2] * n) * reps / dur
        for s in range(int(t), int(t + dur)):
            load[s] = watts
        t += dur
    rows, acc, cnt = [], 0.0, 0
    for s in range(1_000_000, int(t) + 1):
        if s % 20 == 0:
            rows.append({"update_t": s, "AccumulatedSystemLoad": acc, "SystemLoadAccumulatorCount": cnt})
        acc += load.get(s, 0.5) * 1000
        cnt += 1
    f = fit(cell_energy(cells, windows(rows), 0.5))["eq4"]
    assert abs(f["e0_J"] - true[0]) < 0.5 and abs(f["c_pre_J"] - true[1]) < 2e-4 and abs(f["c_dec_J"] - true[2]) < 5e-3, f
    print("check OK", {k: round(v, 5) for k, v in f.items()})


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    elif sys.argv[1] == "grid":
        grid(sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else None)
    elif sys.argv[1] == "blocks":
        blocks(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    else:
        asyncio.run(record(sys.argv[1], float(sys.argv[2])))
