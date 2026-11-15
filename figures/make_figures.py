"""Paper figures. Numbers are the final test results recorded in Results 6, 9, 12, 13, 11 and 14 (strict judge)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4 (validated; low-contrast hues get direct labels)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555"})
B = [1, 2, 4, 8]

# Fig. 1: accuracy vs retrieved-context budget, per reader
lme = {"Qwen3-8B": [None, 62.3, 58.0, 59.0], "Qwen3-4B": [59.0, 59.5, 57.0, 54.0],
       "Gemma 3 4B": [None, 50.7, 50.0, 43.5], "Llama 3.2 3B": [None, 40.5, 35.5, 27.8]}
loc = {"Qwen3-4B": [52.6, 56.8, 58.7, 60.3], "Gemma 3 4B": [None, 55.7, None, 48.7], "Llama 3.2 3B": [None, 47.1, None, 43.4]}
order = ["Qwen3-4B", "Qwen3-8B", "Gemma 3 4B", "Llama 3.2 3B"]  # colour follows the reader in both panels
marker = dict(zip(order, "osD^"))
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
for ax, data, title in ((axes[0], lme, "LongMemEval-S (400)"), (axes[1], loc, "LoCoMo (1,540)")):
    for name, ys in data.items():
        pts = [(b, y) for b, y in zip(B, ys) if y is not None]
        c = C[order.index(name)]
        ax.plot(*zip(*pts), color=c, lw=2, marker=marker[name], ms=5, mec="white", mew=1, label=name)
        ax.annotate(name, pts[-1], xytext=(5, 0), textcoords="offset points", va="center", fontsize=7, color="#333")
    ax.set_xscale("log", base=2)
    ax.set_xticks(B, ["1K", "2K", "4K", "8K"])
    ax.set_xlim(0.85, 16)
    ax.set_title(title, fontsize=8, color="#333")
    ax.set_xlabel("retrieval budget (tokens)")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
axes[0].set_ylabel("accuracy (%)")
h = dict(zip(*axes[0].get_legend_handles_labels()[::-1]))
fig.legend([h[n] for n in order], order, loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
fig.savefig("fig_readers.pdf", bbox_inches="tight")

# Fig. 2: energy per correct answer vs accuracy, Qwen3-4B on LongMemEval (Fortuna model fitted on the M4 Max)
ret = {"1K": (38, 59.0), "2K": (68, 59.5), "4K": (127, 57.0), "8K": (243, 54.0)}  # J per query, accuracy
kv = {"SnapKV 2K of 8K": (243, 53.8), "Quest 2K of 8K": (243, 55.0)}
fig, ax = plt.subplots(figsize=(3.3, 2.4))
x = [e / (a / 100) for e, a in ret.values()]
ax.plot(x, [a for _, a in ret.values()], color=C[0], lw=2, marker="o", ms=5, mec="white", mew=1, label="hybrid retrieval")
for (k, (e, a)), xi in zip(ret.items(), x):
    ax.annotate(k, (xi, a), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color="#333")
for i, (k, (e, a)) in enumerate(kv.items()):
    ax.plot(e / (a / 100), a, color=C[1], marker="s", ms=5, mec="white", mew=1, ls="none", label="KV selection" if i == 0 else None)
    ax.annotate(k, (e / (a / 100), a), xytext=(-8, -4 if i == 0 else 4), textcoords="offset points", ha="right", va="center", fontsize=7, color="#333")
ax.set_xscale("log")
ax.set_xticks([60, 120, 240, 480], ["60", "120", "240", "480"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(53, 60.3)
ax.set_xlabel("energy per correct answer (J, SoC)")
ax.set_ylabel("LongMemEval accuracy (%)")
ax.grid(color="#e5e5e5", lw=0.6)
ax.legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig("fig_energy.pdf", bbox_inches="tight")
# Fig. 3: query time split into prefill (to first token) and decode, per device and budget (experiments/roofline.py)
import json
rf = json.load(open(Path(__file__).resolve().parent.parent / "results/exp5/roofline.json"))
devs = ["M4 Max", "iPhone 18 Pro Max", "iPhone 12 Pro", "Galaxy S25 Ultra"]
rows = [(d, b) for d in devs for b in (1000, 2000, 4000, 8000) if f"{d}|{b}" in rf]
fig, ax = plt.subplots(figsize=(3.4, 3.5))
y = []
for i, (d, b) in enumerate(rows):
    r = rf[f"{d}|{b}"]
    yi = i + devs.index(d) * 0.8  # gap between device groups
    y.append(yi)
    pre = 100 * r["prefill_time_share"]
    ax.barh(yi, pre, color=C[0], height=0.75, edgecolor="white", linewidth=1, label="prefill (to first token)" if i == 0 else None)
    ax.barh(yi, 100 - pre, left=pre, color=C[1], height=0.75, edgecolor="white", linewidth=1, label="decode" if i == 0 else None)
    ax.annotate(f"{r['query_s']:.1f} s", (100, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color="#333")
ax.set_yticks(y, [f"{d.replace('iPhone ', '').replace('Galaxy ', '')}  {b // 1000}K" for d, b in rows], fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("share of query time (%)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.0))
fig.tight_layout()
fig.savefig("fig_time.pdf", bbox_inches="tight")
print("figures written")
