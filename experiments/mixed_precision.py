"""Exp 5.16: per-layer mixed precision at a fixed model size.

Sensitivity (simplified APreQEL signal): mean cosine similarity between each decoder layer's input and output hidden states over
calibration tokens (low similarity = the layer changes the representation more = important). Two calibration sets:
70 WebInstructSub-prometheus samples (instruction + generation) and 70 of our LongMemEval-dev hybrid@2K prompts;
both truncated to 1,024 tokens. Scanned on the bf16 Qwen/Qwen3-4B.
Models, all the size of uniform 4-bit (group 64: b + 0.5 bits/weight; 6x8.5 + 6x4.5 + 24x3.5 = 36x4.5), embeddings 4-bit:
  u4  uniform 4-bit (requantized here, the fair baseline)
  w   6 most important layers 8-bit, next 6 4-bit, 24 least important 3-bit, ranking from WebInstruct
  m   same, ranking from the memory prompts
  r   control: WebInstruct ranking reversed (the 6 least important layers get 8-bit)
Exp 5.16b (APreQEL's regime: raise important layers ABOVE a 4-bit base when memory allows): base uniform 4-bit,
important units raised to 6-bit (WebInstruct ranking):
  x6w / x12w  the 6 / 12 most important decoder blocks at 6-bit      x6r / x12r  control: the 6 / 12 least important blocks
  u8          uniform 8-bit (embeddings too): the precision ceiling
  x12s        attention and FFN ranked separately (sub-layer cosine), most important units at 6-bit up to x12w's parameter budget
Exp 5.16d (headroom is below 4 bits): 3-bit base, APreQEL's memory-priority split for Qwen3-4B
(72% of the 72 attention/FFN units at q3_K, 28% at q4_K = the top 20 units at 4-bit), sub-layer ranking, embeddings 4-bit:
  u3   uniform 3-bit      x3s  top 20 units at 4-bit
  x3r  control: least important units first at 4-bit, up to x3s's 4-bit weight count (same size; x3s's are mostly FFN halves)
Evaluated as readers (`rdr=mp<name>`) on LME test @2K. Usage (from experiments/): python mixed_precision.py check | scan | scansub | build | summary
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import user_memory as um

BF16 = "Qwen/Qwen3-4B"
MP = um.D5 / "mp"
SCAN = um.ROOT / "results" / "exp5" / "mp_scan.json"
N_TOK, N8, N4 = 1024, 6, 6


def bits_for(rank):
    """rank = layer indices, most important first → bits per layer index (6 at 8, 6 at 4, the rest at 3)."""
    bits = np.full(len(rank), 3)
    bits[list(rank[:N8])] = 8
    bits[list(rank[N8:N8 + N4])] = 4
    return bits.tolist()


def raise_top(rank, k, hi=6):
    """5.16b: uniform 4-bit, the first k layers of `rank` at `hi` bits → [[attn, ffn] bits] per layer."""
    bits = [[4, 4] for _ in rank]
    for i in rank[:k]:
        bits[i] = [hi, hi]
    return bits


def raise_units(units, budget, sizes, n_layers, hi=6, lo=4):
    """5.16b: units = (layer, 0=attn | 1=ffn), most important first; raise units to `hi` while their weights fit in `budget`."""
    bits, used = [[lo, lo] for _ in range(n_layers)], 0
    for i, kind in units:
        if used + sizes[kind] <= budget:
            bits[i][kind], used = hi, used + sizes[kind]
    return bits


def base_raise(units, k, n_layers, lo=3, hi=4):
    """5.16d: every unit at `lo` bits, the first k of `units` (layer, 0=attn | 1=ffn) at `hi` → [[attn, ffn] bits] per layer."""
    bits = [[lo, lo] for _ in range(n_layers)]
    for i, kind in units[:k]:
        bits[i][kind] = hi
    return bits


def unit_sizes():
    """Weights per (attention, FFN) block, from the bf16 config (q, k, v, o and gate, up, down projections)."""
    from huggingface_hub import snapshot_download
    c = json.loads((Path(snapshot_download(BF16)) / "config.json").read_text())
    q, kv = c["num_attention_heads"] * c["head_dim"], c["num_key_value_heads"] * c["head_dim"]
    return 2 * c["hidden_size"] * (q + kv), 3 * c["hidden_size"] * c["intermediate_size"]


def calib_texts(tok):
    rows = [r["row"] for r in json.loads((MP / "webinstruct70.json").read_text())["rows"]]
    web = [f"{r['instruction']}\n{r['generation']}" for r in rows]
    mem = [t for _, t, _, _ in um.contexts(um.MAIN, 2000, "lme-dev")[:70]]
    return {"webinstruct": web, "memory": mem}


def scan():
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(BF16)
    out = {}
    for name, texts in calib_texts(tok).items():
        cos = []
        for t in texts:
            h = model.model.embed_tokens(mx.array(tok.encode(t)[:N_TOK])[None])
            row = []
            for layer in model.model.layers:
                h2 = layer(h, "causal" if h.shape[1] > 1 else None)
                a, b = h[0].astype(mx.float32), h2[0].astype(mx.float32)
                c = (a * b).sum(-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1) + 1e-6)
                row.append(float(c.mean()))
                h = h2
            cos.append(row)
        m = np.array(cos).mean(0)
        out[name] = {"n": len(cos), "mean_cos": m.round(4).tolist(), "rank": np.argsort(m, kind="stable").tolist()}
        print(f"{name}: least similar (most important) layers {out[name]['rank'][:8]}, most similar {out[name]['rank'][-6:]}", flush=True)
    a, b = np.array(out["webinstruct"]["rank"]), np.array(out["memory"]["rank"])
    out["top12_overlap"] = len(set(a[:12]) & set(b[:12]))
    SCAN.write_text(json.dumps(out, indent=1))


def scansub():
    """5.16b: cosine(input, output) of the attention and the FFN half of every block separately (WebInstruct set)."""
    import mlx.core as mx
    from mlx_lm import load
    model, tok = load(BF16)
    cos_ = lambda a, b: float(((a * b).sum(-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1) + 1e-6)).mean())
    rows = []
    for t in calib_texts(tok)["webinstruct"]:
        h, row = model.model.embed_tokens(mx.array(tok.encode(t)[:N_TOK])[None]), []
        for L in model.model.layers:  # Qwen3 block: h + attn(norm(h)), then + mlp(norm(.)) — same as L(h)
            a = h + L.self_attn(L.input_layernorm(h), "causal", None)
            out = a + L.mlp(L.post_attention_layernorm(a))
            if not rows and not row:  # the split must reproduce the block exactly
                assert mx.allclose(out, L(h, "causal"), atol=1e-2).item()
            row.append([cos_(h[0].astype(mx.float32), a[0].astype(mx.float32)), cos_(a[0].astype(mx.float32), out[0].astype(mx.float32))])
            h = out
        rows.append(row)
    m = np.array(rows).mean(0)  # layers x 2
    order = np.argsort(m.ravel(), kind="stable")  # least similar first = most important
    s = json.loads(SCAN.read_text())
    s["sub_webinstruct"] = {"n": len(rows), "mean_cos": m.round(4).tolist(), "units": [[int(u // 2), int(u % 2)] for u in order]}
    SCAN.write_text(json.dumps(s, indent=1))
    print("most important units (layer, 0=attn 1=ffn):", s["sub_webinstruct"]["units"][:12], flush=True)


def specs():
    """name -> [[attn bits, ffn bits]] per layer."""
    s = json.loads(SCAN.read_text())
    w, m = s["webinstruct"]["rank"], s["memory"]["rank"]
    pair = lambda bits: [[b, b] for b in bits]
    out = {"u4": pair([4] * len(w)), "w": pair(bits_for(w)), "m": pair(bits_for(m)), "r": pair(bits_for(w[::-1])),
           "u8": pair([8] * len(w)), "x6w": raise_top(w, 6), "x12w": raise_top(w, 12), "x6r": raise_top(w[::-1], 6), "x12r": raise_top(w[::-1], 12)}
    if "sub_webinstruct" in s:
        sizes = unit_sizes()
        out["x12s"] = raise_units(s["sub_webinstruct"]["units"], 12 * sum(sizes), sizes, len(w))
        units, k = s["sub_webinstruct"]["units"], round(0.28 * 2 * len(w))  # APreQEL: 28% of the units at q4_K
        budget = sum(sizes[kind] for _, kind in units[:k])
        out.update(u3=pair([3] * len(w)), x3s=base_raise(units, k, len(w)),
                   x3r=raise_units(units[::-1], budget, sizes, len(w), hi=4, lo=3))
    return out


def build(names=None):
    from mlx_lm import convert
    for name, bits in specs().items():
        if names and name not in names:
            continue
        path = MP / name
        if (path / "config.json").exists():
            print(f"{name}: exists")
            continue
        shutil.rmtree(path, ignore_errors=True)

        def pred(p, module, bits=bits, rest=8 if name == "u8" else 4):  # u8 = the precision ceiling, embeddings 8-bit too
            parts = p.split(".")
            i = int(parts[2]) if len(parts) > 3 and parts[1] == "layers" and parts[2].isdigit() else None
            return {"group_size": 64, "bits": rest if i is None else bits[i][parts[3] == "mlp"], "mode": "affine"}

        convert(BF16, mlx_path=str(path), quantize=True, q_group_size=64, q_bits=4, quant_predicate=pred)
        size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
        print(f"{name}: bits {bits} -> {size / 1e9:.3f} GB", flush=True)


def summary(names=None):
    """Adds (or refreshes) `names` (default: all) in mp_summary.json; GB is read from the built model if it still exists."""
    qs = {q["qid"]: q for q in json.loads(um.QS.read_text())}
    res = {k: json.loads((um.OUT / f"{um.MAIN},rdr=mp{k}@2000~lme-test.json").read_text()) for k in specs()
           if k == "u4" or not names or k in names}
    base = json.loads((um.OUT / f"{um.MAIN}@2000.json").read_text())
    path = um.ROOT / "results" / "exp5" / "mp_summary.json"
    rng, out = np.random.default_rng(0), json.loads(path.read_text()) if path.exists() else {}
    for k, r in res.items():
        if names and k not in names:
            continue
        ks = [q for q in r if q in base and qs[q]["bench"] == "lme"]
        y = np.array([r[q]["judge"] for q in ks], float)
        files = list((MP / k).glob("*.safetensors"))
        row = {"acc": float(y.mean()), "GB": sum(f.stat().st_size for f in files) / 1e9 if files else out.get(k, {}).get("GB")}
        for ref_name, ref in (("vs_u4", res["u4"]), ("vs_community4bit", base)):
            d = y - np.array([ref[q]["judge"] for q in ks], float)
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            row[ref_name] = [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
        row["types"] = {t: float(np.mean([r[q]["judge"] for q in ks if qs[q]["type"] == t])) for t in sorted({qs[q]["type"] for q in ks})}
        out[k] = row
        print(f"{k}: LME {row['acc']:.1%}  {row['GB'] or 0:.3f} GB  vs u4 {row['vs_u4'][0] * 100:+.1f} [{row['vs_u4'][1] * 100:+.1f},{row['vs_u4'][2] * 100:+.1f}]", flush=True)
    path.write_text(json.dumps(out, indent=1))


def check():
    rank = list(range(36))
    b = bits_for(rank)
    assert b[:6] == [8] * 6 and b[6:12] == [4] * 6 and b[12:] == [3] * 24
    assert sum(x + 0.5 for x in b) == 36 * 4.5  # same size as uniform 4-bit (group 64 adds 0.5 bit/weight)
    assert bits_for(rank[::-1])[-6:] == [8] * 6
    x = raise_top(rank, 12)
    assert x[:12] == [[6, 6]] * 12 and x[12:] == [[4, 4]] * 24 and raise_top(rank[::-1], 6)[-6:] == [[6, 6]] * 6
    sizes = (26, 75)  # ~Qwen3-4B attention / FFN weights (M)
    u = raise_units([(0, 1), (1, 0), (2, 1), (3, 0)], 12 * sum(sizes), sizes, 36)
    assert u[0] == [4, 6] and u[1] == [6, 4] and u[2] == [4, 6] and u[3] == [6, 4]
    assert sum(sizes[k] for i in range(36) for k in (0, 1) if raise_units([(j // 2, j % 2) for j in range(72)], 12 * sum(sizes), sizes, 36)[i][k] == 6) <= 12 * sum(sizes)
    b3 = base_raise([(0, 1), (2, 0)], 1, 36)
    assert b3[0] == [3, 4] and b3[2] == [3, 3] and b3[1] == [3, 3]
    assert round(0.28 * 72) == 20
    print("checks pass")


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:]
    {"check": check, "scan": scan, "scansub": scansub, "build": lambda: build(names), "summary": lambda: summary(names)}[cmd]()
