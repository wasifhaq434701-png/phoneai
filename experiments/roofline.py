"""Achieved memory bandwidth (decode) and compute (prefill) of Qwen3-4B-4bit per device, from existing bench runs.

decode: every new token reads all weights once (tied embeddings: the output layer reads the whole embedding table)
        plus the KV cache so far -> GB/s = tok/s * (weight bytes + KV bytes at the mean decode context).
prefill: 2 FLOPs per non-embedding weight per token, plus causal attention (QK^T and AV) at the mean position L/2;
         no logits during prefill (the bench app and mlx-lm skip them).
Run: python experiments/roofline.py  -> prints a table, writes results/exp5/roofline.json
"""
import json
import statistics as st
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results/exp5/phone"
W_BYTES = 2_263_022_529                        # mlx-community/Qwen3-4B-4bit model.safetensors, rev 4dcb3d10
H, LAYERS, FFN, QH, KVH, HD = 2560, 36, 9728, 32, 8, 128
KV_PER_TOK = 2 * LAYERS * KVH * HD * 2         # K and V, fp16 -> 147,456 B
LIN = LAYERS * (H * QH * HD + 2 * H * KVH * HD + QH * HD * H + 3 * H * FFN)
NEW = 128
_E = json.load(open(R.parent / "energy.json"))["per_correct"]
P_OUT = {b: _E[f"hybrid,div=0,edges=0@{b}"]["lme"]["p_out"] for b in (1000, 2000, 4000, 8000)}

GGUF_BYTES = 2_269_269_472                     # data/exp5/android/qwen3-4b-q4_0.gguf (Q4_0 --pure, sha 01b25740)
RUNS = {  # device -> (file under results/exp5, weight bytes); 12 Pro 8K came from a separate probe, left out
    "M4 Max": ("phone/mac_ref_rerun.json", W_BYTES),
    "iPhone 18 Pro Max": ("phone/bench-iPhone19,7-Qwen3-4B-4bit-1793298077.json", W_BYTES),
    "iPhone 12 Pro": ("phone/bench-iPhone13,3-Qwen3-4B-4bit-1792690536.json", W_BYTES),
    # llama.cpp CPU; speed run stopped after 25 of 80 calls (Result 26)
    "Galaxy S25 Ultra": ("android/bench-SM-S938B-qwen3-4b-q4_0-1794038822.json", GGUF_BYTES),
}


def prefill_flops_per_tok(L):
    return 2 * LIN + 2 * 2 * LAYERS * QH * HD * (L / 2)


# llama-bench runs (experiments/mac_llamacpp.sh): same GGUF on the M4 Max's GPU and CPU; prompt lengths -> budgets
BENCH = {"M4 Max llama.cpp GPU": "gpu", "M4 Max llama.cpp CPU": "cpu"}
BUDGET_OF = {1136: 1000, 2156: 2000, 4170: 4000, 8204: 8000}


def llama_bench_calls(name):
    """One pseudo-call per budget from llama-bench JSON (median over its repeats), in the bench apps' call format."""
    d = R.parent / "mac_llamacpp"
    # median over repeats: llama.cpp's Metal decode runs ~40% faster on the first repeat than on the rest
    pre = {r["n_prompt"]: st.median(r["samples_ts"]) for r in json.load(open(d / f"{name}-prefill.json"))}
    dec = {r["n_depth"]: st.median(r["samples_ts"]) for r in json.load(open(d / f"{name}-decode.json"))}
    return [{"budget": BUDGET_OF[L], "prompt_tok": L, "prefill_tok_s": pre[L], "decode_tok_s": dec[L], "ttft_s": L / pre[L]}
            for L in BUDGET_OF]


def main():
    assert LIN == 3_633_315_840 and KV_PER_TOK == 147_456
    out = {}
    runs = [(dev, [c for c in json.load(open(R.parent / f))["calls"] if c.get("phase", "item") == "item"], w)
            for dev, (f, w) in RUNS.items()]
    runs += [(dev, llama_bench_calls(n), GGUF_BYTES) for dev, n in BENCH.items()
             if (R.parent / "mac_llamacpp" / f"{n}-decode.json").exists()]
    for dev, calls, w_bytes in runs:
        for b in sorted({c["budget"] for c in calls}):
            cs = [c for c in calls if c["budget"] == b]
            bw = st.median(c["decode_tok_s"] * (w_bytes + KV_PER_TOK * (c["prompt_tok"] + NEW / 2)) for c in cs) / 1e9
            tf = st.median(c["prefill_tok_s"] * prefill_flops_per_tok(c["prompt_tok"]) for c in cs) / 1e12
            # query time at the reader's real mean answer length on LongMemEval (Result 14), not the bench's forced 128
            p_out = P_OUT[b]
            ttft = st.median(c["ttft_s"] for c in cs)
            dec = p_out / st.median(c["decode_tok_s"] for c in cs)
            out[f"{dev}|{b}"] = {"n": len(cs), "prompt_tok": st.median(c["prompt_tok"] for c in cs),
                                 "decode_GBps": round(bw, 1), "prefill_TFLOPs": round(tf, 2),
                                 "query_s": round(ttft + dec, 1), "prefill_time_share": round(ttft / (ttft + dec), 3)}
            print(f"{dev:18s} {b:5d}  ttft {ttft:6.1f}  pre {st.median(c['prefill_tok_s'] for c in cs):6.0f}  dec {st.median(c['decode_tok_s'] for c in cs):5.1f}  peak {max(c.get('peak_footprint') or 0 for c in cs)/1e9:4.2f}  n={len(cs):2d}  decode {bw:6.1f} GB/s   prefill {tf:5.2f} TFLOP/s"
                  f"   query {ttft + dec:6.1f} s, prefill share {ttft / (ttft + dec):.0%}")
    (R.parent / "roofline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
