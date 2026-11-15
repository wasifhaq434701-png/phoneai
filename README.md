# The Retrieval Budget Is the Knob — code and data

Code, benchmark apps and raw results for the paper *The Retrieval Budget Is the Knob: Characterizing Long-Term Memory
Queries of Small LLMs on Phones*. A memory query retrieves excerpts of a user's history up to a token budget,
prefills them with the question and decodes an answer. We measure accuracy on a Mac, and time, memory and energy on a
Mac and four phones.

```
experiments/   Python and shell scripts (run from the repo root unless noted)
figures/       make_figures.py → fig_readers.pdf, fig_energy.pdf, fig_time.pdf
results/       every result file the tables and figures use (JSON / JSONL), no logs
prompts/       prompts.json: the 80 fixed device prompts as Qwen3-4B token ids
ios/           iPhone benchmark app (Xcode project, MLX Swift)
android/       Android benchmark app (Kotlin + llama.cpp b11836, CPU)
```

## What reproduces what

| paper | script | result files |
|---|---|---|
| Table 2 (budget sweep), Fig. 1 (four readers), LongMemEval-M paragraph | `experiments/user_memory.py run / judge / report`; `experiments/compare55.py` (LongMemEval-M) | `results/exp4/*.json`, `results/exp5/lme_m/*.json` |
| Judge agreement (§3) | `user_memory.py label` | `results/exp4/judge_check.json`, `judge_check_8b.json` |
| RULER probe (§4) | `experiments/ruler_probe.py` | `results/exp5/ruler.json`, `ruler_outputs.json` |
| M4 Max energy fit, Fig. 2, Table 3 (M4 Max) | `experiments/energy.py calib / fit` | `results/exp5/energy.json`, `energy_calls.json` (`*_run1` = an earlier, matching run) |
| iPhone energy (Table 3) | `experiments/iphone_power.py` (record, then `grid`) | `results/exp5/phone/power/` |
| Android energy (Table 3, Pixel 8 prefill) | `experiments/android_energy.py` | `results/exp5/android/grid_fit_*.json`, `android/pixel8/prefill_Pixel8.json` |
| Learned per-question budget (§5) | `experiments/budget_predictor.py feats / train / calib` | `results/exp5/budget_pred.json`, `budget_calib.json` |
| Table 4 (time per device), Fig. 3 | iOS / Android bench apps; `experiments/phone_ref.py ref` (Mac reference); `experiments/roofline.py` | `results/exp5/phone/bench-*.json`, `phone/mac_ref*.json`, `android/bench-*.json`, `results/exp5/roofline.json` |
| Agreement with the M4 Max (§6) | `phone_ref.py compare / judge` | `results/exp5/phone/judge2k.json` |
| What fits (§7) | iOS bench app (memory probe) | `results/exp5/phone/memory_limits.json`, `phone/probe/` |
| Table 5 (notes, SnapKV, Quest, KV bits) | `user_memory.py` options `sum`, `snap`, `quest`, `kvb` | `results/exp4/*{sum,snap,quest,kvb}*` |
| Table 5 (weight precision) | `experiments/mixed_precision.py scan / build / summary` | `results/exp5/mp_scan.json`, `mp_summary.json`, `results/exp4/*rdr=mp*` |
| Other selector components (§8) | `user_memory.py` options `div`, `edges`, `keep`/`gate`, `session`; `user_memory.py tier` | `results/exp4/*` |
| CPU vs GPU control on the M4 Max | `experiments/mac_llamacpp.sh` | `results/exp5/mac_llamacpp/` |

`python3 experiments/roofline.py` and `cd figures && python3 make_figures.py` run from the shipped results alone
(no model, no dataset). Figs. 1 and 2 hard-code the final numbers from Table 2 and Table 3.

Result files are keyed `<selector>@<budget tokens>[~<split>]`, e.g. `hybrid,div=0,edges=0@2000.json` (the main
configuration: cosine + BM25 reciprocal-rank fusion, no diversity rule, no neighbour expansion). Each holds
`{question id: {out, mem_tok, recall, judge}}`. Bench files (schema `membench/1`) hold one entry per call with TTFT,
prefill and decode rates, peak memory, thermal state and the output token ids. In this release, earlier names were
replaced by neutral ones (selector name, app ids, schema string); the numbers are unchanged. Removing the prompt text
from `prompts.json` changed its SHA-256, so the `prompts.sha256` recorded in the bench files is the hash of the
original file, which had the same token ids plus the decoded text.

## Devices

| device | SoC | RAM | runtime |
|---|---|---|---|
| MacBook Pro 14" (2024) | M4 Max | 36 GB | MLX 0.31 / mlx-lm 0.31 (accuracy, energy, reference timings) |
| iPhone 12 Pro | A14 | 6 GB | MLX Swift 0.32.3 + mlx-swift-lm 3.32.3, GPU |
| iPhone 18 Pro Max | A20 Pro | 12 GB | same app |
| Galaxy S25 Ultra | Snapdragon 8 Elite for Galaxy | 12 GB | llama.cpp b11836, CPU, 8 threads |
| Pixel 8 | Tensor G3 | 8 GB | same app, prefill only |

## Mac: accuracy and energy (Python)

Python 3.13 on Apple silicon with `mlx 0.31.2`, `mlx-lm 0.31.3`, `numpy 2.1`, `sentence-transformers 5.5`,
`transformers 5.9`, `torch 2.10` (MPS), `scikit-learn 1.6`, `pandas 2.2`, `scipy 1.15`, `huggingface-hub 1.17`,
`matplotlib 3.10`. Models come from the Hugging Face Hub on first use: `mlx-community/Qwen3-4B-4bit` (reader),
`mlx-community/Qwen3-8B-4bit`, `mlx-community/gemma-3-text-4b-it-4bit`, `mlx-community/Llama-3.2-3B-Instruct-4bit`,
`mlx-community/Qwen3-30B-A3B-4bit` (judge), `sentence-transformers/all-MiniLM-L6-v2` (embeddings),
`Qwen/Qwen3-4B` (bf16, for the GGUF and the precision experiments).

**Data** (not in this repository; it goes in `data/`, which git ignores):

- LongMemEval, cleaned release, MIT: <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>
  (`longmemeval_s_cleaned.json`; `longmemeval_m_cleaned.json` for LongMemEval-M). `user_memory.py ingest` downloads it.
- LoCoMo, CC BY-NC 4.0: <https://github.com/snap-research/locomo>. Download `data/locomo10.json` yourself and
  place it at `data/exp4/raw/locomo10.json`. We do not redistribute it.
- RULER (Apache 2.0): <https://github.com/NVIDIA/RULER>. Generate the five tasks at 8K with its `prepare.py` and the
  Qwen3-4B tokenizer into `data/exp5/ruler/` (see the header of `ruler_probe.py`).
- WebInstructSub-prometheus (calibration text for `mixed_precision.py scan`): the first 70 rows from the Hub's
  datasets-server `/rows` API, saved as `data/exp5/mp/webinstruct70.json`.

```bash
cd experiments
python3 -u user_memory.py check                      # self-test, no data needed
python3 -u user_memory.py ingest && python3 -u user_memory.py embed
python3 -u user_memory.py run "hybrid,div=0,edges=0" 1000 2000 4000 8000 --set test
python3 -u user_memory.py judge && python3 -u user_memory.py report
MEM_STORE=lme_m python3 -u user_memory.py ingest      # LongMemEval-M in its own store (~1M tokens per user)
```

Tuning used only the LongMemEval dev split (every fifth question, `--set lme-dev`). Run one MLX job at a time: two at
once can run out of GPU memory. Throughput on the M4 Max is about 4 s per question at 2K for the 4B reader.

**Mac energy.** `energy.py calib` logs the wall-clock window of each call while `sudo powermetrics -i 100 > pm.txt` runs
in another terminal (CPU + GPU + ANE "Combined Power": SoC only, a lower bound); `energy.py fit pm.txt` aligns and
integrates. `energy.py check` self-tests on synthetic data.

**Mac reference for the phones.** `phone_ref.py ref` runs the 80 prompts exactly as the apps do (feeds the token ids,
greedy, fresh KV cache, 128 tokens with EOS ignored, prefill in chunks of 2,048).

**llama.cpp on the Mac.** `experiments/android_bench.sh build` converts `Qwen/Qwen3-4B` to the Q4_0 GGUF used on the
Android phones (`convert_hf_to_gguf.py` then `llama-quantize --pure … Q4_0`; SHA-256
`01b25740ddcf91454dc8fba52cd8cdd89d14fc8bb6a24834490012157a697de4`, 2,269,269,472 bytes).
`experiments/mac_llamacpp.sh` runs `llama-bench` with it on the M4 Max GPU and CPU. Both expect the llama.cpp b11836
source under `data/exp5/android/src/llama.cpp`.

## iPhone benchmark app (`ios/`)

There is no prebuilt app: an `.ipa` installs only on devices it was signed for, and the app needs the
increased-memory-limit entitlement, which requires a signed build. You need a Mac with Xcode 26 or later (including
its Metal Toolchain component), an Apple developer team, and an iPhone with Developer Mode on.

```bash
TEAM=<your team id> ios/build_bench.sh <device UDID>   # builds Release and installs; UDIDs: xcrun devicectl list devices
ios/push_assets.sh <device UDID>                       # copies prompts.json + mlx-community/Qwen3-4B-4bit from your HF cache
```

If the default bundle id `org.membench.ios` is taken for your team, pass `BUNDLE_ID=<yours>` to both scripts. The app
has one screen: pick the budgets, the number of repeats and the sustained-run length, then Start. Protocol used in the
paper: plugged in, airplane mode, Do Not Disturb, phone untouched (long prefills can trip the iOS GPU watchdog when
the system needs the GPU). Results are written to the app's Documents folder; fetch them with
`xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer --domain-identifier org.membench.ios
--source Documents --destination <dir>`. The energy grid (4 prompts × 16/64/256 forced tokens, each cell ≥ 100 s,
three passes) is a separate button.

**iPhone power.** `experiments/iphone_power.py <out.jsonl> <seconds>` reads the phone's battery telemetry (whole-device
power, 1 Hz samples published every 20 s) over USB through Apple's diagnostics service, using `pymobiledevice3 11.49.1`
in its own virtualenv. Measure with the phone plugged in and held at its 80% charge limit, so it does not charge.
`iphone_power.py grid <grid.json> <power.jsonl> <idle_W>` fits Eq. 4 and Eq. 6.

## Android benchmark app (`android/`)

The APK is attached to this repository's release (arm64, Android 12 or later). It has no network permission.
Details of every button and output key are in `android/README.md`.

```bash
adb install membench-android-bench.apk
adb push qwen3-4b-q4_0.gguf /sdcard/Android/data/org.membench.bench/files/models/   # or "Import model…" in the app
```

Build it yourself with the Android SDK (platform 36, NDK 28.2.13676358, CMake 3.22.1) and the llama.cpp b11836 source
in `third_party/llama.cpp` (`git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp && git -C
third_party/llama.cpp checkout b11836`):

```bash
cd android && ./gradlew :app:assembleRelease :app:testDebugUnitTest
```

Set `BENCH_KEYSTORE=/path/to/keystore.properties` (keys `storeFile`, `alias`, `password`) to sign the release build;
without it Gradle produces an unsigned APK. Energy comes from the battery gauge (current every 0.25 s × voltage); the
app's idle-baseline button measures the power subtracted in the paper. `android_energy.py <grid.json> <idle.json>`
turns a grid file into energy per query.

## Licences of dependencies

MLX, mlx-lm, MLX Swift and mlx-swift-lm (MIT); llama.cpp (MIT); swift-numerics and swift-syntax (Apache 2.0);
sentence-transformers (Apache 2.0); Hugging Face transformers and huggingface-hub (Apache 2.0); PyTorch (BSD-3);
NumPy, SciPy, scikit-learn, pandas (BSD); matplotlib (PSF-based); pymobiledevice3 (GPL-3.0, used as a separate tool, not
linked). Models: Qwen3 (Apache 2.0), Gemma 3 (Gemma Terms of Use), Llama 3.2 (Llama 3.2 Community License),
all-MiniLM-L6-v2 (Apache 2.0). Datasets: LongMemEval (MIT), LoCoMo (CC BY-NC 4.0, not redistributed), RULER (Apache 2.0),
WebInstructSub (see its dataset card).
