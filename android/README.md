# MemBench (Android)

A one-screen benchmark app: it runs Qwen3-4B (Q4_0 GGUF) with llama.cpp on the phone's CPU and records time, memory and
**battery energy** per call. Same protocol and JSON format as the iPhone bench app (`ios/MemBench/Bench/`),
so `experiments/phone_ref.py compare` and `judge` read its results unchanged. The app has no network permission.

## Build (Mac)

Needs the Android SDK (platform 36, NDK 28.2.13676358, CMake 3.22.1) and, outside git, the llama.cpp b11836 source in
`third_party/llama.cpp`. `prompts/prompts.json` is bundled unchanged. For a signed release, point `BENCH_KEYSTORE` at a
`keystore.properties` (keys `storeFile`, `alias`, `password`); without it the release APK is unsigned.

```bash
cd android && BENCH_KEYSTORE=/path/to/keystore.properties ./gradlew :app:assembleRelease :app:testDebugUnitTest
```

APK: `android/app/build/outputs/apk/release/app-release.apk` (arm64 only, minSdk 31 = Android 12).

## Install

- With a cable: `adb install app-release.apk`.
- Without: copy the APK to the phone and open it (allow "install unknown apps" for the file manager when asked).

## Load the model (once)

The model is not in the APK. Copy `qwen3-4b-q4_0.gguf` (2.27 GB, SHA-256 `01b25740ddcf91454dc8fba52cd8cdd89d14fc8bb6a24834490012157a697de4`)
to the phone (e.g. the Download folder), then in the app tap **Import model…** and pick it. The app copies it into its own
storage (needs ~2.3 GB free on top of the file you copied; delete that copy afterwards) and shows the SHA-256 in the log
when done (about a minute). Development shortcut: `adb push qwen3-4b-q4_0.gguf /sdcard/Android/data/org.membench.bench/files/models/`.

## Run

- **Items** (none / up to 1K / 2K / 4K / all = 8K) and **Sustained run: N min**, then **Start**: a warm-up call, the 20 LongMemEval
  questions at each budget up to the chosen one, interleaved (question by question, budgets rotated one step per question,
  same order as the iPhone app), then the first 2K question in a loop for N minutes.
  Every call: fresh KV cache, greedy, 128 new tokens with EOS ignored, prefill in chunks of 2048.
- **Natural answers** (checkbox): the 80 question calls stop at the end of the answer, at most 200 tokens, as on the Mac,
  so `python3 experiments/phone_ref.py judge_all <file>` can score all 80 answers there, and each call's energy is the
  energy of a real query. Warm-up and sustained calls keep the forced 128 tokens.
- **Start at question N**: continues a run that the battery window stopped. The log (and `next_question` in the file)
  says where it stopped; charge, then Start again at that question. Each part has its own file (`…-q<N>-…`), with
  the same order as an uninterrupted run; `judge_all a.json+b.json` joins them (a question measured twice keeps the
  first measurement).
- **Repeats** (default 3): one Start runs them all, each as a complete run with its own results file. Before runs 2 and 3
  the app waits until the phone has cooled down: at least 15 min, thermal status back to "none" and the battery within 1 °C
  of where run 1 started (at most 60 min; the file's `series` object records the wait). Leave the phone untouched. Same rule
  as the iPhone app, except iOS reports no temperature. An energy run is always a single run (the 90% → 30% window).
- **Energy grid**: one prompt per budget (1K/2K/4K/8K) × forced 16 / 64 / 256 output tokens, cells in the same rotated
  order as the iPhone app's grid, each cell repeated until ≥ 3 s,
  3 passes (like the Mac calibration, Result 14). If one charge can't hold all three (a Pixel 8 on the CPU uses ~55% of
  its battery for two), pick "Grid pass 1 only", recharge, then pass 2, then pass 3: each pass keeps its own rotated
  order and writes its own file (`…-grid-p<n>-…`), so the three files add up to the full grid.
- **Idle baseline (10 min, on battery)**: model loaded, nothing running, screen at the run brightness, same 90–30% window
  (`…-idle-…` file; mean W = `idle.energy.decode_j / decode_s`). Run it in the same session as the grid, then
  `python3 experiments/android_energy.py <grid.json> <idle.json> [out.json]` gives energy per query above idle.
- **Threads**: defaults to the phone's fast cores (max clock ≥ 70% of the fastest core). The little cores slow llama.cpp down;
  the choice is recorded in the results.
- **Energy run** checkbox (for both buttons): refuses to start while charging or outside 90–30% battery, and stops cleanly
  when the charger is connected or the battery drops below 30%. For an energy run: unplug, airplane mode on, battery at 90%
  (Samsung: turn "Protect battery" off to charge past 80/85%),
  close other apps, don't touch the phone. The app fixes the screen brightness at 50% and keeps the screen on during every run.

llama.cpp uses every chosen core, so anything else running on the phone (app updates, sync, the screen being touched)
slows it down a lot: restart the phone, wait ~5 minutes, then start the run.

**Accuracy and energy in one run:** Items all, Sustained 0 min, Energy run, Natural answers, Start at question 1. On a
phone's CPU this takes longer than one charge (the S25 Ultra needs ~3.5 h for the 80 calls), so expect two parts.

A full run (all items + 30 min sustained) takes roughly 1–2 h on a recent phone; the grid ~30–60 min. Keep the app in front.

## Results

`/sdcard/Android/data/org.membench.bench/files/bench-<device>-<model>-<unix time>.json` (grid: `…-grid-…`), rewritten after
every call (a crash keeps the earlier calls; `in_progress` names the call that was running). **Share latest results** sends the
newest file through the share sheet; or `adb pull` that folder into `results/exp5/android/`.

Same keys as iOS (`schema` `membench/1`). Android differences and additions:

- `peak_footprint` = peak VmRSS (sampled every 100 ms); `available_memory_after` = system-wide available RAM (Android has no
  per-app limit like iOS jetsam). No `mlx_peak`.
- `thermal_state` = PowerManager thermal status (none / light / moderate / severe / …); `thermal_headroom` when the phone reports it.
- `device`: `soc`, `manufacturer`, `cpu_cores`, `airplane_mode`; `model`: file name, `sha256`, `size_bytes`; `runtime`: llama.cpp
  build, backend (`cpu`), threads, `fast_cores`, `system_info`.
- `energy`: the battery trace (`columns` + `rows`, one row per 250 ms: boot-clock ms, charge counter µAh, CURRENT_NOW µA,
  voltage mV, energy counter nWh, level %, plugged), `boot_to_wall_ms` to convert to wall clock, `counters` (does the phone report
  each counter and how often it changes: CURRENT_NOW often updates only every ~1 s or slower), `energy_run`, `brightness`.
- Per call `energy`: prefill (start → first token) and decode (first token → last token) windows, with `*_j` = ∫ |I| × V dt,
  `*_uah` from the charge counter and `*_counter_j` from the energy counter when the phone has them; `*_under_1s` flags windows too
  short for the battery sensors. Battery energy includes the screen and the rest of the phone, not just the chip.
- Grid calls are in `calls` with `phase: "grid"`, `pass`, `n_new`, `rep` (`compare`/`judge` only read `phase: "item"`).
