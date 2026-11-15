#!/bin/bash
# Control for the Galaxy S25 Ultra result: the same Q4_0 GGUF and llama.cpp b11836 on the M4 Max, once on the GPU (Metal)
# and once on the CPU only (no Metal, no Accelerate), with the Android app's settings (batch = ubatch = 2048, flash
# attention auto). Separates runtime (MLX vs llama.cpp) from processor (GPU vs CPU) on one chip.
# Builds: data/exp5/android/build-macmetal (GGML_METAL=ON), build-maccpu (GGML_METAL=OFF GGML_BLAS=OFF).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A=$ROOT/data/exp5/android
OUT=$ROOT/results/exp5/mac_llamacpp
GGUF=$A/qwen3-4b-q4_0.gguf
PROMPT_TOK="1136,2156,4170,8204"  # same prompt lengths as android_bench.sh / the phone prompts
CPU_THREADS=${CPU_THREADS:-10}     # the M4 Max's performance cores
mkdir -p $OUT

bench() {  # <build> <name> <extra args...>
  local b=$1 n=$2; shift 2
  $A/$b/bin/llama-bench -m $GGUF -b 2048 -ub 2048 -o json -p $PROMPT_TOK -n 0 -r 3 "$@" > $OUT/$n-prefill.json
  $A/$b/bin/llama-bench -m $GGUF -b 2048 -ub 2048 -o json -p 0 -n 128 -d $PROMPT_TOK -r 3 "$@" > $OUT/$n-decode.json
}

echo "start $(date)"
bench build-macmetal gpu -ngl 99
echo "gpu done $(date)"
bench build-maccpu cpu -ngl 0 -t $CPU_THREADS
echo "cpu done $(date)"
