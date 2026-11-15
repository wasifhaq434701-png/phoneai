#!/bin/bash
# Android benchmark (Samsung S25 Ultra or any arm64 phone) with llama.cpp over adb — no app needed.
#   build   Mac: llama-quantize + Q4_0 GGUF from the local Qwen/Qwen3-4B; Android: OpenCL ICD loader + llama.cpp (Adreno OpenCL)
#   run     phone: push, device info, speed sweep (GPU + CPU), 20 x 2K accuracy answers, sustained loop, battery power log
#   clean   phone: delete everything this script put there
# Sources (data/exp5/android/src, gitignored): llama.cpp b11836, KhronosGroup OpenCL-Headers / OpenCL-ICD-Loader v2026.05.29.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A=$ROOT/data/exp5/android
SRC=$A/src
OUT=$ROOT/results/exp5/android
SDK=~/Library/Android/sdk
NDK=$SDK/ndk/28.2.13676358
CMAKE=$SDK/cmake/3.22.1/bin/cmake
NINJA=$SDK/cmake/3.22.1/bin/ninja
HF=~/.cache/huggingface/hub/models--Qwen--Qwen3-4B
GGUF=$A/qwen3-4b-q4_0.gguf
DEV=/data/local/tmp/membench
PROMPT_TOK="1136,2156,4170,8204"  # mean prompt tokens of the 1K/2K/4K/8K items (prompts/prompts.json)

build() {
  # Mac tools for the model file (CPU only: no Metal needed to quantize)
  $CMAKE -S $SRC/llama.cpp -B $A/build-mac -G Ninja -DCMAKE_MAKE_PROGRAM=$NINJA -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
  $CMAKE --build $A/build-mac --target llama-quantize
  if [ ! -f $GGUF ]; then
    rev=$(cat $HF/refs/main)
    python3 $SRC/llama.cpp/convert_hf_to_gguf.py $HF/snapshots/$rev --outtype f16 --outfile $A/qwen3-4b-f16.gguf
    # --pure: every tensor Q4_0 incl. embeddings/LM head, like the all-4-bit MLX model used on the Mac and iPhone
    $A/build-mac/bin/llama-quantize --pure $A/qwen3-4b-f16.gguf $GGUF Q4_0
    rm $A/qwen3-4b-f16.gguf
    echo "Qwen/Qwen3-4B@$rev -> Q4_0 --pure, llama.cpp b11836" > $GGUF.source
  fi
  # Android: OpenCL loader (the phone's vendor driver is loaded at run time), then llama.cpp with the Adreno backend.
  # Paths are passed to CMake instead of copying headers/libs into the NDK as the llama.cpp docs do.
  local TC="-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a"
  $CMAKE -S $SRC/OpenCL-ICD-Loader -B $A/build-icd -G Ninja -DCMAKE_MAKE_PROGRAM=$NINJA -DCMAKE_BUILD_TYPE=Release $TC \
    -DANDROID_PLATFORM=24 -DANDROID_STL=c++_shared -DOPENCL_ICD_LOADER_HEADERS_DIR=$SRC/OpenCL-Headers -DBUILD_TESTING=OFF
  $CMAKE --build $A/build-icd
  # Two phone builds: Adreno OpenCL, and CPU only (fallback if the shell can't load the vendor OpenCL driver).
  # OpenMP off: phones don't ship libomp.so.
  local v flags
  for v in opencl cpu; do
    flags=""
    [ $v = opencl ] && flags="-DGGML_OPENCL=ON -DOpenCL_INCLUDE_DIR=$SRC/OpenCL-Headers -DOpenCL_LIBRARY=$A/build-icd/libOpenCL.so"
    $CMAKE -S $SRC/llama.cpp -B $A/build-$v -G Ninja -DCMAKE_MAKE_PROGRAM=$NINJA -DCMAKE_BUILD_TYPE=Release $TC \
      -DANDROID_PLATFORM=android-28 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_OPENMP=OFF $flags \
      -DCMAKE_C_FLAGS="-march=armv8.2-a+dotprod+i8mm" -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod+i8mm"
    $CMAKE --build $A/build-$v --target llama-bench llama-completion
    $NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip $A/build-$v/bin/llama-bench $A/build-$v/bin/llama-completion
  done
  ls -la $A/build-{opencl,cpu}/bin/llama-* $GGUF
}

sh_dev() { adb shell "cd $DEV && LD_LIBRARY_PATH=/vendor/lib64 $*" </dev/null; }  # vendor OpenCL driver
bench() { local b=$1; shift; sh_dev $b/llama-bench -m model.gguf -o json "$@"; }

# usage: run [sustained minutes, default 15]. Results → results/exp5/android/<model>-<time>/
run() {
  local minutes=${1:-15} R
  [ "$(adb get-state 2>/dev/null)" = device ] || { echo "no adb device (USB debugging on? Mac authorised?)"; exit 1; }
  R=$OUT/$(adb shell getprop ro.product.model | tr -d '\r ')-$(date +%Y%m%d-%H%M)
  mkdir -p $R/answers $R/sustained
  { for p in ro.product.model ro.soc.model ro.build.version.release; do echo "$p=$(adb shell getprop $p)"; done
    adb shell 'grep -m1 Features /proc/cpuinfo; grep -E "MemTotal|MemAvailable" /proc/meminfo; dumpsys battery'
  } > $R/device.txt
  cat $R/device.txt

  echo "== push"
  python3 -I - "$ROOT/prompts/prompts.json" "$A/prompts2k" <<'PY'
import json, pathlib, sys
from transformers import AutoTokenizer
meta = json.load(open(sys.argv[1]))
tok = AutoTokenizer.from_pretrained(meta["model"])
out = pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=True)
for it in meta["items"]:
    if it["budget"] == 2000:  # already chat-templated (thinking off): fed as-is, special tokens parsed
        (out / f"{it['qid']}.txt").write_text(tok.decode(it["ids"]))
PY
  adb shell mkdir -p $DEV/opencl $DEV/cpu
  for v in opencl cpu; do adb push $A/build-$v/bin/llama-bench $A/build-$v/bin/llama-completion $DEV/$v/; done
  adb push $GGUF $DEV/model.gguf
  adb push $A/prompts2k $DEV/
  cp $GGUF.source $R/model.txt

  echo "== battery power log (1 Hz): current µA, voltage µV, temperature 0.1 °C"
  if adb shell cat /sys/class/power_supply/battery/current_now >/dev/null 2>&1; then
    adb shell "cd /sys/class/power_supply/battery && nohup sh -c 'while true; do echo \$(date +%s) \$(cat current_now) \$(cat voltage_now) \$(cat temp); sleep 1; done' > $DEV/power.log 2>&1 &"
  else
    echo "not readable without root: no energy data" | tee $R/power_unavailable.txt
  fi

  local B=opencl ngl=99
  if bench opencl -ngl 99 -p 64 -n 8 -r 1 > $R/gpu_check.json 2> $R/gpu_check.log; then
    echo "== GPU (OpenCL): prefill at the four prompt lengths, then 128-token decode at those context depths"
    bench opencl -ngl 99 -p $PROMPT_TOK -n 0 -r 3 > $R/bench_gpu_prefill.json
    bench opencl -ngl 99 -p 0 -n 128 -d $PROMPT_TOK -r 3 > $R/bench_gpu_decode.json
  else
    echo "OpenCL build failed to run (see gpu_check.log): CPU only from here"; B=cpu ngl=0
  fi
  echo "== CPU only: prefill at the four prompt lengths, decode at 2K depth"
  bench cpu -p $PROMPT_TOK -n 0 -r 2 > $R/bench_cpu_prefill.json || echo "  CPU prefill bench failed"
  bench cpu -p 0 -n 128 -d 2156 -r 2 > $R/bench_cpu_decode.json || echo "  CPU decode bench failed"
  echo "$B" > $R/backend.txt

  echo "== 20 x 2K answers (greedy, 128 tokens, EOS ignored like the Mac/iPhone protocol)"
  for f in $A/prompts2k/*.txt; do
    q=$(basename $f .txt)
    sh_dev $B/llama-completion -m model.gguf -ngl $ngl -c 4096 -f prompts2k/$q.txt -n 128 --ignore-eos --temp 0 \
      -no-cnv -sp --no-display-prompt > $R/answers/$q.txt 2> $R/answers/$q.log || echo "  $q failed"
    grep -E "prompt eval time|eval time" $R/answers/$q.log | tail -2 || true
  done

  echo "== sustained: first 2K prompt in a loop for $minutes min"
  local q end i=0
  q=$(ls $A/prompts2k | head -1 | sed 's/.txt//')
  end=$(( $(date +%s) + minutes * 60 ))
  while [ $(date +%s) -lt $end ]; do
    { date +%s; adb shell 'dumpsys thermalservice | grep -m1 "Thermal Status"' || true; } > $R/sustained/$i.meta
    sh_dev $B/llama-completion -m model.gguf -ngl $ngl -c 4096 -f prompts2k/$q.txt -n 128 --ignore-eos --temp 0 \
      -no-cnv --no-display-prompt > /dev/null 2> $R/sustained/$i.log || echo "  iteration $i failed"
    echo "$i $(grep -E 'prompt eval time|  eval time' $R/sustained/$i.log | awk -F', ' '{print $2}' | tr '\n' ' ')"
    i=$((i + 1))
  done

  adb shell "pkill -f 'cat current_now'" || true
  adb pull $DEV/power.log $R/ 2>/dev/null || true
  echo "results: $R"
}

clean() {
  adb shell "pkill -f 'cat current_now'; rm -rf $DEV" || true
  echo "removed $DEV from the phone; turn off USB debugging in Developer options when done"
}

"$@"
