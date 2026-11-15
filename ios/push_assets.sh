#!/bin/bash
# Copy benchmark assets from this Mac into the MemBench app's Documents on a device (the app must be installed):
# prompts.json + model snapshots from the Hugging Face cache (byte-identical to the weights behind the Mac numbers).
# usage: [BUNDLE_ID=org.membench.ios] ios/push_assets.sh <device-udid> [hf-repo ...]     default repo: mlx-community/Qwen3-4B-4bit
# device ids: xcrun devicectl list devices
set -euo pipefail
dev=$1; shift
repos=("${@:-mlx-community/Qwen3-4B-4bit}")
root=$(cd "$(dirname "$0")/.." && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/models"
cp "$root/prompts/prompts.json" "$stage/"
for repo in "${repos[@]}"; do
  hub=~/.cache/huggingface/hub/models--${repo//\//--}
  rev=$(cat "$hub/refs/main")
  name=${repo#*/}
  cp -cRL "$hub/snapshots/$rev" "$stage/models/$name"  # APFS clone of the blobs: instant, no extra disk
  echo "$repo@$rev" > "$stage/models/$name/source.txt"
done
du -sh "$stage"
xcrun devicectl device copy to --device "$dev" --domain-type appDataContainer \
  --domain-identifier "${BUNDLE_ID:-org.membench.ios}" --source "$stage" --destination Documents
