#!/bin/sh
# Builds the iPhone benchmark app (Release, device) and optionally installs it. Needs a Mac with Xcode 26+ and an Apple
# developer team: the increased-memory-limit and extended-virtual-addressing entitlements need a signed build.
#   TEAM=<your team id> [BUNDLE_ID=org.membench.ios] ios/build_bench.sh [device UDID]   (UDIDs: xcrun devicectl list devices)
# Swift packages are pinned in MemBench.xcodeproj/.../Package.resolved (mlx-swift 0.32.3, mlx-swift-lm 3.32.3).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
: "${TEAM:?set TEAM to your Apple developer team id}"
BUNDLE_ID=${BUNDLE_ID:-org.membench.ios}
DD="$HERE/build"
xcodebuild -project "$HERE/MemBench.xcodeproj" -scheme MemBench -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath "$DD" -allowProvisioningUpdates DEVELOPMENT_TEAM="$TEAM" PRODUCT_BUNDLE_IDENTIFIER="$BUNDLE_ID" build \
  | grep -E "error:|BUILD (SUCCEEDED|FAILED)"
APP="$DD/Build/Products/Release-iphoneos/MemBench.app"
[ -z "$1" ] || xcrun devicectl device install app --device "$1" "$APP" | tail -2
