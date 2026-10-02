#!/usr/bin/env bash
# Build the GPUIX 0.10.0 native addon with the primary-seat GPUI patch
# (patches/gpui-primary-seat.patch) and copy it to vendor/.
#
# Stock @gpuix/native 0.10.0 binds every wl_seat and keeps the last one, so on
# desktops with extra (virtual) seats the window receives no physical input.
#
#   GPUIX_SRC=~/repos/gpuix bash scripts/build-addon.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${1:-${GPUIX_SRC:-$HOME/repos/gpuix}}"
tag="@gpuix/react@0.10.0"
patch="$here/patches/gpui-primary-seat.patch"

if [ ! -d "$src/.git" ]; then
  git clone --depth 1 --branch "$tag" https://github.com/remorses/gpuix "$src"
fi
if [ ! -f "$src/zed/Cargo.toml" ]; then
  git -C "$src" submodule update --init --depth 1 zed
fi
test "$(git -C "$src/zed" rev-parse HEAD)" = '81c99f816b4a5f69d3c014774068034c24d1d7af'
test "$(git -C "$src" rev-parse HEAD)" = "$(git -C "$src" rev-parse "$tag^{commit}")"

client="crates/gpui_linux/src/linux/wayland/client.rs"
if ! grep -q 'ignoring additional wl_seat global' "$src/zed/$client"; then
  git -C "$src/zed" apply "$patch"
fi

(cd "$src" && bun install --frozen-lockfile)
(cd "$src/packages/native" && bunx napi build --platform --release --no-default-features)

mkdir -p "$here/vendor"
cp "$src/packages/native/gpuix-native.linux-x64-gnu.node" "$here/vendor/"
echo "vendor/gpuix-native.linux-x64-gnu.node ($(du -h "$here/vendor/gpuix-native.linux-x64-gnu.node" | cut -f1))"
