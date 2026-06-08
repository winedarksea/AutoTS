#!/usr/bin/env bash
# Build the AutoTS wheel and place it where the PWA worker can fetch it.
#
# Usage:
#   scripts/build_wheel.sh [DEST_DIR] [--force]
#
# DEST_DIR defaults to ./dist (so `trunk serve` serves it at /autots.whl).
# The wheel is cached under pwa/.wheel so repeated `trunk serve` builds are fast;
# pass --force to rebuild.
set -euo pipefail

PWA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PWA_DIR/.." && pwd)"
# When invoked by Trunk, write into the staging dir Trunk will publish as dist/.
DEST_DIR="${1:-${TRUNK_STAGING_DIR:-$PWA_DIR/dist}}"
CACHE_DIR="$PWA_DIR/.wheel"
FORCE="${2:-}"

mkdir -p "$CACHE_DIR" "$DEST_DIR"

if [[ "$FORCE" == "--force" ]]; then
  rm -f "$CACHE_DIR"/autots-*.whl
fi

if ! ls "$CACHE_DIR"/autots-*.whl >/dev/null 2>&1; then
  echo "Building AutoTS wheel (pure-Python, no deps)…"
  python -m pip wheel "$REPO_ROOT" --no-deps -w "$CACHE_DIR" >/dev/null
fi

# micropip requires the real PEP-427 wheel filename, so preserve the basename
# and publish a tiny manifest the client reads to discover the URL.
WHEEL="$(ls -t "$CACHE_DIR"/autots-*.whl | head -1)"
WHEEL_NAME="$(basename "$WHEEL")"
cp "$WHEEL" "$DEST_DIR/$WHEEL_NAME"
# Relative URL so the worker can fetch the wheel under any base path (e.g. /forecasting/app/).
printf '{"url": "%s"}\n' "$WHEEL_NAME" > "$DEST_DIR/autots_wheel.json"
echo "Wheel ready: $DEST_DIR/$WHEEL_NAME  (manifest: autots_wheel.json)"
