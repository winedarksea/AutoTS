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
OPENPYXL_VERSION="3.1.5"
ET_XMLFILE_VERSION="2.0.0"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python is required to build the PWA wheel." >&2
  exit 1
fi

mkdir -p "$CACHE_DIR" "$DEST_DIR"

if [[ "$FORCE" == "--force" ]]; then
  rm -f "$CACHE_DIR"/autots-*.whl \
    "$CACHE_DIR"/openpyxl-*.whl \
    "$CACHE_DIR"/et_xmlfile-*.whl
fi

WHEEL="$(ls -t "$CACHE_DIR"/autots-*.whl 2>/dev/null | head -1 || true)"
if [[ -z "$WHEEL" ]] ||
   find "$REPO_ROOT/autots" "$REPO_ROOT/setup.py" "$REPO_ROOT/pyproject.toml" \
     -type f -newer "$WHEEL" -print -quit | grep -q .; then
  echo "Building AutoTS wheel (pure-Python, no deps)…"
  rm -f "$CACHE_DIR"/autots-*.whl
  "$PYTHON_BIN" -m pip wheel \
    --no-deps \
    --wheel-dir "$CACHE_DIR" \
    "$REPO_ROOT" >/dev/null
fi

# micropip requires the real PEP-427 wheel filename, so preserve the basename
# and publish a tiny manifest the client reads to discover the URL.
WHEEL="$(ls -t "$CACHE_DIR"/autots-*.whl | head -1)"
WHEEL_NAME="$(basename "$WHEEL")"
cp "$WHEEL" "$DEST_DIR/$WHEEL_NAME"

if ! ls "$CACHE_DIR"/openpyxl-"$OPENPYXL_VERSION"-*.whl >/dev/null 2>&1 ||
   ! ls "$CACHE_DIR"/et_xmlfile-"$ET_XMLFILE_VERSION"-*.whl >/dev/null 2>&1; then
  echo "Downloading pinned Excel reader wheels…"
  "$PYTHON_BIN" -m pip download \
    --only-binary=:all: \
    --no-deps \
    --dest "$CACHE_DIR" \
    "openpyxl==$OPENPYXL_VERSION" \
    "et-xmlfile==$ET_XMLFILE_VERSION" >/dev/null
fi

OPENPYXL_WHEEL="$(ls "$CACHE_DIR"/openpyxl-"$OPENPYXL_VERSION"-*.whl | head -1)"
ET_XMLFILE_WHEEL="$(ls "$CACHE_DIR"/et_xmlfile-"$ET_XMLFILE_VERSION"-*.whl | head -1)"
OPENPYXL_NAME="$(basename "$OPENPYXL_WHEEL")"
ET_XMLFILE_NAME="$(basename "$ET_XMLFILE_WHEEL")"
cp "$OPENPYXL_WHEEL" "$DEST_DIR/$OPENPYXL_NAME"
cp "$ET_XMLFILE_WHEEL" "$DEST_DIR/$ET_XMLFILE_NAME"

"$PYTHON_BIN" - "$DEST_DIR/autots_wheel.json" "$WHEEL_NAME" "$ET_XMLFILE_NAME" "$OPENPYXL_NAME" <<'PY'
import json
import sys

manifest_path, autots_wheel, et_xmlfile_wheel, openpyxl_wheel = sys.argv[1:]
with open(manifest_path, "w", encoding="utf-8") as manifest_file:
    json.dump(
        {
            "url": autots_wheel,
            "dependencies": [
                {"name": "et-xmlfile", "url": et_xmlfile_wheel},
                {"name": "openpyxl", "url": openpyxl_wheel},
            ],
        },
        manifest_file,
        indent=2,
    )
    manifest_file.write("\n")
PY

"$PYTHON_BIN" - "$DEST_DIR" <<'PY'
import hashlib
import json
import os
import sys

destination = os.path.abspath(sys.argv[1])
assets = []
version_hasher = hashlib.sha256()
for directory, _, filenames in os.walk(destination):
    for filename in sorted(filenames):
        path = os.path.join(directory, filename)
        relative_path = os.path.relpath(path, destination).replace(os.sep, "/")
        if relative_path == "offline_assets.json":
            continue
        assets.append(relative_path)
        version_hasher.update(relative_path.encode("utf-8"))
        with open(path, "rb") as asset_file:
            for chunk in iter(lambda: asset_file.read(1024 * 1024), b""):
                version_hasher.update(chunk)

with open(
    os.path.join(destination, "offline_assets.json"),
    "w",
    encoding="utf-8",
) as manifest_file:
    json.dump(
        {
            "version": version_hasher.hexdigest()[:16],
            "assets": sorted(assets),
        },
        manifest_file,
        indent=2,
    )
    manifest_file.write("\n")

service_worker_path = os.path.join(destination, "service_worker.js")
if os.path.exists(service_worker_path):
    with open(service_worker_path, "r", encoding="utf-8") as service_worker_file:
        service_worker = service_worker_file.read()
    service_worker = service_worker.replace(
        "__AUTOTS_CACHE_VERSION__",
        version_hasher.hexdigest()[:16],
    )
    with open(service_worker_path, "w", encoding="utf-8") as service_worker_file:
        service_worker_file.write(service_worker)
PY
echo "Wheel ready: $DEST_DIR/$WHEEL_NAME  (manifest: autots_wheel.json)"
