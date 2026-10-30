#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python -m tracesplit serve --root . --port 8765
