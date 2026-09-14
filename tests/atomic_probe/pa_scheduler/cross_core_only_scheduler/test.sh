#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Run bash $ROOT/setup.sh first." >&2
  exit 2
fi
exec "$ROOT/.venv/bin/python" -m unittest discover -s "$ROOT" -p 'test_*.py' -v
