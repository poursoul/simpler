#!/usr/bin/env bash
# Python tools only use the standard library; no network or pip install needed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
python3 -c 'import sys; sys.exit("Python 3.10+ is required") if sys.version_info < (3, 10) else None'
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -c 'import sys; sys.exit("Recreate .venv with Python 3.10+") if sys.version_info < (3, 10) else None'
echo "[READY] $ROOT/.venv/bin/python (standard library only)"
