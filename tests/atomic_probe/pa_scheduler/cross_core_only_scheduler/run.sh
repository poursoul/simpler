#!/usr/bin/env bash
# B256 only. Source the existing CANN environment before using ccec.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
COMMAND="${1:-swimlane}"
BACKEND="${2:-ccec}"
shift "$(( $# >= 2 ? 2 : $# ))"
case "$BACKEND" in cpu|ccec) ;; *) echo "Expected cpu|ccec" >&2; exit 2 ;; esac
case "$COMMAND" in
  build) exec bash "$ROOT/build.sh" "$BACKEND" swimlane ;;
  build-perf-clock) exec bash "$ROOT/build.sh" "$BACKEND" perf-clock ;;
  swimlane|run) VARIANT=swimlane ;;
  perf-clock) VARIANT=perf-clock ;;
  *) echo "Usage: $0 build|build-perf-clock|swimlane|run|perf-clock cpu|ccec [options]" >&2; exit 2 ;;
esac
OUT="$ROOT/build/$BACKEND/$VARIANT"
ARGS=(--batches 256 --runs 1)
if [[ "$BACKEND" == cpu ]]; then
  BIN="$OUT/pa_scheduler_cpu"
else
  BIN="$OUT/pa_scheduler_host"
  ARGS+=(--kernel "$OUT/pa_scheduler_kernel.o")
  if ! command -v task-submit >/dev/null 2>&1; then
    echo "[WARN] task-submit absent; this run is unisolated. Confirm device availability before running." >&2
  fi
fi
if [[ ! -x "$BIN" ]]; then
  echo "Build first: bash $ROOT/build.sh $BACKEND $VARIANT" >&2
  exit 2
fi
run_binary() {
  if [[ "$BACKEND" == ccec ]] && command -v task-submit >/dev/null 2>&1; then
    bash "$ROOT/tools/check_onboard_arch.sh" a5
    local device=0 previous="" argument quoted
    for argument in "$@"; do
      if [[ "$previous" == --device ]]; then device="$argument"; fi
      previous="$argument"
    done
    printf -v quoted '%q ' "$BIN" "$@"
    task-submit --device "$device" --timeout 180 --max-time 180 --run "$quoted"
  else
    "$BIN" "$@"
  fi
}
if [[ "$VARIANT" == perf-clock ]]; then
  run_binary "${ARGS[@]}" --no-swimlane "$@"
  exit
fi
if [[ "$COMMAND" == run ]]; then
  run_binary "${ARGS[@]}" "$@"
  exit
fi
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Run bash $ROOT/setup.sh first to create this directory's .venv." >&2
  exit 2
fi
# Keep intermediate data under ignored build/, not test_record/.
CAPTURE="$(mktemp -d "$OUT/capture.XXXXXX")"
run_binary "${ARGS[@]}" "$@" --swimlane-json "$CAPTURE/raw.json"
"$PYTHON" "$ROOT/swimlane_converter.py" "$CAPTURE/raw.json" -o "$CAPTURE/swimlane.json"
echo "[OUTPUT] $CAPTURE/swimlane.json"
