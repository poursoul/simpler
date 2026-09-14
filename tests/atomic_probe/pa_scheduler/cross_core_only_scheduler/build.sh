#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BACKEND="${1:-cpu}"
VARIANT="${2:-swimlane}"
case "$BACKEND" in cpu|ccec) ;; *) exit 2 ;; esac
case "$VARIANT" in
  swimlane) OBS=(-DPA_BUILD_SWIMLANE=1 -DPA_BUILD_ATOMIC_SWIMLANE=1 -DPA_BUILD_PERF_CLOCK=0) ;;
  perf-clock) OBS=(-DPA_BUILD_SWIMLANE=0 -DPA_BUILD_ATOMIC_SWIMLANE=0 -DPA_BUILD_PERF_CLOCK=1) ;;
  *) exit 2 ;;
esac
OUT="$ROOT/build/$BACKEND/$VARIANT"
mkdir -p "$OUT"
DEFINES=(-DPTO_FDWIC_SHARED_MAP=1 -DPTO_FDWIC_TENSORMAP_RING_CAP=128
  -DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=1 -DPA_BUILD_COMPACT_GENERIC_TRACE=0
  -DPA_BUILD_SUBMIT_PMU=0 "${OBS[@]}")
CXX_BIN="${CXX:-g++}"
if [[ "$BACKEND" == cpu ]]; then
  "$CXX_BIN" -O2 -std=c++17 -pthread -Wall -Wextra -Werror "${DEFINES[@]}" \
    "$ROOT/cpu/main.cpp" "$ROOT/common/host_prepare.cpp" -o "$OUT/pa_scheduler_cpu"
else
  : "${ASCEND_HOME_PATH:?source the existing CANN environment first}"
  for tool in readelf rg awk; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "Required A5 build-check tool is missing: $tool" >&2
      exit 2
    fi
  done
  PTO_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"
  CCEC="$ASCEND_HOME_PATH/bin/ccec"
  # Mixed ELF holds two distinct 1152-byte role objects (2304 bytes total)
  # in one 4 KiB block-local reservation; each Scalar uses only its own role.
  FLAGS=(-c -O3 -g -x cce -Wall -std=c++17 --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-block-local-reserve-size=0x1000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -I"$ROOT/common" -isystem "$PTO_ROOT/include" "${DEFINES[@]}")
  "$CCEC" "${FLAGS[@]}" --cce-aicore-arch=dav-c310-cube -DPA_BUILD_AIC \
    "$ROOT/ccec/kernel.cpp" -o "$OUT/aic.o"
  "$CCEC" "${FLAGS[@]}" --cce-aicore-arch=dav-c310-vec -DPA_BUILD_AIV \
    "$ROOT/ccec/kernel.cpp" -o "$OUT/aiv.o"
  "$ASCEND_HOME_PATH/bin/ld.lld" -m aicorelinux -Ttext=0 -static \
    --version-script="$ROOT/ccec/pa_scheduler_device_exports.map" \
    "$OUT/aic.o" "$OUT/aiv.o" -o "$OUT/pa_scheduler_kernel.o"
  if readelf --symbols --wide "$OUT/pa_scheduler_kernel.o" |
     rg 'FUNC.*(RunSchedulerImpl|BuildDispatchedSharedTask|FinishSharedWinnerSubmitBody|PublishCrossCoreExecTask)' ; then
    echo "Unexpected device Build code in scheduler-only ELF" >&2
    exit 1
  fi
  for role in aic aiv; do
    readelf --symbols --wide "$OUT/pa_scheduler_kernel.o" |
      awk -v name="pa_only_scheduler_stats_$role" '
        $4 == "OBJECT" && $8 == name && $3 == 1152 { found++ }
        END { if (found != 1) exit 1 }
      '
  done
  "$CXX_BIN" -O2 -std=c++17 -pthread -Wall -Wextra -Werror -Wno-deprecated-declarations \
    "${DEFINES[@]}" -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/pkg_inc" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime" -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    "$ROOT/ccec/host.cpp" "$ROOT/common/host_prepare.cpp" \
    -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -lascendcl -lruntime -ldl -o "$OUT/pa_scheduler_host"
fi
echo "[BUILT] $BACKEND/$VARIANT (8 AIC + 8 AIV; no device Build)"
