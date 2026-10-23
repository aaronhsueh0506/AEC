#!/usr/bin/env bash
# pyc_parity_aec.sh -- Python<->C EFFECT parity for the AEC, in two layers (see test/pyc_parity/pyc_parity.py).
#
#   Layer 1  std-math C build  vs Python (mic-path HPF emulated in float32, bit-exact to audiolib's)
#            std-math build = SIMD=0 + -DUSE_STANDARD_MATH (no optimisation-only macros exist in this repo)
#   Layer 2  shipping build (Makefile default: SIMD=1, fast_math)  vs  the std-math build
#
# Grids 8k/256, 16k/256, 16k/512, 48k/1024; presets balanced, aggressive (mild sampled); inputs: the tracked
# aec_record recording (+ 8k/48k resamples) and seeded synthetic echo / double-talk scenes; observer off/on,
# mic HPF on/off, context seam, plus a regenerate-and-diff check of aec_balanced_config.h.
#
# Known gaps are registered as EXPECTED_RED in test/pyc_parity/pyc_parity.py; an expected-red cell that turns green FAILS
# the run (remove the marker). Tolerances are frozen there, and every cell prints the measured numbers beside them.
#
#   usage: pyc_parity_aec.sh [--full | --quick] [--only SUBSTR[,SUBSTR]] [--skip-static] [--json FILE] [--work DIR]
#                            [--keep-work] [--aec-root DIR] [--repeat-check] [--write-pins]
#          make test-pyc-parity [PYC_ARGS="--full"]
#   env  : PYTHON  interpreter with numpy, scipy, soundfile (default python3)
#
#   (default)       the lean suite: 10 scenes, one per mechanism a planted regression is known to move
#   --full          every scenario (the full matrix; `make test-full` runs this)
#   --quick         an older, narrower subset; no aggressive-preset scene and its observer scene misses a +1 hop shift
#   --only          substring filter over the full matrix (comma separated = any)
#   --repeat-check  run the whole evaluation twice from fresh run directories and require byte-identical JSON
#   --write-pins    (maintainers) rewrite test/pyc_parity/inputs.sha256 from the current environment
#
# Everything is built and run under a scratch directory (default: a mktemp dir, removed afterwards unless
# --keep-work); nothing is written into the working tree.
# Exit: 0 pass (incl. EXPECTED_RED cells), 1 a cell failed, 2 setup error.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
AEC_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON=${PYTHON:-python3}
QUICK=""; FULL=""; ONLY=""; JSON=""; WORK=""; KEEP=0; REPEAT=0; WRITEPINS=""; SKIPSTATIC=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --quick) QUICK="--quick" ;;
        --full) FULL="--full" ;;
        --only) ONLY=$2; shift ;;
        --json) JSON=$2; shift ;;
        --work) WORK=$2; KEEP=1; shift ;;
        --keep-work) KEEP=1 ;;
        --aec-root) AEC_ROOT=$(cd "$2" && pwd); shift ;;
        --repeat-check) REPEAT=1 ;;
        --write-pins) WRITEPINS="--write-pins" ;;
        --skip-static) SKIPSTATIC="--skip-static" ;;
        -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1 (try -h)" >&2; exit 2 ;;
    esac
    shift
done
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
C_IMPL=$AEC_ROOT/c_impl
DRV=$C_IMPL/test/pyc_parity/pyc_parity.py

"$PYTHON" -c 'import numpy, scipy, soundfile' 2>/dev/null || { echo "FAIL(setup): $PYTHON needs numpy, scipy and soundfile" >&2; exit 2; }
[ -f "$DRV" ] || { echo "FAIL(setup): $DRV not found" >&2; exit 2; }

if [ -z "$WORK" ]; then WORK=$(mktemp -d "${TMPDIR:-/tmp}/pyc_parity_aec.XXXXXX") || exit 2; fi
mkdir -p "$WORK" || exit 2
WORK=$(cd "$WORK" && pwd)
cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$WORK"; }
trap cleanup EXIT

build() {   # $1 label, rest = make variables; prints the directory holding pyc_aec_run
    local label=$1; shift
    local o="$WORK/build_$label/obj" b="$WORK/build_$label/bin"
    if ! make -s -C "$C_IMPL" OBJ_ROOT="$o" BIN_ROOT="$b" "$@" pyc-parity-driver > "$WORK/build_$label.log" 2>&1; then
        echo "FAIL(setup): build '$label' failed; tail of $WORK/build_$label.log:" >&2
        tail -20 "$WORK/build_$label.log" >&2
        KEEP=1; exit 2
    fi
    make -s -C "$C_IMPL" OBJ_ROOT="$o" BIN_ROOT="$b" "$@" print-bin-dir
}
echo "== building (scratch: $WORK)"
STD_DIR=$(build std SIMD=0 EXTRA_CFLAGS=-DUSE_STANDARD_MATH) || exit 2
SHIP_DIR=$(build ship SIMD=1 EXTRA_CFLAGS=) || exit 2
STD_BIN=$STD_DIR/pyc_aec_run; SHIP_BIN=$SHIP_DIR/pyc_aec_run
[ -x "$STD_BIN" ] && [ -x "$SHIP_BIN" ] || { echo "FAIL(setup): driver binaries missing" >&2; exit 2; }
echo "   std-math build: SIMD=0 EXTRA_CFLAGS=-DUSE_STANDARD_MATH"
echo "   ship build    : SIMD=1 (Makefile defaults)"

drive() {   # $1 work subdir, $2 json path
    "$PYTHON" "$DRV" --aec-root "$AEC_ROOT" --work "$1" --std-bin "$STD_BIN" --ship-bin "$SHIP_BIN" \
        ${QUICK:+$QUICK} ${FULL:+$FULL} ${ONLY:+--only "$ONLY"} ${WRITEPINS:+$WRITEPINS} ${SKIPSTATIC:+$SKIPSTATIC} ${2:+--json "$2"}
}

if [ "$REPEAT" -eq 1 ]; then
    drive "$WORK/r1" "$WORK/result1.json" > "$WORK/stdout1.txt"; rc1=$?
    cat "$WORK/stdout1.txt"
    drive "$WORK/r2" "$WORK/result2.json" > "$WORK/stdout2.txt"; rc2=$?
    if cmp -s "$WORK/result1.json" "$WORK/result2.json" && cmp -s "$WORK/stdout1.txt" "$WORK/stdout2.txt"; then
        echo "DETERMINISM: two runs gave byte-identical JSON and table (sha256 $(shasum -a 256 "$WORK/result1.json" | cut -d' ' -f1))"
        det=0
    else
        echo "DETERMINISM: FAIL (the two runs differ)"; diff "$WORK/stdout1.txt" "$WORK/stdout2.txt" | head -20; det=1
    fi
    [ -n "$JSON" ] && cp "$WORK/result1.json" "$JSON"
    [ "$rc1" -ne 0 ] && exit "$rc1"; [ "$rc2" -ne 0 ] && exit "$rc2"
    exit "$det"
fi
drive "$WORK/r" "$JSON"
exit $?
