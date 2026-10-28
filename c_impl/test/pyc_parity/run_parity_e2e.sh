#!/usr/bin/env bash
# run_parity_e2e.sh -- `make test-parity-e2e`: the repo's end-to-end Python<->C gate (test/parity_aec_e2e.c).
#
#   usage: run_parity_e2e.sh <checker-binary> <aec-root> <workdir> [--json FILE]
#   env  : PYTHON           interpreter with numpy/scipy/soundfile (default python3)
#
# What it does
#   1. Verifies the source wavs against test/golden/parity_e2e_inputs.sha256 (missing file or hash mismatch => FAIL;
#      wav/aec_challenge_blind/ is git-ignored, so the pin is the only link).
#   2. Regenerates every golden with python/diag/gen_aec_e2e_golden.py into <workdir> (never into git) and replays it through
#      the C checker. Pass/fail is the checker's own exit code; its TOL_E2E / TOL_E2E_48K are not re-implemented here.
#   3. 16 kHz and 48 kHz rows must PASS. The 8 kHz row is the expected red (it exceeds the checker's tolerance): it is reported
#      as EXPECTED_RED with the measured numbers, and if it turns green this script FAILS so the marker gets removed.
#   Not covered here: 16 kHz fft=512 (the generator has no --fft; see test/pyc_parity_aec.sh).
#
# Exit: 0 = every row as expected; 1 = a row failed / an expected-red turned green; 2 = setup error.
set -u
if [ "$#" -lt 3 ]; then
    echo "usage: $0 <checker-binary> <aec-root> <workdir> [--json FILE]" >&2
    exit 2
fi
CHECKER=$1; AEC_ROOT=$2; WORK=$3; shift 3
JSON_OUT=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --json) JSON_OUT=${2:-}; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
PYTHON=${PYTHON:-python3}
WAV_ROOT=$AEC_ROOT/wav          # the generator reads <aec-root>/wav and cannot be redirected
PIN=$AEC_ROOT/c_impl/test/golden/parity_e2e_inputs.sha256
GEN=$AEC_ROOT/python/diag/gen_aec_e2e_golden.py
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

if command -v shasum >/dev/null 2>&1; then SHA() { shasum -a 256 "$1" | cut -d' ' -f1; }
else SHA() { sha256sum "$1" | cut -d' ' -f1; }; fi

[ -x "$CHECKER" ] || { echo "FAIL(setup): checker binary not found: $CHECKER" >&2; exit 2; }
[ -f "$PIN" ] || { echo "FAIL(setup): pin file not found: $PIN" >&2; exit 2; }
[ -f "$GEN" ] || { echo "FAIL(setup): golden generator not found: $GEN" >&2; exit 2; }
mkdir -p "$WORK" || exit 2

# ---- 1. pinned inputs -------------------------------------------------------------------------
echo "== inputs (pin: test/golden/parity_e2e_inputs.sha256, wav root: <aec-root>/wav)"
pin_bad=0
while read -r want rel; do
    case "$want" in ''|'#'*) continue ;; esac
    f=$WAV_ROOT/$rel
    if [ ! -f "$f" ]; then
        echo "  MISSING  $rel"
        echo "           pinned source wav not found under <aec-root>/wav. wav/aec_challenge_blind/ is a" >&2
        echo "           git-ignored local dataset: restore it (see wav/download_blind_test.py)." >&2
        pin_bad=1; continue
    fi
    got=$(SHA "$f")
    if [ "$got" != "$want" ]; then
        echo "  MISMATCH $rel"
        echo "           expected $want" >&2
        echo "           got      $got" >&2
        echo "           the golden would be generated from different audio than the one the gate was" >&2
        echo "           calibrated on; refusing to run." >&2
        pin_bad=1
    else
        echo "  ok       $rel  sha256=${got}"
    fi
done < "$PIN"
if [ "$pin_bad" -ne 0 ]; then echo "FAIL: pinned input check failed"; exit 1; fi

# ---- 2/3. rows --------------------------------------------------------------------------------
# id | sr | level | extra flags (given to BOTH generator and checker) | expected
ROWS="e2e_16k_balanced|16000|balanced||GREEN
e2e_16k_balanced_nores|16000|balanced|--no-res|GREEN
e2e_16k_balanced_observer|16000|balanced|--live-observer|GREEN
e2e_48k_balanced|48000|balanced||GREEN
e2e_8k_balanced|8000|balanced||RED:S-19"

echo
printf '%-26s %6s %9s %12s %12s %4s  %s\n' row sr tol linear_max out_max rc status
fail=0
JSON_ROWS=""
while IFS='|' read -r id sr level flags expect; do
    g=$WORK/golden_$id.bin
    log=$WORK/$id.log
    if ! "$PYTHON" "$GEN" "$g" "$level" --sr "$sr" $flags > "$log.gen" 2>&1; then
        echo "FAIL(setup): golden generation failed for $id (see $log.gen)"; tail -5 "$log.gen" >&2; fail=1; continue
    fi
    "$CHECKER" "$g" "$level" $flags > "$log" 2>&1; rc=$?
    tol=$(sed -n 's/.*(tol=\([^)]*\)).*/\1/p' "$log" | head -1)
    lin=$(sed -n 's/^  raw_output(linear) mism=[0-9]* (max=\([^ ]*\) .*/\1/p' "$log" | head -1)
    outm=$(sed -n 's/^  >>> out\[hop\] mism=[0-9]* (max=\([^ ]*\) .*/\1/p' "$log" | head -1)
    case "$expect:$rc" in
        GREEN:0)   status=PASS ;;
        GREEN:1)   status="FAIL (exceeds tolerance)"; fail=1 ;;
        RED:S-19:1) status="EXPECTED_RED(S-19)" ;;
        RED:S-19:0) status="FAIL UNEXPECTED_GREEN(S-19): 8 kHz now within tolerance; remove the marker"; fail=1 ;;
        *)         status="FAIL (checker rc=$rc, setup/replay error; see $log)"; fail=1 ;;
    esac
    printf '%-26s %6s %9s %12s %12s %4s  %s\n' "$id" "$sr" "${tol:--}" "${lin:--}" "${outm:--}" "$rc" "$status"
    JSON_ROWS="${JSON_ROWS}${JSON_ROWS:+,}
    {\"id\":\"$id\",\"sr\":$sr,\"preset\":\"$level\",\"flags\":\"$flags\",\"tol\":\"${tol:-}\",\"linear_max\":\"${lin:-}\",\"out_max\":\"${outm:-}\",\"checker_rc\":$rc,\"status\":\"$status\"}"
done <<EOF
$ROWS
EOF

echo
echo "tolerances are the checker's own constants (TOL_E2E=2e-2, TOL_E2E_48K=1e-1); this wrapper never edits them."
echo "EXPECTED_RED(S-19): the 8 kHz row exceeds TOL_E2E; classification (sync bug vs documented HPF f32/fp64 effect) and"
echo "the tolerance are the user's decision (PY_C_SYNC_AUDIT.md S-19)."
if [ -n "$JSON_OUT" ]; then
    printf '{"gate":"parity_aec_e2e","rows":[%s\n  ]}\n' "$JSON_ROWS" > "$JSON_OUT"
fi
if [ "$fail" -ne 0 ]; then echo "RESULT: FAIL"; exit 1; fi
echo "RESULT: PASS (8 kHz is EXPECTED_RED(S-19))"
exit 0
