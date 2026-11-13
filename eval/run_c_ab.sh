#!/bin/bash
# C-path 90-case blind A/B.
#
# eval_aec_challenge.py drives the Python AEC ("from aec import AEC"), so it
# cannot observe a C-only change; this script drives the C CLI (aec_wav).
#
# Baseline and candidate are separate git WORKTREES, each built from its own
# checkout. Nothing is swapped in the working tree, so a build cannot pick up
# the other side's sources. The worktrees must be SIBLINGS of the repo:
# example/wav_io.h resolves audiolib by relative path.
#
# NO_PREALIGN is irrelevant here -- that flag gates the Python driver's offline
# GCC-PHAT crutch. aec_wav has no pre-align path at all and always runs the
# online estimator, which is the stricter condition.
#
# Completeness checks:
#   - renders are counted by SUCCESS, not by loop iteration;
#   - the rendered, scored and manifest stem SETS must be equal, not merely the
#     same size;
#   - "byte-identical" is decided by eval/ab_compare.py from file digests and
#     per-sample statistics, never inferred from equal AECMOS deltas.
#
# Both sides are built after `make clean` with SIMD=1 WERROR=1; the run aborts
# if the two binaries are byte-identical. PYBIN selects the interpreter.
# OUT_DIR receives out_*/, res_*/, score_*.log, build_*.log and
# wav_comparison_g<fft>.json per grid.
#
# Usage:
#   eval/run_c_ab.sh BASE_WORKTREE CAND_WORKTREE OUT_DIR ["256 512"] [aec_wav args...]
#
# e.g. a change only reachable with the shadow filter disabled:
#   eval/run_c_ab.sh /tmp/wt-base /tmp/wt-cand /tmp/out "256 512" --no-shadow
set -euo pipefail

if [ "$#" -lt 3 ]; then
  sed -n '2,32p' "$0" >&2; exit 2
fi

BASE_WT="$1"; shift
CAND_WT="$1"; shift
OUT="$1"; shift
GRIDS="${1:-256 512}"
if [ "$#" -gt 0 ]; then shift; fi
# Forwarded verbatim to aec_wav. The ${EXTRA[@]+...} guard in render is needed:
# under `set -u` bash 3.2 (the macOS system shell) treats an empty array
# expansion as unbound.
EXTRA=("$@")

AEC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SE=$(dirname "$AEC")
COMMON_DIR="$SE/audiolib/common"
DATASET="$AEC/wav/aec_challenge_blind"
STEMS="$AEC/eval/manifest_90case_stems.txt"
PYBIN="${PYBIN:-$SE/.venv/bin/python3.9}"
command -v "$PYBIN" >/dev/null 2>&1 || PYBIN=python3

for d in "$COMMON_DIR" "$DATASET"; do
  [ -d "$d" ] || { echo "missing: $d" >&2; exit 1; }
done
[ -f "$STEMS" ] || { echo "missing manifest: $STEMS" >&2; exit 1; }
WANT=$(grep -cve '^[[:space:]]*$' -e '^#' "$STEMS")

mkdir -p "$OUT"
echo "harness:  $AEC/eval/run_c_ab.sh"
echo "manifest: $STEMS ($WANT cases)"
echo "extra:    ${EXTRA[*]:-<none>}"

build () {  # build <worktree> -> echoes the aec_wav path
  local wt="$1"
  # WERROR=1: a warning introduced by the candidate must stop the run.
  make -C "$wt/c_impl" clean >/dev/null 2>&1 || true
  make -C "$wt/c_impl" SIMD=1 WERROR=1 COMMON_DIR="$COMMON_DIR" \
      >"$OUT/build_$(basename "$wt").log" 2>&1 || {
    echo "make failed in $wt -- see $OUT/build_$(basename "$wt").log" >&2
    exit 1; }
  # `make clean` above guarantees no stale object survives between builds.
  local bin
  bin=$(ls -t "$wt"/c_impl/bin/ne10-*/aec_wav 2>/dev/null | head -1)
  [ -n "$bin" ] && [ -x "$bin" ] || { echo "no aec_wav under $wt" >&2; exit 1; }
  echo "$bin"
}

render () {  # render <aec_wav> <label> <fft>
  local exe="$1" label="$2" fft="$3"
  local dir="$OUT/out_$label"
  rm -rf "$dir"; mkdir -p "$dir"
  local n=0
  while read -r stem; do
    case "$stem" in ""|\#*) continue;; esac
    # Manifest stems already carry the scenario suffix; the scenario DIRECTORY
    # is the stem's scenario with any _with_movement suffix stripped.
    local sc="${stem#*_}"
    sc="${sc%_with_movement}"
    local mic="$DATASET/$sc/${stem}_mic.wav"
    local lpb="$DATASET/$sc/${stem}_lpb.wav"
    [ -f "$mic" ] && [ -f "$lpb" ] || { echo "  missing source: $stem" >&2; continue; }
    if "$exe" "$mic" "$lpb" "$dir/${stem}_ours.wav" --level 1 --cng \
           --fft-size "$fft" ${EXTRA[@]+"${EXTRA[@]}"} >/dev/null 2>&1; then
      n=$((n+1))
    else
      echo "  RENDER FAILED: $stem" >&2
    fi
  done < "$STEMS"
  if [ "$n" -ne "$WANT" ]; then
    echo "ABORT: rendered $n of $WANT cases for $label" >&2; exit 1
  fi
  echo "  rendered $n/$WANT -> $dir"
}

score () {  # score <label> [baseline-scores.json]
  local label="$1" base="${2:-}"
  local args=(--label "$label")
  [ -n "$base" ] && args+=(--baseline "$base")
  mkdir -p "$OUT/res_$label"
  ( cd "$AEC" && "$PYBIN" python/bench_aecmos.py "$OUT/out_$label" \
      "$OUT/res_$label" "${args[@]}" ) > "$OUT/score_$label.log" 2>&1
  echo "  scored $label"
}

echo "building baseline..."; BASE_EXE=$(build "$BASE_WT")
echo "building candidate..."; CAND_EXE=$(build "$CAND_WT")
echo "  base: $BASE_EXE"
echo "  cand: $CAND_EXE"
if cmp -s "$BASE_EXE" "$CAND_EXE"; then
  echo "ABORT: the two builds produced byte-identical binaries -- the worktrees" \
       "do not differ, so this run would compare a commit against itself" >&2
  exit 1
fi

for fft in $GRIDS; do
  echo "grid fft=$fft"
  render "$BASE_EXE" "base_g$fft" "$fft"
  render "$CAND_EXE" "cand_g$fft" "$fft"
  score "base_g$fft"
  score "cand_g$fft" "$OUT/res_base_g$fft/scores.json"
  # The gate: stem-set equality across manifest/renders/scores, then a
  # per-sample comparison of every output pair.
  "$PYBIN" "$AEC/eval/ab_compare.py" \
      --manifest "$STEMS" \
      --base-dir "$OUT/out_base_g$fft" --cand-dir "$OUT/out_cand_g$fft" \
      --scores-base "$OUT/res_base_g$fft/scores.json" \
      --scores-cand "$OUT/res_cand_g$fft/scores.json" \
      --out "$OUT/wav_comparison_g$fft.json"
done
echo "DONE -> $OUT"
