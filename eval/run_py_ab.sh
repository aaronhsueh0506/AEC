#!/bin/bash
# Python-path 90-case blind A/B.
#
# The C twin is eval/run_c_ab.sh; run BOTH for any change that lands in both
# ports. Neither can see the other's regressions: eval_aec_challenge.py drives
# the Python AEC and cannot observe a C-only change, and aec_wav cannot observe
# a Python-only one.
#
# Baseline and candidate are separate git WORKTREES. The previous version of
# this harness swapped module sources in and out of the working tree around
# each render, which makes the tree transiently un-retimed -- indistinguishable
# from an unfinished implementation to anything else reading it, and impossible
# to attribute afterwards if the run is interrupted.
#
# NO_PREALIGN=1 is mandatory. The eval driver's DEFAULT is an offline GCC-PHAT
# pre-align crutch, which invalidates any timing-related verdict.
#
# The dataset and the manifest come from THIS checkout, not the worktrees, so
# both sides see byte-identical input.
#
# Usage:
#   eval/run_py_ab.sh BASE_WORKTREE CAND_WORKTREE OUT_DIR ["256 512"]
set -euo pipefail

if [ "$#" -lt 3 ]; then sed -n '2,21p' "$0" >&2; exit 2; fi

BASE_WT="$1"; CAND_WT="$2"; OUT="$3"; GRIDS="${4:-256 512}"

AEC=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SE=$(dirname "$AEC")
DATASET="$AEC/wav/aec_challenge_blind"
STEMS="$AEC/eval/manifest_90case_stems.txt"
PYBIN="${PYBIN:-$SE/.venv/bin/python3.9}"
command -v "$PYBIN" >/dev/null 2>&1 || PYBIN=python3
WORKERS="${WORKERS:-6}"

[ -d "$DATASET" ] || { echo "missing dataset: $DATASET" >&2; exit 1; }
[ -f "$STEMS" ] || { echo "missing manifest: $STEMS" >&2; exit 1; }
WANT=$(grep -cve '^[[:space:]]*$' -e '^#' "$STEMS")

mkdir -p "$OUT"
echo "harness:  $AEC/eval/run_py_ab.sh"
echo "manifest: $STEMS ($WANT cases)"

render () {  # render <worktree> <label> <frame_size>
  local wt="$1" label="$2" fft="$3"
  local dir="$OUT/out_$label"
  rm -rf "$dir"; mkdir -p "$dir"
  ( cd "$wt" && NO_PREALIGN=1 AEC_CFG_OVERRIDE="frame_size=$fft" \
      "$PYBIN" python/eval_aec_challenge.py "$DATASET" \
        --preset balanced --filter 52 --cng \
        --cases-list "$STEMS" --workers "$WORKERS" -o "$dir" ) \
    > "$OUT/render_$label.log" 2>&1
  local n
  n=$(ls "$dir"/*_ours.wav 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ne "$WANT" ]; then
    echo "ABORT: rendered $n of $WANT cases for $label -- see $OUT/render_$label.log" >&2
    exit 1
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

for fft in $GRIDS; do
  echo "grid frame_size=$fft"
  render "$BASE_WT" "base_g$fft" "$fft"
  render "$CAND_WT" "cand_g$fft" "$fft"
  score "base_g$fft"
  score "cand_g$fft" "$OUT/res_base_g$fft/scores.json"
  "$PYBIN" "$AEC/eval/ab_compare.py" \
      --manifest "$STEMS" \
      --base-dir "$OUT/out_base_g$fft" --cand-dir "$OUT/out_cand_g$fft" \
      --scores-base "$OUT/res_base_g$fft/scores.json" \
      --scores-cand "$OUT/res_cand_g$fft/scores.json" \
      --out "$OUT/wav_comparison_g$fft.json"
done
echo "DONE -> $OUT"
