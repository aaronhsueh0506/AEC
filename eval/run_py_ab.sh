#!/bin/bash
# Python-path blind A/B over the fixed manifest (eval/manifest_90case_stems.txt).
#
# The C twin is eval/run_c_ab.sh; run BOTH for any change that lands in both
# ports. Neither can see the other's regressions: eval_aec_challenge.py drives
# the Python AEC and cannot observe a C-only change, and aec_wav cannot observe
# a Python-only one.
#
# Baseline and candidate are separate git WORKTREES; no module sources are
# swapped in the working tree, so an interrupted run leaves the tree intact.
#
# NO_PREALIGN=1 is mandatory. The eval driver's DEFAULT is an offline GCC-PHAT
# pre-align crutch, which invalidates any timing-related verdict.
#
# The dataset and the manifest come from THIS checkout, not the worktrees, so
# both sides see byte-identical input. A stem resolves to
# <scenario>/<stem>_{mic,lpb}.wav (eval/manifest_pin.py); the run stops
# naming every unresolved stem, and when a clip or the manifest no longer
# matches eval/manifest_90case.sha256. The resolved clips are linked into
# $OUT/dataset_view, which the renderer and the scorer read. PYBIN selects the interpreter; WORKERS
# (default 6) the number of parallel renders. OUT_DIR receives out_*/, res_*/,
# render_*.log, score_*.log and wav_comparison_g<fft>.json.
#
# Usage:
#   eval/run_py_ab.sh BASE_WORKTREE CAND_WORKTREE OUT_DIR ["256 512"]
set -euo pipefail

if [ "$#" -lt 3 ]; then sed -n '2,27p' "$0" >&2; exit 2; fi

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
VIEW="$OUT/dataset_view"
rm -rf "$VIEW"
"$PYBIN" "$AEC/eval/manifest_pin.py" prepare --dataset "$DATASET" --view "$VIEW" || exit 1
echo "harness:  $AEC/eval/run_py_ab.sh"
echo "manifest: $STEMS ($WANT cases)"

render () {  # render <worktree> <label> <frame_size>
  local wt="$1" label="$2" fft="$3"
  local dir="$OUT/out_$label"
  rm -rf "$dir"; mkdir -p "$dir"
  ( cd "$wt" && NO_PREALIGN=1 AEC_CFG_OVERRIDE="frame_size=$fft" \
      "$PYBIN" python/eval_aec_challenge.py "$VIEW" \
        --preset balanced --filter 832 --cng \
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
      "$OUT/res_$label" --dataset "$VIEW" "${args[@]}" ) > "$OUT/score_$label.log" 2>&1
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
