#!/bin/bash
# C-path 90-case blind A/B.
#
# The repo's eval_aec_challenge.py drives the PYTHON AEC ("from aec import AEC"),
# so it cannot see a C-only change at all -- it would report a spurious zero
# delta. That gap is why a C/Python divergence in alpha_power could sit in main
# without any benchmark noticing. This harness drives the C CLI instead.
#
# Baseline and candidate are separate git WORKTREES, each built from its own
# checkout. Nothing is swapped in the working tree, so a build cannot pick up
# the other side's sources.
#
# NO_PREALIGN is irrelevant here (that flag gates the Python driver's offline
# GCC-PHAT crutch); aec_wav has no pre-align path at all -- it runs the real
# online estimator, which is the stricter condition.
set -euo pipefail

BASE_WT="$1"      # worktree for the baseline commit
CAND_WT="$2"      # worktree for the candidate commit
OUT="$3"          # results root
GRIDS="${4:-256 512}"

AEC=/Users/mingyu/Desktop/novatek/SE/AEC
PY=/Users/mingyu/Desktop/novatek/SE/.venv/bin/python3.9
DATASET="$AEC/wav/aec_challenge_blind"
STEMS="$AEC/eval/manifest_90case_stems.txt"

mkdir -p "$OUT"

build () {  # build <worktree> -> echoes the aec_wav path
  local wt="$1"
  make -C "$wt/c_impl" BACKEND=kiss SIMD=1 AC_DIR=/Users/mingyu/Desktop/novatek/SE/audio_common >/dev/null 2>&1 || {
    echo "make failed in $wt" >&2; exit 1; }
  # print-bin-dir needs AC_LIB resolved the way phase 1 does; globbing the
  # freshly built tree is equivalent and does not depend on that dispatch.
  local bin
  bin=$(ls -t "$wt"/c_impl/bin/kiss-*/aec_wav 2>/dev/null | head -1)
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
    if "$exe" "$mic" "$lpb" "$dir/${stem}_ours.wav" --preset balanced --cng \
           --fft-size "$fft" >/dev/null 2>&1; then
      n=$((n+1))
    else
      echo "  RENDER FAILED: $stem" >&2
    fi
  done < "$STEMS"
  local want
  want=$(grep -cve '^\s*$' -e '^#' "$STEMS")
  if [ "$n" -ne "$want" ]; then
    echo "ABORT: rendered $n of $want cases for $label" >&2; exit 1
  fi
  echo "  rendered $n/$want -> $dir"
}

score () {  # score <label> [baseline-scores.json]
  local label="$1" base="${2:-}"
  local args=(--label "$label")
  [ -n "$base" ] && args+=(--baseline "$base")
  mkdir -p "$OUT/res_$label"
  ( cd "$AEC" && "$PY" python/bench_aecmos.py "$OUT/out_$label" "$OUT/res_$label" \
      "${args[@]}" ) > "$OUT/score_$label.log" 2>&1
  local scored
  scored=$("$PY" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['scores']))" \
             "$OUT/res_$label/scores.json")
  if [ "$scored" -lt 1 ]; then
    echo "ABORT: scorer saw $scored cases for $label" >&2; exit 1
  fi
  echo "  scored $label ($scored cases)"
}

echo "building baseline..."; BASE_EXE=$(build "$BASE_WT")
echo "building candidate..."; CAND_EXE=$(build "$CAND_WT")
echo "  base: $BASE_EXE"
echo "  cand: $CAND_EXE"

for fft in $GRIDS; do
  echo "grid fft=$fft"
  render "$BASE_EXE" "base_g$fft" "$fft"
  render "$CAND_EXE" "cand_g$fft" "$fft"
  score "base_g$fft"
  score "cand_g$fft" "$OUT/res_base_g$fft/scores.json"
done
echo "DONE -> $OUT"
