#!/bin/bash
# ARCHIVED COPY of the script that produced the scores.json files beside it.
# `S` points at the original session's scratchpad and no longer exists; the
# variant sources it swapped are preserved as variant_diff/*.diff. Kept
# verbatim rather than tidied so the recorded numbers stay attributable to an
# exact invocation -- in particular the NO_PREALIGN=1 on the render line.
#
# Round-2 A/B: the five orchestrator/filters/preprocessing timing constants,
# on top of the already-shipped detector retiming.
#
# Baseline and candidate differ ONLY in four files under python/modules/,
# swapped in and out around each render. The baseline variant reproduces the
# authored constants exactly on all four grids (verified before launch).
#
# NO_PREALIGN=1 is mandatory: the eval driver's DEFAULT is the offline GCC-PHAT
# pre-align crutch, which would invalidate any timing-related verdict.
set -euo pipefail

AEC=/Users/mingyu/Desktop/novatek/SE/AEC
S=/private/tmp/claude-501/-Users-mingyu-Desktop-novatek-SE/fdd8dd4d-5e9f-4d28-b239-09e595e85967/scratchpad/bench
FILES="detectors.py orchestrator.py filters.py preprocessing.py"

use () { for f in $FILES; do cp "$S/$1/$f" "$AEC/python/modules/$f"; done; }

render () {   # render <label> <frame_size> <variant-dir>
  local label="$1" fft="$2" variant="$3"
  local out="$S/out_$label"
  rm -rf "$out" "$S/res_$label"; mkdir -p "$out" "$S/res_$label"
  use "$variant"
  ( cd "$AEC" && NO_PREALIGN=1 AEC_CFG_OVERRIDE="frame_size=$fft" \
      python3 python/eval_aec_challenge.py wav/aec_challenge_blind \
        --preset balanced --filter 52 --cng \
        --cases-list eval/manifest_90case_stems.txt \
        --workers 6 -o "$out" ) > "$S/render_$label.log" 2>&1
  echo "rendered $label"
}

score () {    # score <label> [baseline-scores.json]
  local label="$1" base="${2:-}"
  local args=(--label "$label")
  [ -n "$base" ] && args+=(--baseline "$base")
  ( cd "$AEC" && python3 python/bench_aecmos.py "$S/out_$label" "$S/res_$label" \
      "${args[@]}" ) > "$S/score_$label.log" 2>&1
  echo "scored $label"
}

for grid in 256 512; do
  render "b2_base_g$grid" "$grid" v2_legacy
  render "b2_cand_g$grid" "$grid" v2_retimed
  score  "b2_base_g$grid"
  score  "b2_cand_g$grid" "$S/res_b2_base_g$grid/scores.json"
done

use v2_retimed
echo "ALL DONE; working tree restored to the retimed modules"
