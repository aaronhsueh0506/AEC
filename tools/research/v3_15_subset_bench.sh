#!/usr/bin/env bash
# v3.15 Tier-1 subset bench wrapper.
#
# Renders only the ~60 cases listed in tools/research/v3_15_subset_cases.txt
# instead of the full 800-case AEC Challenge corpus. Designed for fast
# threshold-sweep / V1-vs-V2 comparison sprints (~5-10 min vs ~40 min).
# 800-case Tier 2 stays MANDATORY at merge / verdict gates.
#
# Usage:
#   tools/research/v3_15_subset_bench.sh <output_dir> [extra_env]
#
# Examples:
#   tools/research/v3_15_subset_bench.sh /tmp/v3_15_baseline/
#   tools/research/v3_15_subset_bench.sh /tmp/v3_15_arc_m_v3_on/ \
#       'AEC_ARC_M_T_GATED_ENABLED=1'
#   tools/research/v3_15_subset_bench.sh /tmp/v3_15_arc_t_on/ \
#       'AEC_ARC_T_COHORT_DETECTOR=1 AEC_ARC_T_RES_PREEMPT_MODE=1'
#
# Standard 800-case config preserved: preset=balanced / fl=832 (52 ms) / cng=True / parallel
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <output_dir> [extra_env]" >&2
    echo "  e.g. $0 /tmp/v3_15_arc_m_t_gated_on/ 'AEC_ARC_M_T_GATED_ENABLED=1'" >&2
    exit 1
fi

OUT_DIR="$1"
EXTRA_ENV="${2:-}"

# Resolve repo root (this script lives at <repo>/tools/research/)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CASES_LIST="$REPO/tools/research/v3_15_subset_cases.txt"
DATASET="$REPO/wav/aec_challenge_blind"

if [[ ! -d "$DATASET" ]]; then
    # Worktree may share dataset with sibling AEC repo
    ALT="/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind"
    if [[ -d "$ALT" ]]; then
        DATASET="$ALT"
    else
        echo "ERROR: dataset not found at $REPO/wav/aec_challenge_blind or $ALT" >&2
        exit 1
    fi
fi

if [[ ! -f "$CASES_LIST" ]]; then
    echo "ERROR: cases list not found at $CASES_LIST" >&2
    echo "       run tools/research/v3_15_build_subset.py first" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "[subset-bench] dataset    = $DATASET" >&2
echo "[subset-bench] cases-list = $CASES_LIST" >&2
echo "[subset-bench] out_dir    = $OUT_DIR" >&2
echo "[subset-bench] extra_env  = ${EXTRA_ENV:-<none>}" >&2

# shellcheck disable=SC2086
env $EXTRA_ENV python3 "$REPO/python/eval_aec_challenge.py" \
    "$DATASET" \
    --preset balanced \
    --filter 832 \
    --cng \
    -o "$OUT_DIR" \
    --parallel \
    --cases-list "$CASES_LIST"
