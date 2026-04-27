#!/bin/bash
# git bisect run script — FS_echo regression (c2551db → HEAD).
#
# Returns 0 (good) if FS_echo >= THRESHOLD, 1 (bad) otherwise.
# Exit 125 = skip this commit.
#
# Usage:
#   git bisect start
#   git bisect bad HEAD
#   git bisect good c2551db
#   git bisect run python/bisect_run.sh
#
# Depends on:
#   python/smoke_v280.py         (untracked, uses run_ours without shadow_dt_suppress_k)
#   python/eval_aecmos_local.py  (tracked; all commits in range have compatible API)
#   model/Run_1663915512_Stage_0.onnx

set -e

DATASET="/Users/mingyu/Desktop/novatek/SE/AEC/wav/aec_challenge_blind"
OUT_DIR="/tmp/bisect_step_$$"
N_CASES=50
THRESHOLD=2.99   # midpoint: v2.8.0≈3.064, κ-4≈2.934; c2551db TBD

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)

mkdir -p "$OUT_DIR"

echo "[bisect] testing $COMMIT …"

# Run AEC on FS cases only (FS_echo is our metric)
python3 "${SCRIPT_DIR}/smoke_v280.py" "$DATASET" -o "$OUT_DIR" -n "$N_CASES" 2>&1 \
    | grep -v "^$" || { echo "[bisect] smoke failed — skip"; rm -rf "$OUT_DIR"; exit 125; }

# Score
FS_ECHO=$(python3 "${SCRIPT_DIR}/eval_aecmos_local.py" "$DATASET" -o "$OUT_DIR" 2>/dev/null \
    | grep -A4 "FAREND SINGLETALK" | grep "MEAN" | awk '{print $2}')

rm -rf "$OUT_DIR"

if [ -z "$FS_ECHO" ]; then
    echo "[bisect] no score — skip"
    exit 125
fi

RESULT=$(python3 -c "print('GOOD' if float('${FS_ECHO}') >= ${THRESHOLD} else 'BAD')")
echo "[bisect] $COMMIT  FS_echo=${FS_ECHO}  → ${RESULT}"

python3 -c "import sys; sys.exit(0 if float('${FS_ECHO}') >= ${THRESHOLD} else 1)"
