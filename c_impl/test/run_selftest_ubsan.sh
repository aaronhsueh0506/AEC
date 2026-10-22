#!/bin/sh
# Run the AEC shared-kernel self-test under UBSan without building audiolib.
# Usage: ./test/run_selftest_ubsan.sh [--use-standard-math]
# All generated files live in a private scratch directory removed on exit.

set -e
cd "$(dirname "$0")/.."

# An explicit template honors TMPDIR on hosts whose bare mktemp ignores it.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/aec-selftest-ubsan.XXXXXX")"
# Signal handlers exit nonzero; EXIT performs cleanup for every path.
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CC=${CC:-cc}
BIN_DIR="$SCRATCH/bin"
BIN="$BIN_DIR/simd_selftest_aec_ubsan"

# Match the standalone and nested-checkout layouts accepted by the Makefile.
COMMON_DIR=${COMMON_DIR:-}
if [ -z "$COMMON_DIR" ]; then
    if [ -d ../../audiolib/common ]; then
        COMMON_DIR=../../audiolib/common
    elif [ -d ../../../../audiolib/common ]; then
        COMMON_DIR=../../../../audiolib/common
    else
        echo "FATAL: could not auto-detect audiolib (set COMMON_DIR=...)" >&2
        exit 1
    fi
fi

EXTRA_CFLAGS=""
case $# in
    0)
        ;;
    1)
        case "$1" in
            --use-standard-math) EXTRA_CFLAGS="-DUSE_STANDARD_MATH" ;;
            *)
                echo "Usage: $0 [--use-standard-math]" >&2
                exit 2
                ;;
        esac
        ;;
    *)
        echo "Usage: $0 [--use-standard-math]" >&2
        exit 2
        ;;
esac

mkdir -p "$BIN_DIR"

echo "--- AEC simd_selftest_aec UBSan build (COMMON_DIR=$COMMON_DIR EXTRA_CFLAGS=$EXTRA_CFLAGS) ---"
"$CC" -Wall -Wextra -O2 -std=gnu99 -I./include -I./example -I"$COMMON_DIR/include" -I"$COMMON_DIR" \
    -ffp-contract=off -fno-math-errno $EXTRA_CFLAGS \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -o "$BIN" test/simd_selftest_aec.c -lm

echo "--- AEC simd_selftest_aec UBSan run ---"
"$BIN"

echo "UBSan probe: PASS (binary exited 0, no undefined-behavior diagnostic)"
