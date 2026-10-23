#!/bin/sh
# run_counter_saturation_ubsan.sh - UBSan probe for test/test_counter_saturation.c.
#
# Standalone script rather than a Makefile target, like test/run_selftest_ubsan.sh. The test exercises the full Aec type, so this
# compiles every src/*.c TU together with the test under -fsanitize=undefined -fno-sanitize-recover=all and links audiolib's
# ordinary (non-sanitized) archive: UBSan is per-TU instrumentation, and audiolib's internals are covered by its own tests.
# The run must exit 0 with no UBSan diagnostic and print ">>> PASS" (0 failed checks). ASan is not wired here: its runtime is
# not usable on every development host.
#
# Usage: ./test/run_counter_saturation_ubsan.sh [--use-standard-math]   (cd's to the repo root itself)
#
# Everything lives under one mktemp -d SCRATCH, so it cannot race a concurrent build in this repo's bin/ or in audiolib's
# default obj/bin trees; one trap removes it on normal exit and on both signals. OBJ_ROOT/BIN_ROOT are passed with identical
# values to the audiolib `lib` build and to the `print-lib-path` query, so the two cannot resolve to different paths.

set -e
cd "$(dirname "$0")/.."

# Explicit template: a bare `mktemp -d` ignores $TMPDIR on some hosts; the template honors it (falling back to /tmp), so a
# caller can point TMPDIR at a private directory to observe the scratch lifecycle.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/aec-counter-saturation-ubsan.XXXXXX")"
# Separate handlers: a combined `trap '...' EXIT INT TERM` would run the cleanup and then let the script continue, so a signal
# arriving mid-run could end in a false PASS. Each signal handler exits with 128+signum, which triggers the EXIT trap, so
# cleanup always runs.
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CC=${CC:-cc}
BIN_DIR="$SCRATCH/bin"
BIN="$BIN_DIR/test_counter_saturation_ubsan"
AC_OBJ_ROOT="$SCRATCH/ac_obj"
AC_BIN_ROOT="$SCRATCH/ac_bin"

# Same COMMON_DIR auto-detection as the Makefile (audiolib next to this repo, or four levels up in a nested checkout).
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

# Argument validation: no argument, or exactly "--use-standard-math"; anything else is a usage error, because a silently
# ignored typo would run the wrong configuration.
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

echo "--- audiolib lib build (BACKEND=ne10, ordinary/non-sanitized, scratch OBJ_ROOT/BIN_ROOT) ---"
"${MAKE:-make}" -s -C "$COMMON_DIR" BACKEND=ne10 CC="$CC" OBJ_ROOT="$AC_OBJ_ROOT" BIN_ROOT="$AC_BIN_ROOT" lib
COMMON_LIB="$("${MAKE:-make}" -s -C "$COMMON_DIR" BACKEND=ne10 CC="$CC" OBJ_ROOT="$AC_OBJ_ROOT" BIN_ROOT="$AC_BIN_ROOT" print-lib-path)"
test -n "$COMMON_LIB" || { echo "FATAL: could not resolve audiolib archive path" >&2; exit 1; }

echo "--- AEC test_counter_saturation UBSan build (COMMON_DIR=$COMMON_DIR EXTRA_CFLAGS=$EXTRA_CFLAGS) ---"
# shellcheck disable=SC2046
"$CC" -Wall -Wextra -O2 -std=gnu99 -I./include -I./example -I"$COMMON_DIR/include" -I"$COMMON_DIR" \
    -ffp-contract=off -fno-math-errno $EXTRA_CFLAGS \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -o "$BIN" test/test_counter_saturation.c $(find src -name '*.c') \
    "$COMMON_LIB" -lm

echo "--- AEC test_counter_saturation UBSan run (must be invoked from c_impl/, matching the"
echo "    test's own wav/aec_record fixture path convention -- this script already cd's there) ---"
"$BIN"

echo "UBSan probe: PASS (binary exited 0, no undefined-behavior diagnostic, 0 failed checks)"
