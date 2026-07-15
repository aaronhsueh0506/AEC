#!/bin/sh
# run_selftest_ubsan.sh - UBSan probe for test/simd_selftest_aec.c
# (round-3 review B05, item 5).
#
# Standalone script rather than a new Makefile target: this repo's Makefile
# is being reworked concurrently by another task, so wiring a target in
# there risks a conflict; this script needs nothing from it
# (aec_simd_kernels.h is header-only, no audio_common archive link required,
# same as the ordinary `make selftest` recipe).
#
# Compiles+runs the SAME selftest source with
# -fsanitize=undefined -fno-sanitize-recover=all (host UBSan works fine for
# a single small translation unit like this one) and must exit 0 with no
# UBSan diagnostic printed -- any diagnostic here is a real finding, not
# noise.
#
# ASan is intentionally NOT wired here: this development host's ASan runtime
# is broken (see project memory) -- ASan stays a Linux-CI item. -fsanitize=
# undefined only.
#
# Usage: ./test/run_selftest_ubsan.sh [--use-standard-math]
#   (from anywhere; cd's to this repo root itself via the script's own
#   location, no assumption about caller cwd)

set -e
cd "$(dirname "$0")/.."

CC=${CC:-cc}
BIN_DIR=bin
BIN="$BIN_DIR/simd_selftest_aec_ubsan"

# Same AC_DIR auto-detection the Makefile itself uses (audio_common sitting
# next to this repo, or nested deeper e.g. an integration-repo submodule
# checkout) -- kept in sync with the Makefile's own AC_DIR line rather than
# hardcoding one layout.
AC_DIR=${AC_DIR:-}
if [ -z "$AC_DIR" ]; then
    if [ -d ../../audio_common ]; then
        AC_DIR=../../audio_common
    elif [ -d ../../../../audio_common ]; then
        AC_DIR=../../../../audio_common
    else
        echo "FATAL: could not auto-detect audio_common (set AC_DIR=...)" >&2
        exit 1
    fi
fi

EXTRA_CFLAGS=""
case "$1" in
    --use-standard-math) EXTRA_CFLAGS="-DUSE_STANDARD_MATH" ;;
esac

mkdir -p "$BIN_DIR"

echo "--- AEC simd_selftest_aec UBSan build (AC_DIR=$AC_DIR EXTRA_CFLAGS=$EXTRA_CFLAGS) ---"
"$CC" -Wall -Wextra -O2 -std=gnu99 -I./include -I./example -I"$AC_DIR/include" \
    -ffp-contract=off $EXTRA_CFLAGS \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -o "$BIN" test/simd_selftest_aec.c -lm

echo "--- AEC simd_selftest_aec UBSan run ---"
"$BIN"

echo "UBSan probe: PASS (binary exited 0, no undefined-behavior diagnostic)"
