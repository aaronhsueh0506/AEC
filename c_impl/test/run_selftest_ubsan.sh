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
#
# Scratch isolation (same bug class/fix as test/run_counter_saturation_ubsan.sh's
# review finding, applied here for consistency -- this script never calls
# audio_common's make, so the only real-tree pollution risk it had was
# writing its own binary into this repo's real bin/, hence no OBJ_ROOT/
# BIN_ROOT plumbing needed here, just a scratch BIN_DIR): everything this
# run touches now lives under one mktemp -d SCRATCH, removed by a single
# trap covering normal exit and both signals.

set -e
cd "$(dirname "$0")/.."

# Explicit template (round-8 review): a bare `mktemp -d` with no template
# argument ignores $TMPDIR on this host (confirmed by direct repro -- it
# creates under /var/folders/... regardless of $TMPDIR), which silently
# defeated any caller (e.g. this repo's own signal-handling regression
# test) that sets TMPDIR to a private directory specifically to observe
# this script's scratch-dir lifecycle. The explicit template below actually
# honors $TMPDIR (falling back to /tmp if unset).
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/aec-selftest-ubsan.XXXXXX")"
# Separate, explicit handlers (round-7 review: a combined `trap '...' EXIT INT
# TERM` registers the SAME handler -- a bare cleanup command with no exit --
# for all three. On INT/TERM that handler runs to completion and then the
# shell simply continues the script (`set -e` doesn't apply to a trap body
# itself), so a real signal never actually aborts the run: a signal landing
# mid-`"$BIN"` was observed to let the run continue to completion and print
# a false "UBSan probe: PASS" / exit 0, as if nothing happened -- exactly
# what a caller sending SIGTERM/SIGINT to abort a run must not see. Each
# signal handler below calls `exit` with the conventional 128+signum code,
# which in turn triggers the EXIT trap so cleanup still always runs.
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CC=${CC:-cc}
BIN_DIR="$SCRATCH/bin"
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

# Argument validation (same fix applied to run_counter_saturation_ubsan.sh
# for the identical gap): the old bare case had no default/wildcard arm, so
# a typo or bogus flag was silently accepted and ignored. Now: zero args, or
# exactly one arg that is precisely "--use-standard-math", or a usage error
# to stderr + nonzero exit for anything else (including a stray second
# argument).
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

echo "--- AEC simd_selftest_aec UBSan build (AC_DIR=$AC_DIR EXTRA_CFLAGS=$EXTRA_CFLAGS) ---"
"$CC" -Wall -Wextra -O2 -std=gnu99 -I./include -I./example -I"$AC_DIR/include" \
    -ffp-contract=off $EXTRA_CFLAGS \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -o "$BIN" test/simd_selftest_aec.c -lm

echo "--- AEC simd_selftest_aec UBSan run ---"
"$BIN"

echo "UBSan probe: PASS (binary exited 0, no undefined-behavior diagnostic)"
