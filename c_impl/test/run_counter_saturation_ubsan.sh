#!/bin/sh
# run_counter_saturation_ubsan.sh - UBSan probe for
# test/test_counter_saturation.c -- the checked-in counter-saturation
# regression test.
#
# Standalone script rather than a new UBSan-flavoured Makefile target --
# same shape/rationale as test/run_selftest_ubsan.sh (kept consistent with
# that established pattern rather than introducing a second, differently-
# shaped mechanism for "run this one test under UBSan").
#
# Unlike simd_selftest_aec (header-only, no audiolib archive link
# needed), test_counter_saturation.c exercises the full Aec type (its
# aec.c/pbfdkf.c counter paths), so this script compiles EVERY src/*.c TU
# (this repo's own code) under -fsanitize=undefined together with the test
# file, and links against audiolib's ordinary (non-sanitized) archive --
# the code under test is this repo's own, and linking a
# sanitizer-instrumented TU against plain object code is well-supported
# (UBSan is a per-TU instrumentation, not an ABI change); it just means
# audiolib's own internals aren't independently checked here (that is
# audiolib's own test's job, not this repo's).
#
# Compiles+runs the SAME test source with -fsanitize=undefined
# -fno-sanitize-recover=all (host UBSan works fine here, same as
# run_selftest_ubsan.sh) and must exit 0 with no UBSan diagnostic AND print
# ">>> PASS" (0 failed checks) -- either a UBSan trap or a reported check
# failure is a real finding, not noise.
#
# ASan is intentionally NOT wired here: its runtime is not usable on every
# development host -- ASan stays a Linux-CI item. -fsanitize=
# undefined only.
#
# Usage: ./test/run_counter_saturation_ubsan.sh [--use-standard-math]
#   (from anywhere; cd's to this repo root itself via the script's own
#   location, no assumption about caller cwd)
#
# Scratch isolation: everything this run touches lives under one mktemp -d
# SCRATCH, so it cannot race a concurrent real build in this repo's own bin/
# or in audiolib's default obj/bin trees (reached via the `make ... lib` /
# `make ... print-lib-path` calls) -- a developer's own terminal, AEC's/NR's
# own build pulling in audiolib, CI, or a second concurrent invocation of
# this same script with a different --use-standard-math setting. The scratch
# is removed by a single trap covering normal exit and both signals -- the
# same isolation discipline audiolib's own
# scripts/test_build_isolation.sh applies to its own scratch builds.
# OBJ_ROOT/BIN_ROOT are audiolib's OWN placement knobs and are passed
# with IDENTICAL values to both the `lib` build
# call and the `print-lib-path` query call below, so the two invocations can
# never resolve to different paths -- the same query/build-divergence
# discipline this repo's own Makefile enforces for its own
# BACKEND/WERROR/CC/CXX/EXTRA_CFLAGS/NO_STDIO overrides.

set -e
cd "$(dirname "$0")/.."

# Explicit template: a bare `mktemp -d` with no template
# argument ignores $TMPDIR on this host (confirmed by direct repro -- it
# creates under /var/folders/... regardless of $TMPDIR), which silently
# defeated any caller (e.g. this repo's own signal-handling regression
# test) that sets TMPDIR to a private directory specifically to observe
# this script's scratch-dir lifecycle. The explicit template below actually
# honors $TMPDIR (falling back to /tmp if unset).
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/aec-counter-saturation-ubsan.XXXXXX")"
# Separate, explicit handlers (a combined `trap '...' EXIT INT
# TERM` registers the SAME handler -- a bare cleanup command with no exit --
# for all three. On INT/TERM that handler runs to completion and then the
# shell simply continues the script (`set -e` doesn't apply to a trap body
# itself), so a real signal never actually aborts the run: a signal landing
# mid-`"$BIN"` was observed to let the run continue to completion and print
# a false ">>> PASS" / "UBSan probe: PASS" / exit 0, as if nothing happened --
# exactly what a caller sending SIGTERM/SIGINT to abort a run must not see.
# Each signal handler below calls `exit` with the conventional 128+signum
# code, which in turn triggers the EXIT trap so cleanup still always runs.
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CC=${CC:-cc}
BIN_DIR="$SCRATCH/bin"
BIN="$BIN_DIR/test_counter_saturation_ubsan"
AC_OBJ_ROOT="$SCRATCH/ac_obj"
AC_BIN_ROOT="$SCRATCH/ac_bin"

# Same COMMON_DIR auto-detection the Makefile itself uses (audiolib sitting
# next to this repo, or nested deeper e.g. an integration-repo submodule
# checkout) -- kept in sync with the Makefile's own COMMON_DIR line rather than
# hardcoding one layout.
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

# Argument validation: the old bare `case "$1" in --use-standard-math) ...;
# esac` had no default/wildcard arm, so a typo, an extra flag, `-h`, or any
# other bogus argument was silently accepted and ignored, running the wrong
# (non-USE_STANDARD_MATH) configuration instead of failing loudly. Now: zero
# args, or exactly one arg that is precisely "--use-standard-math", or a
# usage error to stderr + nonzero exit for anything else (including a stray
# second argument).
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
