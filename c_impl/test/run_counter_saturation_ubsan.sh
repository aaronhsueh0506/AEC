#!/bin/sh
# run_counter_saturation_ubsan.sh - UBSan probe for
# test/test_counter_saturation.c (round-6 review: the permanent,
# checked-in counter-saturation regression test; Codex explicitly asked for
# a committed test since every prior round's UBSan probe was ad hoc).
#
# Standalone script rather than a new UBSan-flavoured Makefile target --
# same shape/rationale as test/run_selftest_ubsan.sh (kept consistent with
# that established pattern rather than introducing a second, differently-
# shaped mechanism for "run this one test under UBSan").
#
# Unlike simd_selftest_aec (header-only, no audio_common archive link
# needed), test_counter_saturation.c exercises the full Aec type (its
# aec.c/pbfdkf.c-embedded fixes), so this script compiles EVERY src/*.c TU
# (this repo's own code) under -fsanitize=undefined together with the test
# file, and links against audio_common's ordinary (non-sanitized) archive --
# none of this round's fixes touch audio_common's own code, and linking a
# sanitizer-instrumented TU against plain object code is well-supported
# (UBSan is a per-TU instrumentation, not an ABI change); it just means
# audio_common's own internals aren't independently checked here (that is
# audio_common's own test's job, not this repo's).
#
# Compiles+runs the SAME test source with -fsanitize=undefined
# -fno-sanitize-recover=all (host UBSan works fine here, same as
# run_selftest_ubsan.sh) and must exit 0 with no UBSan diagnostic AND print
# ">>> PASS" (0 failed checks) -- either a UBSan trap or a reported check
# failure is a real finding, not noise.
#
# ASan is intentionally NOT wired here: this development host's ASan runtime
# is broken (see project memory) -- ASan stays a Linux-CI item. -fsanitize=
# undefined only.
#
# Usage: ./test/run_counter_saturation_ubsan.sh [--use-standard-math]
#   (from anywhere; cd's to this repo root itself via the script's own
#   location, no assumption about caller cwd)
#
# Scratch isolation (review finding, P2): this script used to build straight
# into this repo's own real bin/ AND drive audio_common's real (default)
# obj/bin trees via its `make ... lib` / `make ... print-lib-path` calls --
# racing any concurrent real build (a developer's own terminal, AEC's/NR's
# own build pulling in audio_common, CI, or even a second concurrent
# invocation of this same script with a different --use-standard-math
# setting). Everything this run touches now lives under one mktemp -d
# SCRATCH, removed by a single trap covering normal exit and both signals --
# the same isolation discipline audio_common's own
# scripts/test_build_isolation.sh applies to its own scratch builds.
# OBJ_ROOT/BIN_ROOT are audio_common's OWN placement knobs (round-6 review,
# that repo) and are passed with IDENTICAL values to both the `lib` build
# call and the `print-lib-path` query call below, so the two invocations can
# never resolve to different paths -- the same query/build-divergence
# discipline this repo's own Makefile enforces for its own
# BACKEND/WERROR/CC/CXX/EXTRA_CFLAGS/NO_STDIO overrides.

set -e
cd "$(dirname "$0")/.."

# Explicit template (round-8 review): a bare `mktemp -d` with no template
# argument ignores $TMPDIR on this host (confirmed by direct repro -- it
# creates under /var/folders/... regardless of $TMPDIR), which silently
# defeated any caller (e.g. this repo's own signal-handling regression
# test) that sets TMPDIR to a private directory specifically to observe
# this script's scratch-dir lifecycle. The explicit template below actually
# honors $TMPDIR (falling back to /tmp if unset).
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/aec-counter-saturation-ubsan.XXXXXX")"
# Separate, explicit handlers (round-7 review: a combined `trap '...' EXIT INT
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

echo "--- audio_common lib build (BACKEND=kiss, ordinary/non-sanitized, scratch OBJ_ROOT/BIN_ROOT) ---"
"${MAKE:-make}" -s -C "$AC_DIR" BACKEND=kiss CC="$CC" OBJ_ROOT="$AC_OBJ_ROOT" BIN_ROOT="$AC_BIN_ROOT" lib
AC_LIB="$("${MAKE:-make}" -s -C "$AC_DIR" BACKEND=kiss CC="$CC" OBJ_ROOT="$AC_OBJ_ROOT" BIN_ROOT="$AC_BIN_ROOT" print-lib-path)"
test -n "$AC_LIB" || { echo "FATAL: could not resolve audio_common archive path" >&2; exit 1; }

echo "--- AEC test_counter_saturation UBSan build (AC_DIR=$AC_DIR EXTRA_CFLAGS=$EXTRA_CFLAGS) ---"
# shellcheck disable=SC2046
"$CC" -Wall -Wextra -O2 -std=gnu99 -I./include -I./example -I"$AC_DIR/include" \
    -ffp-contract=off $EXTRA_CFLAGS \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -o "$BIN" test/test_counter_saturation.c $(find src -name '*.c') \
    "$AC_LIB" -lm

echo "--- AEC test_counter_saturation UBSan run (must be invoked from c_impl/, matching the"
echo "    test's own wav/aec_record fixture path convention -- this script already cd's there) ---"
"$BIN"

echo "UBSan probe: PASS (binary exited 0, no undefined-behavior diagnostic, 0 failed checks)"
