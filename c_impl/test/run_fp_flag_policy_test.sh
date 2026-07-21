#!/bin/sh
# run_fp_flag_policy_test.sh - permanent regression test for the
# quote/escape/response-file rejection + CFLAGS+CXXFLAGS+CPPFLAGS
# exact-token conflict check in ../Makefile's "Conflict detection" block
# (search "FP_INPUT_FLAGS" there).
#
# BACKGROUND: this repo's Makefile used to check ONLY $(CFLAGS), via three
# separate $(findstring PATTERN,$(CFLAGS)) substring searches -- a Codex
# review found this both too narrow (CXXFLAGS/CPPFLAGS could carry
# -Ofast/-ffast-math/-ffp-contract=<x> straight past it, e.g.
# `env CXXFLAGS=-Ofast make ... lib`) and too broad ($(findstring) does a
# plain substring search, so a harmless `-DROUND9_NOTE=-Ofastness` macro
# define -- not the compiler flag -Ofast at all -- used to be rejected
# anyway). audio_common's Makefile carries the fix for both problems
# (scripts/test_build_isolation.sh's S23 series); this script mirrors that
# same fix now replicated into this repo's own Makefile: an exact-TOKEN
# $(filter -Ofast -ffast-math -ffp-contract=%,...) check over the combined
# $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) text, preceded by an outright rejection
# of any shell-quote/backslash/backtick/'$('/semicolon/pipe/ampersand/
# @response-file character appearing anywhere in that combined text (Make
# has zero concept of shell quoting, so a quoted/escaped/response-file
# variant of -Ofast can reach the compiler for real while never matching
# $(filter)'s plain-word comparison -- see the Makefile's own comment for
# the full writeup of each bypass shape).
#
# This repo has no standalone shell-based build-isolation suite of its own
# (searched for "test_build_isolation" anywhere in this repo; the only hit
# is a comment in run_counter_saturation_ubsan.sh referencing audio_common's
# suite by name, not a suite of this repo's own), so these 5 cases -- the
# same 5-case set audio_common's S23b added -- live here instead, as a
# small permanent standalone script, same convention as this repo's other
# test/run_*.sh scripts (run_selftest_ubsan.sh,
# run_counter_saturation_ubsan.sh): cd to the repo root via the script's
# own location, scratch-isolated (OBJ_ROOT/BIN_ROOT under one mktemp -d,
# removed by a trap covering normal exit and both signals), never touching
# this repo's real obj/bin trees.
#
# The 5 cases, each run against the cheap `print-obj-dir` introspection
# target (no real compile/link needed to prove the parse-time gate fires
# or doesn't):
#   1. EXTRA_CFLAGS="'-Ofast'"      (single-quoted)         -> must FAIL,
#      log contains "FP policy conflict" AND "single-quote"
#   2. EXTRA_CFLAGS='"-ffast-math"' (double-quoted)         -> must FAIL,
#      log contains "FP policy conflict" AND "double-quote"
#   3. EXTRA_CFLAGS="-O'f'ast"      (quote-split mid-token) -> must FAIL,
#      log contains "FP policy conflict" AND "single-quote"
#   4. EXTRA_CFLAGS='@flags.rsp'    (response-file)         -> must FAIL,
#      log contains "FP policy conflict" AND "response-file"
#   5. positive control: EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness AND (a
#      second sub-case) EXTRA_CFLAGS=-DTEXT=ffast-math -- both harmless,
#      UNQUOTED tokens that merely CONTAIN -Ofast/ffast-math as a substring
#      of a larger identifier, not the compiler flag itself -> both must
#      SUCCEED (the exact-token $(filter) check must not false-positive on
#      either, exactly the class of false rejection the old
#      $(findstring ...) check used to produce)
#
# Also re-confirms the OTHER two things a naive re-implementation could
# regress: (i) an actual bare -Ofast/-ffast-math/-ffp-contract=<x> token,
# unquoted, in CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS, is still rejected
# (round-3/round-9 behaviour preserved) -- including via CXXFLAGS/CPPFLAGS
# specifically now that the check has been widened from CFLAGS-only; and
# (ii) the pre-existing, unrelated "Command-line override rejection"
# foreach (round-4 review P1-1: a command-line CFLAGS=/CXXFLAGS=/CPPFLAGS=/
# LDFLAGS=/FP_POLICY= override, or `make -e` with any of those five set in
# the environment) still fires first and is untouched by this change.
#
# Usage: ./test/run_fp_flag_policy_test.sh
#   (from anywhere; cd's to this repo root itself via the script's own
#   location, no assumption about caller cwd)
#
# Exit code 0 + "run_fp_flag_policy_test: ALL PASS" means every case
# behaved as expected. Nonzero + at least one "FAIL" line means a
# regression.

set -u
cd "$(dirname "$0")/.."

# Explicit template (same fix applied throughout this repo's other
# test/run_*.sh scripts): a bare `mktemp -d` with no template argument
# ignores $TMPDIR on this host, silently defeating any caller that sets
# TMPDIR to observe this script's scratch-dir lifecycle.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/aec-fp-flag-policy.XXXXXX")"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OBJ_ROOT="$SCRATCH/obj"
BIN_ROOT="$SCRATCH/bin"
MAKE_BIN=${MAKE:-make}

PASS_COUNT=0
FAIL_COUNT=0
pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); echo "  FAIL: $1"; }

echo "=== run_fp_flag_policy_test: quote/escape/response-file bypass cases ==="

# Case 1: single-quoted.
LOG1="$SCRATCH/log_single_quoted"
if env EXTRA_CFLAGS="'-Ofast'" "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG1" 2>&1; then
    fail "case 1: env EXTRA_CFLAGS=\"'-Ofast'\" (single-quoted) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
    cat "$LOG1" >&2
else
    if grep -q "FP policy conflict" "$LOG1" && grep -q "single-quote" "$LOG1"; then
        pass "case 1: env EXTRA_CFLAGS=\"'-Ofast'\" (single-quoted) print-obj-dir correctly FAILS, identifying the single-quote character"
    else
        fail "case 1: env EXTRA_CFLAGS=\"'-Ofast'\" (single-quoted) print-obj-dir failed but did NOT identify the single-quote character specifically"
        cat "$LOG1" >&2
    fi
fi

# Case 2: double-quoted.
LOG2="$SCRATCH/log_double_quoted"
if env EXTRA_CFLAGS='"-ffast-math"' "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG2" 2>&1; then
    fail "case 2: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
    cat "$LOG2" >&2
else
    if grep -q "FP policy conflict" "$LOG2" && grep -q "double-quote" "$LOG2"; then
        pass "case 2: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir correctly FAILS, identifying the double-quote character"
    else
        fail "case 2: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir failed but did NOT identify the double-quote character specifically"
        cat "$LOG2" >&2
    fi
fi

# Case 3: quote-split mid-token.
LOG3="$SCRATCH/log_quote_split"
if env EXTRA_CFLAGS="-O'f'ast" "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG3" 2>&1; then
    fail "case 3: env EXTRA_CFLAGS=\"-O'f'ast\" (quote-split mid-token) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
    cat "$LOG3" >&2
else
    if grep -q "FP policy conflict" "$LOG3" && grep -q "single-quote" "$LOG3"; then
        pass "case 3: env EXTRA_CFLAGS=\"-O'f'ast\" (quote-split mid-token) print-obj-dir correctly FAILS, identifying the single-quote character"
    else
        fail "case 3: env EXTRA_CFLAGS=\"-O'f'ast\" (quote-split mid-token) print-obj-dir failed but did NOT identify the single-quote character specifically"
        cat "$LOG3" >&2
    fi
fi

# Case 4: @response-file syntax.
LOG4="$SCRATCH/log_response_file"
if env EXTRA_CFLAGS='@flags.rsp' "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG4" 2>&1; then
    fail "case 4: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
    cat "$LOG4" >&2
else
    if grep -q "FP policy conflict" "$LOG4" && grep -q "response-file" "$LOG4"; then
        pass "case 4: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir correctly FAILS, identifying the @-prefixed response-file token"
    else
        fail "case 4: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir failed but did NOT identify the response-file token specifically"
        cat "$LOG4" >&2
    fi
fi

# Case 5: positive control -- two harmless, UNQUOTED tokens that merely
# CONTAIN -Ofast/ffast-math as a substring of a larger identifier, neither
# of which is the compiler flag itself; must NOT be rejected.
LOG5A="$SCRATCH/log_positive_round9_note"
if env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness "$MAKE_BIN" -s BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG5A" 2>&1; then
    pass "case 5a: env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness print-obj-dir (positive control, no forbidden characters) succeeds"
else
    fail "case 5a: env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness print-obj-dir unexpectedly FAILED (false-positive on a harmless token)"
    cat "$LOG5A" >&2
fi

LOG5B="$SCRATCH/log_positive_dtext"
if env EXTRA_CFLAGS=-DTEXT=ffast-math "$MAKE_BIN" -s BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG5B" 2>&1; then
    pass "case 5b: env EXTRA_CFLAGS=-DTEXT=ffast-math print-obj-dir (positive control, no forbidden characters) succeeds"
else
    fail "case 5b: env EXTRA_CFLAGS=-DTEXT=ffast-math print-obj-dir unexpectedly FAILED (false-positive on a harmless token)"
    cat "$LOG5B" >&2
fi

echo "=== run_fp_flag_policy_test: prior (round-3/round-9) behaviour must not regress ==="

# A real, unquoted -Ofast/-ffast-math/-ffp-contract=<x> must still be
# rejected -- via EXTRA_CFLAGS (the original, narrower CFLAGS-only check's
# own case) AND now via CXXFLAGS/CPPFLAGS directly (the widened part of
# this fix; a plain environment CXXFLAGS/CPPFLAGS folds in normally via the
# pre-existing "Command-line override rejection" foreach's origin logic,
# never touching EXTRA_CFLAGS at all -- see the Makefile's own "CORRECTED
# UNDERSTANDING" comment).
LOG6="$SCRATCH/log_plain_ofast_extra_cflags"
if env EXTRA_CFLAGS=-Ofast "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG6" 2>&1; then
    fail "case 6: env EXTRA_CFLAGS=-Ofast print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
    cat "$LOG6" >&2
else
    if grep -q "FP policy conflict" "$LOG6" && grep -q -- "-Ofast" "$LOG6"; then
        pass "case 6: env EXTRA_CFLAGS=-Ofast print-obj-dir correctly FAILS (round-3 behaviour preserved)"
    else
        fail "case 6: env EXTRA_CFLAGS=-Ofast print-obj-dir failed but did NOT identify -Ofast specifically"
        cat "$LOG6" >&2
    fi
fi

LOG7="$SCRATCH/log_cxxflags_ofast"
if env CXXFLAGS=-Ofast "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG7" 2>&1; then
    fail "case 7: env CXXFLAGS=-Ofast print-obj-dir unexpectedly SUCCEEDED (widened CXXFLAGS check regressed)"
    cat "$LOG7" >&2
else
    if grep -q "FP policy conflict" "$LOG7" && grep -q -- "-Ofast" "$LOG7"; then
        pass "case 7: env CXXFLAGS=-Ofast print-obj-dir correctly FAILS (widened CXXFLAGS check holds)"
    else
        fail "case 7: env CXXFLAGS=-Ofast print-obj-dir failed but did NOT identify -Ofast specifically"
        cat "$LOG7" >&2
    fi
fi

LOG8="$SCRATCH/log_cppflags_ffastmath"
if env CPPFLAGS=-ffast-math "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG8" 2>&1; then
    fail "case 8: env CPPFLAGS=-ffast-math print-obj-dir unexpectedly SUCCEEDED (widened CPPFLAGS check regressed)"
    cat "$LOG8" >&2
else
    if grep -q "FP policy conflict" "$LOG8" && grep -q -- "-ffast-math" "$LOG8"; then
        pass "case 8: env CPPFLAGS=-ffast-math print-obj-dir correctly FAILS (widened CPPFLAGS check holds)"
    else
        fail "case 8: env CPPFLAGS=-ffast-math print-obj-dir failed but did NOT identify -ffast-math specifically"
        cat "$LOG8" >&2
    fi
fi

# The pre-existing, unrelated command-line-override / `make -e` rejection
# (round-4 review P1-1) must still fire first and untouched -- a command-
# line CFLAGS= override, and `make -e` with FP_POLICY= set in the
# environment, are both still rejected by that separate foreach, never
# reaching (or being affected by) this fix's own conflict-detection block.
LOG9="$SCRATCH/log_cmdline_cflags_override"
if "$MAKE_BIN" CFLAGS=-O3 BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG9" 2>&1; then
    fail "case 9: make CFLAGS=-O3 print-obj-dir unexpectedly SUCCEEDED (command-line override rejection regressed)"
    cat "$LOG9" >&2
else
    if grep -q "cannot be overridden" "$LOG9"; then
        pass "case 9: make CFLAGS=-O3 print-obj-dir correctly FAILS, mentioning 'cannot be overridden' (round-4 P1-1 untouched)"
    else
        fail "case 9: make CFLAGS=-O3 print-obj-dir failed but did NOT mention 'cannot be overridden'"
        cat "$LOG9" >&2
    fi
fi

LOG10="$SCRATCH/log_e_fp_policy_override"
if env FP_POLICY=-ffp-contract=fast "$MAKE_BIN" -e BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG10" 2>&1; then
    fail "case 10: env FP_POLICY=-ffp-contract=fast make -e print-obj-dir unexpectedly SUCCEEDED (make -e FP_POLICY gap regressed)"
    cat "$LOG10" >&2
else
    if grep -q "cannot be overridden" "$LOG10"; then
        pass "case 10: env FP_POLICY=-ffp-contract=fast make -e print-obj-dir correctly FAILS, mentioning 'cannot be overridden' (make -e gap fix untouched)"
    else
        fail "case 10: env FP_POLICY=-ffp-contract=fast make -e print-obj-dir failed but did NOT mention 'cannot be overridden'"
        cat "$LOG10" >&2
    fi
fi

echo
echo "TOTAL: $((PASS_COUNT + FAIL_COUNT))  PASS: $PASS_COUNT  FAIL: $FAIL_COUNT"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "run_fp_flag_policy_test: ALL PASS"
    exit 0
else
    echo "run_fp_flag_policy_test: FAIL"
    exit 1
fi
