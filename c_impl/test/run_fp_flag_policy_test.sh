#!/bin/sh
# run_fp_flag_policy_test.sh - permanent regression test for the
# quote/escape/allow-list rejection + CFLAGS+CXXFLAGS+CPPFLAGS exact-token
# conflict check in ../Makefile's "Conflict detection" block (search
# "FP_INPUT_FLAGS" there).
#
# BACKGROUND: this repo's Makefile used to check ONLY $(CFLAGS), via three
# separate $(findstring PATTERN,$(CFLAGS)) substring searches -- this was
# both too narrow (CXXFLAGS/CPPFLAGS could carry
# -Ofast/-ffast-math/-ffp-contract=<x> straight past it, e.g.
# `env CXXFLAGS=-Ofast make ... lib`) and too broad ($(findstring) does a
# plain substring search, so a harmless `-DROUND9_NOTE=-Ofastness` macro
# define -- not the compiler flag -Ofast at all -- used to be rejected
# anyway). That was fixed with an exact-TOKEN $(filter) check preceded by a
# hand-picked 9-item DENY-list (single quote, double quote, backslash,
# backtick, a literal "$(" sequence, semicolon, pipe, ampersand, and an
# @-prefixed response-file token) rejecting known shell-quoting/escaping/
# expansion/response-file bypass shapes.
#
# ALLOW-LIST REDESIGN (replicated from audio_common's identical fix -- see
# that repo's Makefile for the full writeup): a later audit found the
# 9-item deny-list does NOT catch bare glob characters (*, ?, [, ]), tilde
# (~), or -- most seriously -- real shell REDIRECTION (>, <) or process
# substitution (<(...)). Replaced with a single character-class ALLOW-list
# (see the Makefile's "REDESIGN"/"The allow-list itself" comments) covering
# the input as a whole -- everything NOT in
# [A-Za-z0-9_./=,+- ] (plus space) is rejected outright. The single-quote
# check survives as a Make-native (non-shell) pre-check, kept for a
# DIFFERENT reason (it gates the allow-list's own $(shell)-based
# single-quote-embedding of the untrusted text, so it must run first and
# stay pure $(findstring)) -- not because the allow-list itself fails to
# reject a lone single quote (it does, on its own, too).
#
# This repo has no standalone shell-based build-isolation suite of its own
# (searched for "test_build_isolation" anywhere in this repo; the only hit
# is a comment in run_counter_saturation_ubsan.sh referencing audio_common's
# suite by name, not a suite of this repo's own), so this script mirrors
# BOTH audio_common's S23b (quote/escape/response-file bypass) AND S24
# (allow-list bypass shapes the old deny-list never caught, plus the
# FP-policy check's own internal-variable override exposure) as one small
# permanent standalone script, same convention as this repo's other
# test/run_*.sh scripts (run_selftest_ubsan.sh,
# run_counter_saturation_ubsan.sh): cd to the repo root via the script's
# own location, scratch-isolated (OBJ_ROOT/BIN_ROOT under one mktemp -d,
# removed by a trap covering normal exit and both signals), never touching
# this repo's real obj/bin trees.
#
# Cases 1-10 (original S23b-equivalent set), each run against the cheap
# `print-obj-dir` introspection target (no real compile/link needed to
# prove the parse-time gate fires or doesn't):
#   1. EXTRA_CFLAGS="'-Ofast'"      (single-quoted)         -> must FAIL,
#      log contains "FP policy conflict" AND "single-quote"
#   2. EXTRA_CFLAGS='"-ffast-math"' (double-quoted)         -> must FAIL,
#      log contains "FP policy conflict" AND "outside the allowed set" AND
#      the specific character '"' in the reported disallowed-character set
#      (the allow-list redesign moved this off a dedicated "double-quote"
#      message -- double quote is simply not in the allowed character set)
#   3. EXTRA_CFLAGS="-O'f'ast"      (quote-split mid-token) -> must FAIL,
#      log contains "FP policy conflict" AND "single-quote" (unchanged: the
#      single-quote pre-check still fires first for any single quote)
#   4. EXTRA_CFLAGS='@flags.rsp'    (response-file)         -> must FAIL,
#      log contains "FP policy conflict" AND "outside the allowed set" AND
#      the specific character '@' (same allow-list redesign as case 2 --
#      '@' is simply not in the allowed set)
#   5. positive control: EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness AND (a
#      second sub-case) EXTRA_CFLAGS=-DTEXT=ffast-math -- both harmless,
#      UNQUOTED tokens that merely CONTAIN -Ofast/ffast-math as a substring
#      of a larger identifier, not the compiler flag itself -> both must
#      SUCCEED (the exact-token $(filter) check must not false-positive on
#      either, exactly the class of false rejection the old
#      $(findstring ...) check used to produce, AND every character in
#      both tokens is in the allow-list's own allowed set)
#
# Also re-confirms the OTHER two things a naive re-implementation could
# regress: (i) an actual bare -Ofast/-ffast-math/-ffp-contract=<x> token,
# unquoted, in CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS, is still rejected --
# including via CXXFLAGS/CPPFLAGS specifically now that the check has been
# widened from CFLAGS-only; and (ii) the pre-existing, unrelated
# "Command-line override rejection" foreach (a command-line CFLAGS=/
# CXXFLAGS=/CPPFLAGS=/LDFLAGS=/FP_POLICY=/FP_ALLOWED_CHARS_RE= override, or
# `make -e` with any of those six set in the environment) still fires first
# and is untouched by this change.
#
# Cases 11-17 (allow-list bypass shapes the OLD 9-item deny-list never
# caught -- audio_common S24 equivalent): glob-star, glob-question,
# glob-brackets, tilde, redirect-out, redirect-in, process-subst -- each a
# REAL, live bypass of the old deny-list (none of these characters were
# checked for at all before this redesign), each now REJECTED by the
# character-class allow-list, identifying the specific offending
# character(s).
#
# Case 18: positive control, distinct code path: a
# Make-native ${VAR}-style expansion must still be rejected via the
# EXACT-TOKEN $(filter) check, not the character allow-list -- GNU Make
# treats ${...} identically to $(...) and resolves it to R11_FLAG's value
# (-Ofast) before either FP-policy check ever runs, so the resulting
# FP_INPUT_FLAGS text is the plain, clean, ALL-ALLOWED-CHARACTERS token
# "-Ofast" -- the character allow-list has nothing to catch here; only the
# unchanged FP_CONFLICT_FLAGS $(filter) check can still reject it.
#
# Cases 19-26: the FP-policy check's OWN internal variables
# (FP_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC/FP_ALLOWED_CHARS_RE/
# FP_CONFLICT_FLAGS) must themselves be rejected as a command-line override
# AND under `-e` -- each of the four names, both ways -- exactly like
# FP_POLICY/CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS already are, so overriding an
# internal checker variable can never silently defeat the checks above it
# while the real CFLAGS/CXXFLAGS/CPPFLAGS content still reaches the compiler
# unvalidated. (SHELL_SAFE_ALLOWLIST_RC is this fix's rename of what used to
# be FP_ALLOWLIST_RC -- see the "link-flags character-safety coverage"
# section below.)
#
# LINK-FLAGS (LDFLAGS/EXTRA_LDFLAGS) CHARACTER-SAFETY COVERAGE (Codex
# review, replicated from audio_common's identical fix -- see that repo's
# Makefile/test suite (search "S25") for the full writeup this section
# mirrors): the allow-list above originally validated FP_INPUT_FLAGS only
# (CFLAGS/CXXFLAGS/CPPFLAGS) -- LDFLAGS (a plain `-lm` literal plus whatever
# EXTRA_LDFLAGS folds in) was never inspected at all, even though it is
# embedded in every real link recipe ($(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS))
# exactly the same way CFLAGS is embedded in every compile recipe.
# Confirmed empirically: `make print-obj-dir
# EXTRA_LDFLAGS=';echo LINK_FLAG_INJECTION'` used to pass every check above
# untouched -- no FP policy error at all -- while the exact same payload via
# EXTRA_CFLAGS on the same target was correctly rejected. Fixed in the
# Makefile by introducing SHELL_SAFE_INPUT_FLAGS (= FP_INPUT_FLAGS plus the
# fully-assembled LDFLAGS) and routing the single-quote pre-check and the
# allow-list itself through it instead of FP_INPUT_FLAGS directly
# (FP_ALLOWLIST_RC -> SHELL_SAFE_ALLOWLIST_RC / FP_DISALLOWED_CHARS ->
# SHELL_SAFE_DISALLOWED_CHARS, renamed to make the widened scope explicit).
# The cases further below (after case 26) replicate this file's own
# EXTRA_CFLAGS matrix -- the literal reported injection payload, the
# glob/tilde/redirect/process-substitution set (this file's cases 11-17),
# and a positive control -- against EXTRA_LDFLAGS instead, and extend the
# internal-variable override-rejection coverage to the two names this fix
# introduces/renames (SHELL_SAFE_INPUT_FLAGS is new; SHELL_SAFE_ALLOWLIST_RC
# is FP_ALLOWLIST_RC's rename), this time via an EXTRA_LDFLAGS payload
# specifically (cases 19-26's own EXTRA_CFLAGS payload already covers the
# CFLAGS-side vector; an override here must not be able to let an
# EXTRA_LDFLAGS injection payload back through either).
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
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/aec-fp-flag-policy.XXXXXX")" || {
    echo "FATAL: mktemp failed to create a scratch directory" >&2
    exit 1
}
[ -n "$SCRATCH" ] && [ -d "$SCRATCH" ] || {
    echo "FATAL: mktemp reported success but SCRATCH is empty or not a directory" >&2
    exit 1
}
cleanup() { [ -n "$SCRATCH" ] && [ -d "$SCRATCH" ] && rm -rf "$SCRATCH"; }
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

# Case 2: double-quoted. Allow-list redesign: double quote is caught by the
# character-class allow-list (not a dedicated "double-quote" deny-list
# entry any more) -- the reported disallowed-character set is a single '"'.
LOG2="$SCRATCH/log_double_quoted"
if env EXTRA_CFLAGS='"-ffast-math"' "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG2" 2>&1; then
    fail "case 2: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
    cat "$LOG2" >&2
else
    if grep -q "FP policy conflict" "$LOG2" && grep -q "outside the allowed set" "$LOG2" && grep -qF 'found: """' "$LOG2"; then
        pass "case 2: env EXTRA_CFLAGS='\"-ffast-math\"' (double-quoted) print-obj-dir correctly FAILS, identifying the double-quote character via the allow-list"
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

# Case 4: @response-file syntax. Allow-list redesign: '@' is caught by the
# character-class allow-list (it is simply not in the allowed set), not a
# dedicated "response-file" deny-list entry any more.
LOG4="$SCRATCH/log_response_file"
if env EXTRA_CFLAGS='@flags.rsp' "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG4" 2>&1; then
    fail "case 4: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
    cat "$LOG4" >&2
else
    if grep -q "FP policy conflict" "$LOG4" && grep -q "outside the allowed set" "$LOG4" && grep -qF 'found: "@"' "$LOG4"; then
        pass "case 4: env EXTRA_CFLAGS='@flags.rsp' (response-file) print-obj-dir correctly FAILS, identifying the @-prefixed response-file token via the allow-list"
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

echo "=== run_fp_flag_policy_test: prior behaviour must not regress ==="

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
        pass "case 6: env EXTRA_CFLAGS=-Ofast print-obj-dir correctly FAILS (prior behaviour preserved)"
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
# must still fire first and untouched -- a command-
# line CFLAGS= override, and `make -e` with FP_POLICY= set in the
# environment, are both still rejected by that separate foreach, never
# reaching (or being affected by) this fix's own conflict-detection block.
LOG9="$SCRATCH/log_cmdline_cflags_override"
if "$MAKE_BIN" CFLAGS=-O3 BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG9" 2>&1; then
    fail "case 9: make CFLAGS=-O3 print-obj-dir unexpectedly SUCCEEDED (command-line override rejection regressed)"
    cat "$LOG9" >&2
else
    if grep -q "cannot be overridden" "$LOG9"; then
        pass "case 9: make CFLAGS=-O3 print-obj-dir correctly FAILS, mentioning 'cannot be overridden' (untouched by this fix)"
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
echo "=== run_fp_flag_policy_test: FP-policy allow-list bypass shapes the OLD"
echo "=== 9-item deny-list never caught (audio_common S24 equivalent) ==="

# Cases 11-17: real, live bypasses of the OLD deny-list (verified BEFORE the
# allow-list fix: none of these characters/constructs were checked for at
# all), each now REJECTED by the character-class allow-list, identifying the
# specific offending character(s).
fp_check_rejected() {
    label="$1"; flag_value="$2"; expect_chars="$3"
    log="$SCRATCH/log_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_')"
    if env EXTRA_CFLAGS="$flag_value" "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$log" 2>&1; then
        fail "case $label: EXTRA_CFLAGS='$flag_value' print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
        cat "$log" >&2
    elif grep -q "FP policy conflict" "$log" && grep -q "outside the allowed set" "$log" && grep -qF "found: \"$expect_chars\"" "$log"; then
        pass "case $label: EXTRA_CFLAGS='$flag_value' print-obj-dir correctly FAILS, identifying \"$expect_chars\""
    else
        fail "case $label: EXTRA_CFLAGS='$flag_value' print-obj-dir failed but did NOT identify \"$expect_chars\" specifically"
        cat "$log" >&2
    fi
}

fp_check_rejected "11_glob_star"     '-O*t'           '*'
fp_check_rejected "12_glob_question" '-Ofas?'         '?'
fp_check_rejected "13_glob_brackets" '-Ofas[t]'       '[]'
fp_check_rejected "14_tilde"         '~/pwned'        '~'
fp_check_rejected "15_redirect_out"  '-I>/tmp/evil'   '>'
fp_check_rejected "16_redirect_in"   '-I</etc/passwd' '<'
fp_check_rejected "17_process_subst" '<(echo hi)'     '()<'

# Case 18: positive control (distinct code path) -- a Make-native
# ${VAR}-style expansion must still be rejected via the EXACT-TOKEN
# $(filter) check, not the character allow-list -- GNU Make treats ${...}
# identically to $(...) and resolves it to R11_FLAG's value (-Ofast) before
# either FP-policy check ever runs, so the resulting FP_INPUT_FLAGS text is
# the plain, clean, ALL-ALLOWED-CHARACTERS token "-Ofast" -- the character
# allow-list has nothing to catch here; only the unchanged FP_CONFLICT_FLAGS
# $(filter) check can still reject it. R11_FLAG must be `export`-ed as a
# real shell variable first (a bare `env VAR=x cmd ${VAR}` does NOT make VAR
# visible to the CURRENT shell's own expansion of ${VAR} in that same
# command line), and 'EXTRA_CFLAGS=${R11_FLAG}' must be SINGLE-QUOTED on
# make's own command line so the invoking shell does NOT resolve
# ${R11_FLAG} itself, leaving the literal text for GNU MAKE to resolve
# (Make auto-imports every environment variable, so R11_FLAG is visible to
# it once exported). Deliberately `-n` (a dry run) -- the FP-policy checks
# run at Makefile parse time, before any target executes, so a dry run
# alone still exercises this path.
export R11_FLAG=-Ofast
LOG18="$SCRATCH/log_18_r11_dollar_brace"
if "$MAKE_BIN" -n BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" 'EXTRA_CFLAGS=${R11_FLAG}' lib >"$LOG18" 2>&1; then
    fail "case 18: R11_FLAG=-Ofast (exported) make -n EXTRA_CFLAGS=\${R11_FLAG} lib unexpectedly SUCCEEDED (must be rejected via the exact-token filter)"
    cat "$LOG18" >&2
else
    if grep -q "FP policy conflict" "$LOG18" && grep -q -- "-Ofast" "$LOG18" && ! grep -q "outside the allowed set" "$LOG18"; then
        pass "case 18: R11_FLAG=-Ofast (exported) make -n EXTRA_CFLAGS=\${R11_FLAG} lib correctly FAILS via the exact-token \$(filter) check (not the character allow-list)"
    else
        fail "case 18: R11_FLAG=-Ofast (exported) make -n EXTRA_CFLAGS=\${R11_FLAG} lib failed but not via the expected exact-token filter path"
        cat "$LOG18" >&2
    fi
fi
unset R11_FLAG

echo
echo "=== run_fp_flag_policy_test: FP-policy check's own internal variables"
echo "=== must themselves be override-proof (cmdline + make -e) ==="

# Cases 19-26: the FP-policy check's own INTERNAL variables (FP_INPUT_FLAGS/
# SHELL_SAFE_ALLOWLIST_RC/FP_ALLOWED_CHARS_RE/FP_CONFLICT_FLAGS) were
# themselves overridable from the command line / under `-e`, silently
# defeating every check above while the REAL CFLAGS/CXXFLAGS/CPPFLAGS used
# in actual compile recipes still carried the dangerous content. payload is
# the EXTRA_CFLAGS value that would (absent the override) be caught by the
# check the override under test defeats: FP_INPUT_FLAGS/
# SHELL_SAFE_ALLOWLIST_RC/FP_ALLOWED_CHARS_RE gate the CHARACTER allow-list,
# so a semicolon payload exercises that path; FP_CONFLICT_FLAGS gates the
# SEPARATE
# -Ofast/-ffast-math/-ffp-contract= exact-token filter, so it needs an
# all-allowed-characters payload that is still a real conflict ("-Ofast"
# itself) -- a semicolon payload there would be caught by the (untouched)
# allow-list first and never actually exercise the FP_CONFLICT_FLAGS-
# specific override path at all.
fp_check_override_rejected() {
    varname="$1"; varvalue="$2"; payload="$3"; use_dash_e="$4"
    log="$SCRATCH/log_override_$(printf '%s' "$varname" | tr -c 'A-Za-z0-9' '_')_$use_dash_e"
    if [ "$use_dash_e" = "dashe" ]; then
        if env "$varname=$varvalue" "$MAKE_BIN" -e BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" EXTRA_CFLAGS="$payload" print-obj-dir >"$log" 2>&1; then
            fail "case: env $varname=$varvalue make -e print-obj-dir unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the FP-policy checks)"
            cat "$log" >&2
        elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
            pass "case: env $varname=$varvalue make -e print-obj-dir correctly FAILS, mentioning '$varname cannot be overridden'"
        else
            fail "case: env $varname=$varvalue make -e print-obj-dir failed but did NOT mention '$varname cannot be overridden'"
            cat "$log" >&2
        fi
    else
        if "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" EXTRA_CFLAGS="$payload" "$varname=$varvalue" print-obj-dir >"$log" 2>&1; then
            fail "case: make $varname=$varvalue print-obj-dir unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the FP-policy checks)"
            cat "$log" >&2
        elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
            pass "case: make $varname=$varvalue print-obj-dir correctly FAILS, mentioning '$varname cannot be overridden'"
        else
            fail "case: make $varname=$varvalue print-obj-dir failed but did NOT mention '$varname cannot be overridden'"
            cat "$log" >&2
        fi
    fi
}
fp_check_override_rejected "FP_INPUT_FLAGS"        "clean" '-O2;rm' cmdline
fp_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"   '-O2;rm' cmdline
fp_check_override_rejected "FP_ALLOWED_CHARS_RE"   ".*"    '-O2;rm' cmdline
fp_check_override_rejected "FP_CONFLICT_FLAGS"     ""      '-Ofast' cmdline
fp_check_override_rejected "FP_INPUT_FLAGS"        "clean" '-O2;rm' dashe
fp_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"   '-O2;rm' dashe
fp_check_override_rejected "FP_ALLOWED_CHARS_RE"   ".*"    '-O2;rm' dashe
fp_check_override_rejected "FP_CONFLICT_FLAGS"     ""      '-Ofast' dashe

echo
echo "=== run_fp_flag_policy_test: link-flags (LDFLAGS/EXTRA_LDFLAGS)"
echo "=== character-safety coverage ==="

# The literal reported repro: a semicolon payload via EXTRA_LDFLAGS,
# dry-run-confirmed BEFORE this fix to sail through untouched and land live
# in a real link recipe (`make -n selftest
# EXTRA_LDFLAGS=';echo LINK_FLAG_INJECTION'` -- and, more directly,
# `make -n print-obj-dir` with the same override, which never even reaches
# a compile/link recipe -- both showed no FP-policy rejection at all before
# this fix, unlike the identical payload via EXTRA_CFLAGS).
ldflags_check_rejected() {
    label="$1"; flag_value="$2"; expect_chars="$3"
    log="$SCRATCH/log_ldflags_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_')"
    if env EXTRA_LDFLAGS="$flag_value" "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$log" 2>&1; then
        fail "case ldflags-$label: EXTRA_LDFLAGS='$flag_value' print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
        cat "$log" >&2
    elif grep -q "FP policy conflict" "$log" && grep -q "outside the allowed set" "$log" && grep -qF "found: \"$expect_chars\"" "$log"; then
        pass "case ldflags-$label: EXTRA_LDFLAGS='$flag_value' print-obj-dir correctly FAILS, identifying \"$expect_chars\""
    else
        fail "case ldflags-$label: EXTRA_LDFLAGS='$flag_value' print-obj-dir failed but did NOT identify \"$expect_chars\" specifically"
        cat "$log" >&2
    fi
}

ldflags_check_rejected "injection-semicolon" ';echo LINK_FLAG_INJECTION' ';'

# Same glob/tilde/redirect/process-substitution matrix cases 11-17 run for
# EXTRA_CFLAGS above, replicated verbatim for EXTRA_LDFLAGS.
ldflags_check_rejected "glob-star"        '-L*t'            '*'
ldflags_check_rejected "glob-question"    '-Lfas?'          '?'
ldflags_check_rejected "glob-brackets"    '-Lfas[t]'        '[]'
ldflags_check_rejected "tilde"            '~/pwned'         '~'
ldflags_check_rejected "redirect-out"     '-lm>/tmp/evil'   '>'
ldflags_check_rejected "redirect-in"      '-lm</etc/passwd' '<'
ldflags_check_rejected "process-subst"    '<(echo hi)'      '()<'

# Positive control: a real, legitimate link flag this project's own
# Makefile comment anticipates a consumer needing (the "-Wl,-rpath,dir /
# -Wa,--option pass-through flags" example the allow-list's own char-set
# comment gives) must pass the character check AND actually produce a
# working aec_wav binary. Unlike every case above, this exercises a REAL
# link recipe ($(TARGET), which embeds $(LDFLAGS) directly) rather than the
# cheap print-obj-dir query target, so -- like every other real build this
# repo performs -- it needs audio_common present as a sibling checkout
# (AC_DIR); the query-only cases above never touch audio_common at all.
LOG_LDFLAGS_POS="$SCRATCH/log_ldflags_positive_rpath"
POS_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib'
if env EXTRA_LDFLAGS="$POS_LDFLAGS" "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" all >"$LOG_LDFLAGS_POS" 2>&1; then
    # Resolve the just-built binary's own path from the build's own
    # "aec_wav [<backend>] -> <path>" echo line (the `all` recipe's last
    # line) rather than a separate `print-bin-dir` query invocation: a
    # query-only invocation never resolves AC_LIB, so its CFG_SIG folds in
    # an EMPTY AC_PRODUCER_SIG and resolves to a DIFFERENT (query-only)
    # bin/ directory than the real build above, which does resolve AC_LIB
    # and folds in a real producer signature -- confirmed empirically.
    bin_path="$(sed -n 's/^aec_wav \[[^]]*\] -> //p' "$LOG_LDFLAGS_POS" | tail -1)"
    # `aec_wav --help` is not a recognized option (this binary only takes
    # positional mic/ref/out args) -- it prints its usage banner and exits
    # NON-zero, same as any other invalid invocation. So the correctness bar
    # here is "the binary actually ran and printed its usage text" (proof
    # the -Wl,-rpath/-L payload landed in a real, loadable, executing
    # binary), not a zero exit status.
    RUN_OUT="$SCRATCH/run_ldflags_positive.out"
    "$bin_path" --help >"$RUN_OUT" 2>&1
    if [ -n "$bin_path" ] && [ -x "$bin_path" ] && grep -q "^Usage:" "$RUN_OUT"; then
        pass "case ldflags-positive: EXTRA_LDFLAGS='$POS_LDFLAGS' all builds a real, working aec_wav binary (legitimate link flag not rejected)"
    else
        fail "case ldflags-positive: EXTRA_LDFLAGS='$POS_LDFLAGS' all succeeded but the resulting aec_wav binary did not run"
        cat "$LOG_LDFLAGS_POS" >&2
        cat "$RUN_OUT" >&2
    fi
else
    fail "case ldflags-positive: EXTRA_LDFLAGS='$POS_LDFLAGS' all unexpectedly FAILED (false-positive rejection of a legitimate link flag)"
    cat "$LOG_LDFLAGS_POS" >&2
fi

# Command-line/`-e` override-rejection for the two variables this fix
# introduces/renames (SHELL_SAFE_INPUT_FLAGS is new; SHELL_SAFE_ALLOWLIST_RC
# is FP_ALLOWLIST_RC's rename) -- cases 19-26 above already cover both via
# an EXTRA_CFLAGS payload; this repeats it via EXTRA_LDFLAGS specifically,
# since that is the exact vector this fix exists to close (an override here
# must not be able to let an EXTRA_LDFLAGS injection payload back through).
ldflags_check_override_rejected() {
    varname="$1"; varvalue="$2"; payload="$3"; use_dash_e="$4"
    log="$SCRATCH/log_ldflags_override_$(printf '%s' "$varname" | tr -c 'A-Za-z0-9' '_')_$use_dash_e"
    if [ "$use_dash_e" = "dashe" ]; then
        if env "$varname=$varvalue" "$MAKE_BIN" -e BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" EXTRA_LDFLAGS="$payload" print-obj-dir >"$log" 2>&1; then
            fail "case: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the link-flags character-safety check)"
            cat "$log" >&2
        elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
            pass "case: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) correctly FAILS, mentioning '$varname cannot be overridden'"
        else
            fail "case: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) failed but did NOT mention '$varname cannot be overridden'"
            cat "$log" >&2
        fi
    else
        if "$MAKE_BIN" BACKEND=kiss OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" EXTRA_LDFLAGS="$payload" "$varname=$varvalue" print-obj-dir >"$log" 2>&1; then
            fail "case: make $varname=$varvalue print-obj-dir (EXTRA_LDFLAGS payload) unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the link-flags character-safety check)"
            cat "$log" >&2
        elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
            pass "case: make $varname=$varvalue print-obj-dir (EXTRA_LDFLAGS payload) correctly FAILS, mentioning '$varname cannot be overridden'"
        else
            fail "case: make $varname=$varvalue print-obj-dir (EXTRA_LDFLAGS payload) failed but did NOT mention '$varname cannot be overridden'"
            cat "$log" >&2
        fi
    fi
}
ldflags_check_override_rejected "SHELL_SAFE_INPUT_FLAGS"  "clean" ';rm' cmdline
ldflags_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     ';rm' cmdline
ldflags_check_override_rejected "SHELL_SAFE_INPUT_FLAGS"  "clean" ';rm' dashe
ldflags_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     ';rm' dashe

echo
echo "TOTAL: $((PASS_COUNT + FAIL_COUNT))  PASS: $PASS_COUNT  FAIL: $FAIL_COUNT"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "run_fp_flag_policy_test: ALL PASS"
    exit 0
else
    echo "run_fp_flag_policy_test: FAIL"
    exit 1
fi
