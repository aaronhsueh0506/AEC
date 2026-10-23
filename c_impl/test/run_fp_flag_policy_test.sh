#!/bin/sh
# run_fp_flag_policy_test.sh - regression test for the Makefile's flag-policy gates: the character allow-list and the
# CFLAGS/CXXFLAGS/CPPFLAGS exact-token conflict check (see "FP_INPUT_FLAGS" in ../Makefile).
#
# Contract: a bare -Ofast/-ffast-math/-ffp-contract=<x> token in CFLAGS, CXXFLAGS, CPPFLAGS or EXTRA_CFLAGS is rejected, while a
# harmless token that merely contains such text (-DROUND9_NOTE=-Ofastness) is accepted. Before that check, every character
# outside [A-Za-z0-9_./=,+- ] (plus space) is rejected, which covers quoting, escaping, globs, ~, redirection, process
# substitution and @response-file shapes. The single-quote check stays a separate Make-native pre-check because it gates the
# allow-list's own single-quoted $(shell) embedding.
#
# Cases (mostly against the cheap `print-obj-dir` target, since the gates fire at Makefile parse time):
#   1-4    single-quoted, double-quoted, quote-split and @response-file EXTRA_CFLAGS -> must FAIL, naming the character
#   5      positive control: -DROUND9_NOTE=-Ofastness and -DTEXT=ffast-math -> must SUCCEED
#   6-8    bare -Ofast / -ffast-math via EXTRA_CFLAGS, CXXFLAGS and CPPFLAGS -> must FAIL
#   9-10   command-line CFLAGS= and `make -e` with FP_POLICY= still hit the override-rejection foreach first
#   11-17  glob, ~, redirection and process-substitution payloads -> rejected by the allow-list
#   18     ${VAR} expansion to -Ofast -> rejected by the exact-token $(filter), not the allow-list
#   19-26  the checks' own internal variables must be rejected as a command-line override and under `-e`
#   then   the same payload matrix against EXTRA_LDFLAGS (LDFLAGS is embedded in every link recipe), a positive control
#          with a legitimate link flag, and the internal-variable overrides again via an EXTRA_LDFLAGS payload
#
# Usage: ./test/run_fp_flag_policy_test.sh   (cd's to the repo root itself; scratch-isolated OBJ_ROOT/BIN_ROOT under one
# mktemp -d, removed by a trap on normal exit and both signals; the repo's real obj/bin trees are never touched)
# Exit 0 and "run_fp_flag_policy_test: ALL PASS" mean every case behaved as expected; a "FAIL" line is a regression.

set -u
cd "$(dirname "$0")/.."

# Explicit template: a bare `mktemp -d` ignores $TMPDIR on some hosts; the template honors it.
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
if env EXTRA_CFLAGS="'-Ofast'" "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG1" 2>&1; then
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

# Case 2: double-quoted. The character allow-list catches it; the reported disallowed set is a single '"'.
LOG2="$SCRATCH/log_double_quoted"
if env EXTRA_CFLAGS='"-ffast-math"' "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG2" 2>&1; then
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
if env EXTRA_CFLAGS="-O'f'ast" "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG3" 2>&1; then
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

# Case 4: @response-file syntax. '@' is not in the allowed set, so the allow-list catches it.
LOG4="$SCRATCH/log_response_file"
if env EXTRA_CFLAGS='@flags.rsp' "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG4" 2>&1; then
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
if env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness "$MAKE_BIN" -s BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG5A" 2>&1; then
    pass "case 5a: env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness print-obj-dir (positive control, no forbidden characters) succeeds"
else
    fail "case 5a: env EXTRA_CFLAGS=-DROUND9_NOTE=-Ofastness print-obj-dir unexpectedly FAILED (false-positive on a harmless token)"
    cat "$LOG5A" >&2
fi

LOG5B="$SCRATCH/log_positive_dtext"
if env EXTRA_CFLAGS=-DTEXT=ffast-math "$MAKE_BIN" -s BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG5B" 2>&1; then
    pass "case 5b: env EXTRA_CFLAGS=-DTEXT=ffast-math print-obj-dir (positive control, no forbidden characters) succeeds"
else
    fail "case 5b: env EXTRA_CFLAGS=-DTEXT=ffast-math print-obj-dir unexpectedly FAILED (false-positive on a harmless token)"
    cat "$LOG5B" >&2
fi

echo "=== run_fp_flag_policy_test: prior behaviour must not regress ==="

# A real, unquoted -Ofast/-ffast-math/-ffp-contract=<x> must still be rejected: via EXTRA_CFLAGS (case 6) and directly via
# CXXFLAGS/CPPFLAGS (cases 7-8; a plain environment value folds in through the override-rejection foreach's origin logic, not
# through EXTRA_CFLAGS).
LOG6="$SCRATCH/log_plain_ofast_extra_cflags"
if env EXTRA_CFLAGS=-Ofast "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG6" 2>&1; then
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
if env CXXFLAGS=-Ofast "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG7" 2>&1; then
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
if env CPPFLAGS=-ffast-math "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG8" 2>&1; then
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

# The command-line-override / `make -e` rejection must still fire first: a command-line CFLAGS= (case 9) and `make -e` with
# FP_POLICY= in the environment (case 10).
LOG9="$SCRATCH/log_cmdline_cflags_override"
if "$MAKE_BIN" CFLAGS=-O3 BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG9" 2>&1; then
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
if env FP_POLICY=-ffp-contract=fast "$MAKE_BIN" -e BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$LOG10" 2>&1; then
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
echo "=== 9-item deny-list never caught (audiolib S24 equivalent) ==="

# Cases 11-17: glob, ~, redirection and process-substitution payloads, each rejected by the character allow-list and
# identified by its offending character(s).
fp_check_rejected() {
    label="$1"; flag_value="$2"; expect_chars="$3"
    log="$SCRATCH/log_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_')"
    if env EXTRA_CFLAGS="$flag_value" "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$log" 2>&1; then
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

# Case 18: positive control on a distinct code path. Make resolves a ${VAR} expansion to the plain token "-Ofast" before either
# check runs, so only the exact-token $(filter) check, not the character allow-list, can reject it. R11_FLAG must be exported
# so Make imports it, and EXTRA_CFLAGS='${R11_FLAG}' must be single-quoted so the invoking shell leaves the text for Make.
# Run with -n: the checks fire at Makefile parse time, before any target executes.
export R11_FLAG=-Ofast
LOG18="$SCRATCH/log_18_r11_dollar_brace"
if "$MAKE_BIN" -n BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" 'EXTRA_CFLAGS=${R11_FLAG}' lib >"$LOG18" 2>&1; then
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

# Cases 19-26: the checks' internal variables (FP_INPUT_FLAGS, SHELL_SAFE_ALLOWLIST_RC, FP_ALLOWED_CHARS_RE, FP_CONFLICT_FLAGS)
# must be rejected as a command-line override and under `-e`, because an override would silently defeat the checks above them.
# payload is an EXTRA_CFLAGS value that the check under test would catch: a semicolon for the character allow-list variables,
# and "-Ofast" for FP_CONFLICT_FLAGS (a semicolon would be caught by the allow-list first and never reach the conflict check).
fp_check_override_rejected() {
    varname="$1"; varvalue="$2"; payload="$3"; use_dash_e="$4"
    log="$SCRATCH/log_override_$(printf '%s' "$varname" | tr -c 'A-Za-z0-9' '_')_$use_dash_e"
    if [ "$use_dash_e" = "dashe" ]; then
        if env "$varname=$varvalue" "$MAKE_BIN" -e BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" EXTRA_CFLAGS="$payload" print-obj-dir >"$log" 2>&1; then
            fail "case: env $varname=$varvalue make -e print-obj-dir unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the FP-policy checks)"
            cat "$log" >&2
        elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
            pass "case: env $varname=$varvalue make -e print-obj-dir correctly FAILS, mentioning '$varname cannot be overridden'"
        else
            fail "case: env $varname=$varvalue make -e print-obj-dir failed but did NOT mention '$varname cannot be overridden'"
            cat "$log" >&2
        fi
    else
        if "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" EXTRA_CFLAGS="$payload" "$varname=$varvalue" print-obj-dir >"$log" 2>&1; then
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

# The reported injection payload: a semicolon via EXTRA_LDFLAGS must be rejected like the same payload via EXTRA_CFLAGS.
ldflags_check_rejected() {
    label="$1"; flag_value="$2"; expect_chars="$3"
    log="$SCRATCH/log_ldflags_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_')"
    if env EXTRA_LDFLAGS="$flag_value" "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" print-obj-dir >"$log" 2>&1; then
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

# The glob/tilde/redirect/process-substitution matrix of cases 11-17, run against EXTRA_LDFLAGS.
ldflags_check_rejected "glob-star"        '-L*t'            '*'
ldflags_check_rejected "glob-question"    '-Lfas?'          '?'
ldflags_check_rejected "glob-brackets"    '-Lfas[t]'        '[]'
ldflags_check_rejected "tilde"            '~/pwned'         '~'
ldflags_check_rejected "redirect-out"     '-lm>/tmp/evil'   '>'
ldflags_check_rejected "redirect-in"      '-lm</etc/passwd' '<'
ldflags_check_rejected "process-subst"    '<(echo hi)'      '()<'

# Positive control: a legitimate link flag (-Wl,-rpath,dir) must pass the character check and yield a working aec_wav binary.
# Unlike the cases above it exercises a real link recipe ($(TARGET) embeds $(LDFLAGS)), so it needs audiolib as a sibling
# checkout (COMMON_DIR).
LOG_LDFLAGS_POS="$SCRATCH/log_ldflags_positive_rpath"
POS_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib'
if env EXTRA_LDFLAGS="$POS_LDFLAGS" "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" all >"$LOG_LDFLAGS_POS" 2>&1; then
    # Take the binary's path from the build's own "aec_wav [<backend>] -> <path>" line (the last line of the `all` recipe): a
    # query-only `print-bin-dir` does not resolve COMMON_LIB, so its CFG_SIG differs and it names a different bin/ directory.
    bin_path="$(sed -n 's/^aec_wav \[[^]]*\] -> //p' "$LOG_LDFLAGS_POS" | tail -1)"
    # `aec_wav --help` is not a recognized option (the binary takes positional mic/ref/out args): it prints its usage banner and
    # exits non-zero. The bar is that the binary ran and printed usage, which proves the -Wl,-rpath/-L payload produced a
    # loadable executable; the exit status is not checked.
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

# Override rejection for SHELL_SAFE_INPUT_FLAGS and SHELL_SAFE_ALLOWLIST_RC via an EXTRA_LDFLAGS payload (cases 19-26 cover the
# EXTRA_CFLAGS vector): an override must not let an EXTRA_LDFLAGS injection payload through.
ldflags_check_override_rejected() {
    varname="$1"; varvalue="$2"; payload="$3"; use_dash_e="$4"
    log="$SCRATCH/log_ldflags_override_$(printf '%s' "$varname" | tr -c 'A-Za-z0-9' '_')_$use_dash_e"
    if [ "$use_dash_e" = "dashe" ]; then
        if env "$varname=$varvalue" "$MAKE_BIN" -e BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" EXTRA_LDFLAGS="$payload" print-obj-dir >"$log" 2>&1; then
            fail "case: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the link-flags character-safety check)"
            cat "$log" >&2
        elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
            pass "case: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) correctly FAILS, mentioning '$varname cannot be overridden'"
        else
            fail "case: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) failed but did NOT mention '$varname cannot be overridden'"
            cat "$log" >&2
        fi
    else
        if "$MAKE_BIN" BACKEND=ne10 OBJ_ROOT="$OBJ_ROOT" BIN_ROOT="$BIN_ROOT" EXTRA_LDFLAGS="$payload" "$varname=$varvalue" print-obj-dir >"$log" 2>&1; then
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
