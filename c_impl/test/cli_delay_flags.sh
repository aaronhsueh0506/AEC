#!/usr/bin/env bash
#
# cli_delay_flags.sh -- test of aec_wav's --delay-mode / --delay-num-filters / --fixed-delay / --print-mem-size options.
#
# Scope: CLI plumbing only. A value typed on the command line must land in the AecConfig handed to aec_create() and
# aec_get_mem_breakdown(); an illegal mode/field combination must be rejected, not clamped (aec_resolve is the
# single authority); and --print-mem-size must report the number the library itself returns. Bank-size geometry, the
# ring-size formula and the mode x field validation matrix belong to test_delay_num_filters.c, test_linear_context.c and
# test_config_validation.c.
#
# Run: cd c_impl && make && test/cli_delay_flags.sh   (or `make test-cli-delay-flags`, which builds both binaries first)
# Exit 0 and "cli_delay_flags: PASS" mean the options reach the engine.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIMPL="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$CIMPL/.." && pwd)"

MIC="$REPO/wav/aec_record/aec_record_mic_10s.wav"
REF="$REPO/wav/aec_record/aec_record_ref_10s.wav"
for f in "$MIC" "$REF"; do
    [ -f "$f" ] || { echo "cli_delay_flags: FAIL — missing input $f"; exit 1; }
done
MIC_SR=16000   # the fixture pair's actual sample rate — kept in sync with
               # print_mem_size_ref's first positional argument below.

BIN_DIR="$(cd "$CIMPL" && make print-bin-dir 2>/dev/null | tail -1)"
AEC_WAV="$BIN_DIR/aec_wav"
PMR="$BIN_DIR/print_mem_size_ref"
[ -x "$AEC_WAV" ] || { echo "cli_delay_flags: FAIL — no aec_wav at $AEC_WAV (run make first)"; exit 1; }
[ -x "$PMR" ]     || { echo "cli_delay_flags: FAIL — no print_mem_size_ref at $PMR (run make first)"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fails=0
ck() {  # ck <ok|no> <description>
    if [ "$1" = "ok" ]; then
        echo "  PASS  $2"
    else
        echo "  FAIL  $2"
        fails=$((fails + 1))
    fi
}

# ============================================================================
# 1. Plumbing: no flags -> level defaults are reported (options are opt-in)
# ============================================================================
out="$("$AEC_WAV" "$MIC" "$REF" "$TMP/o1.wav" --cng 2>&1 || true)"
if grep -q "delay_mode=matched num_filters=5 fixed_delay_samples=-1" <<<"$out"; then
    ck ok "no flags -> delay_mode=matched num_filters=5 fixed_delay_samples=-1"
else
    ck no "no flags -> delay_mode=matched num_filters=5 fixed_delay_samples=-1 (got: $(grep -o 'delay_mode=[a-z]* num_filters=[0-9-]* fixed_delay_samples=[0-9-]*' <<<"$out" || echo none))"
fi

# ============================================================================
# 2. --delay-mode reaches AecConfig (each legal on its own)
# ============================================================================
out="$("$AEC_WAV" "$MIC" "$REF" "$TMP/o2a.wav" --cng --delay-mode matched 2>&1 || true)"
if grep -q "delay_mode=matched" <<<"$out"; then ck ok "--delay-mode matched reaches cfg"
else ck no "--delay-mode matched reaches cfg"; fi

out="$("$AEC_WAV" "$MIC" "$REF" "$TMP/o2b.wav" --cng --delay-mode fixed --fixed-delay 1600 2>&1 || true)"
if grep -q "delay_mode=fixed" <<<"$out"; then ck ok "--delay-mode fixed reaches cfg"
else ck no "--delay-mode fixed reaches cfg"; fi

out="$("$AEC_WAV" "$MIC" "$REF" "$TMP/o2c.wav" --cng --delay-mode external 2>&1 || true)"
if grep -q "delay_mode=external" <<<"$out"; then ck ok "--delay-mode external reaches cfg"
else ck no "--delay-mode external reaches cfg"; fi

out="$("$AEC_WAV" "$MIC" "$REF" "$TMP/o2d.wav" --cng --delay-mode bogus 2>&1 || true)"
if grep -q "unknown --delay-mode" <<<"$out"; then ck ok "--delay-mode bogus rejected at CLI parse"
else ck no "--delay-mode bogus rejected at CLI parse"; fi

# ============================================================================
# 3. --delay-num-filters reaches AecConfig (MATCHED, legal range)
# ============================================================================
for n in 1 3 5; do
    out="$("$AEC_WAV" "$MIC" "$REF" "$TMP/o3.wav" --cng --delay-num-filters "$n" 2>&1 || true)"
    if grep -q "num_filters=$n " <<<"$out"; then ck ok "--delay-num-filters $n reaches cfg"
    else ck no "--delay-num-filters $n reaches cfg (got: $(grep -o 'num_filters=[0-9]*' <<<"$out" || echo none))"; fi
done

# ============================================================================
# 4. --fixed-delay reaches AecConfig (paired with --delay-mode fixed)
# ============================================================================
for d in 0 1600 8000; do
    out="$("$AEC_WAV" "$MIC" "$REF" "$TMP/o4.wav" --cng --delay-mode fixed --fixed-delay "$d" 2>&1 || true)"
    if grep -q "fixed_delay_samples=$d" <<<"$out"; then ck ok "--fixed-delay $d reaches cfg"
    else ck no "--fixed-delay $d reaches cfg (got: $(grep -o 'fixed_delay_samples=[0-9-]*' <<<"$out" || echo none))"; fi
done

# ============================================================================
# 5. Illegal mode/field combinations are REJECTED (aec_resolve is the sole authority): a bad combination must fail
#    the run, never be clamped or ignored into a different configuration. Check (a) below is the mutation target.
# ============================================================================
# (a) --fixed-delay negative under FIXED (the "unset" sentinel is -1; FIXED demands >= 0, so every negative value, -1
#     included, must be rejected). Clamping a negative value to 0 in aec_wav.c before the cfg assignment turns this check
#     and the checks of sections 6 and 9 red.
for bad in -1 -2 -100; do
    if "$AEC_WAV" "$MIC" "$REF" "$TMP/o5a.wav" --cng --delay-mode fixed --fixed-delay "$bad" >/dev/null 2>&1; then
        ck no "--delay-mode fixed --fixed-delay $bad rejected"
    else
        ck ok "--delay-mode fixed --fixed-delay $bad rejected"
    fi
done
# (b) --fixed-delay given without --delay-mode fixed (mode stays MATCHED,
#     which demands the sentinel -1 — see aec_resolve's switch).
if "$AEC_WAV" "$MIC" "$REF" "$TMP/o5b.wav" --cng --fixed-delay 1600 >/dev/null 2>&1; then
    ck no "--fixed-delay without --delay-mode fixed rejected"
else
    ck ok "--fixed-delay without --delay-mode fixed rejected"
fi
# (c) --delay-mode fixed given without --fixed-delay (fixed_delay_samples
#     stays the -1 default, which FIXED cannot honour).
if "$AEC_WAV" "$MIC" "$REF" "$TMP/o5c.wav" --cng --delay-mode fixed >/dev/null 2>&1; then
    ck no "--delay-mode fixed without --fixed-delay rejected"
else
    ck ok "--delay-mode fixed without --fixed-delay rejected"
fi
# (d) --delay-num-filters out of [1,5].
for bad in 0 6 99 -1; do
    if "$AEC_WAV" "$MIC" "$REF" "$TMP/o5d.wav" --cng --delay-num-filters "$bad" >/dev/null 2>&1; then
        ck no "--delay-num-filters $bad rejected"
    else
        ck ok "--delay-num-filters $bad rejected"
    fi
done
# (e) --delay-num-filters other than the default under a mode with no bank
#     to size (FIXED / external): a non-default n outside
#     MATCHED would silently promise a compute saving that mode already
#     gives in full.
if "$AEC_WAV" "$MIC" "$REF" "$TMP/o5e.wav" --cng --delay-mode fixed --fixed-delay 1600 --delay-num-filters 2 >/dev/null 2>&1; then
    ck no "--delay-mode fixed --delay-num-filters 2 rejected"
else
    ck ok "--delay-mode fixed --delay-num-filters 2 rejected"
fi
if "$AEC_WAV" "$MIC" "$REF" "$TMP/o5f.wav" --cng --delay-mode external --delay-num-filters 2 >/dev/null 2>&1; then
    ck no "--delay-mode external --delay-num-filters 2 rejected"
else
    ck ok "--delay-mode external --delay-num-filters 2 rejected"
fi

# ============================================================================
# 6. A rejected run must not leave a half-written output
# ============================================================================
rm -f "$TMP/o6.wav"
"$AEC_WAV" "$MIC" "$REF" "$TMP/o6.wav" --cng --delay-mode fixed --fixed-delay -1 >/dev/null 2>&1 || true
if [ -s "$TMP/o6.wav" ]; then ck no "rejected config writes no output"
else ck ok "rejected config writes no output"; fi

# ============================================================================
# 7. The default run is byte-identical to explicit --delay-mode matched --delay-num-filters 5 (the documented defaults):
#    the options are behaviour-neutral at their default values rather than taking a different code path
# ============================================================================
"$AEC_WAV" "$MIC" "$REF" "$TMP/def.wav"      --cng >/dev/null 2>&1
"$AEC_WAV" "$MIC" "$REF" "$TMP/explicit.wav" --cng --delay-mode matched --delay-num-filters 5 >/dev/null 2>&1
if cmp -s "$TMP/def.wav" "$TMP/explicit.wav"; then
    ck ok "no-flag == --delay-mode matched --delay-num-filters 5 (byte-identical)"
else
    ck no "no-flag == --delay-mode matched --delay-num-filters 5 (byte-identical)"
fi

# ============================================================================
# 8. --print-mem-size: prints and exits WITHOUT touching any audio
# ============================================================================
rm -f "$TMP/o8.wav"
out="$("$AEC_WAV" "$MIC" "$REF" "$TMP/o8.wav" --print-mem-size 2>&1)"
rc=$?
if [ "$rc" -eq 0 ]; then ck ok "--print-mem-size exits 0"
else ck no "--print-mem-size exits 0 (got $rc)"; fi
if [ -e "$TMP/o8.wav" ]; then ck no "--print-mem-size creates no output file"
else ck ok "--print-mem-size creates no output file"; fi
if grep -q "^mem: " <<<"$out"; then ck ok "--print-mem-size prints a 'mem: ' line"
else ck no "--print-mem-size prints a 'mem: ' line (got: $out)"; fi
if grep -qE "Processed [0-9]+ frames|^duty: " <<<"$out"; then
    ck no "--print-mem-size does not process audio (found a Processed/duty line)"
else
    ck ok "--print-mem-size does not process audio (no Processed/duty line)"
fi

# --- 8b. every required field is present -----------------
line="$(grep '^mem: ' <<<"$out")"
for field in sr= fft= hop= mode= n= fixed_delay_samples= total_bytes= estimator_bytes= ring_bytes=; do
    if grep -q "$field" <<<"$line"; then ck ok "print-mem-size line has $field"
    else ck no "print-mem-size line has $field (line: $line)"; fi
done

# ============================================================================
# 9. --print-mem-size on an illegal combination fails the SAME way (library
#    validation, fail-fast) as actually processing that config would.
# ============================================================================
if "$AEC_WAV" "$MIC" "$REF" "$TMP/o9.wav" --print-mem-size --delay-mode fixed --fixed-delay -5 >/dev/null 2>&1; then
    ck no "--print-mem-size rejects an illegal combination"
else
    ck ok "--print-mem-size rejects an illegal combination"
fi

# ============================================================================
# 10. MUTATION TARGET: the number --print-mem-size reports for total_bytes
#     must equal what the library's aec_get_mem_size()/aec_get_mem_breakdown()
#     actually return. Cross-checked against print_mem_size_ref, an
#     INDEPENDENT tool (test/print_mem_size_ref.c) that calls the same
#     library entry point from its own source file, so this catches a bug in
#     aec_wav.c's printf specifically (wrong field, transposed values, a
#     stale hardcoded number) — not a bug in the library itself.
# ============================================================================
get_field() {  # get_field <field-name> <line>
    grep -oE "$1=[0-9]+" <<<"$2" | head -1 | cut -d= -f2
}

check_mem_case() {  # check_mem_case <description> <ref-args...> -- <cli-args...>
    local desc="$1"; shift
    local ref_args=()
    while [ "$1" != "--" ]; do ref_args+=("$1"); shift; done
    shift
    local cli_args=("$@")

    local cli_out cli_line ref_out
    # "${arr[@]+"${arr[@]}"}" rather than a bare "${arr[@]}": bash 3.2
    # (macOS's default /bin/bash) treats expanding an EMPTY array under
    # `set -u` as an unbound-variable error (fixed upstream in bash 4.4+,
    # but this script has to run on the stock macOS shell too).
    cli_out="$("$AEC_WAV" "$MIC" "$REF" "$TMP/mem_case.wav" --print-mem-size "${cli_args[@]+"${cli_args[@]}"}" 2>&1)"
    cli_line="$(grep '^mem: ' <<<"$cli_out" || true)"
    ref_out="$("$PMR" "$MIC_SR" "${ref_args[@]+"${ref_args[@]}"}")"

    local cli_total cli_est cli_ring ref_total ref_est ref_ring
    cli_total="$(get_field total_bytes "$cli_line")"
    cli_est="$(get_field estimator_bytes "$cli_line")"
    cli_ring="$(get_field ring_bytes "$cli_line")"
    ref_total="$(get_field total_bytes "$ref_out")"
    ref_est="$(get_field estimator_bytes "$ref_out")"
    ref_ring="$(get_field ring_bytes "$ref_out")"

    if [ -n "$cli_total" ] && [ "$cli_total" = "$ref_total" ] \
            && [ "$cli_est" = "$ref_est" ] && [ "$cli_ring" = "$ref_ring" ]; then
        ck ok "$desc: total/estimator/ring == print_mem_size_ref ($cli_total/$cli_est/$cli_ring)"
    else
        ck no "$desc: total/estimator/ring == print_mem_size_ref (CLI: $cli_total/$cli_est/$cli_ring, ref: $ref_total/$ref_est/$ref_ring)"
    fi
}

check_mem_case "MATCHED n=5 (default), 16k/256"       --
check_mem_case "MATCHED n=1, 16k/256"                 --delay-num-filters 1 -- --delay-num-filters 1
check_mem_case "FIXED 1600, 16k/256"                  --delay-mode fixed --fixed-delay 1600 -- --delay-mode fixed --fixed-delay 1600
check_mem_case "EXTERNAL_ALIGNED, 16k/256"            --delay-mode external -- --delay-mode external
check_mem_case "MATCHED n=5 (default), 16k/512"       --fft-size 512 -- --fft-size 512

# ============================================================================
# 11. Integer options are parsed strictly: a typo exits 2 instead of reading as 0
# ============================================================================
for bad in "--fft-size abc" "--fft-size 512x" "--delay-num-filters 3x" "--delay-num-filters ''" "--fixed-delay 12.5" "--fixed-delay 99999999999"; do
    rc=0
    eval "\"\$AEC_WAV\" \"\$MIC\" \"\$REF\" \"\$TMP/o5.wav\" --cng $bad" >"$TMP/o5.log" 2>&1 || rc=$?
    if [ "$rc" = 2 ] && grep -q "is not an integer" "$TMP/o5.log"; then ck ok "$bad rejected at CLI parse (exit 2)"
    else ck no "$bad rejected at CLI parse (exit $rc)"; fi
done

if [ "$fails" -eq 0 ]; then
    echo "cli_delay_flags: PASS"
    exit 0
fi
echo "cli_delay_flags: FAIL ($fails check(s))"
exit 1
