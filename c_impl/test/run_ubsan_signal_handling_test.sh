#!/usr/bin/env bash
# run_ubsan_signal_handling_test.sh - permanent regression test for the
# signal-handling fix in test/run_counter_saturation_ubsan.sh and
# test/run_selftest_ubsan.sh (regression review).
#
# THE BUG THIS GUARDS AGAINST: both scripts used to register one combined
# trap for normal exit AND both signals:
#     trap 'rm -rf "$SCRATCH"' EXIT INT TERM
# The INT/TERM handler is just the cleanup command -- no explicit `exit`.
# Bash defers running a pending trap until the shell's current foreground
# child returns control to it, and then (absent an explicit exit) simply
# resumes the script at the next statement. Confirmed via direct repro
# (see session history): sending SIGTERM while the compiled test binary
# itself was running let the run continue to completion and print a false
# ">>> PASS" / "UBSan probe: PASS", exit 0, as if nothing had interrupted
# it -- exactly what a caller (timeout wrapper, supervisor doing graceful
# shutdown) must not see. Sending the signal during the earlier
# audio_common-build phase instead corrupted the run differently: the
# (no-op-for-continuation) trap deleted the scratch dir mid-flight,
# so the next step's archive-link failed with an "archive not found"-style
# message and rc=1 -- also not the correct, intentional signal-exit-code
# contract.
#
# THE FIX (identical in both scripts): explicit, separate handlers so a
# real signal always terminates the script with the conventional
# 128+signum code, while cleanup still always runs via the EXIT trap
# (`exit` from the INT/TERM handler re-triggers EXIT):
#     cleanup() { rm -rf "$SCRATCH"; }
#     trap cleanup EXIT
#     trap 'exit 130' INT
#     trap 'exit 143' TERM
#
# WHAT THIS TEST DOES, for BOTH scripts, sending SIGINT and separately
# SIGTERM at two distinct moments -- (i) during the audio_common-build
# phase (run_counter_saturation_ubsan.sh only; run_selftest_ubsan.sh has no
# such phase) and (ii) while the compiled test binary itself is executing:
#   - asserts the script's own final exit code is exactly 130 (SIGINT) /
#     143 (SIGTERM) at BOTH moments;
#   - asserts the captured stdout+stderr log does NOT contain the exact
#     string "UBSan probe: PASS" for any interrupted run (a substring match
#     on bare "PASS" would also catch the compiled test binary's OWN
#     legitimate internal ">>> PASS" / per-kernel "PASS foo" lines, which
#     are not the bug this guards against -- only the wrapper script's own
#     final success line is);
#   - asserts the scratch directory is fully removed afterward.
# It also re-confirms a normal, uninterrupted run of each script still
# prints "UBSan probe: PASS" and exits 0 (both default and
# --use-standard-math), and that this repo's own c_impl/obj + c_impl/bin
# and audio_common's obj + bin trees are byte-for-byte untouched
# (sha256 + mtime snapshot before/after) across ALL of the above runs, not
# just the normal one -- these two UBSan scripts are scratch-isolated
# specifically so a run (interrupted or not) can never touch a real build
# tree, and this is the permanent check that the isolation holds even
# under interruption.
#
# HOW SIGNALS ARE DELIVERED (read before "fixing" this if it ever looks
# racy): each trial launches the target script as `sh script &`, then waits
# for a readiness signal specific to the phase being tested, then sends
# `kill -s SIG` to *only* the script's own top-level PID (never its child
# make/cc/binary, never the process group). That is deliberate, not an
# oversight -- it is what makes the OLD script's bug (and the FIX)
# reproduce faithfully:
#   - Bash only acts on a pending trap once the *current foreground child*
#     of the target script returns control to it. Signalling the script's
#     PID while, say, the compiled binary is still running does not kill
#     that binary -- it runs to completion regardless, and only once it
#     exits does the (old, buggy or new, fixed) trap actually fire. This is
#     exactly why the old bug looked like "the run continued to completion
#     and printed a false PASS" instead of an abrupt kill.
#   - For the audio_common-build phase specifically, this test overrides
#     MAKE=<a tiny wrapper> to insert an artificial delay before the real
#     `make ... lib` call -- both scripts already support MAKE=... as an
#     explicit override hook, so this needs no edit to either script under
#     test, and it turns a build phase that is normally too fast (~0.3 s on
#     this host) to reliably land a signal in in into a wide, deterministic
#     window, without changing what "during the audio_common-build phase"
#     means from the target script's point of view (it is still the same
#     single foreground `"${MAKE:-make}" ... lib` line, just slower).
#   - `set -m` (job control) is REQUIRED in this harness. POSIX/bash forces
#     SIGINT and SIGQUIT to be ignored for asynchronous ("&") commands
#     started from a shell running WITHOUT job control, and a signal
#     ignored at shell-startup can never be trapped by that child
#     regardless of its own `trap ... INT` -- so without `set -m`,
#     `kill -INT` on the backgrounded script would be silently swallowed,
#     producing a false "nothing happened" reading for reasons entirely
#     unrelated to the fix under test. Verified empirically in this
#     session: an otherwise-identical harness without `set -m` reports "no
#     signal received, rc=0" for a trivial `trap 'exit 130' INT; sleep 5`
#     script; with `set -m` it correctly reports rc=130. SIGTERM is
#     unaffected by this auto-ignore rule (only INT/QUIT are), which is why
#     a SIGTERM-only harness would not have surfaced this at all.
#
# Usage: ./test/run_ubsan_signal_handling_test.sh
#   (from anywhere; cd's to this repo root itself via the script's own
#   location, same convention as the two scripts under test)
#
# Exit code 0 + "run_ubsan_signal_handling_test: ALL PASS" means every
# assertion above held. Nonzero + at least one "FAIL" line means a
# regression.
#
# ADDITIONAL REGRESSION COVERAGE (three more false-pass mechanisms verification identified in this
# test itself, not in the two scripts under test):
#   1. Both scripts' `SCRATCH="$(mktemp -d)"` (no template) silently ignores
#      $TMPDIR on this host, so the scratch-cleanup check below was
#      checking a directory that was ALWAYS empty regardless of the
#      script's real (elsewhere-located) scratch dir. Fixed in both scripts
#      to use an explicit `"${TMPDIR:-/tmp}/aec-<name>-ubsan.XXXXXX"`
#      template; `run_normal_trial` now asserts the emptiness too (not just
#      `run_signal_trial`), for symmetry.
#   2. `wait_for_binary` used a system-wide `pgrep -f`, so an unrelated
#      concurrent process whose command line happens to match the pattern
#      made it falsely report "ready" -- a signal could then land during
#      the wrong phase while every assertion still spuriously passed. Fixed
#      by scoping the match to an actual descendant of the trial's own
#      backgrounded pid (`is_descendant_of`, ancestry walk via
#      `ps -o ppid=`), with that pid threaded in as an explicit parameter.
#      `run_decoy_trial` below empirically proves the old code was fooled
#      by a same-pattern decoy and the new code is not.
#   3. `snapshot_real_trees`/`_sha` used BSD-only `stat -f`/`shasum`, each
#      silently swallowed via `2>/dev/null` -- on a host where neither
#      works (e.g. plain GNU/Linux), BOTH before/after snapshots would
#      become empty strings and trivially compare equal: a false PASS with
#      zero assurance, exactly when the tree-isolation check matters most.
#      Fixed with an up-front, fail-closed platform probe (real throwaway
#      file in WORK_ROOT) that picks a working stat/sha256 syntax once
#      before any trial runs, or FATAL-exits immediately if neither works.
#
# ADDITIONAL REGRESSION COVERAGE (two more independently confirmed issues, this time in
# THIS test's own process-cleanup and tree-snapshot logic):
#   4. run_decoy_trial's end-of-trial cleanup used a SYSTEM-WIDE
#      `pkill -9 -f "$binpat"` plus a `pgrep -f "$binpat"` leftover check,
#      keyed only on a fixed pattern string ("decoy_trial_ubsan_target") --
#      capable of killing (or being fooled by) an unrelated process
#      anywhere on the system whose command line happens to contain that
#      same text (another user's process, or a second concurrent run of
#      this very test suite). Fixed two ways: (a) the decoy/real_bin
#      pattern now carries a per-invocation-unique component (this script's
#      own $$ plus two RANDOM draws by default, or an explicit caller-
#      supplied id -- see run_decoy_trial/start_decoy_instance), so even a
#      fully independent concurrent run of this suite never shares a
#      literal pattern; (b) cleanup itself no longer does ANY name-based
#      pgrep/pkill -- it captures target_pid's children via
#      `pgrep -P "$target_pid"` while target_pid is still alive (an
#      orphaned child reparents once its parent dies, so the same query
#      against an already-dead target_pid would find nothing), kills each
#      captured child, target_pid, and decoy_pid individually by PID, then
#      busy-polls `kill -0` on those SPECIFIC PIDs until every one is
#      confirmed gone (see pid_scoped_cleanup). run_pkill_immunity_trial and
#      run_concurrent_decoy_trial below are permanent regression tests for,
#      respectively, an unrelated same-pattern process surviving a decoy
#      trial's cleanup untouched, and two concurrent decoy-trial-style
#      process sets (deliberately SHARING one pattern, to stress the
#      PID-scoped fix rather than lean on the name-uniqueness
#      defense-in-depth) never touching each other's processes.
#   5. snapshot_real_trees/_sha only ever walked `-type f`, so a new/
#      changed directory (e.g. a freshly created, still-empty keyed obj
#      dir) or symlink (new or retargeted) was invisible to the snapshot;
#      and a stat_line/sha_line failure on any individual file (permission
#      denied, a race with a concurrent writer, a tool bug) was silently
#      swallowed inside the `find | while read` pipeline -- the failure
#      never propagated, so a real unaccounted-for change (or a broken
#      verification) could leave "before" and "after" spuriously matching.
#      Fixed by snapshot_one_entry/snapshot_tree_root: each of the 4 real
#      roots gets an explicit PRESENT/ABSENT record; a PRESENT root is
#      walked for every directory/regular-file/symlink entry (not just
#      files) via `find ... -mindepth 1 \( -type d -o -type f -o -type l
#      \)`; directories just record path+type, symlinks record their
#      TARGET STRING via `readlink` (never by following the link), and
#      regular files also get a content hash via the existing sha_line/
#      stat_line helpers; and ANY failure -- readlink, stat/sha, or an
#      entry that is none of the three expected types -- FATAL-exits
#      immediately instead of silently shortening the snapshot. The walk
#      uses `while ... done < <(find ...)` (process substitution), NOT
#      `find | while ...` (a piped-into subshell would swallow that exit at
#      the pipeline boundary -- empirically confirmed on this host's bash,
#      see run_snapshot_negative_tests case (c) below, which is itself the
#      permanent proof); the two real call sites (snapshot_real_trees/_sha,
#      still captured via `$(...)`, which is ITS OWN subshell boundary)
#      explicitly check the capture's exit status and FATAL the whole
#      script immediately if it's nonzero, so the fail-closed guarantee
#      survives that second subshell hop too. run_snapshot_negative_tests
#      below is the permanent regression test, against a throwaway
#      synthetic tree under WORK_ROOT (never the real obj/bin trees): a new
#      empty subdirectory and a new symlink must both change the snapshot,
#      and a chmod-000 (unreadable) file must make the snapshot function
#      FATAL rather than silently omit it.
#
# ADDITIONAL REGRESSION COVERAGE (two more independently confirmed issues, again in
# THIS test's own process-cleanup and tree-snapshot logic):
#   6. pid_scoped_cleanup only ever walked ONE level of target_pid's
#      descendants (a single `pgrep -P "$target_pid"` call), leaking any
#      grandchild-or-deeper descendant -- verification confirmed a real 'sleep 60'
#      process with PPID=1 left running after a full 48/48 PASS run (a
#      descendant reparented to init once its own parent, itself only a
#      one-level-removed child of target_pid, was killed). Fixed by
#      collect_descendants_post_order: a recursive `pgrep -P` walk that
#      builds the COMPLETE descendant list for both target_pid and
#      decoy_pid -- entirely before issuing any kill, since a `pgrep -P`
#      query against an already-dead pid finds nothing (the dead process's
#      children are reparented away from it the instant it exits) -- then
#      kills every collected pid in the exact (already post-order: deepest
#      descendants first, target_pid/decoy_pid last) order collected, and
#      busy-polls `kill -0` until EVERY one of them, not just
#      target_pid/decoy_pid, is confirmed gone. old_one_level_cleanup below
#      is a verbatim copy of the ORIGINAL one-level mechanics (never called
#      by any real cleanup path) -- run_multilevel_subtree_trial uses it to
#      empirically prove the OLD code really does leave a
#      >=3-level-deep synthetic chain's deeper levels running, and the NEW
#      pid_scoped_cleanup does not; run_multilevel_pkill_immunity_trial and
#      run_concurrent_multilevel_trial extend the existing decoy-trial
#      coverage (unrelated same-name immunity, concurrent-instance
#      isolation) to this deeper chain shape. A final PID sweep near the
#      end of this script (record_pid/ALL_TRACKED_PIDS) additionally
#      re-checks EVERY pid this whole test run ever recorded across every
#      trial, not just the pids each trial's own assertion happens to look
#      at -- exactly the check that would have caught the PPID=1 orphan.
#      pid_scoped_cleanup ALSO uses collect_full_subtree_settled (not a
#      single collect_descendants_post_order pass) before killing anything:
#      a pre-existing `/bin/sh` fixture (decoy_bin/real_bin, kept
#      deliberately non-`exec`'d so pgrep -f can keep matching their own
#      script path by name throughout a trial -- an `exec`'d replacement
#      was tried and found to intermittently break that matching instead)
#      can still be mid-fork of its own last command at the exact instant a
#      newly-discovered node is first queried for its own children; direct
#      instrumentation showed even "stop once two consecutive passes
#      agree" (and, in a later refinement, "keep going until $SECONDS
#      merely changes once") can both still stop one pass too early --
#      $SECONDS is a plain 1-second-granularity counter, so "changes once"
#      can mean almost no real time actually passed, if the first pass
#      happened to land a millisecond before a second boundary.
#      collect_full_subtree_settled instead requires $SECONDS to advance
#      by at least 2 from its own start, GUARANTEEING at least one full
#      real second has elapsed (never near-zero) across repeated
#      whole-tree passes -- still no arbitrary `sleep`, and still nothing
#      killed until it returns. The SAME gap existed, undetected until
#      instrumented, in run_pkill_immunity_trial's own end-of-trial
#      cleanup for unrelated_bin (a plain `kill -9` with no descendant walk
#      at all, even under regression); fixed the same way.
#   7. snapshot_tree_root's find call, even after regression's fix, still
#      pre-filtered to `-type d -o -type f -o -type l`, so a FIFO/socket/
#      device-node entry anywhere under the tree was silently DROPPED from
#      find's own output before snapshot_one_entry ever got a chance to see
#      it (and refuse it) -- invisible, not FATAL, the same class of blind
#      spot regression fixed for "only -type f" moved one level up the
#      filter. find's own exit status was also still folded into (and
#      implicitly trusted by) the process-substitution read, never checked
#      as its own explicit step. Fixed by restructuring snapshot_tree_root
#      into an explicit two-stage form: stage 1 runs find with NO type
#      filter at all into a real list_file (via mktemp under $WORK_ROOT --
#      already covered by this script's one top-level cleanup trap), with
#      find's exit status checked immediately and a clear FATAL naming the
#      root on failure; stage 2 reads list_file (a real file, not a live
#      pipe/process substitution) in a `while ... done < list_file
#      > out_file` loop, which -- like the process-substitution form it
#      replaces -- still runs in the CURRENT shell (no subshell), so an
#      `exit` inside snapshot_one_entry still propagates all the way up.
#      snapshot_one_entry itself gains an explicit "vanished between
#      listing and processing" branch (distinct from the final catch-all)
#      and now names the unexpected type (FIFO/socket/other) it refuses.
#      run_snapshot_negative_tests gains new cases: an unreadable
#      directory forcing find itself to fail (verified independently
#      first, so the test doesn't trust an assumption it hasn't checked
#      holds on this host), a FIFO, a best-effort Unix-domain-socket case,
#      and a symlink removed between listing and processing (made directly
#      testable specifically because the two-stage split now exposes a
#      real seam between "list" and "process").

set -uo pipefail
set -m

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."   # c_impl/

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
AC_DIR="$(cd "$AC_DIR" && pwd)"
C_IMPL_DIR="$(pwd)"

PASS_COUNT=0
FAIL_COUNT=0

pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); echo "  FAIL: $1"; }

WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ubsan_sigtest.XXXXXX")"
trap 'rm -rf "$WORK_ROOT"' EXIT

# --- platform-detection probe (regression review, must run BEFORE any trial;
# fail CLOSED, not open) --------------------------------------------------
# snapshot_real_trees{,_sha} below need a `stat` invocation and a sha256
# tool that actually work on this host. The original code hardcoded BSD
# `stat -f ...` (macOS-only syntax; GNU stat needs `-c` with different
# format specifiers) and `shasum -a 256` (absent on many minimal Linux
# images, which have `sha256sum` instead), each swallowed through
# `2>/dev/null` -- so on a host where neither works, EVERY file's line
# silently disappears from both the "before" and "after" snapshots, which
# then trivially compare equal: a false PASS with zero assurance the real
# trees were untouched. Fix: detect ONCE, up front, against a real
# throwaway probe file in our own WORK_ROOT (try BSD syntax first, since
# this is macOS; fall back to GNU). If NEITHER syntax works for stat, or
# NEITHER works for the hash tool, FATAL + exit 1 immediately -- before any
# trial runs -- rather than letting the snapshot functions silently
# produce empty/partial output later.
PROBE_FILE="$WORK_ROOT/.stat_sha_probe"
: > "$PROBE_FILE"

STAT_MODE=""
if stat -f "%N %m %z" "$PROBE_FILE" >/dev/null 2>&1; then
    STAT_MODE="bsd"
elif stat -c "%n %Y %s" "$PROBE_FILE" >/dev/null 2>&1; then
    STAT_MODE="gnu"
else
    echo "FATAL: neither 'stat -f ...' (BSD) nor 'stat -c ...' (GNU) works on this host -- cannot verify the real obj/bin trees are untouched, refusing to run any trial" >&2
    exit 1
fi

SHA_TOOL=""
if shasum -a 256 "$PROBE_FILE" >/dev/null 2>&1; then
    SHA_TOOL="shasum"
elif sha256sum "$PROBE_FILE" >/dev/null 2>&1; then
    SHA_TOOL="sha256sum"
else
    echo "FATAL: neither 'shasum -a 256' nor 'sha256sum' works on this host -- cannot verify the real obj/bin trees are untouched, refusing to run any trial" >&2
    exit 1
fi

echo "=== platform detection: stat=$STAT_MODE sha=$SHA_TOOL ==="
pass "platform probe: selected working stat ($STAT_MODE) and sha256 ($SHA_TOOL) syntax up front"

# stat_line/sha_line: the single point both snapshot functions call through,
# so the rest of the script never hardcodes a platform-specific tool
# invocation again.
stat_line() {
    case "$STAT_MODE" in
        bsd) stat -f "%N %m %z" "$1" ;;
        gnu) stat -c "%n %Y %s" "$1" ;;
    esac
}

sha_line() {
    case "$SHA_TOOL" in
        shasum) shasum -a 256 "$1" ;;
        sha256sum) sha256sum "$1" ;;
    esac
}

# Tiny MAKE=... wrapper (test-harness-only technique, not a script edit):
# delays the audio_common "lib" build call so the build-phase trials get a
# wide, deterministic window. Touches SLOWMAKE_MARKER right before the
# delay so the harness can detect "we are now inside this foreground
# command" without guessing at timing.
SLOW_MAKE="$WORK_ROOT/slow_make.sh"
cat > "$SLOW_MAKE" <<'EOF'
#!/bin/sh
set -e
last=""
for a in "$@"; do last="$a"; done
if [ "$last" = "lib" ]; then
    : > "${SLOWMAKE_MARKER:?}"
    sleep "${SLOWMAKE_DELAY:-3}"
fi
exec make "$@"
EOF
chmod +x "$SLOW_MAKE"

# snapshot_one_entry <path> <mode:mtime|sha>
# Emits exactly one canonical line for ONE filesystem entry (directory,
# regular file, or symlink) -- or FATAL-exits the WHOLE script immediately
# on any failure (regression review: the old code let stat_line/sha_line fail
# silently inside a `find | while read` pipeline, which just produced a
# shorter-than-expected snapshot with no error -- a before/after pair could
# spuriously match even though verification itself was broken, or a real
# unaccounted-for change had occurred). A directory has no content to
# record beyond its own existence + path. A symlink records its TARGET
# STRING via `readlink` -- never by following the link and hashing/stat'ing
# whatever it points at, which would conflate "symlink changed" with
# "target's content changed" and could also walk outside the tree being
# snapshotted. Anything that is none of the three (a fifo, socket, device
# node) is refused outright: it should never legitimately appear in a
# build's obj/bin tree, and silently ignoring it would reopen the exact
# blind spot this fix closes.
snapshot_one_entry() {
    local path="$1" mode="$2" leaf rc kind

    if [ -L "$path" ]; then
        leaf="$(readlink "$path" 2>&1)"; rc=$?
        if [ "$rc" -ne 0 ] || [ -z "$leaf" ]; then
            echo "FATAL: snapshot: readlink failed for symlink '$path' (rc=$rc)" >&2
            exit 1
        fi
        echo "L $path -> $leaf"
    elif [ -d "$path" ]; then
        echo "D $path"
    elif [ -f "$path" ]; then
        case "$mode" in
            mtime) leaf="$(stat_line "$path" 2>&1)"; rc=$? ;;
            sha)   leaf="$(sha_line "$path" 2>&1)"; rc=$? ;;
            *) echo "FATAL: snapshot: unknown mode '$mode'" >&2; exit 1 ;;
        esac
        if [ "$rc" -ne 0 ] || [ -z "$leaf" ]; then
            echo "FATAL: snapshot: failed to stat/hash regular file '$path' (mode=$mode rc=$rc) -- refusing to produce a silently-shortened snapshot" >&2
            exit 1
        fi
        echo "F $path $leaf"
    elif [ ! -e "$path" ]; then
        # regression review: this path was recorded by find's own listing
        # (stage 1) but no longer exists by the time this function runs
        # (stage 2) -- e.g. a symlink removed out from under a concurrent
        # snapshot, or any other entry that vanished mid-walk. Silently
        # dropping it would produce a snapshot that is quietly SHORTER
        # than what find actually saw; FATAL instead.
        echo "FATAL: snapshot: '$path' was recorded by find but no longer exists (removed between listing and processing) -- refusing to silently drop it from the snapshot" >&2
        exit 1
    else
        # regression review: reachable at all now only because
        # snapshot_tree_root's find call no longer pre-filters by -type --
        # a fifo/socket/device-node/anything-else entry used to be
        # silently EXCLUDED from find's own output before this function
        # ever ran on it (invisible, not FATAL); now it is listed and
        # explicitly refused here, identifying which kind it is where
        # that's cheap to determine.
        kind="an entry of an unrecognized type"
        [ -p "$path" ] 2>/dev/null && kind="a FIFO"
        [ -S "$path" ] 2>/dev/null && kind="a Unix domain socket"
        echo "FATAL: snapshot: '$path' is $kind (not a directory, regular file, or symlink) -- refusing to silently skip an unexpected entry type" >&2
        exit 1
    fi
}

# snapshot_tree_root <root_path> <mode:mtime|sha>
# One root's full contribution to the overall snapshot: a PRESENT/ABSENT
# record line (regression review: so a root going from absent to
# newly-created-but-empty is ITSELF a detectable, recorded change), plus --
# for a PRESENT root -- one canonical line per entry anywhere under it,
# classified by snapshot_one_entry. `-mindepth 1` excludes the root
# directory entry itself, already covered by the PRESENT/ABSENT line
# above. Output is sorted for a stable, order-independent comparison.
#
# regression review: restructured into an explicit TWO-STAGE form (the old
# single `find ... -print0 | ...` / `< <(find ...)` form (a) pre-filtered
# find itself to `-type d -o -type f -o -type l`, silently DROPPING any
# fifo/socket/device-node entry from find's own output before
# snapshot_one_entry ever got a chance to see -- and refuse -- it, and (b)
# never checked find's own exit status as a distinct step, only ever
# implicitly trusting whatever the process substitution happened to
# deliver):
#   stage 1: run find with NO type filter at all (every entry under root,
#     any type) into list_file -- a REAL file via mktemp under $WORK_ROOT,
#     which this script's one top-level `trap 'rm -rf "$WORK_ROOT"' EXIT`
#     already covers -- and check find's own exit status IMMEDIATELY,
#     right here, as its own explicit step: nonzero means a clear FATAL
#     naming the root, no snapshot produced from a possibly-incomplete
#     listing.
#   stage 2: read list_file (NOT a live pipe/process substitution of find
#     directly) in a `while ... done < list_file` loop that calls
#     snapshot_one_entry per entry -- classifying EVERY entry type now
#     that find no longer pre-filters, FATALing on anything it doesn't
#     explicitly recognize (see snapshot_one_entry) -- with the loop's own
#     output redirected into out_file.
# Reading from a plain file with `< list_file` keeps this while loop in
# the CURRENT shell, same as the process-substitution form it replaces
# (NOT `find ... | while ...`: a pipe's right-hand side runs in a SUBSHELL
# in bash, so an `exit` inside snapshot_one_entry, called from the loop
# body, would only terminate that subshell and this function would sail on
# as if nothing failed) -- so an `exit` inside snapshot_one_entry still
# propagates all the way up through this function to its own `$(...)`
# caller. The loop's output is captured via a plain `> "$out_file"`
# redirection, deliberately NOT `| sort`: piping the loop's output into
# `sort` would put the loop back on the LEFT side of a pipe, which bash
# also runs in a subshell, silently reopening the exact hole this whole
# fix closes (confirmed empirically while developing this: an `exit` inside
# the loop only escaped to the caller through the `> file` form, not the
# `| sort` form). `sort` runs afterward as an ordinary, non-piped command
# instead. This really does propagate an `exit` up through this function --
# empirically confirmed on this host's bash by run_snapshot_negative_tests
# below (an unreadable file/directory, a FIFO, or a symlink removed between
# listing and processing must each make this whole function FATAL, not
# just omit one line), which is the permanent, always-run proof of this
# rather than a one-off manual check.
snapshot_tree_root() {
    local root="$1" mode="$2" path list_file out_file rc

    if [ ! -e "$root" ]; then
        echo "ROOT $root ABSENT"
        return 0
    fi
    echo "ROOT $root PRESENT"

    list_file="$(mktemp "$WORK_ROOT/snap_list.XXXXXX")" || {
        echo "FATAL: snapshot: mktemp failed while preparing to list '$root'" >&2
        exit 1
    }
    out_file="$(mktemp "$WORK_ROOT/snap_entries.XXXXXX")" || {
        echo "FATAL: snapshot: mktemp failed while preparing to walk '$root'" >&2
        exit 1
    }

    find "$root" -mindepth 1 -print0 > "$list_file"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "FATAL: snapshot: find exited nonzero (rc=$rc) while listing '$root' -- refusing to produce a snapshot from a possibly-incomplete listing" >&2
        exit 1
    fi

    while IFS= read -r -d '' path; do
        snapshot_one_entry "$path" "$mode"
    done < "$list_file" > "$out_file"

    sort "$out_file"
    rm -f "$list_file" "$out_file"
}

snapshot_real_trees() {
    snapshot_tree_root "$C_IMPL_DIR/obj" mtime
    snapshot_tree_root "$C_IMPL_DIR/bin" mtime
    snapshot_tree_root "$AC_DIR/obj" mtime
    snapshot_tree_root "$AC_DIR/bin" mtime
}

snapshot_real_trees_sha() {
    snapshot_tree_root "$C_IMPL_DIR/obj" sha
    snapshot_tree_root "$C_IMPL_DIR/bin" sha
    snapshot_tree_root "$AC_DIR/obj" sha
    snapshot_tree_root "$AC_DIR/bin" sha
}

# --- readiness detection -----------------------------------------------
# Busy-poll (no sleep -- keeps the detection window tight relative to the
# artificial build-phase delay, and costs nothing extra on the naturally
# ~1s-plus binary-run window) bounded by both an iteration cap and a
# wall-clock cap so a genuinely broken host fails fast instead of hanging.
wait_for_marker() {
    local marker="$1"
    local start=$SECONDS
    while [ ! -e "$marker" ] && [ $((SECONDS - start)) -lt 30 ]; do :; done
    [ -e "$marker" ]
}

# wait_for_file <path> <timeout_secs>
# Busy-polls (no sleep) until <path> exists and is non-empty, or the
# timeout elapses. Used (regression review) to synchronize on a descendant
# process in a multilevel chain actually having started and recorded its
# own PID (see build_multilevel_chain/start_multilevel_chain below) before
# the caller reads that PID back -- ground truth, independent of the
# collection/cleanup logic under test.
wait_for_file() {
    local path="$1" timeout="$2"
    local start=$SECONDS
    while [ ! -s "$path" ] && [ $((SECONDS - start)) -lt "$timeout" ]; do :; done
    [ -s "$path" ]
}

# is_descendant_of <ancestor_pid> <candidate_pid>
# Walks candidate_pid's own parent chain (via repeated `ps -o ppid=`
# lookups) up to pid 1, succeeding iff ancestor_pid appears anywhere in
# that chain (including as candidate_pid's immediate parent). This is what
# lets wait_for_binary tell "a real descendant of the specific backgrounded
# trial script" apart from "any unrelated process anywhere on the system
# whose command line happens to match" (regression review).
is_descendant_of() {
    local ancestor="$1" candidate="$2"
    local cur="$candidate" ppid hops=0
    while [ -n "$cur" ] && [ "$cur" != "0" ] && [ "$cur" != "1" ]; do
        [ "$cur" = "$ancestor" ] && return 0
        ppid="$(ps -o ppid= -p "$cur" 2>/dev/null | tr -d ' ')"
        [ -z "$ppid" ] && return 1
        cur="$ppid"
        hops=$((hops + 1))
        [ "$hops" -gt 200 ] && return 1   # guard against a ppid-chain bug/cycle
    done
    [ "$cur" = "$ancestor" ]
}

# wait_for_binary <target_pid> <binpat>
# Only reports "ready" once some pgrep -f match is target_pid itself or an
# actual descendant of it -- fixes the system-wide false-positive bug
# (regression review): the target pid is threaded in explicitly by every call
# site (the same `pid=$!` each trial already captures) rather than being
# read from an outer-scope variable.
wait_for_binary() {
    local target_pid="$1" binpat="$2"
    local start=$SECONDS
    local cand
    while [ $((SECONDS - start)) -lt 30 ]; do
        for cand in $(pgrep -f "$binpat" 2>/dev/null); do
            if [ "$cand" = "$target_pid" ] || is_descendant_of "$target_pid" "$cand"; then
                return 0
            fi
        done
    done
    return 1
}

# --- one signal trial ----------------------------------------------------
# run_signal_trial <script_name> <phase:build|run> <sig:INT|TERM> <expected_rc>
run_signal_trial() {
    local script_name="$1" phase="$2" sig="$3" expected_rc="$4"
    local label="$script_name phase=$phase sig=$sig"
    local trial_dir tmpdir_for_script log marker pid rc caught

    trial_dir="$(mktemp -d "$WORK_ROOT/trial.XXXXXX")"
    tmpdir_for_script="$trial_dir/tmpdir"
    mkdir -p "$tmpdir_for_script"
    log="$trial_dir/log.txt"
    marker="$trial_dir/marker"

    case "$script_name" in
        run_counter_saturation_ubsan.sh) binpat="test_counter_saturation_ubsan\$" ;;
        run_selftest_ubsan.sh)           binpat="simd_selftest_aec_ubsan\$" ;;
        *) fail "$label: unknown script_name"; return ;;
    esac

    if [ "$phase" = "build" ]; then
        TMPDIR="$tmpdir_for_script" MAKE="$SLOW_MAKE" SLOWMAKE_MARKER="$marker" SLOWMAKE_DELAY=3 \
            ./test/"$script_name" > "$log" 2>&1 &
        pid=$!
        wait_for_marker "$marker"; caught=$?
    else
        TMPDIR="$tmpdir_for_script" ./test/"$script_name" > "$log" 2>&1 &
        pid=$!
        wait_for_binary "$pid" "$binpat"; caught=$?
    fi

    if [ "$caught" != 0 ]; then
        fail "$label: never reached the intended phase within the timeout (infra problem, not the fix under test)"
        kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        return
    fi

    if ! kill -s "$sig" "$pid" 2>/dev/null; then
        fail "$label: kill -s $sig failed to reach pid $pid (process already gone -- infra race, not the fix under test)"
        wait "$pid" 2>/dev/null
        return
    fi

    wait "$pid" 2>/dev/null
    rc=$?

    if [ "$rc" -eq "$expected_rc" ]; then
        pass "$label: exit code == $expected_rc"
    else
        fail "$label: exit code == $rc, expected $expected_rc"
    fi

    if grep -qF "UBSan probe: PASS" "$log"; then
        fail "$label: log contains the false-PASS string 'UBSan probe: PASS'"
    else
        pass "$label: log does not contain 'UBSan probe: PASS'"
    fi

    local leftover
    leftover="$(ls -A "$tmpdir_for_script" 2>/dev/null)"
    if [ -z "$leftover" ]; then
        pass "$label: scratch directory fully removed"
    else
        fail "$label: scratch directory NOT removed (leftover: $leftover)"
    fi
}

# --- one normal (uninterrupted) run --------------------------------------
run_normal_trial() {
    local script_name="$1"; shift
    local extra_desc="$*"
    [ -z "$extra_desc" ] && extra_desc="default"
    local trial_dir tmpdir_for_script log rc

    trial_dir="$(mktemp -d "$WORK_ROOT/normal.XXXXXX")"
    tmpdir_for_script="$trial_dir/tmpdir"
    mkdir -p "$tmpdir_for_script"
    log="$trial_dir/log.txt"

    TMPDIR="$tmpdir_for_script" ./test/"$script_name" "$@" > "$log" 2>&1
    rc=$?

    if [ "$rc" -eq 0 ]; then
        pass "$script_name ($extra_desc): normal run exit code == 0"
    else
        fail "$script_name ($extra_desc): normal run exit code == $rc, expected 0"
    fi

    if grep -qF "UBSan probe: PASS" "$log"; then
        pass "$script_name ($extra_desc): normal run prints 'UBSan probe: PASS'"
    else
        fail "$script_name ($extra_desc): normal run did NOT print 'UBSan probe: PASS'"
    fi

    # Symmetry with run_signal_trial's existing scratch-cleanup check
    # (regression review): an uninterrupted run deserves the same proof its
    # scratch dir was fully cleaned up, not just interrupted ones.
    local leftover
    leftover="$(ls -A "$tmpdir_for_script" 2>/dev/null)"
    if [ -z "$leftover" ]; then
        pass "$script_name ($extra_desc): normal run scratch directory fully removed"
    else
        fail "$script_name ($extra_desc): normal run scratch directory NOT removed (leftover: $leftover)"
    fi
}

# --- decoy-process trial (guards wait_for_binary's descendant scoping,
# regression review) ---------------------------------------------------------
# old_unscoped_wait_for_binary is a VERBATIM copy of the ORIGINAL,
# system-wide-matching wait_for_binary this test used to call (pre-fix).
# It is never used by any real trial above -- it exists solely so
# run_decoy_trial can empirically prove the OLD code really would have been
# fooled by a same-pattern decoy process, the same "reproduce the old bug,
# don't just assert the new behaviour" discipline this file already applies
# to the trap bug (see header).
old_unscoped_wait_for_binary() {
    local binpat="$1"
    local start=$SECONDS
    while ! pgrep -f "$binpat" >/dev/null 2>&1 && [ $((SECONDS - start)) -lt 30 ]; do :; done
    pgrep -f "$binpat" >/dev/null 2>&1
}

# --- PID-scoped cleanup helpers (regression review P2 fix; full-subtree
# post-order rewrite regression review) ----------------------------------------
# wait_pids_gone <timeout_secs> <pid...>
# Busy-polls (no sleep, consistent with this file's zero-sleep discipline)
# until `kill -0` fails (ESRCH) for EVERY pid given, or the timeout elapses.
# A single point-in-time check right after the kills would be racy: a
# just-killed process can briefly still answer `kill -0` successfully while
# it is a zombie awaiting reap by its parent (or by pid 1, once
# reparented).  Empty/blank arguments are skipped (callers may pass an
# empty "captured children" list through unquoted).
wait_pids_gone() {
    local timeout="$1"; shift
    local start=$SECONDS
    local all_gone p
    while [ $((SECONDS - start)) -lt "$timeout" ]; do
        all_gone=1
        for p in "$@"; do
            [ -z "$p" ] && continue
            if kill -0 "$p" 2>/dev/null; then
                all_gone=0
                break
            fi
        done
        [ "$all_gone" -eq 1 ] && return 0
    done
    return 1
}

# --- Global PID bookkeeping (regression review) -------------------------------
# record_pid <pid>: appends pid to ALL_TRACKED_PIDS, a single running list of
# every PID this ENTIRE test script has ever spawned as a detached/background
# fixture process across every trial (decoy processes, multilevel chains,
# unrelated-process-immunity fixtures -- anything outside a script-under-
# test's own self-contained lifecycle, which already reaps itself via
# `wait`). The final sweep near the end of this script asserts `kill -0`
# fails for every single one of these, not just the specific PIDs each
# trial's own narrower assertion happens to check -- exactly the check that
# would have caught the PPID=1 orphan verification identified (a leftover descendant no
# single trial's own assertion was looking at).
ALL_TRACKED_PIDS=""
record_pid() {
    [ -n "${1:-}" ] && ALL_TRACKED_PIDS="$ALL_TRACKED_PIDS $1"
}

# collect_descendants_post_order <pid>
# Recursively walks pid's FULL descendant subtree via repeated `pgrep -P`
# lookups, queried while everything is still alive -- the fix for the
# regression cleanup's one-level blind spot (regression coverage: a 'sleep 60'
# process with PPID=1 was found still running after a full 48/48 PASS run
# -- a grandchild-or-deeper descendant the single `pgrep -P "$target_pid"`
# call never captured, since an orphaned process's children are reparented
# AWAY from it the instant it exits, meaning a `pgrep -P` query against an
# already-dead pid finds nothing). For a given pid, recurses into each of
# ITS children FIRST (each of which does the same for its own children
# first), and only emits (echoes) the pid itself once every one of its
# descendants has already been emitted -- a genuine depth-first POST-ORDER
# sequence (deepest descendants first, this pid last), with no separate
# bookkeeping needed: the recursion order alone determines the emission
# order. This function only ever QUERIES (pgrep -P) -- it never kills
# anything -- so a caller can build the COMPLETE list for every pid it
# cares about entirely BEFORE issuing a single kill, exactly the ordering
# the fix requires (interleaving a kill with an as-yet-incomplete walk
# would risk the same reparenting blind spot this fix closes).
collect_descendants_post_order() {
    local pid="$1" c
    for c in $(pgrep -P "$pid" 2>/dev/null); do
        collect_descendants_post_order "$c"
    done
    echo "$pid"
}

# collect_full_subtree_settled <root_pid...>
# Calls collect_descendants_post_order for EVERY root pid given, in
# argument order, and repeats that WHOLE multi-root collection across a
# real, non-zero span of wall-clock time -- until $SECONDS has advanced by
# at least 1 since the very first pass -- before returning the LAST
# result. No `sleep`: the repeated pgrep-driven passes (each root's own
# recursive walk forks a real `pgrep` process per tree node) themselves
# consume the real time.
#
# Residual-race hardening (found empirically while developing this fix): a
# throwaway `/bin/sh` fixture whose non-exec last command this host's
# shell forks rather than execs (e.g. decoy_bin/real_bin above) can still
# be mid-fork of that child at the exact instant a NEWLY-discovered
# intermediate node (e.g. real_bin itself, just found as target_pid's own
# child) is first queried for ITS children -- confirmed by direct
# instrumentation: a plain "stop once two consecutive passes agree" loop
# can still stop one pass too early, because both of those passes can land
# before the fork completes, agreeing on a still-incomplete answer. Only a
# real, non-zero MINIMUM span of elapsed wall-clock time (not merely "N
# identical samples", which does not bound how little real time separates
# them) gives the fork a genuine chance to finish before the final sample
# is taken. This still satisfies (does not weaken) "collect fully before
# killing anything": nothing is killed until this function returns.
collect_full_subtree_settled() {
    local settle_start=$SECONDS result pid
    result="$(for pid in "$@"; do collect_descendants_post_order "$pid"; done)"
    # `-lt 2`, not `-lt 1` (found empirically: $SECONDS is a plain integer
    # counter with 1-second granularity, so requiring it to merely change
    # ONCE from settle_start guarantees almost nothing -- if the first
    # collection above happens to complete a single millisecond before a
    # real second boundary ticks over, the loop below would exit almost
    # immediately, having waited close to zero real time. Requiring the
    # DIFFERENCE to reach 2 guarantees at least one full real second has
    # actually elapsed (at most just under three), regardless of where
    # within the current second the first pass happened to land.
    while [ $((SECONDS - settle_start)) -lt 2 ]; do
        result="$(for pid in "$@"; do collect_descendants_post_order "$pid"; done)"
    done
    printf '%s\n' "$result"
}

# old_one_level_cleanup <decoy_pid> <target_pid>
# VERBATIM copy of the ORIGINAL (regression) one-level pid_scoped_cleanup's own
# kill/wait MECHANICS -- never called by any real cleanup path in this
# script. Exists solely so run_multilevel_subtree_trial can empirically
# prove the OLD code really does leave a grandchild-or-deeper descendant
# running (the verified bug), the same "reproduce the old bug, don't
# just assert the new behaviour" discipline this file already applies to
# the trap bug (see header) and to wait_for_binary's decoy immunity
# (old_unscoped_wait_for_binary above). Deliberately has no internal
# pass/fail call of its own (same convention as old_unscoped_wait_for_binary)
# -- the CALLER inspects the resulting process state, since it alone knows
# about the deeper levels this function is blind to.
old_one_level_cleanup() {
    local decoy_pid="$1" target_pid="$2"
    local target_children c

    target_children="$(pgrep -P "$target_pid" 2>/dev/null)"

    for c in $target_children; do
        kill -9 "$c" 2>/dev/null
    done
    kill -9 "$target_pid" 2>/dev/null
    wait "$target_pid" 2>/dev/null
    kill -9 "$decoy_pid" 2>/dev/null
    wait "$decoy_pid" 2>/dev/null
}

# pid_scoped_cleanup <label> <decoy_pid> <target_pid>
# Explicit-PID-tracking cleanup: NEVER a system-wide pkill/pgrep-by-name.
# FIXED (regression review P2): the regression version (see old_one_level_cleanup
# above for its preserved mechanics) walked only ONE level of target_pid's
# descendants, leaking any grandchild-or-deeper descendant -- regression
# confirmed a real 'sleep 60' process with PPID=1 left running after a full
# 48/48 PASS run. Now: (1) build the COMPLETE post-order descendant list
# for BOTH target_pid and decoy_pid via collect_descendants_post_order,
# entirely BEFORE issuing any kill (see that function's own comment for why
# this separation matters -- a pgrep -P query against an already-dead pid
# finds nothing); (2) kill every PID in that list in the EXACT order
# collected, which is already a true post-order sequence (deepest
# descendants of target_pid first, target_pid itself immediately after its
# own last child, then decoy_pid's subtree the same way, decoy_pid last);
# (3) busy-poll `kill -0` until EVERY one of those PIDs -- not just
# target_pid/decoy_pid -- is confirmed gone. Never a name-based
# `pgrep -f`/`pkill -f` scan, which could kill (or be fooled by, or falsely
# blame) an unrelated process that merely shares the same command-line
# text.
#
# Step (1) uses collect_full_subtree_settled (see its own comment) rather
# than a single collect_descendants_post_order pass, to close a residual
# race found empirically while developing this fix.
pid_scoped_cleanup() {
    local label="$1" decoy_pid="$2" target_pid="$3"
    local full_list p

    full_list="$(collect_full_subtree_settled "$target_pid" "$decoy_pid")"

    for p in $full_list; do
        kill -9 "$p" 2>/dev/null
    done

    # Reap target_pid/decoy_pid specifically -- both this script's own
    # direct children; any deeper descendant was never our own child, so
    # there is nothing for US to `wait` on for it (its own former parent,
    # or -- once reparented -- pid 1, reaps it instead).
    wait "$target_pid" 2>/dev/null
    wait "$decoy_pid" 2>/dev/null

    if wait_pids_gone 10 $full_list; then
        pass "$label: decoy + target + FULL descendant subtree (post-order: ${full_list:-none}) fully cleaned up (PID-scoped, no system-wide pkill/pgrep-by-name)"
    else
        fail "$label: at least one PID in the full collected subtree ($full_list) still answers kill -0 after cleanup"
    fi
}

run_decoy_trial() {
    local id="${1:-$$_${RANDOM}${RANDOM}}"
    local label="wait_for_binary decoy immunity ($id)"
    # Per-invocation-unique pattern (regression review P1-ish fix, defense-in-
    # depth on top of pid_scoped_cleanup above -- NOT a replacement for it):
    # even a second, fully independent concurrent run of this whole test
    # script never has to share this literal process-name text with an
    # unrelated decoy trial by accident. The default id folds in this
    # script's own top-level PID ($$) plus two RANDOM draws; callers below
    # (run_pkill_immunity_trial, run_concurrent_decoy_trial) pass an
    # explicit id when they deliberately WANT a collision, to prove the
    # PID-scoped cleanup holds even then.
    local pattern="decoy_trial_ubsan_target_${id}"
    local binpat="${pattern}\$"
    local trial_dir decoy_bin real_bin target_script decoy_pid target_pid
    local old_start old_elapsed old_rc new_start new_elapsed new_rc
    local c found_pid

    trial_dir="$(mktemp -d "$WORK_ROOT/decoy.XXXXXX")"

    # The decoy: a REAL, running process whose entire command line matches
    # binpat, started directly by this harness -- a SIBLING of, not a
    # descendant of, the "target" pid below. Implemented as a throwaway
    # script literally named to satisfy the anchored `<pattern>$` regex
    # (pgrep -f matches binpat against the full command line, same
    # convention as the real binpat values in run_signal_trial -- no
    # trailing argv tokens allowed after the name).
    # NOTE (regression review): deliberately a BARE `sleep 60`, not
    # `exec sleep 60` -- unlike build_multilevel_chain's throwaway
    # decoy.sh/unrelated_script below (which are never pattern-matched by
    # name), decoy_bin/real_bin's own SCRIPT PATH is exactly what binpat
    # anchors against via `pgrep -f` in wait_for_binary/is_descendant_of/
    # old_unscoped_wait_for_binary; confirmed empirically that `exec`
    # replaces this process's own command line with "sleep 60" at some
    # UNPREDICTABLE later moment (racy: sometimes near-instant, sometimes
    # not, depending on host load), which would silently and
    # intermittently break every one of those pgrep -f lookups mid-trial.
    # The residual risk this bare (non-exec) form re-opens -- decoy_bin/
    # real_bin's own /bin/sh wrapper forking an untracked grandchild --
    # is handled instead by pid_scoped_cleanup's own stabilizing re-collect
    # loop below (collects until two consecutive passes agree, entirely
    # before any kill), rather than by removing the fork here.
    mkdir -p "$trial_dir/decoy" "$trial_dir/real"
    decoy_bin="$trial_dir/decoy/$pattern"
    cat > "$decoy_bin" <<'EOF'
#!/bin/sh
sleep 60
EOF
    chmod +x "$decoy_bin"
    "$decoy_bin" &
    decoy_pid=$!
    record_pid "$decoy_pid"

    # The "real descendant": a same-named binary that only appears after a
    # delay, as a child of a background target script started the same way
    # run_signal_trial starts the scripts under test (`... & pid=$!`) --
    # simulating "the backgrounded target has not yet reached the phase
    # we're watching for".
    real_bin="$trial_dir/real/$pattern"
    cat > "$real_bin" <<'EOF'
#!/bin/sh
sleep 60
EOF
    chmod +x "$real_bin"

    target_script="$trial_dir/target.sh"
    cat > "$target_script" <<EOF
#!/bin/sh
sleep 3
"$real_bin" &
wait
EOF
    chmod +x "$target_script"
    "$target_script" &
    target_pid=$!
    record_pid "$target_pid"

    # OLD (unscoped) behavior: called while ONLY the decoy is running (the
    # target's real descendant cannot possibly exist yet -- it is asleep
    # for 3s). Must return "ready" almost instantly, purely because the
    # system-wide decoy matches -- THIS is the bug.
    old_start=$SECONDS
    old_unscoped_wait_for_binary "$binpat"; old_rc=$?
    old_elapsed=$((SECONDS - old_start))

    if [ "$old_rc" -eq 0 ] && [ "$old_elapsed" -le 1 ]; then
        pass "$label: reproduced the OLD bug (unscoped wait_for_binary fooled by the decoy in ${old_elapsed}s, before the real descendant could possibly exist)"
    else
        fail "$label: could NOT reproduce the OLD bug (unscoped call rc=$old_rc elapsed=${old_elapsed}s) -- decoy setup is unreliable, this trial needs fixing"
    fi

    # NEW (scoped) behavior: called against the SAME decoy plus the SAME
    # not-yet-ready target_pid. Must NOT be fooled by the decoy -- it can
    # only return "ready" once target_pid's own real descendant exists,
    # i.e. no sooner than the target script's 3s delay.
    new_start=$SECONDS
    wait_for_binary "$target_pid" "$binpat"; new_rc=$?
    new_elapsed=$((SECONDS - new_start))

    if [ "$new_rc" -eq 0 ] && [ "$new_elapsed" -ge 2 ]; then
        pass "$label: fixed wait_for_binary ignored the decoy and correctly waited for target_pid's real descendant (${new_elapsed}s)"
    else
        fail "$label: fixed wait_for_binary rc=$new_rc elapsed=${new_elapsed}s -- expected success at >=2s (the decoy must not short-circuit it)"
    fi

    # Belt-and-suspenders on top of the timing evidence: confirm the match
    # it eventually accepted really is a descendant of target_pid, not the
    # decoy.
    found_pid=""
    for c in $(pgrep -f "$binpat" 2>/dev/null); do
        if is_descendant_of "$target_pid" "$c"; then found_pid="$c"; break; fi
    done
    record_pid "$found_pid"
    if [ -n "$found_pid" ] && [ "$found_pid" != "$decoy_pid" ]; then
        pass "$label: accepted match ($found_pid) is a real descendant of target_pid ($target_pid), not the decoy ($decoy_pid)"
    else
        fail "$label: could not confirm the accepted match was target_pid's own descendant (found_pid='$found_pid' decoy_pid=$decoy_pid)"
    fi

    # cleanup: decoy + target's whole subtree (never leave these running) --
    # PID-scoped (regression review P2 fix), see pid_scoped_cleanup's own
    # comment for why this is no longer a system-wide pkill/pgrep-by-name.
    pid_scoped_cleanup "$label" "$decoy_pid" "$target_pid"
}

# --- unrelated same-pattern process immunity trial (regression review P2) -----
# Proves the OTHER half of the pkill/pgrep-by-name bug: not only could the
# old cleanup be FOOLED (wait_for_binary, already covered above), it could
# actively KILL an unrelated process anywhere on the system that merely
# shares the decoy pattern's command-line text. Starts a real, independent
# background process using the EXACT SAME pattern text a decoy trial will
# use (by passing that same id to run_decoy_trial), then confirms it is
# still running, completely untouched, once that trial's own cleanup has
# run -- and cleans it up itself afterward (it is not part of any trial's
# own tracked PIDs).
run_pkill_immunity_trial() {
    local label="pkill/pgrep immunity: unrelated same-pattern process"
    local id="pkill_immunity_shared_$$_${RANDOM}"
    local trial_dir pattern unrelated_bin unrelated_pid unrelated_list p2

    trial_dir="$(mktemp -d "$WORK_ROOT/pkillimm.XXXXXX")"
    pattern="decoy_trial_ubsan_target_${id}"
    unrelated_bin="$trial_dir/$pattern"
    # Bare (non-exec) `sleep 60` -- see run_decoy_trial's decoy_bin/real_bin
    # comment above: this pattern must remain pgrep -f-matchable by its own
    # script path for the duration of the trial (run_decoy_trial, invoked
    # below with the SAME pattern, depends on it).
    cat > "$unrelated_bin" <<'EOF'
#!/bin/sh
sleep 60
EOF
    chmod +x "$unrelated_bin"
    "$unrelated_bin" &
    unrelated_pid=$!
    record_pid "$unrelated_pid"

    # A full decoy trial that happens to use the EXACT SAME pattern text
    # (same id) as the unrelated process above. If run_decoy_trial's own
    # cleanup still did a system-wide `pkill -9 -f "$binpat"`, it would
    # kill this unrelated process purely because its command line matches
    # -- even though it has nothing to do with the trial.
    run_decoy_trial "$id"

    if kill -0 "$unrelated_pid" 2>/dev/null; then
        pass "$label: unrelated same-pattern process ($unrelated_pid) is still running, untouched, after the decoy trial's cleanup"
    else
        fail "$label: unrelated same-pattern process ($unrelated_pid) was killed by the decoy trial's cleanup -- system-wide pkill/pgrep-by-name regression"
    fi

    # regression review: unrelated_bin is the SAME bare (non-exec) `sleep 60`
    # shape as decoy_bin/real_bin, so it can equally well have forked its
    # own untracked "sleep 60" child by now -- a plain `kill -9
    # "$unrelated_pid"` alone (the old shape of this cleanup) would leave
    # that child running, reparented to init, once unrelated_pid is gone.
    # Uses the same settled full-subtree collection as pid_scoped_cleanup
    # (just for this one pid, no decoy counterpart needed here) and
    # verifies every collected pid is actually gone afterward, same as
    # pid_scoped_cleanup's own contract.
    unrelated_list="$(collect_full_subtree_settled "$unrelated_pid")"
    for p2 in $unrelated_list; do
        kill -9 "$p2" 2>/dev/null
    done
    wait "$unrelated_pid" 2>/dev/null
    for p2 in $unrelated_list; do
        record_pid "$p2"
    done
    if wait_pids_gone 10 $unrelated_list; then
        pass "$label: unrelated same-pattern process + its own full descendant subtree (${unrelated_list:-none}) fully cleaned up afterward"
    else
        fail "$label: at least one PID in unrelated_pid's own collected subtree ($unrelated_list) still answers kill -0 after cleanup"
    fi
}

# start_decoy_instance <id> <trial_dir>
# Starts one decoy+target(+later real_bin) process trio, same shape as
# run_decoy_trial's own setup (minus the old/new wait_for_binary timing
# comparison, irrelevant here), parameterized so more than one instance can
# exist at once. Deliberately NOT invoked via command substitution (which
# would fork a subshell and start these background jobs as the SUBSHELL's
# children, not this script's own -- the `wait`/`kill -0` bookkeeping in
# pid_scoped_cleanup and run_concurrent_decoy_trial below relies on them
# being this script's own direct children) -- it sets the *_OUT globals
# below directly instead of trying to return multiple values (this
# targets bash 3.2, which has no `local -n` namerefs).
START_DECOY_INSTANCE_DECOY_PID=""
START_DECOY_INSTANCE_TARGET_PID=""
START_DECOY_INSTANCE_PATTERN=""
start_decoy_instance() {
    local id="$1" trial_dir="$2"
    local pattern="decoy_trial_ubsan_target_${id}"
    local decoy_bin real_bin target_script

    # Bare (non-exec) `sleep 60` -- see run_decoy_trial's own decoy_bin/
    # real_bin comment above for why `exec` is deliberately NOT used here
    # (it would break pgrep -f pattern matching, racily and
    # intermittently).
    mkdir -p "$trial_dir/decoy" "$trial_dir/real"
    decoy_bin="$trial_dir/decoy/$pattern"
    cat > "$decoy_bin" <<'EOF'
#!/bin/sh
sleep 60
EOF
    chmod +x "$decoy_bin"
    "$decoy_bin" &
    START_DECOY_INSTANCE_DECOY_PID=$!
    record_pid "$START_DECOY_INSTANCE_DECOY_PID"

    real_bin="$trial_dir/real/$pattern"
    cat > "$real_bin" <<'EOF'
#!/bin/sh
sleep 60
EOF
    chmod +x "$real_bin"

    target_script="$trial_dir/target.sh"
    cat > "$target_script" <<EOF
#!/bin/sh
sleep 1
"$real_bin" &
wait
EOF
    chmod +x "$target_script"
    "$target_script" &
    START_DECOY_INSTANCE_TARGET_PID=$!
    record_pid "$START_DECOY_INSTANCE_TARGET_PID"

    START_DECOY_INSTANCE_PATTERN="$pattern"
}

# wait_for_child_of <parent_pid> <timeout_secs>
# Busy-polls (no sleep) until `pgrep -P parent_pid` reports at least one
# child, or the timeout elapses.
wait_for_child_of() {
    local parent_pid="$1" timeout="$2"
    local start=$SECONDS
    while [ $((SECONDS - start)) -lt "$timeout" ]; do
        [ -n "$(pgrep -P "$parent_pid" 2>/dev/null)" ] && return 0
    done
    return 1
}

# --- concurrent decoy-trial-style cleanup isolation trial (regression review
# P2 item 3b) -----------------------------------------------------------
# Runs TWO independent decoy+target process trios AT THE SAME TIME, and
# deliberately gives them the SAME literal pattern text (id "conc_shared")
# -- the real safety property under test is pid_scoped_cleanup's PID
# scoping, not the pattern-uniqueness defense-in-depth (fix 1), so this
# trial intentionally does not lean on fix 1 at all. Cleans up instance A
# while instance B is still fully alive and confirms B is untouched, then
# cleans up B and confirms both are gone -- i.e. each instance's own
# tracked PIDs are the only ones ever affected by its own cleanup.
run_concurrent_decoy_trial() {
    local label="concurrent decoy-trial cleanup isolation"
    local trial_dir_a trial_dir_b
    local decoy_pid_a target_pid_a decoy_pid_b target_pid_b

    trial_dir_a="$(mktemp -d "$WORK_ROOT/concA.XXXXXX")"
    trial_dir_b="$(mktemp -d "$WORK_ROOT/concB.XXXXXX")"

    start_decoy_instance "conc_shared" "$trial_dir_a"
    decoy_pid_a="$START_DECOY_INSTANCE_DECOY_PID"
    target_pid_a="$START_DECOY_INSTANCE_TARGET_PID"

    start_decoy_instance "conc_shared" "$trial_dir_b"
    decoy_pid_b="$START_DECOY_INSTANCE_DECOY_PID"
    target_pid_b="$START_DECOY_INSTANCE_TARGET_PID"

    # Both instances are now running concurrently (decoy_pid_a/target_pid_a
    # alongside decoy_pid_b/target_pid_b), sharing one process-name pattern.
    # Let each target's real_bin child actually appear before cleaning up,
    # so instance A's cleanup below has a real captured child to kill too.
    wait_for_child_of "$target_pid_a" 30
    wait_for_child_of "$target_pid_b" 30

    # Clean up instance A only, while B is still fully alive.
    pid_scoped_cleanup "$label: instance A cleanup" "$decoy_pid_a" "$target_pid_a"

    if kill -0 "$decoy_pid_b" 2>/dev/null && kill -0 "$target_pid_b" 2>/dev/null; then
        pass "$label: instance B's processes (decoy=$decoy_pid_b target=$target_pid_b) are untouched by instance A's cleanup"
    else
        fail "$label: instance B's processes were affected by instance A's cleanup (decoy=$decoy_pid_b target=$target_pid_b)"
    fi

    # Now clean up B and confirm both instances are fully gone.
    pid_scoped_cleanup "$label: instance B cleanup" "$decoy_pid_b" "$target_pid_b"
}

# --- Full-subtree post-order cleanup trials (regression review P2 fix) -------
# build_multilevel_chain <trial_dir>
# Creates (but does not start) three chained POSIX shell scripts under
# trial_dir, forming a >=3-level-deep descendant chain BELOW whatever
# process the caller starts as the root (the caller starts
# MULTILEVEL_ROOT_SCRIPT itself via `"$MULTILEVEL_ROOT_SCRIPT" &`, getting
# that process's own PID directly via `$!` -- this function does not start
# anything):
#   root (caller-started)
#     -> child            (level 1, writes its own $$ to child_pidfile)
#          -> grandchild        (level 2, writes its own $$ to
#                                 grandchild_pidfile)
#               -> great-grandchild (level 3, writes its own $$ to
#                                     ggc_pidfile, then `exec`s into a long
#                                     sleep -- so `ps` shows the process
#                                     itself as "sleep 999", the same shape
#                                     as the real orphan verification identified)
# Each pidfile is written by the level itself -- ground truth, independent
# of collect_descendants_post_order/pid_scoped_cleanup, the very mechanism
# under test -- using bash-3.2-friendly OUT-globals (no `local -n`
# namerefs, same convention as start_decoy_instance above).
build_multilevel_chain() {
    local trial_dir="$1"
    local root child grandchild ggc
    local child_pidfile grandchild_pidfile ggc_pidfile

    mkdir -p "$trial_dir"
    root="$trial_dir/root.sh"
    child="$trial_dir/child.sh"
    grandchild="$trial_dir/grandchild.sh"
    ggc="$trial_dir/greatgrandchild.sh"
    child_pidfile="$trial_dir/child.pid"
    grandchild_pidfile="$trial_dir/grandchild.pid"
    ggc_pidfile="$trial_dir/ggc.pid"

    cat > "$ggc" <<EOF
#!/bin/sh
echo \$\$ > "$ggc_pidfile"
exec sleep 999
EOF
    chmod +x "$ggc"

    cat > "$grandchild" <<EOF
#!/bin/sh
echo \$\$ > "$grandchild_pidfile"
"$ggc" &
wait
EOF
    chmod +x "$grandchild"

    cat > "$child" <<EOF
#!/bin/sh
echo \$\$ > "$child_pidfile"
"$grandchild" &
wait
EOF
    chmod +x "$child"

    cat > "$root" <<EOF
#!/bin/sh
"$child" &
wait
EOF
    chmod +x "$root"

    MULTILEVEL_ROOT_SCRIPT="$root"
    MULTILEVEL_CHILD_PIDFILE="$child_pidfile"
    MULTILEVEL_GRANDCHILD_PIDFILE="$grandchild_pidfile"
    MULTILEVEL_GGC_PIDFILE="$ggc_pidfile"
}

# start_multilevel_chain <trial_dir>
# Builds + starts one full multilevel chain (build_multilevel_chain above)
# plus a throwaway decoy sleep process alongside it (matching real
# pid_scoped_cleanup usage shape: <decoy_pid> <target_pid>), waits for
# every level to have actually started (ground-truth pidfiles, not
# inferred from process listings), records every spawned PID via
# record_pid, and sets MULTILEVEL_*_PID out-globals. Sets
# MULTILEVEL_START_OK=1 on success or 0 if the chain never fully
# materialized within the timeout (an infra problem, not a real assertion
# target for the caller).
start_multilevel_chain() {
    local trial_dir="$1"

    build_multilevel_chain "$trial_dir"

    "$MULTILEVEL_ROOT_SCRIPT" &
    MULTILEVEL_ROOT_PID=$!
    record_pid "$MULTILEVEL_ROOT_PID"

    if ! wait_for_file "$MULTILEVEL_GGC_PIDFILE" 30; then
        MULTILEVEL_START_OK=0
        return
    fi

    MULTILEVEL_CHILD_PID="$(cat "$MULTILEVEL_CHILD_PIDFILE")"
    MULTILEVEL_GRANDCHILD_PID="$(cat "$MULTILEVEL_GRANDCHILD_PIDFILE")"
    MULTILEVEL_GGC_PID="$(cat "$MULTILEVEL_GGC_PIDFILE")"
    record_pid "$MULTILEVEL_CHILD_PID"
    record_pid "$MULTILEVEL_GRANDCHILD_PID"
    record_pid "$MULTILEVEL_GGC_PID"

    # `exec sleep 60` (not a bare `sleep 60`, same technique ggc.sh above
    # uses): confirmed empirically on this host that /bin/sh does NOT
    # tail-call-optimize a plain last command into an exec on its own --
    # a bare `sleep 60` forks a genuine child of the /bin/sh wrapper,
    # which would otherwise make MULTILEVEL_DECOY_PID's own descendant set
    # depend on a race between that fork completing and this function's
    # (deliberately zero-sleep) collection/assertions running -- `exec`
    # makes decoy_pid unconditionally the sleep process itself, with no
    # descendants, ever.
    printf '#!/bin/sh\nexec sleep 60\n' > "$trial_dir/decoy.sh"
    chmod +x "$trial_dir/decoy.sh"
    "$trial_dir/decoy.sh" &
    MULTILEVEL_DECOY_PID=$!
    record_pid "$MULTILEVEL_DECOY_PID"

    MULTILEVEL_START_OK=1
}

# run_multilevel_subtree_trial
# Part A: reproduce the OLD one-level bug against a fresh >=3-level chain
# (child -> grandchild -> great-grandchild long sleep) using
# old_one_level_cleanup -- must leave the grandchild and great-grandchild
# still running (empirically proving the bug is real, not assumed). Part B:
# the SAME shape, a FRESH chain, cleaned up via the real (fixed)
# pid_scoped_cleanup -- must remove every level.
run_multilevel_subtree_trial() {
    local label="full-subtree post-order cleanup (>=3 levels deep)"
    local trial_dir
    local root_pid child_pid grandchild_pid ggc_pid decoy_pid

    trial_dir="$(mktemp -d "$WORK_ROOT/multilevel.XXXXXX")"

    # --- Part A: OLD one-level cleanup must NOT reach the deeper levels ---
    start_multilevel_chain "$trial_dir/a"
    if [ "$MULTILEVEL_START_OK" != 1 ]; then
        fail "$label (old-code repro): chain never fully started within the timeout (infra problem, not the fix under test)"
    else
        root_pid="$MULTILEVEL_ROOT_PID"; child_pid="$MULTILEVEL_CHILD_PID"
        grandchild_pid="$MULTILEVEL_GRANDCHILD_PID"; ggc_pid="$MULTILEVEL_GGC_PID"
        decoy_pid="$MULTILEVEL_DECOY_PID"

        old_one_level_cleanup "$decoy_pid" "$root_pid"

        if kill -0 "$grandchild_pid" 2>/dev/null && kill -0 "$ggc_pid" 2>/dev/null; then
            pass "$label: reproduced the OLD one-level bug (grandchild $grandchild_pid and great-grandchild $ggc_pid survive old_one_level_cleanup)"
        else
            fail "$label: could NOT reproduce the OLD bug (grandchild/great-grandchild already gone after old_one_level_cleanup -- this trial's chain setup is unreliable)"
        fi

        # Manual force-cleanup of whatever old_one_level_cleanup missed, so
        # this deliberate bug reproduction never itself leaks a process out
        # of the test run.
        kill -9 "$grandchild_pid" "$ggc_pid" 2>/dev/null
        wait_pids_gone 10 "$child_pid" "$grandchild_pid" "$ggc_pid" "$root_pid" "$decoy_pid"
    fi

    # --- Part B: the FIXED pid_scoped_cleanup must remove ALL levels ---
    start_multilevel_chain "$trial_dir/b"
    if [ "$MULTILEVEL_START_OK" != 1 ]; then
        fail "$label (new-code): chain never fully started within the timeout (infra problem, not the fix under test)"
        return
    fi
    root_pid="$MULTILEVEL_ROOT_PID"; child_pid="$MULTILEVEL_CHILD_PID"
    grandchild_pid="$MULTILEVEL_GRANDCHILD_PID"; ggc_pid="$MULTILEVEL_GGC_PID"
    decoy_pid="$MULTILEVEL_DECOY_PID"

    pid_scoped_cleanup "$label (new-code)" "$decoy_pid" "$root_pid"

    if kill -0 "$root_pid" 2>/dev/null || kill -0 "$child_pid" 2>/dev/null \
       || kill -0 "$grandchild_pid" 2>/dev/null || kill -0 "$ggc_pid" 2>/dev/null \
       || kill -0 "$decoy_pid" 2>/dev/null; then
        fail "$label: fixed pid_scoped_cleanup left at least one level alive (root=$root_pid child=$child_pid grandchild=$grandchild_pid ggc=$ggc_pid decoy=$decoy_pid)"
    else
        pass "$label: fixed pid_scoped_cleanup removed ALL THREE descendant levels + root + decoy (root=$root_pid child=$child_pid grandchild=$grandchild_pid ggc=$ggc_pid decoy=$decoy_pid)"
    fi
}

# run_multilevel_pkill_immunity_trial
# Proves an unrelated, independently-started process sharing one of the
# multilevel chain's own script basenames is UNAFFECTED by a real
# pid_scoped_cleanup run -- the fixed cleanup is purely PID-ancestry-based
# (pgrep -P, never pgrep/pkill -f by name), so a name collision must never
# matter.
run_multilevel_pkill_immunity_trial() {
    local label="multilevel cleanup: unrelated same-name process immunity"
    local trial_dir unrelated_script unrelated_pid

    trial_dir="$(mktemp -d "$WORK_ROOT/mlpkill.XXXXXX")"
    # Deliberately the SAME basename build_multilevel_chain uses for its
    # own great-grandchild script.
    unrelated_script="$trial_dir/greatgrandchild.sh"
    printf '#!/bin/sh\nexec sleep 60\n' > "$unrelated_script"   # exec: see
                                                                # start_multilevel_chain's
                                                                # decoy.sh comment
    chmod +x "$unrelated_script"
    "$unrelated_script" &
    unrelated_pid=$!
    record_pid "$unrelated_pid"

    start_multilevel_chain "$trial_dir/chain"
    if [ "$MULTILEVEL_START_OK" != 1 ]; then
        fail "$label: chain never fully started within the timeout (infra problem, not the fix under test)"
    else
        pid_scoped_cleanup "$label" "$MULTILEVEL_DECOY_PID" "$MULTILEVEL_ROOT_PID"

        if kill -0 "$unrelated_pid" 2>/dev/null; then
            pass "$label: unrelated same-name process ($unrelated_pid) untouched by the multilevel cleanup"
        else
            fail "$label: unrelated same-name process ($unrelated_pid) was killed -- regression"
        fi
    fi

    kill -9 "$unrelated_pid" 2>/dev/null
    wait "$unrelated_pid" 2>/dev/null
}

# run_concurrent_multilevel_trial
# Runs TWO independent multilevel chains at the same time and confirms
# each instance's cleanup never touches the other's still-alive subtree.
run_concurrent_multilevel_trial() {
    local label="concurrent multilevel-chain cleanup isolation"
    local trial_dir_a trial_dir_b
    local root_a child_a grandchild_a ggc_a decoy_a
    local root_b child_b grandchild_b ggc_b decoy_b

    trial_dir_a="$(mktemp -d "$WORK_ROOT/mlconcA.XXXXXX")"
    trial_dir_b="$(mktemp -d "$WORK_ROOT/mlconcB.XXXXXX")"

    start_multilevel_chain "$trial_dir_a"
    if [ "$MULTILEVEL_START_OK" != 1 ]; then
        fail "$label: instance A never fully started within the timeout (infra problem, not the fix under test)"
        return
    fi
    root_a="$MULTILEVEL_ROOT_PID"; child_a="$MULTILEVEL_CHILD_PID"
    grandchild_a="$MULTILEVEL_GRANDCHILD_PID"; ggc_a="$MULTILEVEL_GGC_PID"
    decoy_a="$MULTILEVEL_DECOY_PID"

    start_multilevel_chain "$trial_dir_b"
    if [ "$MULTILEVEL_START_OK" != 1 ]; then
        fail "$label: instance B never fully started within the timeout (infra problem, not the fix under test)"
        kill -9 "$root_a" "$child_a" "$grandchild_a" "$ggc_a" "$decoy_a" 2>/dev/null
        wait_pids_gone 10 "$root_a" "$child_a" "$grandchild_a" "$ggc_a" "$decoy_a"
        return
    fi
    root_b="$MULTILEVEL_ROOT_PID"; child_b="$MULTILEVEL_CHILD_PID"
    grandchild_b="$MULTILEVEL_GRANDCHILD_PID"; ggc_b="$MULTILEVEL_GGC_PID"
    decoy_b="$MULTILEVEL_DECOY_PID"

    # Clean up instance A only, while B is still fully alive.
    pid_scoped_cleanup "$label: instance A cleanup" "$decoy_a" "$root_a"

    if kill -0 "$root_b" 2>/dev/null && kill -0 "$child_b" 2>/dev/null \
       && kill -0 "$grandchild_b" 2>/dev/null && kill -0 "$ggc_b" 2>/dev/null \
       && kill -0 "$decoy_b" 2>/dev/null; then
        pass "$label: instance B's full subtree (root=$root_b child=$child_b grandchild=$grandchild_b ggc=$ggc_b decoy=$decoy_b) is untouched by instance A's cleanup"
    else
        fail "$label: instance B's subtree was affected by instance A's cleanup (root=$root_b child=$child_b grandchild=$grandchild_b ggc=$ggc_b decoy=$decoy_b)"
    fi

    # Now clean up B and confirm both instances are fully gone.
    pid_scoped_cleanup "$label: instance B cleanup" "$decoy_b" "$root_b"
}

# --- snapshot fail-closed / comprehensiveness negative tests (regression
# review P2 item 3; extended regression review for the two-stage,
# no-type-filter rewrite) ------------------------------------------------
# A THROWAWAY synthetic tree under WORK_ROOT (never the real obj/bin
# trees): snapshots it, makes ONE controlled change, re-snapshots, and
# asserts the snapshot correctly differs (cases a/b) or correctly FATALs
# (cases c/d/e/f/g).
run_snapshot_negative_tests() {
    local synth="$WORK_ROOT/snap_synth"
    local before after rc
    local find_probe_rc sock_path py_rc g_list p

    mkdir -p "$synth/sub"
    echo "hello" > "$synth/sub/file.txt"

    # (a) new empty subdirectory -> snapshot must differ. (regression review:
    # re-confirmed under the two-stage rewrite -- unchanged behaviour.)
    before="$(snapshot_tree_root "$synth" mtime)"
    mkdir -p "$synth/new_empty_dir"
    after="$(snapshot_tree_root "$synth" mtime)"
    if [ "$before" != "$after" ]; then
        pass "snapshot negative test (a): a new empty subdirectory changes the snapshot"
    else
        fail "snapshot negative test (a): a new empty subdirectory did NOT change the snapshot"
    fi
    rmdir "$synth/new_empty_dir"

    # (b) new symlink -> snapshot must differ.
    before="$(snapshot_tree_root "$synth" mtime)"
    ln -s "sub/file.txt" "$synth/new_link"
    after="$(snapshot_tree_root "$synth" mtime)"
    if [ "$before" != "$after" ]; then
        pass "snapshot negative test (b): a new symlink changes the snapshot"
    else
        fail "snapshot negative test (b): a new symlink did NOT change the snapshot"
    fi
    rm -f "$synth/new_link"

    # (c) chmod a file to be unreadable (000) -> a hash attempt on it fails
    # -> the snapshot function must FATAL/exit nonzero immediately instead
    # of silently producing a shorter snapshot. Run inside an EXPLICIT,
    # deliberate `( ... )` subshell so the induced `exit 1` only terminates
    # that subshell, not this whole test script -- this is a controlled
    # test of the failure path, not a real failure. (mtime mode would NOT
    # reproduce this: `stat` only needs directory lookup, not read
    # permission on the file itself -- only sha mode, which must actually
    # read the file's bytes, is affected, matching the real-world failure
    # mode this guards.)
    chmod 000 "$synth/sub/file.txt"
    ( snapshot_tree_root "$synth" sha >/dev/null 2>/dev/null )
    rc=$?
    chmod 644 "$synth/sub/file.txt"   # restore BEFORE asserting, so a
                                       # failed assertion below never leaves
                                       # a permanently-broken synthetic tree
    if [ "$rc" -ne 0 ]; then
        pass "snapshot negative test (c): an unreadable file (chmod 000) makes the snapshot function FATAL/exit nonzero instead of silently omitting it"
    else
        fail "snapshot negative test (c): an unreadable file did NOT make the snapshot function fail (rc=$rc) -- silent-omission regression"
    fi

    # (d) regression review: an unreadable DIRECTORY (permission-denied while
    # DESCENDING, as opposed to (c)'s unreadable regular file) forces
    # `find` ITSELF to exit nonzero mid-listing. Verified INDEPENDENTLY,
    # with a bare `find` call, BEFORE trusting the assertion below -- a
    # host running this as root (chmod 000 does not block root's own
    # traversal) would make chmod 000 a no-op for this purpose, which would
    # make the assertion below vacuous rather than a real proof.
    mkdir -p "$synth/blocked_dir"
    echo "secret" > "$synth/blocked_dir/inner.txt"
    chmod 000 "$synth/blocked_dir"

    find "$synth/blocked_dir" -mindepth 1 -print0 >/dev/null 2>/dev/null
    find_probe_rc=$?
    if [ "$find_probe_rc" -eq 0 ]; then
        echo "NOTE: chmod 000 did not make a bare 'find' fail on this host (running as root/sudo?) -- skipping the forced-find-failure/unreadable-directory negative test (d) as unreliable here" >&2
    else
        ( snapshot_tree_root "$synth" mtime >/dev/null 2>/dev/null )
        rc=$?
        if [ "$rc" -ne 0 ]; then
            pass "snapshot negative test (d): an unreadable directory (chmod 000, permission-denied while descending) forces find itself to fail (confirmed independently first), and the snapshot function correctly FATALs instead of silently omitting its contents"
        else
            fail "snapshot negative test (d): an unreadable directory did NOT make the snapshot function fail (rc=$rc) -- forced-find-failure regression"
        fi
    fi
    chmod 755 "$synth/blocked_dir"   # restore BEFORE removal
    rm -rf "$synth/blocked_dir"

    # (e) regression review: a FIFO under the tree must FATAL as an
    # unexpected type. Before this round's fix, find's own `-type d -o
    # -type f -o -type l` filter silently EXCLUDED a fifo from its output
    # entirely -- invisible, not FATAL -- so this case would have PASSED
    # (spuriously, by never even reaching snapshot_one_entry) against the
    # OLD code; it correctly FAILS to reproduce that old behaviour here
    # because this test only runs against the current (fixed) function.
    if mkfifo "$synth/a_fifo" 2>/dev/null; then
        ( snapshot_tree_root "$synth" mtime >/dev/null 2>/dev/null )
        rc=$?
        rm -f "$synth/a_fifo"
        if [ "$rc" -ne 0 ]; then
            pass "snapshot negative test (e): a FIFO under the tree makes the snapshot function FATAL (unexpected type) instead of being silently excluded"
        else
            fail "snapshot negative test (e): a FIFO under the tree did NOT make the snapshot function fail -- unexpected-type regression"
        fi
    else
        echo "NOTE: mkfifo unavailable/failed on this host -- skipping the FIFO negative test (e)" >&2
    fi

    # (f) regression review: a Unix domain socket, best-effort via python3's
    # socket module (skipped cleanly if python3 or the bind is unavailable
    # -- this sub-case is explicitly optional per the task, not a required
    # pass/fail).
    sock_path="$synth/a_socket"
    py_rc=1
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$sock_path" <<'PYEOF' 2>/dev/null
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(sys.argv[1])
PYEOF
        py_rc=$?
    fi
    if [ "$py_rc" -eq 0 ] && [ -S "$sock_path" ]; then
        ( snapshot_tree_root "$synth" mtime >/dev/null 2>/dev/null )
        rc=$?
        rm -f "$sock_path"
        if [ "$rc" -ne 0 ]; then
            pass "snapshot negative test (f): a Unix domain socket under the tree makes the snapshot function FATAL (unexpected type) instead of being silently excluded"
        else
            fail "snapshot negative test (f): a Unix domain socket under the tree did NOT make the snapshot function fail -- unexpected-type regression"
        fi
    else
        rm -f "$sock_path" 2>/dev/null
        echo "NOTE: could not create a Unix domain socket on this host (no python3, or bind failed) -- skipping the Unix-domain-socket negative test (f) as genuinely impractical here" >&2
    fi

    # (g) regression review: a symlink where readlink itself effectively
    # fails, forced via the exact mechanism this two-stage rewrite exposes
    # a real seam for -- list the tree (stage 1), THEN remove the symlink
    # from disk BEFORE processing (stage 2). Exercised by calling the two
    # stages directly (not through the single snapshot_tree_root entry
    # point), so the removal can land exactly between them.
    ln -sf "sub/file.txt" "$synth/vanish_link"
    g_list="$(mktemp "$WORK_ROOT/snap_list_g.XXXXXX")"
    find "$synth" -mindepth 1 -print0 > "$g_list"
    rm -f "$synth/vanish_link"
    (
        while IFS= read -r -d '' p; do
            snapshot_one_entry "$p" mtime
        done < "$g_list"
    ) >/dev/null 2>/dev/null
    rc=$?
    rm -f "$g_list"
    if [ "$rc" -ne 0 ]; then
        pass "snapshot negative test (g): a symlink removed between listing and processing makes the snapshot function FATAL instead of silently dropping it"
    else
        fail "snapshot negative test (g): a symlink removed between listing and processing did NOT make the snapshot function fail (rc=$rc)"
    fi

    rm -rf "$synth"
}

echo "=== run_ubsan_signal_handling_test: snapshot fail-closed / comprehensiveness negative tests ==="
run_snapshot_negative_tests

echo "=== run_ubsan_signal_handling_test: snapshotting real obj/bin trees (before) ==="
# rc checked explicitly (regression review fix): snapshot_real_trees/_sha are
# still captured via `$(...)`, which is ITS OWN subshell boundary -- an
# `exit 1` deep inside (snapshot_one_entry, via snapshot_tree_root) makes
# THAT subshell exit non-zero, which becomes this assignment's own exit
# status, but nothing auto-aborts the script from that alone (this script
# uses `set -uo pipefail`, not `-e`). Checking `rc` and exiting explicitly
# here is what makes the fail-closed guarantee actually reach the top level.
SNAP_MTIME_BEFORE="$(snapshot_real_trees)"; rc=$?
if [ "$rc" -ne 0 ]; then
    echo "FATAL: snapshot_real_trees (before, mtime) failed (rc=$rc) -- aborting immediately rather than risk comparing against an incomplete snapshot" >&2
    exit 1
fi
SNAP_SHA_BEFORE="$(snapshot_real_trees_sha)"; rc=$?
if [ "$rc" -ne 0 ]; then
    echo "FATAL: snapshot_real_trees_sha (before, sha) failed (rc=$rc) -- aborting immediately rather than risk comparing against an incomplete snapshot" >&2
    exit 1
fi

echo "=== run_ubsan_signal_handling_test: normal (uninterrupted) runs ==="
run_normal_trial run_selftest_ubsan.sh
run_normal_trial run_selftest_ubsan.sh --use-standard-math
run_normal_trial run_counter_saturation_ubsan.sh
run_normal_trial run_counter_saturation_ubsan.sh --use-standard-math

echo "=== run_ubsan_signal_handling_test: signal trials ==="
run_signal_trial run_counter_saturation_ubsan.sh run   TERM 143
run_signal_trial run_counter_saturation_ubsan.sh run   INT  130
run_signal_trial run_counter_saturation_ubsan.sh build TERM 143
run_signal_trial run_counter_saturation_ubsan.sh build INT  130
run_signal_trial run_selftest_ubsan.sh           run   TERM 143
run_signal_trial run_selftest_ubsan.sh           run   INT  130

echo "=== run_ubsan_signal_handling_test: decoy-process trial (wait_for_binary descendant scoping) ==="
run_decoy_trial

echo "=== run_ubsan_signal_handling_test: unrelated same-pattern process immunity (pkill/pgrep scoping) ==="
run_pkill_immunity_trial

echo "=== run_ubsan_signal_handling_test: concurrent decoy-trial cleanup isolation ==="
run_concurrent_decoy_trial

echo "=== run_ubsan_signal_handling_test: full-subtree post-order cleanup (>=3 levels deep) ==="
run_multilevel_subtree_trial

echo "=== run_ubsan_signal_handling_test: multilevel cleanup unrelated same-name immunity ==="
run_multilevel_pkill_immunity_trial

echo "=== run_ubsan_signal_handling_test: concurrent multilevel-chain cleanup isolation ==="
run_concurrent_multilevel_trial

echo "=== run_ubsan_signal_handling_test: snapshotting real obj/bin trees (after) ==="
SNAP_MTIME_AFTER="$(snapshot_real_trees)"; rc=$?
if [ "$rc" -ne 0 ]; then
    echo "FATAL: snapshot_real_trees (after, mtime) failed (rc=$rc) -- aborting immediately rather than risk comparing against an incomplete snapshot" >&2
    exit 1
fi
SNAP_SHA_AFTER="$(snapshot_real_trees_sha)"; rc=$?
if [ "$rc" -ne 0 ]; then
    echo "FATAL: snapshot_real_trees_sha (after, sha) failed (rc=$rc) -- aborting immediately rather than risk comparing against an incomplete snapshot" >&2
    exit 1
fi

if [ "$SNAP_MTIME_BEFORE" = "$SNAP_MTIME_AFTER" ]; then
    pass "real c_impl/obj+bin and audio_common obj+bin mtimes unchanged across all runs"
else
    fail "real c_impl/obj+bin and audio_common obj+bin mtimes CHANGED across the runs above"
fi

if [ "$SNAP_SHA_BEFORE" = "$SNAP_SHA_AFTER" ]; then
    pass "real c_impl/obj+bin and audio_common obj+bin sha256 unchanged across all runs"
else
    fail "real c_impl/obj+bin and audio_common obj+bin sha256 CHANGED across the runs above"
fi

# --- Final full-sweep leftover-process check (regression review) -------------
# Independently re-checks EVERY PID this ENTIRE test run ever recorded via
# record_pid, across every trial -- not a spot-check of just the specific
# PIDs each trial's own narrower assertion happened to look at. This is
# exactly the check that would have caught the verified PPID=1
# orphan: a leftover descendant that no single trial's own assertion,
# taken alone, was inspecting.
echo "=== run_ubsan_signal_handling_test: final full-sweep leftover-process check (all recorded PIDs) ==="
leftover_count=0
recorded_count=0
for p in $ALL_TRACKED_PIDS; do
    recorded_count=$((recorded_count + 1))
    if kill -0 "$p" 2>/dev/null; then
        leftover_count=$((leftover_count + 1))
        fail "final sweep: recorded PID $p is STILL ALIVE after its own trial's cleanup"
    fi
done
if [ "$leftover_count" -eq 0 ]; then
    pass "final sweep: zero leftover processes among all $recorded_count PIDs recorded across every trial"
else
    fail "final sweep: $leftover_count leftover process(es) found among $recorded_count PIDs recorded across every trial"
fi

echo
echo "TOTAL: $((PASS_COUNT + FAIL_COUNT))  PASS: $PASS_COUNT  FAIL: $FAIL_COUNT"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "run_ubsan_signal_handling_test: ALL PASS"
    exit 0
else
    echo "run_ubsan_signal_handling_test: FAIL"
    exit 1
fi
