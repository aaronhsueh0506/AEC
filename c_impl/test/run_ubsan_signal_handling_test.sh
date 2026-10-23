#!/usr/bin/env bash
# run_ubsan_signal_handling_test.sh - regression test for signal handling in test/run_counter_saturation_ubsan.sh and
# test/run_selftest_ubsan.sh.
#
# The failure it guards against is one combined `trap 'rm -rf "$SCRATCH"' EXIT INT TERM`. Bash runs a pending trap only when the
# current foreground child returns, and without an explicit `exit` the script then resumes at the next statement: a SIGTERM
# during the compiled test binary let the run finish and print a false "UBSan probe: PASS" with exit 0, and a signal during the
# audiolib build deleted the scratch directory mid-flight. The scripts use separate handlers instead (`trap cleanup EXIT`,
# `trap 'exit 130' INT`, `trap 'exit 143' TERM`): a signal ends the script with 128+signum and the EXIT trap still cleans up.
#
# For both scripts, SIGINT and separately SIGTERM are sent (i) during the audiolib-build phase (run_counter_saturation_ubsan.sh
# only) and (ii) while the compiled test binary runs. Each trial asserts that the exit code is exactly 130 / 143, that the log
# does not contain "UBSan probe: PASS" (a bare "PASS" would also match the test binary's own lines) and that the scratch
# directory is gone. A normal run of each script (default and --use-standard-math) must still print "UBSan probe: PASS" and
# exit 0, and c_impl/obj + bin and audiolib's obj + bin must be unchanged (hash + mtime snapshot) across all runs.
#
# Signal delivery: each trial runs the target as `sh script &`, waits for a phase-specific readiness signal, then sends
# `kill -s SIG` to the script's own PID only, not its children or process group. Bash acts on the trap only after the foreground
# child returns, so signalling the script while the binary runs does not kill the binary. The build phase is widened with
# MAKE=<wrapper> (both scripts honor MAKE=), which delays the real `make ... lib`. `set -m` is required: without job control
# SIGINT is ignored for `&` commands and cannot be trapped by the child, so `kill -INT` would be swallowed.
#
# The helpers carry their own negative tests: a decoy process shows that unscoped pgrep matching is fooled and
# descendant-scoped matching is not; PID-scoped cleanup must reach a deep process chain, leave an unrelated same-name process
# alone and keep concurrent trials apart; the tree snapshot must fail closed (run_snapshot_negative_tests).
#
# Usage: ./test/run_ubsan_signal_handling_test.sh   (cd's to the repo root itself)
# Exit 0 and "run_ubsan_signal_handling_test: ALL PASS" mean every assertion held; a "FAIL" line is a regression.

set -uo pipefail
set -m

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."   # c_impl/

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
COMMON_DIR="$(cd "$COMMON_DIR" && pwd)"
C_IMPL_DIR="$(pwd)"

PASS_COUNT=0
FAIL_COUNT=0

pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); echo "  FAIL: $1"; }

WORK_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/ubsan_sigtest.XXXXXX")"
trap 'rm -rf "$WORK_ROOT"' EXIT

# --- platform-detection probe (must run before any trial; fails closed) ---
# snapshot_real_trees{,_sha} need a `stat` and a sha256 tool that work on this host: BSD `stat -f` / `shasum -a 256` or GNU
# `stat -c` / `sha256sum`. Detect once against a throwaway probe file in WORK_ROOT. If neither syntax works, FATAL immediately:
# silently empty before/after snapshots would compare equal and prove nothing.
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

# stat_line/sha_line: the single point through which both snapshot functions call the platform-specific tools.
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

# Test-harness MAKE=... wrapper: delays the audiolib "lib" build call so the build-phase trials get a wide, deterministic
# window. It touches SLOWMAKE_MARKER right before the delay so the harness can tell the target is inside that foreground
# command.
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
# Emits one canonical line for one directory, regular file or symlink, or FATAL-exits the whole script on any failure, so a
# failed stat/sha cannot produce a shorter snapshot that still compares equal. A directory records its path and type; a symlink
# records its target string via `readlink` (never followed); anything else (fifo, socket, device node) is refused, since it
# should never appear in a build's obj/bin tree.
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
        # Listed by stage 1 but gone by the time stage 2 runs (e.g. a symlink removed mid-walk). Dropping it would make a
        # quietly shorter snapshot, so FATAL.
        echo "FATAL: snapshot: '$path' was recorded by find but no longer exists (removed between listing and processing) -- refusing to silently drop it from the snapshot" >&2
        exit 1
    else
        # Reached for every other entry type, because snapshot_tree_root's find does not pre-filter by -type: refuse it and
        # name the kind where that is cheap to determine.
        kind="an entry of an unrecognized type"
        [ -p "$path" ] 2>/dev/null && kind="a FIFO"
        [ -S "$path" ] 2>/dev/null && kind="a Unix domain socket"
        echo "FATAL: snapshot: '$path' is $kind (not a directory, regular file, or symlink) -- refusing to silently skip an unexpected entry type" >&2
        exit 1
    fi
}

# snapshot_tree_root <root_path> <mode:mtime|sha>
# One root's contribution to the snapshot: a PRESENT/ABSENT record (so an absent root becoming a new empty one is itself a
# detectable change) plus, for a PRESENT root, one line per entry under it, classified by snapshot_one_entry and sorted for a
# stable comparison. `-mindepth 1` leaves out the root itself, which the PRESENT/ABSENT line covers.
# Two stages:
#   1. find with no type filter into a real list_file under $WORK_ROOT, with find's exit status checked right there (FATAL
#      naming the root on failure), so an incomplete listing never produces a snapshot.
#   2. `while ... done < list_file > out_file`, calling snapshot_one_entry per entry. Reading from a file and redirecting the
#      output keeps the loop in the current shell, so an `exit` inside snapshot_one_entry propagates to the caller; a pipe on
#      either side (`find | while` or `| sort`) would run the loop in a subshell and swallow it. `sort` runs afterwards.
# run_snapshot_negative_tests checks that propagation.
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
    snapshot_tree_root "$COMMON_DIR/obj" mtime
    snapshot_tree_root "$COMMON_DIR/bin" mtime
}

snapshot_real_trees_sha() {
    snapshot_tree_root "$C_IMPL_DIR/obj" sha
    snapshot_tree_root "$C_IMPL_DIR/bin" sha
    snapshot_tree_root "$COMMON_DIR/obj" sha
    snapshot_tree_root "$COMMON_DIR/bin" sha
}

# --- readiness detection ---
# Busy-poll (no sleep, to keep the detection window tight against the artificial build delay), bounded by an iteration cap and a
# wall-clock cap so a broken host fails fast instead of hanging.
wait_for_marker() {
    local marker="$1"
    local start=$SECONDS
    while [ ! -e "$marker" ] && [ $((SECONDS - start)) -lt 30 ]; do :; done
    [ -e "$marker" ]
}

# wait_for_file <path> <timeout_secs>
# Busy-polls until <path> exists and is non-empty, or the timeout elapses. Used to wait for a descendant in a multilevel chain
# to start and record its own PID, which is ground truth independent of the collection/cleanup logic under test.
wait_for_file() {
    local path="$1" timeout="$2"
    local start=$SECONDS
    while [ ! -s "$path" ] && [ $((SECONDS - start)) -lt "$timeout" ]; do :; done
    [ -s "$path" ]
}

# is_descendant_of <ancestor_pid> <candidate_pid>
# Walks candidate_pid's parent chain (repeated `ps -o ppid=`) up to pid 1; succeeds iff ancestor_pid appears in it. This tells a
# real descendant of the backgrounded trial script from an unrelated process whose command line happens to match.
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
# Reports "ready" only once a pgrep -f match is target_pid itself or one of its descendants; target_pid is passed in explicitly
# by every call site (the trial's own `pid=$!`).
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

# --- one signal trial ---
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

# --- one normal (uninterrupted) run ---
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

    # An uninterrupted run gets the same scratch-cleanup proof as the interrupted ones.
    local leftover
    leftover="$(ls -A "$tmpdir_for_script" 2>/dev/null)"
    if [ -z "$leftover" ]; then
        pass "$script_name ($extra_desc): normal run scratch directory fully removed"
    else
        fail "$script_name ($extra_desc): normal run scratch directory NOT removed (leftover: $leftover)"
    fi
}

# --- decoy-process trial (guards wait_for_binary's descendant scoping) ---
# old_unscoped_wait_for_binary is a verbatim copy of a system-wide-matching wait_for_binary. No real trial uses it; it lets
# run_decoy_trial show that unscoped matching really is fooled by a same-pattern decoy.
old_unscoped_wait_for_binary() {
    local binpat="$1"
    local start=$SECONDS
    while ! pgrep -f "$binpat" >/dev/null 2>&1 && [ $((SECONDS - start)) -lt 30 ]; do :; done
    pgrep -f "$binpat" >/dev/null 2>&1
}

# --- PID-scoped cleanup helpers ---
# wait_pids_gone <timeout_secs> <pid...>
# Busy-polls until `kill -0` fails (ESRCH) for every pid given, or the timeout elapses. A single check right after the kills is
# racy: a just-killed process can still answer `kill -0` while it is a zombie awaiting reap. Blank arguments are skipped.
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

# --- Global PID bookkeeping ---
# record_pid <pid>: appends pid to ALL_TRACKED_PIDS, the list of every detached/background fixture process this script spawns
# across all trials (decoys, multilevel chains, unrelated-process fixtures; a script under test reaps itself via `wait`). The
# final sweep asserts `kill -0` fails for each of them, which also catches a leftover descendant that no single trial's own
# assertion inspects.
ALL_TRACKED_PIDS=""
record_pid() {
    [ -n "${1:-}" ] && ALL_TRACKED_PIDS="$ALL_TRACKED_PIDS $1"
}

# collect_descendants_post_order <pid>
# Recursively walks pid's full descendant subtree with `pgrep -P`, depth-first post-order (deepest descendants first, pid last).
# It only queries, never kills, so a caller can build the complete list for every pid before the first kill: once a process
# exits, its children are reparented away and `pgrep -P` on the dead pid finds nothing.
collect_descendants_post_order() {
    local pid="$1" c
    for c in $(pgrep -P "$pid" 2>/dev/null); do
        collect_descendants_post_order "$c"
    done
    echo "$pid"
}

# collect_full_subtree_settled <root_pid...>
# Runs collect_descendants_post_order for every root, in argument order, and repeats the whole collection until $SECONDS has
# advanced by at least 2 since the first pass (so at least one full real second has elapsed), then returns the last result. No
# `sleep`: the pgrep passes themselves consume the time. This closes a race: a /bin/sh fixture that forks its last command
# (decoy_bin/real_bin) can be mid-fork when a newly found node is first queried for children, and "two consecutive passes agree"
# can then agree on an incomplete answer. Nothing is killed until this function returns.
collect_full_subtree_settled() {
    local settle_start=$SECONDS result pid
    result="$(for pid in "$@"; do collect_descendants_post_order "$pid"; done)"
    # `-lt 2`, not `-lt 1`: $SECONDS has 1-second granularity, so requiring it to change once can mean almost no real time
    # passed (the first pass may end just before a second boundary). A difference of 2 guarantees at least one full second.
    while [ $((SECONDS - settle_start)) -lt 2 ]; do
        result="$(for pid in "$@"; do collect_descendants_post_order "$pid"; done)"
    done
    printf '%s\n' "$result"
}

# old_one_level_cleanup <decoy_pid> <target_pid>
# Verbatim copy of a one-level pid_scoped_cleanup (kill/wait mechanics only). No real cleanup path calls it; it lets
# run_multilevel_subtree_trial show that one-level cleanup leaves grandchild-or-deeper descendants running. It makes no
# pass/fail call of its own: the caller inspects the process state.
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
# Explicit-PID cleanup, never a name-based pkill/pgrep that could kill, or be fooled by, an unrelated process with the same
# command-line text:
#   (1) build the complete post-order descendant list for target_pid and decoy_pid (collect_full_subtree_settled) before any
#       kill;
#   (2) kill every pid in that order (deepest descendants first, target_pid and decoy_pid last);
#   (3) busy-poll `kill -0` until every one of those pids, not just target_pid/decoy_pid, is gone.
pid_scoped_cleanup() {
    local label="$1" decoy_pid="$2" target_pid="$3"
    local full_list p

    full_list="$(collect_full_subtree_settled "$target_pid" "$decoy_pid")"

    for p in $full_list; do
        kill -9 "$p" 2>/dev/null
    done

    # Reap target_pid/decoy_pid, this script's own direct children; deeper descendants are reaped by their former parent or,
    # once reparented, by pid 1.
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
    # Per-invocation-unique pattern, defense in depth on top of pid_scoped_cleanup: a second independent run of this script
    # never shares this process-name text with a decoy trial. The default id folds in $$ and two RANDOM draws; callers pass an
    # explicit id when they deliberately want a collision, to show that the PID-scoped cleanup holds even then.
    local pattern="decoy_trial_ubsan_target_${id}"
    local binpat="${pattern}\$"
    local trial_dir decoy_bin real_bin target_script decoy_pid target_pid
    local old_start old_elapsed old_rc new_start new_elapsed new_rc
    local c found_pid

    trial_dir="$(mktemp -d "$WORK_ROOT/decoy.XXXXXX")"

    # The decoy: a real running process whose whole command line matches binpat, started by this harness as a sibling of the
    # target pid below. It is a throwaway script named to satisfy the anchored `<pattern>$` regex (pgrep -f matches the full
    # command line, so no trailing argv tokens).
    # It is a bare `sleep 60`, not `exec sleep 60`: decoy_bin/real_bin's script path is what binpat anchors against in
    # wait_for_binary / is_descendant_of / old_unscoped_wait_for_binary, and `exec` would replace the command line with
    # "sleep 60" at an unpredictable later moment, breaking those pgrep -f lookups intermittently. The /bin/sh wrapper may fork
    # an untracked child; pid_scoped_cleanup's settled collection handles that.
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

    # The "real descendant": a same-named binary that appears only after a delay, as a child of a background target script
    # started the way run_signal_trial starts the scripts under test (`... & pid=$!`).
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

    # Unscoped matching, called while only the decoy runs (the target's real descendant is still asleep for 3s): it returns
    # "ready" almost instantly purely because the system-wide decoy matches.
    old_start=$SECONDS
    old_unscoped_wait_for_binary "$binpat"; old_rc=$?
    old_elapsed=$((SECONDS - old_start))

    if [ "$old_rc" -eq 0 ] && [ "$old_elapsed" -le 1 ]; then
        pass "$label: reproduced the OLD bug (unscoped wait_for_binary fooled by the decoy in ${old_elapsed}s, before the real descendant could possibly exist)"
    else
        fail "$label: could NOT reproduce the OLD bug (unscoped call rc=$old_rc elapsed=${old_elapsed}s) -- decoy setup is unreliable, this trial needs fixing"
    fi

    # Scoped matching, called against the same decoy and the same not-yet-ready target_pid: it must not be fooled, and can
    # return "ready" only once target_pid's own real descendant exists, no sooner than the target script's 3s delay.
    new_start=$SECONDS
    wait_for_binary "$target_pid" "$binpat"; new_rc=$?
    new_elapsed=$((SECONDS - new_start))

    if [ "$new_rc" -eq 0 ] && [ "$new_elapsed" -ge 2 ]; then
        pass "$label: fixed wait_for_binary ignored the decoy and correctly waited for target_pid's real descendant (${new_elapsed}s)"
    else
        fail "$label: fixed wait_for_binary rc=$new_rc elapsed=${new_elapsed}s -- expected success at >=2s (the decoy must not short-circuit it)"
    fi

    # Confirm that the match it accepted really is a descendant of target_pid, not the decoy.
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

    # Cleanup of the decoy and the target's whole subtree is PID-scoped (see pid_scoped_cleanup).
    pid_scoped_cleanup "$label" "$decoy_pid" "$target_pid"
}

# --- unrelated same-pattern process immunity trial ---
# A name-based cleanup could kill an unrelated process that merely shares the decoy pattern's command-line text. This starts an
# independent background process with the same pattern (the same id is passed to run_decoy_trial), then confirms it is still
# running, untouched, after that trial's cleanup; it removes the process itself afterwards (it is not one of the trial's PIDs).
run_pkill_immunity_trial() {
    local label="pkill/pgrep immunity: unrelated same-pattern process"
    local id="pkill_immunity_shared_$$_${RANDOM}"
    local trial_dir pattern unrelated_bin unrelated_pid unrelated_list p2

    trial_dir="$(mktemp -d "$WORK_ROOT/pkillimm.XXXXXX")"
    pattern="decoy_trial_ubsan_target_${id}"
    unrelated_bin="$trial_dir/$pattern"
    # Bare (non-exec) `sleep 60`, see the decoy_bin/real_bin comment in run_decoy_trial: the pattern must stay pgrep -f-matchable
    # by its script path for the trial that runs with the same pattern.
    cat > "$unrelated_bin" <<'EOF'
#!/bin/sh
sleep 60
EOF
    chmod +x "$unrelated_bin"
    "$unrelated_bin" &
    unrelated_pid=$!
    record_pid "$unrelated_pid"

    # A full decoy trial using the same pattern text (same id) as the unrelated process above. A system-wide
    # `pkill -9 -f "$binpat"` in its cleanup would kill that process purely because the command line matches.
    run_decoy_trial "$id"

    if kill -0 "$unrelated_pid" 2>/dev/null; then
        pass "$label: unrelated same-pattern process ($unrelated_pid) is still running, untouched, after the decoy trial's cleanup"
    else
        fail "$label: unrelated same-pattern process ($unrelated_pid) was killed by the decoy trial's cleanup -- system-wide pkill/pgrep-by-name regression"
    fi

    # unrelated_bin has the same bare `sleep 60` shape as decoy_bin/real_bin, so it may have forked an untracked child; a plain
    # `kill -9 "$unrelated_pid"` would leave that child running under init. Use the same settled full-subtree collection as
    # pid_scoped_cleanup (this one pid, no decoy counterpart) and verify every collected pid is gone.
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
# Starts one decoy + target (+ later real_bin) trio like run_decoy_trial's setup, parameterized so several instances can exist at
# once. It must not be called through command substitution: that would start the background jobs as the subshell's children, but
# the `wait`/`kill -0` bookkeeping in pid_scoped_cleanup and run_concurrent_decoy_trial needs them to be this script's own. It
# sets the *_OUT globals instead of returning multiple values (bash 3.2 has no `local -n`).
START_DECOY_INSTANCE_DECOY_PID=""
START_DECOY_INSTANCE_TARGET_PID=""
START_DECOY_INSTANCE_PATTERN=""
start_decoy_instance() {
    local id="$1" trial_dir="$2"
    local pattern="decoy_trial_ubsan_target_${id}"
    local decoy_bin real_bin target_script

    # Bare (non-exec) `sleep 60`, see the decoy_bin/real_bin comment in run_decoy_trial.
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
# Busy-polls until `pgrep -P parent_pid` reports at least one child, or the timeout elapses.
wait_for_child_of() {
    local parent_pid="$1" timeout="$2"
    local start=$SECONDS
    while [ $((SECONDS - start)) -lt "$timeout" ]; do
        [ -n "$(pgrep -P "$parent_pid" 2>/dev/null)" ] && return 0
    done
    return 1
}

# --- concurrent decoy-trial cleanup isolation trial ---
# Runs two independent decoy + target trios at the same time with the same literal pattern (id "conc_shared"), so it tests the
# PID scoping of pid_scoped_cleanup rather than pattern uniqueness. It cleans up instance A while B is fully alive and checks
# that B is untouched, then cleans up B and checks that both are gone.
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

    # Both instances run concurrently and share one process-name pattern. Wait for each target's real_bin child to appear, so
    # instance A's cleanup has a real captured child to kill.
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

# --- Full-subtree post-order cleanup trials ---
# build_multilevel_chain <trial_dir>
# Creates (does not start) three chained POSIX shell scripts under trial_dir, forming a 3-level descendant chain below whatever
# process the caller starts as the root (the caller runs MULTILEVEL_ROOT_SCRIPT itself and takes its PID from `$!`):
#   root (caller-started)
#     -> child               (level 1, writes its $$ to child_pidfile)
#          -> grandchild         (level 2, writes its $$ to grandchild_pidfile)
#               -> great-grandchild (level 3, writes its $$ to ggc_pidfile, then `exec`s a long sleep)
# Each level writes its own pidfile, which is ground truth independent of collect_descendants_post_order / pid_scoped_cleanup.
# Results come back in OUT-globals (bash 3.2: no `local -n`), as in start_decoy_instance.
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
# Builds and starts one multilevel chain plus a throwaway decoy sleep process (the <decoy_pid> <target_pid> shape of
# pid_scoped_cleanup), waits for every level's pidfile, records every spawned PID via record_pid and sets the MULTILEVEL_*_PID
# out-globals. MULTILEVEL_START_OK is 1 on success, or 0 if the chain did not fully start in time (an infrastructure problem,
# not an assertion target).
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

    # `exec sleep 60`, like ggc.sh above: /bin/sh does not turn a plain last command into an exec, so a bare `sleep 60` would
    # fork a child of the wrapper and make MULTILEVEL_DECOY_PID's descendant set depend on a race. `exec` makes decoy_pid the
    # sleep process itself, with no descendants.
    printf '#!/bin/sh\nexec sleep 60\n' > "$trial_dir/decoy.sh"
    chmod +x "$trial_dir/decoy.sh"
    "$trial_dir/decoy.sh" &
    MULTILEVEL_DECOY_PID=$!
    record_pid "$MULTILEVEL_DECOY_PID"

    MULTILEVEL_START_OK=1
}

# run_multilevel_subtree_trial
# Part A: old_one_level_cleanup against a fresh 3-level chain must leave the grandchild and great-grandchild running (the
# one-level blind spot is real). Part B: the same shape, cleaned up by the real pid_scoped_cleanup, must remove every level.
run_multilevel_subtree_trial() {
    local label="full-subtree post-order cleanup (>=3 levels deep)"
    local trial_dir
    local root_pid child_pid grandchild_pid ggc_pid decoy_pid

    trial_dir="$(mktemp -d "$WORK_ROOT/multilevel.XXXXXX")"

    # --- Part A: the one-level cleanup must NOT reach the deeper levels ---
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

        # Force-clean whatever old_one_level_cleanup missed, so this deliberate reproduction leaks no process.
        kill -9 "$grandchild_pid" "$ggc_pid" 2>/dev/null
        wait_pids_gone 10 "$child_pid" "$grandchild_pid" "$ggc_pid" "$root_pid" "$decoy_pid"
    fi

    # --- Part B: pid_scoped_cleanup must remove ALL levels ---
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
# An unrelated, independently started process sharing one of the chain's script basenames must be unaffected by a real
# pid_scoped_cleanup run: the cleanup follows pgrep -P ancestry, never a name, so a name collision cannot matter.
run_multilevel_pkill_immunity_trial() {
    local label="multilevel cleanup: unrelated same-name process immunity"
    local trial_dir unrelated_script unrelated_pid

    trial_dir="$(mktemp -d "$WORK_ROOT/mlpkill.XXXXXX")"
    # Deliberately the same basename build_multilevel_chain uses for its great-grandchild script.
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
# Runs two independent multilevel chains at the same time and checks that each instance's cleanup leaves the other's subtree
# alone.
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

    # Now clean up B and confirm both instances are gone.
    pid_scoped_cleanup "$label: instance B cleanup" "$decoy_b" "$root_b"
}

# --- snapshot fail-closed / comprehensiveness negative tests ---
# A throwaway synthetic tree under WORK_ROOT (never the real obj/bin trees): snapshot it, make one controlled change,
# re-snapshot, and assert that the snapshot differs (cases a, b) or FATALs (cases c-g).
run_snapshot_negative_tests() {
    local synth="$WORK_ROOT/snap_synth"
    local before after rc
    local find_probe_rc sock_path py_rc g_list p

    mkdir -p "$synth/sub"
    echo "hello" > "$synth/sub/file.txt"

    # (a) a new empty subdirectory -> the snapshot must differ.
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

    # (c) an unreadable file (chmod 000) makes the hash attempt fail -> the snapshot function must FATAL/exit nonzero instead of
    # producing a shorter snapshot. It runs in an explicit `( ... )` subshell so the induced `exit 1` ends only that subshell.
    # Only sha mode reads file contents; mtime mode needs only directory lookup and would not reproduce this.
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

    # (d) an unreadable DIRECTORY (permission denied while descending, unlike (c)'s unreadable file) makes `find` itself exit
    # nonzero mid-listing. A bare `find` call checks first that chmod 000 really blocks traversal on this host (it does not for
    # root), so the assertion below is not vacuous.
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

    # (e) a FIFO under the tree must FATAL as an unexpected type. A `-type d -o -type f -o -type l` filter would drop it from
    # find's output before snapshot_one_entry could refuse it.
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

    # (f) a Unix domain socket, best effort via python3's socket module (skipped cleanly if python3 or the bind is
    # unavailable; this sub-case is optional).
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

    # (g) a symlink removed between listing (stage 1) and processing (stage 2). The two stages are called directly rather than
    # through snapshot_tree_root, so the removal lands exactly between them.
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
# Check rc explicitly: snapshot_real_trees/_sha are captured via `$(...)`, a subshell boundary, and this script uses
# `set -uo pipefail`, not `-e`, so a nonzero exit from snapshot_one_entry would not abort the script on its own. Exiting here
# is what lets the fail-closed guarantee reach the top level.
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
    pass "real c_impl/obj+bin and audiolib obj+bin mtimes unchanged across all runs"
else
    fail "real c_impl/obj+bin and audiolib obj+bin mtimes CHANGED across the runs above"
fi

if [ "$SNAP_SHA_BEFORE" = "$SNAP_SHA_AFTER" ]; then
    pass "real c_impl/obj+bin and audiolib obj+bin sha256 unchanged across all runs"
else
    fail "real c_impl/obj+bin and audiolib obj+bin sha256 CHANGED across the runs above"
fi

# --- Final full-sweep leftover-process check ---
# Re-checks every PID recorded via record_pid across all trials, not just the PIDs each trial's own assertion looks at, so a
# leftover descendant that no single trial inspects is caught.
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
