# AEC C Implementation — Makefile

CC      ?= gcc
CXX     ?= c++
AR      ?= ar
RANLIB  ?= ranlib

# CFG_SIG payload-coverage placeholders (round-3 review B01). CXXFLAGS/
# CPPFLAGS/EXTRA_LDFLAGS/DEBUG are still not consumed by this Makefile (this
# repo has no C++ TUs of its own, and no DEBUG-gated code path -- `make
# debug` is a distinct recipe, see below), but the CFG_SIG payload
# unconditionally includes them so the payload string has the same stable
# shape as audio_common's and NR's, and so a future use of any of them here
# keys builds correctly instead of silently aliasing with a build that
# didn't set it. NO_STDIO is no longer a placeholder (round-3 review B03):
# NO_STDIO=1 adds -DAEC_NO_STDIO to CFLAGS below and excludes src/aec_debug.c
# from the library sources, producing a stdio-free libaec.a -- see the
# NO_STDIO block further down and the `audit-no-stdio` target.
#
# CXXFLAGS and CPPFLAGS are different: they are also two of the five names
# the "Command-line override rejection" foreach further below rejects a
# command-line/`-e` override of (a Codex review finding -- the same class
# of bug audio_common's CPPFLAGS had two rounds ago, fixed there the same
# way). `?=` only ever assigns when the variable's origin is still
# "undefined" -- if either already has a value from the environment, `?=`
# is a complete no-op (GNU Make never attempts the assignment at all), so
# the origin never flips to "environment override" under `make -e` no
# matter how early this line sits (confirmed empirically:
# `env FP_POLICY=-ffp-contract=fast make -e -n BACKEND=kiss lib` used to
# sail straight through with no error). `+=` fixes this while keeping the
# externally-visible behaviour identical: appending an empty string is a
# no-op on the VALUE (a plain-environment CXXFLAGS/CPPFLAGS still folds in
# normally -- origin becomes "file" -- and still participates in
# CFG_SIG_PAYLOAD unchanged), but `+=` is an unconditional assignment
# ATTEMPT, so under `-e` GNU Make's origin bookkeeping now correctly flips
# it to "environment override", which the foreach rejects. See
# audio_common's Makefile for the identical CPPFLAGS fix and full writeup.
CXXFLAGS      +=
CPPFLAGS      +=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

# TOOLCHAIN_CHECK=0 skips the CC/CXX -dumpmachine target-coherence guard in
# _cfg_guard below (round-4 review P1-2). Participates in CFG_SIG, so a
# guard-skipped build can never alias a guarded one.
TOOLCHAIN_CHECK ?= 1

# DIST_ROOT / HOSTCC (round-5 review): where `make publish` writes releases
# (test scripts point this at a throwaway mktemp dir so they never touch the
# real dist/), and the HOST compiler for the publish-time atomic
# symlink-swap helper (never $(CC): that may be a cross compiler). Neither
# is in CFG_SIG -- neither affects what is built. See audio_common's
# Makefile for the full writeups.
#
# OBJ_ROOT / BIN_ROOT (round-6 review P1): placement knobs for the keyed
# obj/bin trees -- declared further down next to their only consumers,
# $(OBJ_DIR)/$(BIN_DIR) (see that declaration's comment for the full
# writeup). Listed here only so the "four knobs, none in CFG_SIG" group
# (DIST_ROOT/HOSTCC/OBJ_ROOT/BIN_ROOT) reads as one round-5/round-6 family.
#
# ALLOW_DIRTY_PUBLISH (round-6 review P2, scope narrowed in round-7):
# `make publish` REFUSES by default when EITHER this checkout OR the
# resolved audio_common PRODUCER checkout (AC_DIR below -- the shared FFT
# layer this repo links against, whose commit/dirty state the attestation
# also fingerprints) has uncommitted TRACKED changes (`git status
# --porcelain -uno` non-empty) or is not a git checkout at all (commit
# unknown) -- a release whose ATTEST provenance can't name the exact
# source state of BOTH repos that produced it is not a board deliverable.
# ALLOW_DIRTY_PUBLISH=1 is the explicit escape hatch for dev publishes; it
# is recorded in the attestation together with a sha256 of
# `git diff --binary HEAD` for whichever repo(s) are dirty, so the
# deviation itself is traceable. It covers TRACKED changes only --
# untracked files have their own dimension below (round-7 review: the
# tracked diff hash says nothing about untracked bytes, so lumping the two
# under one flag let two different untracked source states publish with
# identical provenance).
#
# ALLOW_UNTRACKED_PUBLISH (round-7 review; encoding + identity handling
# hardened by the round-7 Stage 2c delta): publish also REFUSES by default
# when EITHER this checkout OR the resolved audio_common PRODUCER checkout
# contains ANY untracked, non-ignored file (`git ls-files --others
# --exclude-standard` non-empty; gitignored build output is excluded by
# design and never part of this provenance) -- an untracked source a
# (possibly dirty) Makefile references can change the artifact without
# leaving a trace in the tracked diff. ALLOW_UNTRACKED_PUBLISH=1 is the
# separate escape hatch; it records untracked_tree_sha256 (this repo) /
# audio_common_untracked_tree_sha256 (the producer) in the attestation:
# sha256 over sorted FIXED-FIELD per-file records -- "L <sha256(path)>
# <sha256(readlink output)>" for symlinks, "F <mode> <sha256(path)>
# <sha256(content)>" for regular files -- every variable-length value is
# itself hashed, so records concatenate unambiguously (raw space-joined
# path/target strings were collision-prone: "a b"->"c" vs "a"->"b c"
# encoded identically) and any stat/readlink/shasum failure downgrades the
# entry to an X record => FATAL, never an empty hash (fail-closed). A path
# that is neither a regular file nor a symlink (an embedded git checkout, a
# fifo) is always refused, with a diagnostic naming it and which repo it
# came from. A checkout with NO git identity at all is refused
# UNCONDITIONALLY -- neither knob admits it (provenance is impossible), for
# EITHER this repo or the audio_common producer. Neither knob is in
# CFG_SIG: they never change what is built, only whether publish proceeds.
#
# ATTEST_STAMP (round-6 review P2, TEST-ONLY): overrides the UTC timestamp
# embedded in the attestation FILENAME so the isolation tests can
# deterministically force same-second publish collisions and prove the
# no-clobber retry path. Never use it on a real publish -- though note a
# forged stamp is within the already-accepted threat model: attestations
# are unsigned and provide traceability, NOT authenticity against an
# attacker with filesystem access. Not in CFG_SIG or MANIFEST.
DIST_ROOT ?= dist
HOSTCC    ?= cc
ALLOW_DIRTY_PUBLISH ?= 0
ALLOW_UNTRACKED_PUBLISH ?= 0
ATTEST_STAMP ?=

CFLAGS  = -Wall -Wextra -O2 -std=gnu99
CFLAGS += -I./include -I./example
CFLAGS += $(EXTRA_CFLAGS)
LDFLAGS = -lm
LDFLAGS += $(EXTRA_LDFLAGS)

SRC_DIR     = src
EXAMPLE_DIR = example
TEST_DIR    = test

# Shared DSP layer (FFT backend + fast_math) lives in audio_common, vendored
# out of this repo (same API this repo used to own: Complex, fft_create,
# fft_get_mem_size/fft_init, fft_forward/inverse/power/apply_gain, ALIGN16).
# AC_DIR resolves whether audio_common sits next to this repo (SE/audio_common)
# or nested deeper (e.g. an Audio_ALG submodule checkout).
AC_DIR ?= $(firstword $(wildcard ../../audio_common ../../../../audio_common))

# FFT backend selection. KISS/NE10 compute in float32 -> ~1e-5 vs numpy's fp64
# np.fft (a documented tolerance; the non-FFT C logic stays bit-exact to Python).
#   BACKEND=kiss (default) -> portable KISS FFT, deterministic (tests use this)
#   BACKEND=ne10           -> ARM NEON NE10 R2C/C2R (the embedded deliverable)
BACKEND ?= kiss
CFLAGS  += -I$(AC_DIR)/include

# Link driver (round-4 review P1-2): the ne10 audio_common archive carries
# one C++ TU (NE10's generic-radix .cpp kernel), so binaries linking it use
# the C++ driver -- which supplies the correct C++ runtime itself (libc++ on
# macOS/clang, libstdc++ on GNU/Linux gcc). The old `LDFLAGS += -lc++` here
# was a macOS-ism (duplicate of what the driver adds there, a link error on
# GNU/Linux) and is gone. Declared before CFG_SIG so it participates in the
# payload (P2-1).
LINK = $(CC)
ifeq ($(BACKEND),ne10)
  LINK = $(CXX)
endif

# -Werror opt-in for this repo's OWN sources (default off; CI can flip it on
# with `make WERROR=1`). AEC compiles no vendored TUs itself (the FFT vendor
# code lives in audio_common), so every object built by this Makefile is
# "our code" -- no separate warning isolation is needed here.
WERROR ?= 0
ifeq ($(WERROR),1)
  CFLAGS += -Werror
endif

# NO_STDIO=1 (round-3 review B03): compile-time gate that produces a
# stdio-free libaec.a — the trace/debug plumbing (src/aec_debug.c, the ONLY
# stdio TU in this library: FILE* g_fp, fprintf/vfprintf/fputc) is compiled
# out entirely and its one call site in src/aec.c (the per-frame trace
# block, runtime-gated only -- a bare NULL-FILE* test that still pulled
# aec_debug.o's fprintf/stderr references into every link) is `#ifndef
# AEC_NO_STDIO`'d out. This block MUST run before CFG_SIG_PAYLOAD is
# computed below (same reasoning as the WERROR/BACKEND flags immediately
# above) so a NO_STDIO=1 build keys to its own obj/bin directory instead of
# aliasing with a NO_STDIO=0 build of otherwise-identical flags. See
# aec_debug.h/aec_debug.c for the header/TU side and the `audit-no-stdio`
# target further down for the verification recipe.
#
# Only the LIBRARY objects ($(OBJS), i.e. `make lib`) are meant to be built
# with NO_STDIO=1 -- the CLI (example/aec_wav.c) calls the FILE*-based debug
# API directly (--debug-log/--debug-trace) and is not expected to compile
# under this macro; `audit-no-stdio` below only ever builds the `lib` goal
# with NO_STDIO=1, never `all`/`bench`.
ifeq ($(NO_STDIO),1)
  CFLAGS += -DAEC_NO_STDIO
endif

# --- FP-contraction policy: unified across all four repos (round-3 review
# B04). This flag USED TO be the third token in the base CFLAGS assignment
# above (`-Wall -Wextra -O2 -ffp-contract=off -std=gnu99`) -- i.e. BEFORE
# EXTRA_CFLAGS/BACKEND/WERROR/NO_STDIO were folded in, so a caller passing
# EXTRA_CFLAGS=-ffp-contract=fast (or -Ofast/-ffast-math, which imply
# contraction) would land LATER on the compile line and silently win,
# re-enabling FMA reassociation despite this repo's documented
# build-determinism policy (see CLAUDE.md / README's build-flags section).
# Moved here -- appended LAST, after every other CFLAGS append in this file
# (EXTRA_CFLAGS, the BACKEND-conditional -I$(AC_DIR)/include, WERROR,
# NO_STDIO), and BEFORE CFG_SIG_PAYLOAD is computed below (so it
# automatically participates in CFG_SIG, no separate payload entry needed)
# -- so nothing can override it any more. See audio_common's Makefile for
# the full cross-repo design writeup and scripts/audit_fp_contract.sh (in
# audio_common) for the disassembly-level verification.
#
# --- Bare-literal policy flag, defined early (Codex review: `make -e`
# environment-override gap) --------------------------------------------------
# FP_POLICY is a plain, dependency-free `:=` literal (no reference to CFLAGS
# or anything else computed above) that used to be defined much further down
# this file, immediately ahead of CFG_SIG_PAYLOAD -- for a reason that still
# applies and is unaffected by this move (see the "FP-contraction policy"
# comment above: FP_POLICY's CFLAGS append still needs to happen AFTER every
# other CFLAGS append in this file, and BEFORE CFG_SIG_PAYLOAD is computed --
# both still true with only the bare literal relocated here). FP_POLICY is
# ALSO listed in the "Command-line override rejection" foreach directly
# below, and that combination used to be a real gap under `make -e`,
# empirically confirmed: `env FP_POLICY=-ffp-contract=fast make -e -n
# BACKEND=kiss lib` used to sail straight through with no error (rc=0),
# silently dropping -ffp-contract=off from the dry-run compile line (see the
# foreach's own comment for the full mechanism -- $(origin FP_POLICY) only
# flips from plain "environment" to "environment override" once a makefile
# assignment to FP_POLICY has actually been PARSED, which never happened
# before the foreach ran while the bare `:=` sat all the way down at its old
# position, well after this foreach). Duplicating only the bare literal up
# here -- ahead of the foreach in file order -- fixes it; the original site
# further below still holds its OTHER job unchanged (`CFLAGS +=
# $(FP_POLICY)`, appended LAST), so there is exactly one `:=` definition of
# FP_POLICY in this file, just moved. Same fix audio_common already applied
# to its own FP_POLICY/FFT_WRAPPER_ALIAS_CFLAGS two rounds ago.
FP_POLICY := -ffp-contract=off

# --- Command-line override rejection (round-4 review P1-1) ------------------
# See audio_common's Makefile for the full writeup: a command-line
# CFLAGS=/CXXFLAGS=/CPPFLAGS=/LDFLAGS=/FP_POLICY= silently disables every one
# of this file's own assignments/appends to that variable (here that would
# strip -ffp-contract=off, -DAEC_NO_STDIO, the -I paths, and -lm) while still
# producing artifacts. EXTRA_CFLAGS / EXTRA_LDFLAGS are the supported hooks.
$(foreach v,CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FP_POLICY,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy, NO_STDIO, include paths); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead)))

# Conflict detection (round-3 review B04; widened to CXXFLAGS/CPPFLAGS by a
# later Codex review, mirroring audio_common's identical fix): reject
# outright rather than let a caller silently re-enable contraction. Checked
# against the fully-assembled $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) (each already
# includes EXTRA_CFLAGS folded in above, AND any CFLAGS=/CXXFLAGS=/CPPFLAGS=
# command-line override, since a command-line-set value is what every prior
# `+=` in this file appended to) -- BEFORE FP_POLICY is appended below and
# BEFORE CFG_SIG is computed, so a rejected build never creates an obj/bin
# directory.
#
# CORRECTED UNDERSTANDING (this used to check $(CFLAGS) alone, reasoning that
# this repo has no C++ TUs of its own so CXXFLAGS/CPPFLAGS "didn't matter" --
# a Codex review found the gap that reasoning missed: CXXFLAGS/CPPFLAGS are
# still two of the five names the "Command-line override rejection" foreach
# above rejects a command-line/`-e` override of, and both already
# participate in CFG_SIG_PAYLOAD below (see this file's own top comment) --
# a PLAIN ENVIRONMENT CXXFLAGS or CPPFLAGS is deliberately allowed to fold in
# normally via that same foreach's origin logic, never going through
# EXTRA_CFLAGS at all, so `env CXXFLAGS=-Ofast make BACKEND=kiss lib` used to
# sail straight through this block entirely: no check here ever inspected
# $(CXXFLAGS), so the day either variable starts feeding a real compile
# recipe (this file already carries the placeholder plumbing for that, see
# the top-of-file comment), a live -Ofast/-ffast-math/-ffp-contract=<x>
# could ride in completely unchecked. The check below inspects all three
# variables together so no combination of CFLAGS/CXXFLAGS/CPPFLAGS --
# command-line-folded, EXTRA_CFLAGS-folded, or a plain environment value --
# can carry any of these three patterns past this gate, today or once either
# variable is actually consumed. This position (after every CFLAGS/CXXFLAGS/
# CPPFLAGS append earlier in this file and before FP_POLICY's own append
# just below) is the same position already relied on to see CFLAGS's
# fully-composed value, and is equally late enough for CXXFLAGS and
# CPPFLAGS: neither is ever appended to anywhere else in this file (each is
# a single bare `+=` at the top of this file -- see that line's own
# comment), both fully resolved well before this point.
#
# (round-9-class fix, applied here from the start rather than retrofitted)
# $(findstring ...) does a plain SUBSTRING search over the whole concatenated
# text, so it false-positives on any token that merely CONTAINS one of these
# patterns as part of a larger identifier -- e.g. a harmless
# `-DROUND9_NOTE=-Ofastness` macro definition is not the compiler flag
# -Ofast at all, yet a naive $(findstring -Ofast,...) check (as this block
# used to be) would reject it anyway. $(filter PATTERN...,TEXT) instead
# splits TEXT on whitespace into WORDS and matches each whole word against
# PATTERN (a trailing %% is a single wildcard, used here for
# -ffp-contract=<anything>), so it only ever matches a complete flag token,
# never a substring buried inside an unrelated token.
FP_INPUT_FLAGS := $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS)

# (Codex review) $(filter) directly above does whole-WORD matching against
# GNU Make's OWN text representation of CFLAGS/CXXFLAGS/CPPFLAGS -- but Make
# has zero concept of shell quoting. A value like EXTRA_CFLAGS="'-Ofast'" is
# stored by Make as the literal 9-character text '-Ofast', quote characters
# included, which is not equal to the bare word -Ofast, so $(filter) can
# never match it -- yet when this same text later appears on an actual
# compile recipe line and that line is handed to $(SHELL), the shell strips
# the quote characters and the compiler genuinely receives -Ofast as a real
# argv token. Three confirmed-live bypass shapes: EXTRA_CFLAGS="'-Ofast'"
# (single-quoted), EXTRA_CFLAGS='"-ffast-math"' (double-quoted), and
# EXTRA_CFLAGS="-O'f'ast" (quote-split mid-token -- the shell concatenates
# the pieces back into -Ofast once it strips the quotes). A fourth family,
# @response-file syntax (EXTRA_CFLAGS=@flags.rsp), is unrelated to quoting
# but equally invisible to a plain whole-word check: many compilers treat a
# bare @file argv token as "read more arguments from this file", which could
# inject anything at all with zero text ever appearing in CFLAGS/
# EXTRA_CFLAGS itself.
#
# Since Make cannot parse shell quoting, this block does not try to detect
# "a quoted variant of -Ofast" specifically. Instead, the ONLY supported
# external-input format for CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS is plain,
# unquoted, whitespace-separated compiler argv tokens -- and ANY appearance
# of shell quoting/escaping/expansion/response-file syntax in the combined
# text is rejected outright, regardless of what follows it, BEFORE the
# $(filter ...) exact-token conflict check below even runs, so that check
# only ever sees guaranteed-unquoted input. (This paragraph intentionally
# says nothing about -Ofast/-ffast-math/-ffp-contract itself: whether a
# token IS one of those three is decided solely by the unchanged $(filter)
# check; a harmless token that merely contains innocuous characters, e.g.
# -DROUND9_NOTE=-Ofastness or -DTEXT=ffast-math, contains none of the
# characters checked for below and so is untouched by this block.)
#
# A literal "$(" -- Make's own function/variable-reference opener, and also
# the two-character shell command/variable-substitution opener -- cannot
# safely be written as raw punctuation inline in a $(findstring ...) call:
# GNU Make's parser matches parentheses while scanning a function call's
# argument text, so an unbalanced literal "(" typed inline would confuse
# that scan. Built instead via small helper variables (one holding a literal
# "$" via the doubled-dollar escape, one holding a literal "(", concatenated
# into one holding the two-char sequence) and that variable is referenced as
# the findstring pattern -- confirmed by direct testing to parse cleanly
# and to actually match (a raw unbalanced "(" typed inline here does NOT
# parse cleanly; the helper-variable indirection does). Same mechanism
# audio_common's Makefile uses for its own identical check.
FP_DOLLAR_CHAR      := $$
FP_OPEN_PAREN_CHAR   := (
FP_DOLLAR_PAREN_SEQ  := $(FP_DOLLAR_CHAR)$(FP_OPEN_PAREN_CHAR)

ifneq ($(findstring ',$(FP_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains a single-quote character; this repo requires plain, unquoted, whitespace-separated compiler argv tokens -- shell quoting cannot be safely detected here by GNU Make and is rejected outright)
endif

ifneq ($(findstring ",$(FP_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains a double-quote character; this repo requires plain, unquoted, whitespace-separated compiler argv tokens -- shell quoting cannot be safely detected here by GNU Make and is rejected outright)
endif

ifneq ($(findstring \,$(FP_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains a backslash character; this repo requires plain, unquoted, whitespace-separated compiler argv tokens -- shell escaping cannot be safely detected here by GNU Make and is rejected outright)
endif

ifneq ($(findstring `,$(FP_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains a backtick character; this repo requires plain, unquoted, whitespace-separated compiler argv tokens -- shell command substitution cannot be safely detected here by GNU Make and is rejected outright)
endif

ifneq ($(findstring $(FP_DOLLAR_PAREN_SEQ),$(FP_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains a '$$(' sequence (command/variable substitution); this repo requires plain, unquoted, whitespace-separated compiler argv tokens -- shell/make expansion cannot be safely detected here by GNU Make and is rejected outright)
endif

ifneq ($(findstring ;,$(FP_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains a semicolon character; this repo requires plain, unquoted, whitespace-separated compiler argv tokens -- shell command separators cannot be safely detected here by GNU Make and are rejected outright)
endif

ifneq ($(findstring |,$(FP_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains a pipe character; this repo requires plain, unquoted, whitespace-separated compiler argv tokens -- shell pipelines cannot be safely detected here by GNU Make and are rejected outright)
endif

ifneq ($(findstring &,$(FP_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains an ampersand character; this repo requires plain, unquoted, whitespace-separated compiler argv tokens -- shell backgrounding/AND-lists cannot be safely detected here by GNU Make and are rejected outright)
endif

FP_RESPONSE_FILE_TOKENS := $(filter @%,$(FP_INPUT_FLAGS))
ifneq ($(FP_RESPONSE_FILE_TOKENS),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains an @-prefixed response-file token ($(FP_RESPONSE_FILE_TOKENS)); this repo requires plain, unquoted, whitespace-separated compiler argv tokens -- @file compiler response-file syntax cannot be safely expanded/inspected here and is rejected outright)
endif

FP_CONFLICT_FLAGS := $(filter -Ofast -ffast-math -ffp-contract=%,$(FP_INPUT_FLAGS))
ifneq ($(FP_CONFLICT_FLAGS),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains $(FP_CONFLICT_FLAGS); this repo pins -ffp-contract=off; remove $(FP_CONFLICT_FLAGS) from CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS)
endif

# (Codex review / `make -e` gap fix: FP_POLICY's own bare `:=` literal now
# lives earlier in this file, directly above the "Command-line override
# rejection" foreach -- see that foreach's "Bare-literal policy flag"
# comment for why. Only FP_POLICY's OTHER job -- appending it to CFLAGS,
# LAST, exactly as the "FP-contraction policy" comment above describes --
# still lives at this position; there is exactly one `:=` definition of
# FP_POLICY in this file, not two.)
CFLAGS    += $(FP_POLICY)

.DEFAULT_GOAL := all

# =============================================================================
# Consumer two-phase audio_common resolution (round-3 review B01)
# =============================================================================
# Problem this replaces: the old Makefile hardcoded
# `$(AC_DIR)/bin/$(BACKEND)/libaudio_common.a` directly into LDFLAGS and used
# a plain order-only `audio_common_lib: $(MAKE) -C $(AC_DIR) BACKEND=$(BACKEND)
# lib` prerequisite. Two problems: (1) audio_common's own archive is now
# CFG_SIG-keyed (bin/<backend>-<sig>/), so there is no fixed path to hardcode
# any more; (2) only BACKEND was forwarded to that sub-make, so an
# audio_common built with a different WERROR/EXTRA_CFLAGS/compiler than this
# invocation's own (a real "config skew" risk under concurrent/differently-
# flagged builds) was silently linked in without any of that showing up in
# THIS repo's own build identity.
#
# The fix is a two-phase recursive-make dispatch, driven by whether AC_LIB
# (the resolved absolute path to audio_common's archive) is already known:
#
#   Phase 1 (this file parsed with AC_LIB unset -- the normal case for any
#   top-level `make ...` invocation): goals that need the archive (`all`,
#   `bench`) get a recipe that queries audio_common's own `print-lib-path`
#   query target (forwarding the exact BACKEND/WERROR/CC/CXX/EXTRA_CFLAGS/
#   NO_STDIO this invocation is using) and then re-invokes itself with
#   AC_LIB set to the answer.
#
#   Phase 2 (AC_LIB known, either because phase 1 supplied it or because a
#   caller who already knows it passed it directly): the REAL rules for
#   $(TARGET)/$(BENCH_TARGET) are defined, with $(AC_LIB) as a NORMAL (not
#   order-only) prerequisite -- so relinking happens iff audio_common's
#   archive mtime actually advanced, exactly like any other prerequisite.
#
# Goals that never need audio_common at all (`lib` -- this repo's OWN
# archive has no FFT/audio_common code in it; `selftest` -- header-only,
# links with a plain -lm; `clean`; the print-*/help query targets) get their
# real rules OUTSIDE this dispatch entirely, further down this file --
# invoking any of them never shells out to audio_common, never needs
# AC_DIR to even exist, and (see `make -n` note below) never touches disk
# under a dry run.
#
# `make -n`: GNU Make always executes (never merely prints) a recipe LINE
# that textually mentions $(MAKE), even under -n/-q/-t (documented
# behaviour, "Recursive make" section of the manual) -- so the dispatch
# recipe below genuinely runs its sub-`make` calls under `make -n all` too.
# But those sub-`make` invocations inherit -n right back (MAKEFLAGS is
# environment-propagated to every child `make`, regardless of how it was
# invoked), so the query sub-make's own `@echo`-only recipe is itself just
# printed, and the final phase-2 sub-make's compile/link recipes are also
# just printed -- verified empirically: `make -n all` spawns real
# `make` processes but creates zero files, same as a non-recursive recipe.
AC_FWD = BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK)

ifeq ($(AC_LIB),)
.PHONY: all bench test-counter-saturation
all bench test-counter-saturation:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) $(AC_FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audio_common archive path (AC_DIR=$(AC_DIR) -- is audio_common present as a sibling of this repo?)" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory $@ AC_LIB="$$ac"
else
# Producer-identity component of this consumer's own CFG_SIG (see the
# CFG_SIG section below): the LAST path component of $(AC_LIB)'s directory
# (e.g. "kiss-2139547423"), NOT the full (absolute) path -- so a consumer
# built against a DIFFERENT audio_common config (different flags, or the
# same flags after an audio_common source/header edit bumped its own
# CFG_SIG) keys to its own separate obj/bin directory here too, instead of
# forwarding only BACKEND and risking silent config skew (the actual bug
# class this whole redesign targets). $(AC_DIR) itself (an abspath, used
# only as the -C argument to the sub-make above) never appears in any
# payload or manifest -- only this short backend-sig component does.
AC_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(AC_LIB))))
endif

# Algorithm sources (auto-detected — drop a new file in src/ to add it).
# fft_wrapper*.c no longer live here; they're part of the audio_common archive
# linked above. Declared HERE, before CFG_SIG_PAYLOAD (round-4 review P1-4),
# because the payload folds in the sorted source list -- adding or removing a
# source file must key to a fresh obj/bin directory (see the fresh-archive
# recipe below for the other half of the stale-member fix).
SRCS = $(wildcard $(SRC_DIR)/*.c)

# NO_STDIO=1 (round-3 review B03): belt-and-braces exclusion of the one
# stdio TU from the library sources -- aec_debug.c already compiles to an
# empty file under -DAEC_NO_STDIO (see that file), so this filter-out is
# redundant with the source-level gate by design (either alone keeps
# aec_debug.o's fprintf/vfprintf/stderr out of $(OBJS)/the archive; both
# together means neither can regress silently). `ar -t` on the resulting
# libaec.a must not list aec_debug.o -- verified by `audit-no-stdio` below.
ifeq ($(NO_STDIO),1)
  SRCS := $(filter-out $(SRC_DIR)/aec_debug.c,$(SRCS))
endif

# --- Hash-keyed object/binary directory + full-coverage CFG_SIG ------------
# obj/<backend>-<sig>/ AND (round-3 review B01) bin/<backend>-<sig>/ -- see
# audio_common's Makefile for the full rationale (identical design, applied
# here). BIN_DIR used to be flat (`bin`), so an ne10 build after a kiss
# build -- or two same-backend builds differing only in EXTRA_CFLAGS/WERROR,
# or a stale mtime from a differently-flagged prior build -- could silently
# deliver the wrong `aec_wav`/`libaec.a` to whatever last happened to write
# that flat path. Payload coverage (USER-MANDATED): CC CXX AR RANLIB CFLAGS
# CXXFLAGS CPPFLAGS LDFLAGS EXTRA_CFLAGS EXTRA_LDFLAGS BACKEND WERROR
# NO_STDIO DEBUG, PLUS (consumer-specific) AC_PRODUCER_SIG -- the resolved
# producer's own config identity (empty when AC_LIB isn't resolved yet, e.g.
# for goals like `lib`/`selftest` that never touch audio_common at all).
#
# This MUST be defined after CFLAGS/LDFLAGS are fully assembled (including
# the BACKEND-conditional and WERROR appends above) and after the
# AC_LIB ifeq/else above resolves AC_PRODUCER_SIG, but before the first rule
# below that mentions $(OBJ_DIR)/$(BIN_DIR)/anything derived from them --
# GNU Make expands a rule's target/prerequisite list immediately when
# parsed, not deferred like a recipe body, so a forward reference here would
# silently resolve using CFG_SIG's value at parse time (empty), landing
# objects in a hash-less directory instead of the real one.
#
# Deliberately NOT $(strip)'d (see audio_common's Makefile for the false-miss
# vs false-hit rationale). cksum (POSIX) computes the hash; `cut -d' ' -f1`
# isolates the checksum from cksum's "<checksum> <byte-count>" stdout.
# Round-4 review payload additions: LINK (P2-1), TOOLCHAIN_CHECK (the
# guard-skip knob must not alias a guarded build), SRCS (P1-4: sorted source
# list, post-NO_STDIO-filter, so source add/remove keys to a fresh directory).
CFG_SIG_PAYLOAD := CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) LDFLAGS=$(LDFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) AC_PRODUCER_SIG=$(AC_PRODUCER_SIG) SRCS=$(sort $(SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

# OBJ_ROOT / BIN_ROOT (round-6 review P1): where the keyed obj/bin trees
# live (defaults: obj/ and bin/ in this directory -- the default expansion
# is byte-identical to the previous hardcoded paths). Same rationale as
# DIST_ROOT above: a placement knob, never a build-content knob, so
# deliberately NOT in CFG_SIG. The isolation tests use these to run
# tamper/injection scenarios (corrupt a config.manifest, inject a foreign
# archive member) against a scratch-directory build of the real worktree,
# so the REAL obj/ and bin/ artifacts are never modified -- not even
# transiently.
OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR  = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR  = $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

OBJS  = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

EXAMPLE_SRCS = $(EXAMPLE_DIR)/aec_wav.c
EXAMPLE_OBJS = $(patsubst $(EXAMPLE_DIR)/%.c,$(OBJ_DIR)/%.o,$(EXAMPLE_SRCS))

TARGET     = $(BIN_DIR)/aec_wav
LIB_TARGET = $(BIN_DIR)/libaec.a
BENCH_SRCS = $(TEST_DIR)/bench_rtf.c
BENCH_OBJS = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(BENCH_SRCS))
BENCH_TARGET = $(BIN_DIR)/bench_rtf

# Permanent counter-saturation regression test (round-6 review: replaces the
# ad hoc/throwaway UBSan probes rounds 4-6 each wrote to verify their counter
# overflow fixes with a checked-in test that runs going forward). Needs the
# full Aec type (aec.c/pbfdkf.c's FFT-backed sections), so -- unlike
# SELFTEST_TARGET below -- this lives in the two-phase/AC_LIB-dependent part
# of this Makefile, same as $(TARGET)/$(BENCH_TARGET).
TEST_CS_SRCS   = $(TEST_DIR)/test_counter_saturation.c
TEST_CS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_CS_SRCS))
TEST_CS_TARGET = $(BIN_DIR)/test_counter_saturation

# Standalone bit-exactness + microbench harness for the AEC-only kernel split
# (aec_simd_kernels.h). Header-only + <math.h> -- no audio_common archive
# link needed (unlike bench_rtf/aec_wav), so this target does not depend on
# audio_common's archive and lives outside the two-phase dispatch above.
SELFTEST_SRC    = $(TEST_DIR)/simd_selftest_aec.c
SELFTEST_TARGET = $(BIN_DIR)/simd_selftest_aec

# --- CFG_SIG collision guard + directory creation (round-3 review B01) -----
# Single order-only prerequisite for every compile/link/archive rule below
# (replacing the old bare "| $(OBJ_DIR)"/"| $(BIN_DIR)" prerequisites and
# their own `@mkdir -p $@` rule). .PHONY, so its recipe runs on every `make`
# invocation that needs it, but GNU Make builds any one target at most once
# per invocation even when several rules depend on it, so under `make -j`
# it still runs exactly once and (being an ordinary, if order-only,
# dependency-graph edge) completes before any dependent rule starts:
# race-free. See audio_common's Makefile for the full design writeup (same
# mechanism, applied here) -- write config.manifest first time, verify
# (hard error on mismatch = CFG_SIG collision or payload coverage gap) every
# time after, atomically (temp file + mv).
.PHONY: _cfg_guard
_cfg_guard:
	@if [ "$(BACKEND)" = "ne10" ] && [ "$(TOOLCHAIN_CHECK)" = "1" ]; then \
	  cc_m="$$($(CC) -dumpmachine 2>/dev/null)"; cxx_m="$$($(CXX) -dumpmachine 2>/dev/null)"; \
	  if [ -z "$$cc_m" ] || [ "$$cc_m" != "$$cxx_m" ]; then \
	    echo "FATAL: CC/CXX compile for different targets (CC='$(CC)' -> '$$cc_m', CXX='$(CXX)' -> '$$cxx_m')." >&2; \
	    echo "  BACKEND=ne10 links a C++ TU (audio_common's NE10 archive); pass the FULL toolchain:" >&2; \
	    echo "    make BACKEND=ne10 CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ AR=aarch64-linux-gnu-ar RANLIB=aarch64-linux-gnu-ranlib" >&2; \
	    echo "  (TOOLCHAIN_CHECK=0 skips this check; it participates in CFG_SIG.)" >&2; \
	    exit 1; \
	  fi; \
	fi
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)
	@for d in $(OBJ_DIR) $(BIN_DIR); do \
	  m="$$d/config.manifest"; \
	  if [ -f "$$m" ]; then \
	    old="$$(cat "$$m")"; \
	    if [ "$$old" != "$(CFG_SIG_PAYLOAD)" ]; then \
	      echo "FATAL: CFG_SIG collision or coverage gap in $$m" >&2; \
	      echo "  CFG_SIG=$(CFG_SIG)" >&2; \
	      echo "  stored : $$old" >&2; \
	      echo "  current: $(CFG_SIG_PAYLOAD)" >&2; \
	      exit 1; \
	    fi; \
	  else \
	    tmp="$$d/.config.manifest.tmp.$$$$"; \
	    printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$tmp"; \
	    mv -f "$$tmp" "$$m"; \
	  fi; \
	done

# -MD -MP: emits $(OBJ_DIR)/<name>.d per object (round-3 review B01, header
# dependency tracking) -- included near the bottom of this file via
# `-include $(wildcard $(OBJ_DIR)/*.d)`.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/%.o: $(EXAMPLE_DIR)/%.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

.PHONY: lib
lib: $(LIB_TARGET)

# `lib` needs only THIS repo's own objects -- no audio_common archive is
# linked into libaec.a -- so, per the two-phase design above, it's a real
# rule living entirely outside the AC_LIB dispatch (never shells out to
# audio_common, works even if audio_common isn't present at all).
# Fresh-archive discipline (round-4 review P1-4): always built from scratch
# into a temp file and renamed into place -- never `ar rc` onto an existing
# archive (which only adds/replaces members and would keep a removed source's
# stale member forever). One chained shell line with a PID-suffixed temp
# (round-5 review P2): two same-config processes racing this rule each write
# their OWN temp, final mv is atomic last-writer-wins. See audio_common's
# Makefile for the full writeup.
$(LIB_TARGET): $(OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rc "$$t" $(OBJS) && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "aec [$(BACKEND)] -> $@"

# Bit-exactness + microbench self-test for the AEC-only kernel split
# (include/aec_simd_kernels.h). Links with -lm only (no audio_common
# archive, no AC_LIB dependency) -- the header is self-contained aside from
# transitively pulling in simd_kernels.h via CFLAGS' -I$(AC_DIR)/include.
.PHONY: selftest
selftest: $(SELFTEST_TARGET)
	"$(abspath $(SELFTEST_TARGET))"

$(SELFTEST_TARGET): $(SELFTEST_SRC) | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $< -lm

# =============================================================================
# Phase 2 real rules: only defined once AC_LIB is known (see the dispatch
# block above). $(AC_LIB): FORCE always re-invokes audio_common's own build
# (a no-op if it's already up to date -- audio_common's own Makefile decides
# that), but only actually touches $(AC_LIB)'s mtime if audio_common really
# rebuilt something, so $(TARGET)/$(BENCH_TARGET) only relink when it did.
# =============================================================================
ifneq ($(AC_LIB),)
.PHONY: all bench FORCE test-counter-saturation
all: $(TARGET)
bench: $(BENCH_TARGET)

# Builds AND runs (like `selftest` below, not like `bench` which only
# builds) -- this is the permanent regression test, meant to be run every
# time, not just built. Must be invoked from c_impl/ (the delay_aec3
# section reads a checked-in fixture at a path relative to that cwd; see
# test/test_counter_saturation.c's own top comment) -- same cwd convention
# `make` recipes already run under.
test-counter-saturation: $(TEST_CS_TARGET)
	"$(abspath $(TEST_CS_TARGET))"

FORCE:

$(AC_LIB): FORCE
	+$(MAKE) -C $(AC_DIR) $(AC_FWD) lib
	@test -f '$@' || { echo "ERROR: audio_common did not produce $@ (query/build divergence -- only BACKEND/WERROR/CC/CXX/EXTRA_CFLAGS/NO_STDIO overrides are supported)"; exit 1; }

# $(LINK), not $(CC) (round-4 review P1-2): the C++ driver for BACKEND=ne10
# supplies the C++ runtime the NE10 archive's one C++ TU needs (the old
# hardcoded -lc++ is gone). Same pattern as NR's/pipelines' link recipes.
$(TARGET): $(OBJS) $(EXAMPLE_OBJS) $(AC_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "aec_wav [$(BACKEND)] -> $@"

$(BENCH_TARGET): $(OBJS) $(BENCH_OBJS) $(AC_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "bench_rtf [$(BACKEND)] -> $@"

$(TEST_CS_TARGET): $(OBJS) $(TEST_CS_OBJS) $(AC_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_counter_saturation [$(BACKEND)] -> $@"
endif

# --- Query targets (round-3 review B01) -------------------------------------
# print-obj-dir / print-lib-path report THIS invocation's own build identity
# (AEC's own flags; empty AC_PRODUCER_SIG unless AC_LIB was passed in) -- they
# never shell out to audio_common, matching where a bare `make lib` /
# `make selftest` actually places its artifacts.
# print-bin-dir is different: the binaries (`aec_wav`, `bench_rtf`) only ever
# exist in the RESOLVED-producer directory `make all`/`make bench` deliver
# into, so a bare `make -s print-bin-dir` dispatches through the same phase-1
# resolution as `all`/`bench` and prints that exact directory. (Passing
# AC_LIB= explicitly skips the extra query sub-make, same protocol as
# phase 1.)
.PHONY: print-bin-dir print-obj-dir print-lib-path
ifeq ($(AC_LIB),)
print-bin-dir:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) $(AC_FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audio_common archive path (AC_DIR=$(AC_DIR))" >&2; exit 1; }; \
	 $(MAKE) -s --no-print-directory print-bin-dir AC_LIB="$$ac"
else
print-bin-dir:
	@echo $(abspath $(BIN_DIR))
endif
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))
print-lib-path:
	@echo $(abspath $(LIB_TARGET))

# =============================================================================
# publish v4: immutable, content-addressed release under $(DIST_ROOT)
# (round-3 B01 -> round-4 P2-2 -> round-5 P2 -> round-6 P2 -> round-7) --
# same design as audio_common's publish; see that Makefile for the full
# layout/semantics writeup (deterministic MANIFEST + byte-verified
# idempotent republish incl. MANIFEST tamper detection, rename(2)-atomic
# `current` swap via the swapln helper built with $(HOSTCC)). Round-6
# additions (mirrored from audio_common verbatim, with one AEC-specific
# extension noted below):
#   - Dry-run guard: -n/-q/-t never touch $(DIST_ROOT), the publish lock, or
#     any stage/helper/MANIFEST/ATTEST write (see the MAKEFLAGS word-scan in
#     the `publish` recipe below; round-7 tightened -t further -- see the
#     round-7 bullet below).
#   - ATTEST is one-event-one-file: every publish event (including an
#     idempotent republish) writes exactly one NEW
#     attest-<utc>-<commit>[-dirty]-<seq>.txt, installed via the swapln
#     helper's `--excl-install` mode (write-temp + link(2), the atomic
#     no-clobber equivalent of O_CREAT|O_EXCL -- link(2) fails EEXIST if the
#     name is already taken). A same-second name collision regenerates the
#     event id with the next <seq> and retries; an existing attestation is
#     NEVER overwritten (round-5 used `mv -f`, so same-second republishes
#     could silently clobber the prior record). `git_commit` is always the
#     FULL 40-hex object id (round-5 used `git rev-parse --short`).
#   - AEC-specific: the attestation ALSO records the audio_common PRODUCER's
#     own commit/dirty/untracked state (`audio_common_git_commit`/
#     `audio_common_git_dirty`[`/audio_common_dirty_diff_sha256`]/
#     `audio_common_git_untracked`[`/audio_common_untracked_tree_sha256`]) --
#     this repo's release links against audio_common's archive, so "which
#     source state produced this release" spans both repos, not just this
#     one. $(AC_DIR) is guarded (`[ -n ... ] && [ -d ... ]`) before use: its
#     wildcard default (see AC_DIR above) can resolve to the EMPTY string,
#     and `git -C '' ...` silently operates on the CURRENT (i.e. THIS) repo
#     instead of failing -- which would misreport audio_common's identity
#     as this repo's own. The guard makes an unresolved AC_DIR report
#     audio_common_git_commit=unknown (and audio_common_git_untracked=unknown)
#     instead.
#   - Dirty/untracked policy (round-6 P2, split into two dimensions by
#     round-7; identity refusal hardened by the round-7 Stage 2c delta):
#     publish FATALs by default when EITHER this checkout OR the resolved
#     audio_common PRODUCER checkout (a) has uncommitted TRACKED changes
#     (`git status --porcelain -uno` non-empty), (b) contains ANY
#     untracked, non-ignored file (`git ls-files --others --exclude-standard`
#     non-empty; gitignored build output is excluded by design and never
#     part of this provenance), or (c) has no git identity at all -- the
#     FATAL names exactly which repo(s) and which dimension triggered the
#     refusal. (c) is checked, and FATALs, BEFORE (a)/(b) are even computed,
#     and is UNCONDITIONAL for BOTH repos -- no escape hatch admits an
#     identity-less checkout on either side, because neither the commit nor
#     the untracked enumeration can be named for it (Stage 2c fix: the prior
#     shape folded "unknown" into the same dirty-violation flag that
#     ALLOW_DIRTY_PUBLISH=1 short-circuits, so a dirty-publish escape hatch
#     could previously admit an identity-less checkout -- closed by making
#     the identity check its own unconditional gate with no knob reaching
#     it). The two escape hatches for (a)/(b) remain separate and
#     orthogonal: ALLOW_DIRTY_PUBLISH=1 admits tracked changes and records
#     dirty_diff_sha256 (sha256 of `git diff --binary HEAD`) for whichever
#     repo(s) are dirty; ALLOW_UNTRACKED_PUBLISH=1 admits untracked files
#     and records untracked_tree_sha256 (this repo) /
#     audio_common_untracked_tree_sha256 (the producer) -- sha256 over
#     sorted FIXED-FIELD records (see the knob comment up top for the exact
#     encoding and its fail-closed I/O semantics) -- round-7: the tracked
#     diff hash says nothing about untracked bytes, so two different
#     untracked source states must not share a provenance record. An
#     untracked path that is neither a regular file nor a symlink (an
#     embedded git checkout, a fifo, ...) cannot be hashed deterministically
#     and is always refused, by name, naming which repo it came from.
#   - Round-7 -t tightening: `make -t publish` now prints a one-line no-op
#     note and exits 0 WITHOUT recursing into `_publish_impl` (round-6 had
#     -n and -t share one recursion-only branch; recursing under -t let GNU
#     make's standard touch semantics bump the mtimes of libaec.a/aec_wav/
#     bench_rtf -- and, via the AC_LIB dispatch, audio_common's own archive
#     -- real writes for a phony ACTION target that produces no build
#     product of its own). -n and -q are unchanged.
#   - Attestations remain UNSIGNED: they provide traceability, not
#     authenticity -- anyone with filesystem access could forge one; do not
#     treat ATTEST/ as tamper-proof under an attacker model (MANIFEST
#     byte-verification is the integrity check; ATTEST is the provenance
#     log).
# `publish` is the locked driver; _publish_impl is INTERNAL (direct
# invocation skips the lock) and goes through the same two-phase AC_LIB
# dispatch as `all`/`bench`.
# =============================================================================
PUBLISH_ARTIFACTS = $(LIB_TARGET) $(TARGET) $(BENCH_TARGET)
DIST_BACKEND_DIR  = $(DIST_ROOT)/$(BACKEND)
DIST_LOCK         = $(DIST_ROOT)/.lock.$(BACKEND)

.PHONY: publish
# Dry-run guard (round-6 review P2-3, -t tightened by round-7): this
# recipe mentions $(MAKE), so GNU make executes it for real under -n/-q/-t
# (all three share that rule) -- which used to mkdir $(DIST_ROOT)
# persistently and take/release the real publish lock on every dry run.
# The word-scan below detects those flags in MAKEFLAGS and branches:
#   -t  prints a one-line note and exits 0 WITHOUT recursing (round-7:
#       recursing let standard touch semantics bump the mtimes of
#       libaec.a/aec_wav/bench_rtf (and, via the AC_LIB dispatch,
#       audio_common's own archive) -- real writes. publish is a phony
#       ACTION, not a build product; touching its artifact prerequisites
#       without attesting would fabricate state, so touch mode is an
#       explicit no-op here). Checked FIRST: in a COMBINED invocation like
#       `make -nt publish`, recursing for -n would hand the child make
#       both flags and GNU make then really applies touch semantics
#       (reproduced) -- so any t wins over n, conservatively;
#   -q  exits 1 directly (question mode's documented "needs updating"
#       answer -- publish is phony, so it always would run -- no output);
#   -n  exec a recursion-only path (the child make inherits -n and merely
#       PRINTS _publish_impl's commands).
# All three are now ZERO-WRITE: no DIST_ROOT, no lock, no stage/helper/
# MANIFEST/ATTEST writes, and no artifact mtime changes. MAKEFLAGS parsing
# (verified against GNU make 3.81: 17 flag combinations incl. long forms):
# bundled no-dash short flags may appear as the first word ("n", "sn") OR
# as separate dashed words ("-n" after a long option); long options start
# with "--"; a literal "--" separates flags from command-line variable
# overrides (which can themselves contain n/q/t -- BACKEND=ne10 -- hence
# the explicit skips). Normal invocations keep the round-5 lock-FIRST
# discipline unchanged. See audio_common's Makefile for the full writeup
# (identical mechanism).
publish:
	+@has_n=0; has_q=0; has_t=0; \
	for w in $(MAKEFLAGS); do \
	  case "$$w" in --) break ;; --*|*=*) continue ;; -n|-q|-t) ;; -*) continue ;; esac; \
	  case "$$w" in *n*) has_n=1 ;; esac; \
	  case "$$w" in *q*) has_q=1 ;; esac; \
	  case "$$w" in *t*) has_t=1 ;; esac; \
	done; \
	if [ "$$has_t" = 1 ]; then echo "publish: -t (touch) is a no-op here -- publish is an action, not a build product; nothing was touched"; exit 0; fi; \
	if [ "$$has_q" = 1 ]; then exit 1; fi; \
	if [ "$$has_n" = 1 ]; then exec $(MAKE) --no-print-directory _publish_impl; fi; \
	mkdir -p $(DIST_ROOT); \
	if ! mkdir "$(DIST_LOCK)" 2>/dev/null; then \
	  echo "FATAL: publish lock $(DIST_LOCK) already held (concurrent 'make publish BACKEND=$(BACKEND)'?)" >&2; \
	  exit 1; \
	fi; \
	trap 'rmdir "$(DIST_LOCK)" 2>/dev/null' EXIT INT TERM; \
	$(MAKE) --no-print-directory _publish_impl

ifeq ($(AC_LIB),)
.PHONY: _publish_impl
_publish_impl:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) $(AC_FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audio_common archive path" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory _publish_impl AC_LIB="$$ac"
else
.PHONY: _publish_impl
_publish_impl: $(PUBLISH_ARTIFACTS)
	@set -e; \
	work="$$(mktemp -d)"; \
	tmp="$(DIST_BACKEND_DIR)/.stage.$$$$"; \
	trap 'rm -rf "$$tmp" "$$work"' EXIT INT TERM; \
	hash_untracked() { \
	  LC_ALL=C sort | while IFS= read -r p; do \
	    fp="$$1$$p"; \
	    ph="$$(printf '%s' "$$p" | shasum -a 256 | cut -d' ' -f1)"; \
	    [ "$${#ph}" -eq 64 ] || { printf 'X %s\n' "$$p"; continue; }; \
	    if [ -L "$$fp" ]; then \
	      tgt="$$(readlink "$$fp")" || { printf 'X %s\n' "$$p"; continue; }; \
	      th="$$(printf '%s' "$$tgt" | shasum -a 256 | cut -d' ' -f1)"; \
	      [ "$${#th}" -eq 64 ] || { printf 'X %s\n' "$$p"; continue; }; \
	      printf 'L %s %s\n' "$$ph" "$$th"; \
	    elif [ -f "$$fp" ]; then \
	      m="$$(stat -f '%Lp' "$$fp" 2>/dev/null || stat -c '%a' "$$fp" 2>/dev/null)"; \
	      ch="$$(shasum -a 256 "$$fp" 2>/dev/null | cut -d' ' -f1)"; \
	      { [ -n "$$m" ] && [ "$${#ch}" -eq 64 ]; } || { printf 'X %s\n' "$$p"; continue; }; \
	      printf 'F %s %s %s\n' "$$m" "$$ph" "$$ch"; \
	    else printf 'X %s\n' "$$p"; fi; \
	  done; \
	}; \
	commit="$$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- aec has no git identity here (not a git checkout)." >&2; \
	  echo "  Without an identity the attestation cannot name the source state OR enumerate untracked" >&2; \
	  echo "  content, so NO escape hatch admits this (round-7: ALLOW_DIRTY_PUBLISH used to). Publish" >&2; \
	  echo "  from a real checkout or a scratch clone with a disposable commit instead." >&2; \
	  exit 1; \
	fi; \
	dirty=0; [ -n "$$(git status --porcelain -uno 2>/dev/null)" ] && dirty=1; \
	unt_list="$$(git ls-files --others --exclude-standard 2>/dev/null)"; \
	untracked=0; [ -n "$$unt_list" ] && untracked=1; \
	if [ -n "$(AC_DIR)" ] && [ -d "$(AC_DIR)" ]; then \
	  ac_commit="$$(git -C $(AC_DIR) rev-parse HEAD 2>/dev/null || echo unknown)"; \
	else \
	  ac_commit=unknown; \
	fi; \
	if [ "$$ac_commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- audio_common has no git identity here (not a git checkout, or AC_DIR did not resolve to a real directory: AC_DIR=$(AC_DIR))." >&2; \
	  echo "  Without an identity the attestation cannot name the source state OR enumerate untracked" >&2; \
	  echo "  content, so NO escape hatch admits this (round-7: ALLOW_DIRTY_PUBLISH used to). Publish" >&2; \
	  echo "  against a real audio_common checkout, or point AC_DIR at a scratch clone with a" >&2; \
	  echo "  disposable commit, instead." >&2; \
	  exit 1; \
	fi; \
	ac_dirty=0; [ -n "$$(git -C $(AC_DIR) status --porcelain -uno 2>/dev/null)" ] && ac_dirty=1; \
	ac_unt_list="$$(git -C $(AC_DIR) ls-files --others --exclude-standard 2>/dev/null)"; \
	ac_untracked=0; [ -n "$$ac_unt_list" ] && ac_untracked=1; \
	adp=0; [ "$(ALLOW_DIRTY_PUBLISH)" = "1" ] && adp=1; \
	aup=0; [ "$(ALLOW_UNTRACKED_PUBLISH)" = "1" ] && aup=1; \
	violation=0; [ "$$dirty" = "1" ] && violation=1; [ "$$ac_dirty" = "1" ] && violation=1; \
	if [ "$$adp" != "1" ] && [ "$$violation" = "1" ]; then \
	  [ "$$dirty" = "1" ] && echo "FATAL: publish refused -- aec working tree has uncommitted TRACKED changes." >&2; \
	  [ "$$ac_dirty" = "1" ] && echo "FATAL: publish refused -- audio_common working tree has uncommitted TRACKED changes." >&2; \
	  echo "  Commit (or clone a clean checkout) for a board deliverable, or pass ALLOW_DIRTY_PUBLISH=1" >&2; \
	  echo "  for a dev publish -- the escape hatch is recorded in the attestation." >&2; \
	  exit 1; \
	fi; \
	u_violation=0; [ "$$untracked" = "1" ] && u_violation=1; [ "$$ac_untracked" = "1" ] && u_violation=1; \
	if [ "$$aup" != "1" ] && [ "$$u_violation" = "1" ]; then \
	  if [ "$$untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- aec contains UNTRACKED files (they can change the artifact without a trace in the tracked diff):" >&2; \
	    printf '%s\n' "$$unt_list" | head -5 | sed 's/^/    /' >&2; \
	  fi; \
	  if [ "$$ac_untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- audio_common contains UNTRACKED files (they can change the artifact without a trace in the tracked diff):" >&2; \
	    printf '%s\n' "$$ac_unt_list" | head -5 | sed 's/^/    /' >&2; \
	  fi; \
	  echo "  Track/remove them for a board deliverable, or pass ALLOW_UNTRACKED_PUBLISH=1 -- the escape" >&2; \
	  echo "  hatch is recorded in the attestation with untracked_tree_sha256 over their exact contents." >&2; \
	  exit 1; \
	fi; \
	ddiff=""; [ "$$dirty" = "1" ] && ddiff="$$(git diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	ac_ddiff=""; [ "$$ac_dirty" = "1" ] && ac_ddiff="$$(git -C $(AC_DIR) diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	uhash=""; \
	if [ "$$untracked" = "1" ]; then \
	  printf '%s\n' "$$unt_list" | hash_untracked "./" > "$$work/untracked.records"; \
	  if grep -q '^X ' "$$work/untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  uhash="$$(shasum -a 256 < "$$work/untracked.records" | cut -d' ' -f1)"; \
	fi; \
	ac_uhash=""; \
	if [ "$$ac_untracked" = "1" ]; then \
	  printf '%s\n' "$$ac_unt_list" | hash_untracked "$(AC_DIR)/" > "$$work/ac_untracked.records"; \
	  if grep -q '^X ' "$$work/ac_untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in audio_common cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/ac_untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  ac_uhash="$$(shasum -a 256 < "$$work/ac_untracked.records" | cut -d' ' -f1)"; \
	fi; \
	dateutc="$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	stamp='$(ATTEST_STAMP)'; [ -n "$$stamp" ] || stamp="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	$(HOSTCC) -O2 -o "$$work/swapln" $(AC_DIR)/tools/atomic_symlink_swap.c || { echo "FATAL: could not build the atomic symlink-swap helper with HOSTCC=$(HOSTCC) (pass HOSTCC=<host compiler>)" >&2; exit 1; }; \
	mkdir -p "$(DIST_BACKEND_DIR)"; \
	rm -rf "$$tmp"; mkdir -p "$$tmp"; \
	cp $(PUBLISH_ARTIFACTS) "$$tmp/"; \
	shas="$$( cd "$$tmp" && for f in *; do shasum -a 256 "$$f"; done )"; \
	content12="$$(printf '%s\n' "$$shas" | sort | shasum -a 256 | cut -c1-12)"; \
	rel_id="$(CFG_SIG)-$$content12"; \
	rel_dir="$(DIST_BACKEND_DIR)/$$rel_id"; \
	{ echo "repo=aec"; \
	  echo "release_id=$$rel_id"; \
	  echo "backend=$(BACKEND)"; \
	  echo "cfg_sig=$(CFG_SIG)"; \
	  echo "producer_cfg_sig=$(AC_PRODUCER_SIG)"; \
	  echo "cc=$(CC)"; echo "cxx=$(CXX)"; echo "ar=$(AR)"; echo "ranlib=$(RANLIB)"; echo "link=$(LINK)"; \
	  echo "cflags=$(CFLAGS)"; \
	  echo "extra_cflags=$(EXTRA_CFLAGS)"; \
	  echo "werror=$(WERROR)"; echo "no_stdio=$(NO_STDIO)"; echo "debug=$(DEBUG)"; \
	  echo "$$shas"; \
	} > "$$tmp/MANIFEST.txt"; \
	suffix=""; [ "$$dirty" = "1" ] && suffix="-dirty"; \
	if [ -d "$$rel_dir" ]; then \
	  for f in "$$tmp"/*; do \
	    bn="$$(basename "$$f")"; \
	    cmp -s "$$f" "$$rel_dir/$$bn" || { echo "FATAL: existing release $$rel_id differs from staged content for $$bn (content-address collision, or the stored release/MANIFEST was modified on disk)" >&2; exit 1; }; \
	  done; \
	  rm -rf "$$tmp"; \
	  echo "aec: $$rel_id already published (byte-verified, incl. MANIFEST) -- appending attestation + repointing current"; \
	else \
	  mkdir "$$tmp/ATTEST"; \
	  mv "$$tmp" "$$rel_dir"; \
	fi; \
	mkdir -p "$$rel_dir/ATTEST"; \
	n=1; \
	while :; do \
	  attest_name="attest-$$stamp-$$commit$$suffix-$$(printf '%03d' "$$n").txt"; \
	  { echo "event_id=$${attest_name%.txt}"; \
	    echo "repo=aec"; \
	    echo "release_id=$$rel_id"; \
	    echo "cfg_sig=$(CFG_SIG)"; \
	    echo "backend=$(BACKEND)"; \
	    echo "git_commit=$$commit"; echo "git_dirty=$$dirty"; \
	    [ -n "$$ddiff" ] && echo "dirty_diff_sha256=$$ddiff"; \
	    echo "git_untracked=$$untracked"; \
	    [ -n "$$uhash" ] && echo "untracked_tree_sha256=$$uhash"; \
	    echo "audio_common_git_commit=$$ac_commit"; echo "audio_common_git_dirty=$$ac_dirty"; \
	    [ -n "$$ac_ddiff" ] && echo "audio_common_dirty_diff_sha256=$$ac_ddiff"; \
	    echo "audio_common_git_untracked=$$ac_untracked"; \
	    [ -n "$$ac_uhash" ] && echo "audio_common_untracked_tree_sha256=$$ac_uhash"; \
	    echo "date_utc=$$dateutc"; \
	    echo "allow_dirty_publish=$$adp"; \
	    echo "allow_untracked_publish=$$aup"; \
	  } > "$$work/attest.txt"; \
	  rc=0; "$$work/swapln" --excl-install "$$work/attest.txt" "$$rel_dir/ATTEST/$$attest_name" || rc=$$?; \
	  [ "$$rc" -eq 0 ] && break; \
	  [ "$$rc" -eq 2 ] || { echo "FATAL: could not install attestation $$attest_name" >&2; exit 1; }; \
	  n=$$((n+1)); \
	  [ "$$n" -le 999 ] || { echo "FATAL: 999 attestation name collisions for $$stamp-$$commit (runaway?)" >&2; exit 1; }; \
	done; \
	"$$work/swapln" "$$rel_id" "$(DIST_BACKEND_DIR)/current"; \
	echo "aec: published $(BACKEND)/$$rel_id -> $(DIST_BACKEND_DIR)/current (attested: $$attest_name)"
endif

# =============================================================================
# audit-no-stdio: verifies a NO_STDIO=1 libaec.a is genuinely stdio-free
# (round-3 review B03). Self-contained recipe-time sub-makes (same pattern as
# `publish`'s AC_LIB-unset branch above) -- builds its own NO_STDIO=1 `lib`
# regardless of what NO_STDIO this invocation itself used, so a bare `make
# audit-no-stdio` just works.
#
# Checks, in order:
#   1. key-isolation  -- the NO_STDIO=1 print-lib-path differs from the
#      NO_STDIO=0 one (distinct CFG_SIG -> distinct obj/bin directory; a
#      NO_STDIO build can never silently alias a non-NO_STDIO one).
#   2. `ar -t`         -- aec_debug.o (the one stdio TU) is absent from the
#      NO_STDIO=1 archive.
#   3. `nm -A` (archive)-- no fprintf/vfprintf/printf/puts/fputs/fputc/fwrite/
#      stderr/__stderrp anywhere in libaec.a, defined or undefined. This is
#      the strong, unconditionally-true claim: AEC's OWN objects carry zero
#      stdio references under NO_STDIO=1, on either FFT backend.
#   4. minimal-main ELF gate -- link test/no_stdio_main.c (itself stdio-free:
#      no <stdio.h>, no printf) against the NO_STDIO=1 libaec.a + the
#      resolved audio_common archive, `nm -A` the EXECUTABLE, then run it.
#      Calibration (empirically measured on this host, both backends):
#        - BACKEND=ne10: the executable is FULLY stdio-free -- zero hits,
#          same strict set as the archive check. This is the embedded/board
#          deliverable backend; it sets the real bar.
#        - BACKEND=kiss: audio_common's vendored KISS FFT (lib/kiss_fft/
#          kiss_fft_log.h's KISS_FFT_ERROR macro, used on kiss_fft.c's
#          malloc-failure diagnostic paths) unconditionally references
#          __stderrp/fputc/fwrite -- a pre-existing dependency entirely
#          outside this AEC-side change (confirmed: `nm -u` on kiss_fft.o
#          alone shows exactly these three; libaec.a alone shows none). The
#          honest, achievable assertion on this backend is therefore that
#          NONE of fprintf/vfprintf/printf/puts/fputs (the exact symbols
#          aec_debug.c itself used to contribute -- confirmed by diffing
#          this same nm against a NO_STDIO=0 link, where exactly those two
#          symbols additionally appear) show up; the three KISS-FFT-owned
#          symbols are reported, not gated.
#   5. Run the linked executable; exit code must be 0 (aec_create +
#      one hop of aec_process + aec_destroy all succeeded with no stdio
#      call of its own).
# =============================================================================
AUDIT_STRICT_ALL  = fprintf|vfprintf|printf|puts|fputs|fputc|fwrite|stderr|__stderrp
AUDIT_STRICT_OWN  = fprintf|vfprintf|printf|puts|fputs
AUDIT_BIN_DIR     = $(BIN_ROOT)/audit-no-stdio/$(BACKEND)

.PHONY: audit-no-stdio
audit-no-stdio:
	@echo "== audit-no-stdio: BACKEND=$(BACKEND) WERROR=$(WERROR) CC=$(CC) =="
	+@$(MAKE) --no-print-directory NO_STDIO=1 BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' lib
	@lib_ns1="$$($(MAKE) -s --no-print-directory NO_STDIO=1 BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' print-lib-path)"; \
	 lib_ns0="$$($(MAKE) -s --no-print-directory NO_STDIO=0 BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' print-lib-path)"; \
	 echo "-- NO_STDIO=1 lib: $$lib_ns1"; \
	 echo "-- NO_STDIO=0 lib (key-isolation reference): $$lib_ns0"; \
	 test "$$lib_ns1" != "$$lib_ns0" || { echo "FATAL: NO_STDIO=1/0 resolved to the SAME path -- CFG_SIG key-isolation broken" >&2; exit 1; }; \
	 echo "PASS: key-isolation ($$lib_ns1 != $$lib_ns0)"; \
	 echo "-- ar -t $$lib_ns1 --"; \
	 ar -t "$$lib_ns1"; \
	 if ar -t "$$lib_ns1" | grep -qx 'aec_debug.o'; then \
	   echo "FATAL: aec_debug.o present in NO_STDIO=1 archive" >&2; exit 1; \
	 fi; \
	 echo "PASS: aec_debug.o absent from NO_STDIO=1 archive"; \
	 hits="$$(nm -A "$$lib_ns1" 2>/dev/null | grep -iE '$(AUDIT_STRICT_ALL)' || true)"; \
	 if [ -n "$$hits" ]; then \
	   echo "FATAL: stdio symbols found in NO_STDIO=1 archive:" >&2; echo "$$hits" >&2; exit 1; \
	 fi; \
	 echo "PASS: nm -A $$lib_ns1 -- no fprintf/vfprintf/printf/puts/fputs/fputc/fwrite/stderr/__stderrp (defined or undefined)"; \
	 echo "== audit-no-stdio: minimal-main ELF gate (BACKEND=$(BACKEND)) =="; \
	 $(MAKE) --no-print-directory -C $(AC_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 lib; \
	 ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 print-lib-path)"; \
	 test -n "$$ac" && test -f "$$ac" || { echo "FATAL: could not resolve/build audio_common archive (AC_DIR=$(AC_DIR))" >&2; exit 1; }; \
	 mkdir -p $(AUDIT_BIN_DIR); \
	 exe="$(AUDIT_BIN_DIR)/no_stdio_main"; \
	 $(CC) $(CFLAGS) -c -o "$(AUDIT_BIN_DIR)/no_stdio_main.o" $(TEST_DIR)/no_stdio_main.c; \
	 $(LINK) -o "$$exe" "$(AUDIT_BIN_DIR)/no_stdio_main.o" "$$lib_ns1" "$$ac" $(LDFLAGS); \
	 echo "PASS: linked $$exe"; \
	 exe_hits="$$(nm -A "$$exe" 2>/dev/null | grep -iE '$(AUDIT_STRICT_ALL)' || true)"; \
	 exe_own_hits="$$(printf '%s\n' "$$exe_hits" | grep -iE '$(AUDIT_STRICT_OWN)' || true)"; \
	 if [ -n "$$exe_own_hits" ]; then \
	   echo "FATAL: AEC-attributable stdio symbols found in linked executable:" >&2; echo "$$exe_own_hits" >&2; exit 1; \
	 fi; \
	 if [ -n "$$exe_hits" ]; then \
	   echo "CALIBRATED (BACKEND=$(BACKEND)): executable still carries these stdio-family symbols:"; \
	   echo "$$exe_hits"; \
	   echo "  -- attributed to audio_common's vendored KISS FFT (lib/kiss_fft/kiss_fft_log.h's"; \
	   echo "     KISS_FFT_ERROR malloc-failure diagnostic macro), NOT to AEC or this NO_STDIO"; \
	   echo "     gate -- pre-existing, out of scope for this AEC-side change. Verified: none of"; \
	   echo "     fprintf/vfprintf/printf/puts/fputs (the exact symbols aec_debug.c itself"; \
	   echo "     referenced) appear; only __stderrp/fputc/fwrite from kiss_fft.c's own error"; \
	   echo "     path remain. BACKEND=ne10 (the embedded/board deliverable) carries no such"; \
	   echo "     vendor residual and links fully stdio-free on this host."; \
	 else \
	   echo "PASS: linked executable is FULLY stdio-free (zero fprintf/vfprintf/printf/puts/fputs/fputc/fwrite/stderr/__stderrp, defined or undefined)"; \
	 fi; \
	 echo "== audit-no-stdio: running $$exe =="; \
	 "$$exe"; rc=$$?; \
	 echo "exit code: $$rc"; \
	 test "$$rc" -eq 0 || { echo "FATAL: no_stdio_main exited $$rc (expected 0)" >&2; exit 1; }; \
	 echo "audit-no-stdio: PASS (BACKEND=$(BACKEND))"

# Build Configuration Flags (pass via EXTRA_CFLAGS)
#
# USE_STANDARD_MATH - Use standard math library functions (for debugging/parity check)
#   Default: fast_math.h approximations for expf/sqrtf in hot paths
#   Example: make EXTRA_CFLAGS="-DUSE_STANDARD_MATH"

# debug delegates to `all` with -g -DAEC_DEBUG folded into EXTRA_CFLAGS
# (round-3 review B01): the old `debug: CFLAGS += -g -DAEC_DEBUG` mutated
# CFLAGS via a target-specific variable, which is only visible to the
# recipe, AFTER CFG_SIG had already been computed from the un-mutated
# CFLAGS -- a debug build's objects were compiled with -g -DAEC_DEBUG but
# keyed (CFG_SIG-wise) as if they weren't, silently sharing a directory with
# a non-debug build of otherwise-identical flags. Recursing through
# EXTRA_CFLAGS instead means the flag addition happens BEFORE CFG_SIG is
# computed (EXTRA_CFLAGS folds into CFLAGS at the top of this file), so a
# debug build gets its own directory like any other distinct config.
debug:
	+@$(MAKE) EXTRA_CFLAGS='$(EXTRA_CFLAGS) -g -DAEC_DEBUG' all

clean:
	rm -rf $(OBJ_ROOT) $(BIN_ROOT)

help:
	@echo "Targets:"
	@echo "  all     Build CLI binary (aec_wav)  [default]"
	@echo "  lib     Build static library (libaec.a)"
	@echo "  bench   Build RTF timing harness (bench_rtf)"
	@echo "  selftest Build + run the AEC-only kernel-split self-test (simd_selftest_aec)"
	@echo "  test-counter-saturation  Build + run the permanent counter-saturation"
	@echo "                  regression test (test_counter_saturation) -- run from c_impl/"
	@echo "  debug   Build with -g -DAEC_DEBUG"
	@echo "  publish Copy this build's artifacts to dist/<backend>/<config-hash>/ + update dist/<backend>/current"
	@echo "  audit-no-stdio  Build lib with NO_STDIO=1 and verify it (and a minimal linked"
	@echo "                  binary) carry no fprintf/vfprintf/stderr/etc -- see Makefile comment"
	@echo "  clean   Remove obj/ and bin/ (dist/ survives -- publish is a deliberate release, not a build cache)"
	@echo ""
	@echo "Usage:"
	@echo "  make print-bin-dir   # -> this invocation's bin/<backend>-<config-hash>/"
	@echo "  make print-lib-path  # -> this invocation's libaec.a path"
	@echo "  (to get the path 'make all' itself will produce, first resolve"
	@echo "   AC_LIB the same way phase 1 does -- see the two-phase dispatch"
	@echo "   comment in this Makefile -- and pass it: make print-bin-dir AC_LIB=\"...\")"
	@echo "  ./bin/<backend>-<config-hash>/aec_wav <mic.wav> <ref.wav> <out.wav> [options]"
	@echo "  ./bin/<backend>-<config-hash>/bench_rtf [--seconds N] [--repeat M] [--preset mild|balanced|aggressive]"
	@echo "  make BACKEND=kiss|ne10   (default: kiss; ne10 is the embedded deliverable)"
	@echo "  make AC_DIR=<path>       (override audio_common location)"
	@echo "  make WERROR=1            (opt-in: -Werror for this repo's own sources)"
	@echo "  make lib NO_STDIO=1      (opt-in: stdio-free libaec.a for board/no-stdio builds --"
	@echo "                            only the 'lib' goal; CLI/bench are not built under this flag)"
	@echo ""
	@echo "Objects/binaries live in obj|bin/<backend>-<config-hash>/ -- switching"
	@echo "BACKEND, EXTRA_CFLAGS, or WERROR always lands in a fresh directory"
	@echo "automatically; no manual clean needed, and two differently-configured"
	@echo "builds can run concurrently in this same tree."

dirs: _cfg_guard

.PHONY: all dirs clean debug help

# --- Header dependency tracking (round-3 review B01) ------------------------
# See audio_common's Makefile for the full rationale (-MD not -MMD, -MP
# guards against deleted-header stale-.d errors). $(wildcard ...) makes this
# evaluate to nothing (not an error) before any objects exist yet for this
# CFG_SIG.
-include $(wildcard $(OBJ_DIR)/*.d)
