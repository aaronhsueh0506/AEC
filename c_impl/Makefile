# AEC C Implementation — Makefile

CC      ?= gcc
CXX     ?= c++
AR      ?= ar
RANLIB  ?= ranlib

# CFG_SIG payload-coverage placeholders. CXXFLAGS/
# CPPFLAGS/EXTRA_LDFLAGS/DEBUG are still not consumed by this Makefile (this
# repo has no C++ TUs of its own, and no DEBUG-gated code path -- `make
# debug` is a distinct recipe, see below), but the CFG_SIG payload
# unconditionally includes them so the payload string has the same stable
# shape as audiolib's and NR's, and so a future use of any of them here
# keys builds correctly instead of silently aliasing with a build that
# didn't set it. NO_STDIO is a live switch:
# NO_STDIO=1 adds -DAEC_NO_STDIO to CFLAGS below and excludes src/aec_debug.c
# from the library sources, producing a stdio-free libaec.a -- see the
# NO_STDIO block further down and the `audit-no-stdio` target.
#
# CXXFLAGS and CPPFLAGS are different: they are also two of the five names
# the "Command-line override rejection" foreach further below rejects a
# command-line/`-e` override of -- the same class of bug audiolib's CPPFLAGS had, fixed
# there the same way. `?=` only ever assigns when the variable's origin is still
# "undefined" -- if either already has a value from the environment, `?=`
# is a complete no-op (GNU Make never attempts the assignment at all), so
# the origin never flips to "environment override" under `make -e` no
# matter how early this line sits (confirmed empirically:
# `env FP_POLICY=-ffp-contract=fast make -e -n BACKEND=ne10 lib` used to
# sail straight through with no error). `+=` fixes this while keeping the
# externally-visible behaviour identical: appending an empty string is a
# no-op on the VALUE (a plain-environment CXXFLAGS/CPPFLAGS still folds in
# normally -- origin becomes "file" -- and still participates in
# CFG_SIG_PAYLOAD unchanged), but `+=` is an unconditional assignment
# ATTEMPT, so under `-e` GNU Make's origin bookkeeping now correctly flips
# it to "environment override", which the foreach rejects. See
# audiolib's Makefile for the identical CPPFLAGS fix and full writeup.
CXXFLAGS      +=
CPPFLAGS      +=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

# TOOLCHAIN_CHECK=0 skips the CC/CXX -dumpmachine target-coherence guard in
# _cfg_guard below. Participates in CFG_SIG, so a
# guard-skipped build can never alias a guarded one.
TOOLCHAIN_CHECK ?= 1

# DIST_ROOT / HOSTCC: where `make publish` writes releases
# (test scripts point this at a throwaway mktemp dir so they never touch the
# real dist/), and the HOST compiler for the publish-time atomic
# symlink-swap helper (never $(CC): that may be a cross compiler). Neither
# is in CFG_SIG -- neither affects what is built. See audiolib's
# Makefile for the full writeups.
#
# OBJ_ROOT / BIN_ROOT: placement knobs for the keyed
# obj/bin trees -- declared further down next to their only consumers,
# $(OBJ_DIR)/$(BIN_DIR) (see that declaration's comment for the full
# writeup). Listed here only so the "four knobs, none in CFG_SIG" group
# (DIST_ROOT/HOSTCC/OBJ_ROOT/BIN_ROOT) reads as one family.
#
# ALLOW_DIRTY_PUBLISH: `make publish` REFUSES by default when EITHER this checkout OR the
# resolved audiolib PRODUCER checkout (COMMON_DIR below -- the shared FFT
# layer this repo links against, whose commit/dirty state the attestation
# also fingerprints) has uncommitted TRACKED changes (`git status
# --porcelain -uno` non-empty) or is not a git checkout at all (commit
# unknown) -- a release whose ATTEST provenance can't name the exact
# source state of BOTH repos that produced it is not a board deliverable.
# ALLOW_DIRTY_PUBLISH=1 is the explicit escape hatch for dev publishes; it
# is recorded in the attestation together with a sha256 of
# `git diff --binary HEAD` for whichever repo(s) are dirty, so the
# deviation itself is traceable. It covers TRACKED changes only --
# untracked files have their own dimension below (the
# tracked diff hash says nothing about untracked bytes, so lumping the two
# under one flag let two different untracked source states publish with
# identical provenance).
#
# ALLOW_UNTRACKED_PUBLISH: publish also REFUSES by default
# when EITHER this checkout OR the resolved audiolib PRODUCER checkout
# contains ANY untracked, non-ignored file (`git ls-files --others
# --exclude-standard` non-empty; gitignored build output is excluded by
# design and never part of this provenance) -- an untracked source a
# (possibly dirty) Makefile references can change the artifact without
# leaving a trace in the tracked diff. ALLOW_UNTRACKED_PUBLISH=1 is the
# separate escape hatch; it records untracked_tree_sha256 (this repo) /
# audiolib_untracked_tree_sha256 (the producer) in the attestation:
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
# EITHER this repo or the audiolib producer. Neither knob is in
# CFG_SIG: they never change what is built, only whether publish proceeds.
#
# ATTEST_STAMP (TEST-ONLY): overrides the UTC timestamp
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

# PROFILE=1 turns on the per-hop stage timing (aec_get_last_timing(), aec.h).
# Off by default: a diagnostic that has to be switched off is one that ships
# on by accident, and a release build takes none of the clock reads. The
# header carries the per-hop count, and is the only place that does.
#
# Folded into CFG_SIG both by name and through CFLAGS, so a profile build and
# a release build occupy different obj/bin directories and cannot hand each
# other a stale object.
#
# Same spelling as the two pipeline Makefiles that consume this library, so a
# profile build of the whole chain is one word everywhere rather than a
# hand-written -D here and a PROFILE there.
#
# Orthogonal to DEBUG/`make debug`, which builds with -g -DAEC_DEBUG rather
# than measuring -- profiling a debug build measures a binary nobody ships.
#
# A target whose libc has no POSIX CLOCK_MONOTONIC combines this with
# -DAEC_NOW_US=<fn>; see the stamp's comment in src/aec.c.
PROFILE ?= 0
ifneq ($(words $(PROFILE)),1)
$(error PROFILE must be exactly 0 or 1, got '$(PROFILE)')
endif
ifneq ($(filter-out 0 1,$(PROFILE)),)
$(error PROFILE must be exactly 0 or 1, got '$(PROFILE)')
endif
# Set through PROFILE_CFLAGS -> CFLAGS rather than EXTRA_CFLAGS, mirroring
# SIMD right below. EXTRA_CFLAGS is forwarded verbatim to the producer
# sub-makes and folded into CFG_SIG, so smuggling the -D through it made the
# flag arrive twice whenever a consumer that also has PROFILE drove this
# build, splitting byte-identical objects across two keys, and re-keyed
# producers that never reference the macro at all. CFLAGS is already
# protected from a command-line assignment (see the FP policy guard below),
# so the knob cannot be silently dropped either.
ifeq ($(PROFILE),1)
PROFILE_CFLAGS := -DAEC_STAGE_TIMING=1
else
PROFILE_CFLAGS :=
endif

SIMD ?= 1
ifneq ($(words $(SIMD)),1)
$(error SIMD must be exactly 0 or 1, got '$(SIMD)')
endif
ifneq ($(filter-out 0 1,$(SIMD)),)
$(error SIMD must be exactly 0 or 1, got '$(SIMD)')
endif
ifeq ($(SIMD),0)
SIMD_CFLAGS := -DSIMD_KERNELS_FORCE_SCALAR
else
SIMD_CFLAGS :=
endif

CFLAGS  = -Wall -Wextra -O2 -std=gnu99
CFLAGS += -I./include -I./example
CFLAGS += $(EXTRA_CFLAGS)
CFLAGS += $(SIMD_CFLAGS)
CFLAGS += $(PROFILE_CFLAGS)
LDFLAGS = -lm
LDFLAGS += $(EXTRA_LDFLAGS)

SRC_DIR     = src
EXAMPLE_DIR = example
TEST_DIR    = test

# Shared DSP layer (FFT backend + fast_math) lives in audiolib, vendored
# out of this repo (API: Complex, fft_create,
# fft_get_mem_size/fft_init, fft_forward/inverse/power/apply_gain, ALIGN16).
# COMMON_DIR resolves whether audiolib sits next to this repo (SE/audiolib)
# or nested deeper (e.g. an Audio_ALG submodule checkout).
COMMON_DIR ?= $(firstword $(wildcard ../../audiolib/common ../../../../audiolib/common))

# FFT backend: ARM NEON NE10 R2C/C2R, the only backend (audiolib rejects any
# other value). It computes in float32 -> ~1e-5 vs numpy's fp64 np.fft (a
# documented tolerance; the non-FFT C logic stays bit-exact to Python).
# SIMD=0 keeps the backend and routes it through NE10's scalar C kernels.
# BACKEND stays a variable because it names the keyed obj/bin directories.
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): the KISS FFT backend has been removed)
endif

# Announce the resolved backend on every build (same MAKECMDGOALS guard as
# audiolib's own banner: print-bin-dir/print-obj-dir/print-lib-path are
# consumed via `$$(make -s ... print-lib-path)` command substitution
# elsewhere, and this must not add stray stdout to that captured value).
ifeq ($(filter print-%,$(MAKECMDGOALS)),)
$(info AEC: FFT backend = $(BACKEND))
endif

CFLAGS  += -I$(COMMON_DIR)/include -I$(COMMON_DIR)

# Link driver: the audiolib archive carries one C++ TU (NE10's
# generic-radix .cpp kernel), so binaries linking it use the C++ driver, which
# supplies the correct C++ runtime itself (libc++ on macOS/clang, libstdc++ on
# GNU/Linux gcc). Declared before CFG_SIG so it participates in the payload.
LINK = $(CXX)

# -Werror opt-in for this repo's OWN sources (default off; CI can flip it on
# with `make WERROR=1`). AEC compiles no vendored TUs itself (the FFT vendor
# code lives in audiolib), so every object built by this Makefile is
# "our code" -- no separate warning isolation is needed here.
WERROR ?= 0
ifeq ($(WERROR),1)
  CFLAGS += -Werror
endif

# AEC_TESTING=1: compile the test-only hooks declared in test/aec_test_hooks.h.
# They reach internals that genuinely cannot be observed through the public API
# (see that header for the per-hook justification) and must NOT ship: the
# production library exports none of them.
#
# Like WERROR/NO_STDIO this appends to CFLAGS BEFORE CFG_SIG_PAYLOAD is
# computed, so an AEC_TESTING=1 build keys to its own obj/bin directory instead
# of aliasing with a production build of otherwise-identical flags -- the whole
# point, since the two must never share objects. `make test-no-testing-symbols`
# asserts the key isolation AND the symbol's absence/presence in both
# directions.
AEC_TESTING ?= 0
ifeq ($(AEC_TESTING),1)
  CFLAGS += -DAEC_TESTING
endif

# NO_STDIO=1: compile-time gate that produces a
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

# --- FP-contraction policy: unified across all four repos. FP_POLICY is
# appended to CFLAGS LAST, after every other CFLAGS append in this file
# (EXTRA_CFLAGS, the BACKEND-conditional -I$(COMMON_DIR)/include, WERROR,
# NO_STDIO), and BEFORE CFG_SIG_PAYLOAD is computed below. A caller's
# EXTRA_CFLAGS=-ffp-contract=fast (or -Ofast/-ffast-math, which imply
# contraction) therefore cannot land later on the compile line and win, and
# the policy participates in CFG_SIG without a separate payload entry. See
# README's build-flags section, audiolib's Makefile for the cross-repo
# design and scripts/audit_fp_contract.sh (in audiolib) for the
# disassembly-level verification.
#
# --- Bare-literal policy flag, defined early ---------------------------------
# FP_POLICY is a plain, dependency-free `:=` literal (no reference to CFLAGS
# or anything else computed above). It is defined ahead of the "Command-line
# override rejection" foreach below, which also lists FP_POLICY:
# $(origin FP_POLICY) only flips from plain "environment" to "environment
# override" once a makefile assignment to FP_POLICY has been PARSED, so
# `env FP_POLICY=-ffp-contract=fast make -e ...` is rejected only when the
# assignment precedes the foreach. This is the only `:=` definition of
# FP_POLICY; `CFLAGS += $(FP_POLICY)` further below is its other use.
# Keep scalar/NEON operation ordering stable and let audiolib's guarded
# AArch64 sqrtf calls lower to FSQRT without an errno-checking libm fallback.
FP_POLICY := -ffp-contract=off -fno-math-errno

# FP_ALLOWED_CHARS_RE (added alongside the FP-policy allow-list redesign
# below -- replicated from audiolib's identical fix, see the "REDESIGN"
# comment further down for the full writeup) is a SECOND bare,
# dependency-free `:=` literal with the exact same `make -e` exposure as
# FP_POLICY above -- it lives here, fully relocated (this is its ONLY
# definition in this file), so its origin is already final by the time the
# foreach below queries it. The regex it holds is documented in full where
# it is actually used (the FP-policy allow-list block, further below).
FP_ALLOWED_CHARS_RE := ^[A-Za-z0-9_./=,+ -]*$$

# --- Command-line override rejection ------------------------------------
# See audiolib's Makefile for the full writeup: a command-line
# CFLAGS=/CXXFLAGS=/CPPFLAGS=/LDFLAGS=/FP_POLICY=/FP_ALLOWED_CHARS_RE=
# silently disables every one of this file's own assignments/appends to that
# variable (here that would strip -ffp-contract=off, -DAEC_NO_STDIO, the -I
# paths, and -lm) while still producing artifacts -- and a command-line/`-e`
# override of FP_ALLOWED_CHARS_RE specifically could widen the allowed
# character set (e.g. to `.*`, matching anything) and silently neuter the
# allow-list below. EXTRA_CFLAGS / EXTRA_LDFLAGS are the supported hooks.
$(foreach v,CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FP_POLICY FP_ALLOWED_CHARS_RE,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy, NO_STDIO, include paths); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead)))

# Conflict detection (widened to CXXFLAGS/CPPFLAGS, mirroring
# audiolib's identical fix): reject outright rather than let a caller
# silently re-enable contraction. Checked
# against the fully-assembled $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) (each already
# includes EXTRA_CFLAGS folded in above, AND any CFLAGS=/CXXFLAGS=/CPPFLAGS=
# command-line override, since a command-line-set value is what every prior
# `+=` in this file appended to) -- BEFORE FP_POLICY is appended below and
# BEFORE CFG_SIG is computed, so a rejected build never creates an obj/bin
# directory.
#
# CXXFLAGS and CPPFLAGS are checked as well as CFLAGS: both are two of the
# five names the "Command-line override rejection" foreach above covers,
# both participate in CFG_SIG_PAYLOAD below (see this file's own top
# comment), and a plain environment CXXFLAGS or CPPFLAGS folds in normally
# via that foreach's origin logic, never going through EXTRA_CFLAGS. A live
# -Ofast/-ffast-math/-ffp-contract=<x> in either would otherwise ride in
# unchecked once either variable feeds a real compile recipe (the
# top-of-file comment describes that plumbing). The check below inspects all
# three variables together, so no combination of CFLAGS/CXXFLAGS/CPPFLAGS --
# command-line-folded, EXTRA_CFLAGS-folded, or a plain environment value --
# can carry any of these three patterns past this gate. This position (after
# every CFLAGS/CXXFLAGS/CPPFLAGS append earlier in this file and before
# FP_POLICY's own append just below) sees CFLAGS's fully-composed value, and
# is equally late enough for CXXFLAGS and CPPFLAGS: neither is appended to
# anywhere else in this file (each is a single bare `+=` at the top of this
# file -- see that line's own comment).
#
# The check matches whole words: $(findstring ...) does a plain SUBSTRING
# search over the whole concatenated text, so it would reject any token that
# merely CONTAINS one of these patterns as part of a larger identifier --
# e.g. a harmless `-DBUILD_NOTE=-Ofastness` macro definition is not the
# compiler flag -Ofast at all. $(filter PATTERN...,TEXT) instead
# splits TEXT on whitespace into WORDS and matches each whole word against
# PATTERN (a trailing %% is a single wildcard, used here for
# -ffp-contract=<anything>), so it only ever matches a complete flag token,
# never a substring buried inside an unrelated token.
FP_INPUT_FLAGS := $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS)

# --- link-flags character-safety coverage -----------------------------------
# The allow-list redesign below originally validated FP_INPUT_FLAGS only --
# CFLAGS/CXXFLAGS/CPPFLAGS -- leaving LDFLAGS (built from a plain `-lm` literal plus whatever
# EXTRA_LDFLAGS folds in, see the `LDFLAGS += $(EXTRA_LDFLAGS)` line near the
# top of this file) completely unchecked, even though LDFLAGS is embedded in
# every real link recipe further below ($(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS),
# used by $(TARGET)/$(BENCH_TARGET)/$(TEST_CS_TARGET), and again in the
# audit-no-stdio helper binary further down) exactly the same way CFLAGS is
# embedded in every compile recipe. Confirmed empirically:
# `make print-obj-dir EXTRA_LDFLAGS=';echo LINK_FLAG_INJECTION'` passed every
# check below untouched -- no FP policy error at all -- while the exact same
# payload via EXTRA_CFLAGS on the same target is correctly rejected: the
# exact same class of exposure the allow-list below exists to close for
# CFLAGS/CXXFLAGS/CPPFLAGS, just left open on this one variable (the
# identical gap independently found and fixed in audiolib's Makefile;
# replicated here from that fix). Fixed by introducing this union: the two
# checks that only care about CHARACTER safety (the single-quote pre-check
# and the allow-list itself, immediately below) now run against
# SHELL_SAFE_INPUT_FLAGS instead of FP_INPUT_FLAGS directly, so LDFLAGS/
# EXTRA_LDFLAGS content goes through the exact same grep-based gate with no
# separate/duplicated $(shell) invocation. The FP-semantic conflict check
# (FP_CONFLICT_FLAGS, further below -- -Ofast/-ffast-math/-ffp-contract=% are
# meaningless as LINK flags, not just unnecessary to check there) stays
# scoped to FP_INPUT_FLAGS alone, unchanged.
SHELL_SAFE_INPUT_FLAGS := $(FP_INPUT_FLAGS) $(LDFLAGS)

# $(filter) directly above does whole-WORD matching against
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
# -DBUILD_NOTE=-Ofastness or -DTEXT=ffast-math, contains none of the
# characters checked for below and so is untouched by this block.)
#
# An ALLOW-list is used rather than a character DENY-list. A deny-list
# (single quote, double quote, backslash, backtick, a literal "$(" sequence,
# semicolon, pipe, ampersand, an @-prefixed response-file token) cannot keep
# up: it does not catch bare glob characters (*, ?, [, ]), tilde (~), or --
# most seriously -- real shell REDIRECTION (>, <) or process substitution
# (<(...)). None of these are Make's own syntax, so Make cannot pre-resolve
# or reject them the way it handles ${VAR}-style expansions (GNU Make treats
# ${...} identically to $(...) and resolves it itself before any check
# runs). If a recipe line containing an unrejected '>' or '<' is executed by
# a real shell (not a dry run), it redirects the compiler invocation's own
# stdout/stdin to an arbitrary path -- a real, unconditional risk,
# independent of any -Ofast/-ffast-math/-ffp-contract semantics. A literal
# "$(" entry would be dead code for well-formed input:
# `make ... 'CFLAGS=-O2 $(shell echo INJECTED)'` lets GNU Make's OWN parser
# resolve the well-formed $(...) reference at command-line-parsing time --
# "INJECTED" lands in CFLAGS as plain text, no literal "$(" substring ever
# reaches the check -- while an UNBALANCED `$(unbalanced` (no closing paren)
# makes Make itself abort with its own "unterminated variable reference.
# Stop." before any makefile-level check runs at all.
#
# The block therefore applies a single ALLOW-list covering the input as a
# whole.
# GNU Make has no native regex/character-class function, so this
# necessarily shells out to `grep -E` -- a $(shell ...) call, the same
# technique this file already uses for CFG_SIG's cksum computation below.
# Embedding untrusted, Make-supplied text inside a constructed shell
# command is exactly the kind of thing that can reintroduce the very
# problem being fixed if done carelessly, so the embedding strategy is
# deliberately narrow and was verified directly (not assumed):
# SHELL_SAFE_INPUT_FLAGS (CFLAGS/
# CXXFLAGS/CPPFLAGS/LDFLAGS, fully-assembled -- see the "link-flags
# character-safety coverage" comment above for why LDFLAGS is folded in
# here too) is embedded SINGLE-QUOTED in the shell command text below
# (`printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | ...`). Inside a single-quoted
# POSIX shell string, EVERY character is literal except a single quote
# itself -- double quote, backslash, backtick, "$(", semicolon, pipe, and
# ampersand are all completely inert there. So the ONLY character that can
# break this embedding is a literal single quote in SHELL_SAFE_INPUT_FLAGS
# itself (which would close the quoted string early and hand whatever
# follows to the shell as live syntax) -- which is why the single-quote
# check below is kept ahead of the allow-list, for this structural reason
# (safe shell-embedding), not merely because it's also a disallowed
# character (the allow-list below would reject it either way). It MUST run
# first, and MUST stay a pure
# $(findstring) check with zero shell involvement, so it can gate the
# shell-embedding step that follows it.
ifneq ($(findstring ',$(SHELL_SAFE_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a single-quote character; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from the allowed character set (see the allow-list below) -- a single quote specifically must be rejected by this Make-native check BEFORE the allow-list's own $(shell) call can safely single-quote-embed the rest of the text)
endif

# The allow-list itself. Allowed set: letters, digits, underscore, dot,
# slash, equals, comma, plus, minus -- plus space (the token SEPARATOR
# between flags; SHELL_SAFE_INPUT_FLAGS is always multiple whitespace-joined
# tokens, never a single one, so space must be permitted even though it is
# not part of any individual flag's own character set). Chosen (same audit
# audiolib's Makefile performed for its own identical allow-list) by
# checking every legitimate flag THIS Makefile's own invocations actually
# use: -O0/-O2/-O3, -Wall/-Wextra, -std=gnu99, -Iinclude/-I./include/
# -I./example, -I$(COMMON_DIR)/include, -DAEC_NO_STDIO (letters/digits/
# underscore, and slash/dot for the paths) -- every one of these tokens is
# built ONLY from this set. Comma and plus are included for headroom beyond
# this Makefile's current own usage (e.g. -Wl,-rpath,dir / -Wa,--option
# pass-through flags a consumer might legitimately need), on the same audit
# basis: neither character appears anywhere in this Makefile's own flags
# today, so including them cannot regress anything currently working, and
# both are inert to the shell (unlike the characters actually excluded
# below).
#
# LC_ALL=C avoids any locale-dependent surprises in the [A-Za-z0-9...]
# ranges (verified identical under C locale on this host's BSD grep/tr; the
# same invocation is expected to behave identically under GNU grep/tr on a
# Linux host, since POSIX bracket-expression ranges are locale-defined by
# design and C locale pins them to the unambiguous ASCII ordering on both).
#
# The usual targets of a character deny-list (single quote, double quote,
# backslash, backtick, a literal "$(", semicolon, pipe, ampersand,
# @-response-file) are characters NOT in this allowed set, so all of them
# are rejected by construction; test/run_fp_flag_policy_test.sh carries the
# negative cases. The single-quote check above stays separate for the
# shell-embedding reason given there, not because the allow-list fails to
# cover it (a lone single quote is ALSO rejected by the allow-list on its
# own).
#
# FP_ALLOWED_CHARS_RE itself is NOT defined here -- it is a bare, early
# `:=` literal (see the "Bare-literal policy flag" / "Command-line override
# rejection" comments near the top of this file, where it is protected
# against a command-line/`-e` override the exact same way FP_POLICY is).
# Only referenced here.
SHELL_SAFE_ALLOWLIST_RC := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C grep -Eq '$(FP_ALLOWED_CHARS_RE)'; echo $$?)
ifneq ($(SHELL_SAFE_ALLOWLIST_RC),0)
# Diagnostic-only second $(shell) call (only ever runs on the already-doomed
# error path above, never on a passing build): deletes every ALLOWED
# character from the same single-quote-embedded text, leaving only the
# disallowed ones, then dedupes/sorts them into a short, stable string --
# so the $(error ...) below names the SPECIFIC offending character(s)
# (e.g. ">" or "~" or "*") instead of a generic "invalid format" message.
SHELL_SAFE_DISALLOWED_CHARS := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C tr -d 'A-Za-z0-9_./=,+ -' | fold -w1 | LC_ALL=C sort -u | tr -d '\n')
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a character outside the allowed set [A-Za-z0-9_./=,+- ] (letters/digits/underscore/dot/slash/equals/comma/plus/minus, plus space as the token separator) -- disallowed character(s) found: "$(SHELL_SAFE_DISALLOWED_CHARS)"; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from that set -- remove/replace the disallowed character(s) shown above in CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS)
endif

FP_CONFLICT_FLAGS := $(filter -Ofast -ffast-math -ffp-contract=%,$(FP_INPUT_FLAGS))
ifneq ($(FP_CONFLICT_FLAGS),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains $(FP_CONFLICT_FLAGS); this repo pins -ffp-contract=off; remove $(FP_CONFLICT_FLAGS) from CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS)
endif

# Command-line/`-e` override rejection for the FP-policy/shell-safety
# checks' remaining FOUR internal variables (replicated from audiolib's
# identical fix, found live while that repo implemented the allow-list
# above; SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC added, same
# treatment, when the character-safety check was widened to LDFLAGS -- see
# the "link-flags character-safety coverage" comment above):
# FP_INPUT_FLAGS/SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC/
# FP_CONFLICT_FLAGS are each computed from the fully-assembled CFLAGS/
# CXXFLAGS/CPPFLAGS (SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC also
# fold in the fully-assembled LDFLAGS; SHELL_SAFE_ALLOWLIST_RC via a
# $(shell) call keyed off SHELL_SAFE_INPUT_FLAGS) -- unlike FP_POLICY/
# FP_ALLOWED_CHARS_RE above, none of the four is a bare, dependency-free
# literal, so none can be relocated earlier to ride in the six-name foreach
# near the top of this file (that foreach's own comment explains why
# bare-literal-and-early is required for the `-e` case). But each has the
# EXACT SAME command-line-override exposure as every name in that foreach:
# e.g. `make ... EXTRA_LDFLAGS=';echo INJECTED' SHELL_SAFE_ALLOWLIST_RC=0`
# (or the equivalent with SHELL_SAFE_INPUT_FLAGS=<anything clean>,
# FP_INPUT_FLAGS=<anything clean>, or FP_CONFLICT_FLAGS=) would silently win
# over this file's own computed value and turn every ifneq/$(error) gate
# above into a complete no-op -- while the REAL $(CFLAGS)/$(CXXFLAGS)/
# $(CPPFLAGS)/$(LDFLAGS) used in every actual compile/link recipe still
# carry the dangerous, unvalidated content unmodified. Fixed the same way
# as the `-e` gap elsewhere in this file, mirrored for a LATE-computed
# (not early-literal) variable: the override-rejection check must itself
# run AFTER the variable's own real assignment (never before), so the
# origin-flip -- for both plain "command line" and, under `-e`,
# "environment override" -- has already happened by the time
# $(origin ...) is queried here. This is why this second foreach sits HERE,
# immediately after FP_CONFLICT_FLAGS (the last of the four to be
# assigned), rather than folded into the earlier foreach (which runs too
# early in the file for any of these four to have been assigned yet).
# SHELL_SAFE_DISALLOWED_CHARS is deliberately NOT included here: it is pure
# diagnostic text folded into the $(error ...) message on the failure path
# only, never a condition any ifneq tests, so overriding it cannot bypass
# anything -- at worst it would garble the reported character list on a
# build that was going to fail anyway.
$(foreach v,FP_INPUT_FLAGS SHELL_SAFE_INPUT_FLAGS SHELL_SAFE_ALLOWLIST_RC FP_CONFLICT_FLAGS,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently defeat the FP-policy/shell-safety allow-list/conflict checks above, leaving the real CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS content unvalidated; this variable is internal to those checks and has no supported external use)))

# (`make -e` gap fix: FP_POLICY's own bare `:=` literal now
# lives earlier in this file, directly above the "Command-line override
# rejection" foreach -- see that foreach's "Bare-literal policy flag"
# comment for why. Only FP_POLICY's OTHER job -- appending it to CFLAGS,
# LAST, exactly as the "FP-contraction policy" comment above describes --
# still lives at this position; there is exactly one `:=` definition of
# FP_POLICY in this file, not two.)
CFLAGS    += $(FP_POLICY)

.DEFAULT_GOAL := all

# =============================================================================
# Consumer two-phase audiolib resolution
# =============================================================================
# Problem this replaces: the old Makefile hardcoded
# `$(COMMON_DIR)/bin/$(BACKEND)/libaudio_common.a` directly into LDFLAGS and used
# a plain order-only `audio_common_lib: $(MAKE) -C $(COMMON_DIR) BACKEND=$(BACKEND)
# lib` prerequisite. Two problems: (1) audiolib's own archive is now
# CFG_SIG-keyed (bin/<backend>-<sig>/), so there is no fixed path to hardcode
# any more; (2) only BACKEND was forwarded to that sub-make, so an
# audiolib built with a different WERROR/EXTRA_CFLAGS/compiler than this
# invocation's own (a real "config skew" risk under concurrent/differently-
# flagged builds) was silently linked in without any of that showing up in
# THIS repo's own build identity.
#
# The fix is a two-phase recursive-make dispatch, driven by whether COMMON_LIB
# (the resolved absolute path to audiolib's archive) is already known:
#
#   Phase 1 (this file parsed with COMMON_LIB unset -- the normal case for any
#   top-level `make ...` invocation): goals that need the archive (`all`,
#   `bench`) get a recipe that queries audiolib's own `print-lib-path`
#   query target (forwarding the exact BACKEND/WERROR/CC/CXX/EXTRA_CFLAGS/
#   NO_STDIO this invocation is using) and then re-invokes itself with
#   COMMON_LIB set to the answer.
#
#   Phase 2 (COMMON_LIB known, either because phase 1 supplied it or because a
#   caller who already knows it passed it directly): the REAL rules for
#   $(TARGET)/$(BENCH_TARGET) are defined, with $(COMMON_LIB) as a NORMAL (not
#   order-only) prerequisite -- so relinking happens iff audiolib's
#   archive mtime actually advanced, exactly like any other prerequisite.
#
# Goals that never need audiolib at all (`lib` -- this repo's OWN
# archive has no FFT/audiolib code in it; `selftest` -- header-only,
# links with a plain -lm; `clean`; the print-*/help query targets) get their
# real rules OUTSIDE this dispatch entirely, further down this file --
# invoking any of them never shells out to audiolib, never needs
# COMMON_DIR to even exist, and (see `make -n` note below) never touches disk
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
COMMON_FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK)

ifeq ($(COMMON_LIB),)
.PHONY: all bench test-counter-saturation test-rate-structural test-config-validation test-process-context test-avg-reverb-slots test-linear-context test-stream-sim test-res-nearend-gate test-runtime-preset test-delay-num-filters test-duty-census test-external-realign test-cli-delay-flags test-shared-far-spec test-spectral-reuse test-heap-alignment test-shared-fft-handle test-static-aec test-memory-layout test-reset-parity test-zero-heap test-aec-res test-detectors-parity test-detectors-parity-all test-parity-e2e pyc-parity-driver
all bench test-counter-saturation test-rate-structural test-config-validation test-process-context test-avg-reverb-slots test-linear-context test-stream-sim test-res-nearend-gate test-runtime-preset test-delay-num-filters test-duty-census test-external-realign test-cli-delay-flags test-shared-far-spec test-spectral-reuse test-heap-alignment test-shared-fft-handle test-static-aec test-memory-layout test-reset-parity test-zero-heap test-aec-res test-detectors-parity test-detectors-parity-all test-parity-e2e pyc-parity-driver:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audiolib archive path (COMMON_DIR=$(COMMON_DIR) -- is audiolib present as a sibling of this repo?)" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory $@ COMMON_LIB="$$ac"
else
# Producer-identity component of this consumer's own CFG_SIG (see the
# CFG_SIG section below): the LAST path component of $(COMMON_LIB)'s directory
# (e.g. "ne10-2139547423"), NOT the full (absolute) path -- so a consumer
# built against a DIFFERENT audiolib config (different flags, or the
# same flags after an audiolib source/header edit bumped its own
# CFG_SIG) keys to its own separate obj/bin directory here too, instead of
# forwarding only BACKEND and risking silent config skew (the actual bug
# class this whole redesign targets). $(COMMON_DIR) itself (an abspath, used
# only as the -C argument to the sub-make above) never appears in any
# payload or manifest -- only this short backend-sig component does.
COMMON_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(COMMON_LIB))))
endif

# Algorithm sources (auto-detected — drop a new file in src/ to add it).
# The FFT wrapper sources are part of the audiolib archive linked above,
# not of src/. Declared HERE, before CFG_SIG_PAYLOAD,
# because the payload folds in the sorted source list -- adding or removing a
# source file must key to a fresh obj/bin directory (see the fresh-archive
# recipe below for the other half of the stale-member fix).
SRCS = $(wildcard $(SRC_DIR)/*.c)

# NO_STDIO=1: belt-and-braces exclusion of the one
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
# obj/<backend>-<sig>/ AND bin/<backend>-<sig>/ -- see
# audiolib's Makefile for the full rationale (identical design, applied
# here). A flat bin/ would let two builds differing only in SIMD/EXTRA_CFLAGS/
# WERROR, or a stale mtime from a differently-flagged prior build, silently
# deliver the wrong `aec_wav`/`libaec.a`; the hash-keyed directories prevent
# that. Payload
# coverage: CC CXX AR RANLIB CFLAGS
# CXXFLAGS CPPFLAGS LDFLAGS EXTRA_CFLAGS EXTRA_LDFLAGS BACKEND WERROR
# NO_STDIO DEBUG, PLUS (consumer-specific) COMMON_PRODUCER_SIG -- the resolved
# producer's own config identity (empty when COMMON_LIB isn't resolved yet, e.g.
# for goals like `lib`/`selftest` that never touch audiolib at all).
#
# This MUST be defined after CFLAGS/LDFLAGS are fully assembled (including
# the BACKEND-conditional and WERROR appends above) and after the
# COMMON_LIB ifeq/else above resolves COMMON_PRODUCER_SIG, but before the first rule
# below that mentions $(OBJ_DIR)/$(BIN_DIR)/anything derived from them --
# GNU Make expands a rule's target/prerequisite list immediately when
# parsed, not deferred like a recipe body, so a forward reference here would
# silently resolve using CFG_SIG's value at parse time (empty), landing
# objects in a hash-less directory instead of the real one.
#
# Deliberately NOT $(strip)'d (see audiolib's Makefile for the false-miss
# vs false-hit rationale). cksum (POSIX) computes the hash; `cut -d' ' -f1`
# isolates the checksum from cksum's "<checksum> <byte-count>" stdout.
# Additional CFG_SIG payload coverage: LINK, TOOLCHAIN_CHECK (the
# guard-skip knob must not alias a guarded build), SRCS (sorted source
# list, post-NO_STDIO-filter, so source add/remove keys to a fresh directory).
# The same CC command can select a different SDK through PATH.
# Include effective pointer/FP/ISA predefines to avoid cross-ABI cache reuse.
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) LDFLAGS=$(LDFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) SIMD=$(SIMD) PROFILE=$(PROFILE) WERROR=$(WERROR) AEC_TESTING=$(AEC_TESTING) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) COMMON_PRODUCER_SIG=$(COMMON_PRODUCER_SIG) SRCS=$(sort $(SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

# OBJ_ROOT / BIN_ROOT: where the keyed obj/bin trees
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

# Counter-saturation regression test (counter-overflow behaviour; also run
# under UBSan by test/run_counter_saturation_ubsan.sh). Needs the
# full Aec type (aec.c/pbfdkf.c's FFT-backed sections), so -- unlike
# SELFTEST_TARGET below -- this lives in the two-phase/COMMON_LIB-dependent part
# of this Makefile, same as $(TARGET)/$(BENCH_TARGET).
TEST_CS_SRCS   = $(TEST_DIR)/test_counter_saturation.c
TEST_CS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_CS_SRCS))
TEST_CS_TARGET = $(BIN_DIR)/test_counter_saturation

# Permanent AecLinearContext seam regression test (aec_get_linear_context):
# aligned-far alias + byte-equal content vs the caller's delayed raw far,
# generation bumps at acquisition/shift/reset (including the flagless
# soft-recovery write site), UNLOCKED honesty under enable_delay_est=0, and
# the ref_ring_filled freeze-at-capacity fix. Needs the full Aec type.
TEST_LC_SRCS   = $(TEST_DIR)/test_linear_context.c
TEST_LC_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_LC_SRCS))
TEST_LC_TARGET = $(BIN_DIR)/test_linear_context

# Streaming-API regression test (aec_analyze_render / aec_process_capture):
# FIFO capacity rule, lockstep byte-equality with aec_process(), exact-count
# drop-new overrun, underrun recovery and consumer catch-up, on every
# supported grid. The source keeps its historical name (test/stream_sim.c).
# Needs the full Aec type.
TEST_SS_SRCS   = $(TEST_DIR)/stream_sim.c
TEST_SS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SS_SRCS))
TEST_SS_TARGET = $(BIN_DIR)/stream_sim

# RES double-talk floor uses the existing energy-based near-recent latch;
# the test also pins the exported per-hop decision and existing stationarity
# policy (test/test_res_nearend_gate.c). Needs the full Aec type.
TEST_RNG_SRCS   = $(TEST_DIR)/test_res_nearend_gate.c
TEST_RNG_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_RNG_SRCS))
TEST_RNG_TARGET = $(BIN_DIR)/test_res_nearend_gate

# Permanent runtime-strength-retarget regression test
# (suppression_gain_set_split_floor_far_active_db + aec_set_preset). Proves the
# immediate retarget lands on the constructed float, that a ramp advances once
# per aec_process and lands exactly on the predicted hop, that a retarget
# mid-ramp continues from the live value, that reset drops ramp progress but
# keeps the target, that rejection writes nothing, and that a
# spatial_linear_context lane never advances the ramp at all. Needs the full
# Aec and SuppressionGain types.
TEST_RP_SRCS   = $(TEST_DIR)/test_runtime_preset.c
TEST_RP_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_RP_SRCS))
TEST_RP_TARGET = $(BIN_DIR)/test_runtime_preset

# Permanent matched-filter-bank-size regression test
# (AecConfig::delay_num_filters -> delay_aec_get_mem_size/delay_aec_init ->
# the runtime loop bound in da_matched_filter_update). Proves the default is
# still the 5-filter geometry, that a shorter bank really loses reach end
# to end (a 350 ms echo locks at n=5 and does not at n=2, while 150 ms locks at
# both), that the pool the estimator is carved into scales with n, and that
# out-of-range values are REJECTED at every layer -- aec_validate_config above
# and the pool-first delay_aec_get_mem_size/delay_aec_init below, neither of
# which clamps. Needs the full Aec type.
TEST_DNF_SRCS   = $(TEST_DIR)/test_delay_num_filters.c
TEST_DNF_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_DNF_SRCS))
TEST_DNF_TARGET = $(BIN_DIR)/test_delay_num_filters

# Permanent matched-filter duty-cycle engagement census test (productization
# plan §7 measurement method): measures what the duty machine actually
# decimated on a real run (Aec::duty_hops_total/duty_hops_run, surfaced via
# AecDebugStatus), instead of trusting the "~90% of the matched-filter cost"
# design-note estimate the duty-cycle doc comment in aec.h carries.
TEST_DUC_SRCS   = $(TEST_DIR)/test_duty_census.c
TEST_DUC_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_DUC_SRCS))
TEST_DUC_TARGET = $(BIN_DIR)/test_duty_census

TEST_XRA_SRCS   = $(TEST_DIR)/test_external_realign.c
TEST_XRA_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_XRA_SRCS))
TEST_XRA_TARGET = $(BIN_DIR)/test_external_realign

# Independent reference tool for cli_delay_flags.sh's print-mem-size
# cross-check: calls aec_get_mem_breakdown() from its OWN source file (not
# aec_wav.c/example/), so the shell test's "the CLI's number == the
# library's number" assertion exercises a genuinely separate code path from
# aec_wav.c's --print-mem-size printf. Not itself a `make test-*` target --
# it only exists as a test-cli-delay-flags prerequisite.
TEST_PMR_SRCS   = $(TEST_DIR)/print_mem_size_ref.c
TEST_PMR_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_PMR_SRCS))
TEST_PMR_TARGET = $(BIN_DIR)/print_mem_size_ref

# Per-rate structural regression test: gates the supported grids, including
# the hop-authored top-level constant retiming that its check (d) covers.
# Needs the full Aec type, same as TEST_CS_TARGET above -- reuses the
# already-built $(OBJS); test_rate_structural.c's own header comment gives a
# standalone `$(find src -name '*.c')` recipe for a from-scratch manual
# build.
TEST_RS_SRCS   = $(TEST_DIR)/test_rate_structural.c
TEST_RS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_RS_SRCS))
TEST_RS_TARGET = $(BIN_DIR)/test_rate_structural

# Config-validation regression test. Its section
# test_direct_init_rejects_bad_rate gates the pbfdaf_init_static/
# pbfdkf_init_static/saturation_init sample_rate preconditions.
TEST_CV_SRCS   = $(TEST_DIR)/test_config_validation.c
TEST_CV_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_CV_SRCS))
TEST_CV_TARGET = $(BIN_DIR)/test_config_validation

# Permanent context-only-processing regression test (aec_process_context()):
# proves the two entry points may be interleaved freely on one instance, and
# that every AecResContext field a real caller reads is byte-identical
# between them. Needs the full Aec type, same as
# TEST_CS_TARGET/TEST_RS_TARGET above.
TEST_PC_SRCS   = $(TEST_DIR)/test_process_context.c
TEST_PC_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_PC_SRCS))
TEST_PC_TARGET = $(BIN_DIR)/test_process_context

# Average-render-reverb ring-slot selection (aec_post_avg_reverb_slots): the
# delayed and the one-older partition must never alias the newest one.
TEST_ARS_SRCS   = $(TEST_DIR)/test_avg_reverb_slots.c
TEST_ARS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_ARS_SRCS))
TEST_ARS_TARGET = $(BIN_DIR)/test_avg_reverb_slots

# Permanent shared-far-end-FFT regression test
# (aec_process_context_shared_far()): proves borrowing an externally-computed
# far-end spectrum produces byte-identical AecResContext to computing it
# independently, that the borrowed spectrum is genuinely consumed (not
# silently ignored), that aec_reset() clears a pending borrow cleanly, and
# that the real-FFT-compute counter actually reflects borrow-vs-compute.
# Same "was standalone-gcc-only, never wired into a make target" gap as
# TEST_PC_TARGET above.
TEST_SFS_SRCS   = $(TEST_DIR)/test_shared_far_spec.c
TEST_SFS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SFS_SRCS))
TEST_SFS_TARGET = $(BIN_DIR)/test_shared_far_spec

TEST_SRU_SRCS   = $(TEST_DIR)/test_spectral_reuse.c
TEST_SRU_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SRU_SRCS))
TEST_SRU_TARGET = $(BIN_DIR)/test_spectral_reuse

TEST_HAL_TARGET = $(BIN_DIR)/test_heap_alignment
TEST_HAL_OBJ = $(OBJ_DIR)/test_heap_alignment.o

# Permanent shared-FFT-handle regression test: proves main filter, shadow
# filter, and the post-filter block all borrow the exact SAME FftHandle
# (pointer identity, not just equal size) instead of each owning a private
# one, on both the heap-arena (aec_create) and caller-pool (aec_init) paths,
# surviving a real process() hop and aec_reset(). Mutation-tested: fails
# (33/63) when the sharing is reverted to per-sub-module private handles.
TEST_SFH_SRCS   = $(TEST_DIR)/test_shared_fft_handle.c
TEST_SFH_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SFH_SRCS))
TEST_SFH_TARGET = $(BIN_DIR)/test_shared_fft_handle

# Heap-vs-caller-pool byte-equality test on a complete real recording.
TEST_STATIC_SRCS   = $(TEST_DIR)/test_static_aec.c
TEST_STATIC_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_STATIC_SRCS))
TEST_STATIC_TARGET = $(BIN_DIR)/test_static_aec
TEST_STATIC_MIC ?= ../wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk_mic.wav
TEST_STATIC_REF ?= ../wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk_lpb.wav
TEST_STATIC_SR  ?= 16000

# Permanent aec_reset() fresh-instance gate: a never-warmed instance and a
# warmed-then-reset one, on the same config, must produce byte-identical
# out[hop] / AecResContext.formed_hop / AecResContext.error_spec over the
# whole post-reset run. aec.h documented that equivalence for a long time
# while state (the filters' startup gates, the Kalman H_error, the
# inst-ERLE slope ring, the RES-context stash) actually survived the call and
# diverged the first post-reset hop. Runs the three product grids x both
# consumption modes.
TEST_RPY_SRCS   = $(TEST_DIR)/test_reset_parity.c
TEST_RPY_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_RPY_SRCS))
TEST_RPY_TARGET = $(BIN_DIR)/test_reset_parity

# Python/C parity gate for the three detectors (RenderActivity /
# FilterConvergence / DoubleTalk). detectors.c includes only detectors.h and
# <math.h>, so this links against nothing else -- no $(OBJS), no audiolib
# archive (aec_scale.c is the one extra TU, for the retiming helpers). The
# golden is generated on demand from the Python reference rather
# than checked in, matching the other parity harnesses.
# This target existed as an orphan for eight weeks with no Makefile entry and
# no generator, which is how the SHADOW_FRAME_GATE 50-vs-20 Python/C
# divergence survived undetected. Keep it wired.
TEST_DET_SRC    = $(TEST_DIR)/modules/parity_detectors.c
TEST_DET_TARGET = $(BIN_DIR)/parity_detectors
TEST_DET_HOP   ?= 128
TEST_DET_SR    ?= 16000
TEST_DET_GOLDEN = $(BIN_DIR)/det_golden.bin

# Allocator-hook acceptance test for the complete caller-pool lifecycle.
TEST_ZH_SRCS   = $(TEST_DIR)/test_zero_heap_aec.c
TEST_ZH_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_ZH_SRCS))
TEST_ZH_TARGET = $(BIN_DIR)/test_zero_heap_aec

# Standalone RES stage (aec_res.h): configuration equality with an Aec lane,
# gain and comfort-noise equality with the multi-lane pipelines' stage, API
# contract. The test includes src/aec_res.c for the struct, so it links the
# library objects without aec_res.o.
TEST_AR_OBJS    = $(OBJ_DIR)/test_aec_res.o
TEST_AR_TARGET  = $(BIN_DIR)/test_aec_res

# --- Python<->C parity harness (test/pyc_parity_aec.sh, test/pyc_parity/) --------
# test-parity-e2e : the repo's end-to-end gate (test/parity_aec_e2e.c) built and run against goldens
#                   regenerated on the fly by python/diag/gen_aec_e2e_golden.py into $(BIN_DIR)
#                   (never into git); source wavs pinned by test/golden/parity_e2e_inputs.sha256.
#                   16/48 kHz must pass; the known-red 8 kHz row is reported EXPECTED_RED(S-19).
# pyc-parity-driver : builds test/pyc_parity/aec_run.c (the C side of the harness) against this
#                   configuration; test/pyc_parity_aec.sh calls it once per build (std-math, ship).
# test-pyc-parity : the full Python<->C parity suite (two layers; see test/pyc_parity_aec.sh -h).
#                   Extra arguments: make test-pyc-parity PYC_ARGS="--quick".
# PYTHON must provide numpy, scipy and soundfile.
PYTHON ?= python3
PYC_DIR        = $(TEST_DIR)/pyc_parity
PYC_ARGS      ?=
TEST_PE_OBJS   = $(OBJ_DIR)/parity_aec_e2e.o
TEST_PE_TARGET = $(BIN_DIR)/parity_aec_e2e
PYC_RUN_OBJ    = $(OBJ_DIR)/pyc_aec_run.o
PYC_RUN_TARGET = $(BIN_DIR)/pyc_aec_run

$(PYC_RUN_OBJ): $(PYC_DIR)/aec_run.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# Does its own builds (one std-math, one ship) through `make pyc-parity-driver`, so it needs no
# audiolib dispatch of its own.
.PHONY: test-pyc-parity
test-pyc-parity:
	@PYTHON="$(PYTHON)" bash "$(TEST_DIR)/pyc_parity_aec.sh" $(PYC_ARGS)

# Header-only AEC kernel correctness/microbench harness (utility/simd_aec.h).
# Links with libm, without the audiolib archive or two-phase dispatch.
SELFTEST_SRC    = $(TEST_DIR)/simd_selftest_aec.c
SELFTEST_TARGET = $(BIN_DIR)/simd_selftest_aec

# Permanent delay_aec_reset() consistency regression test (reset() used to
# clear only the outer DaResample48 sidechain,
# leaving the inner DaEstimator's decimators/ring buffer/pending samples
# stale). Needs only delay_aec.c + its aec_scale.c helper dependency --
# NOT the full Aec type/audiolib archive (unlike test-counter-saturation)
# -- so, like `selftest` above, this target lives outside the two-phase
# COMMON_LIB dispatch and reuses the same $(OBJ_DIR)/%.o pattern rules (sharing
# objects with a full `all`/`lib` build under the same CFG_SIG, if present).
TEST_DR_SRCS   = $(SRC_DIR)/delay_aec.c $(SRC_DIR)/aec_scale.c $(TEST_DIR)/test_delay_reset.c
TEST_DR_OBJS   = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(filter $(SRC_DIR)/%.c,$(TEST_DR_SRCS))) \
                  $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(filter $(TEST_DIR)/%.c,$(TEST_DR_SRCS)))
TEST_DR_TARGET = $(BIN_DIR)/test_delay_reset

TEST_DDS_SRCS   = $(SRC_DIR)/delay_aec.c $(SRC_DIR)/aec_scale.c $(TEST_DIR)/test_delay_dominant_selection.c
TEST_DDS_OBJS   = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(filter $(SRC_DIR)/%.c,$(TEST_DDS_SRCS))) \
                  $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(filter $(TEST_DIR)/%.c,$(TEST_DDS_SRCS)))
TEST_DDS_TARGET = $(BIN_DIR)/test_delay_dominant_selection

# test_delay_kernels #includes src/delay_aec.c (the shipped TU, so its
# static kernels are testable) -- it must NOT link delay_aec.o on top, only
# aec_scale.o for delay_aec_init's one external call. Its object gets its
# own compile rule: -I. so the quoted include resolves from the c_impl root,
# and per-object -MD so an edit to the INCLUDED delay_aec.c rebuilds it (a
# single link-line -MF would record only one TU's deps and leave this test
# reusing a stale binary across kernel edits -- the exact failure it exists
# to prevent).
TEST_DKN_OBJ    = $(OBJ_DIR)/test_delay_kernels.o
TEST_DKN_TARGET = $(BIN_DIR)/test_delay_kernels
TEST_DMN_OBJ    = $(OBJ_DIR)/test_delay_matched_neon.o
TEST_DMN_TARGET = $(BIN_DIR)/test_delay_matched_neon
TEST_NT_OBJ     = $(OBJ_DIR)/test_neon_twins.o
TEST_NT_TARGET  = $(BIN_DIR)/test_neon_twins

# --- CFG_SIG collision guard + directory creation ---------------------------
# Single order-only prerequisite for every compile/link/archive rule below
# (replacing the old bare "| $(OBJ_DIR)"/"| $(BIN_DIR)" prerequisites and
# their own `@mkdir -p $@` rule). .PHONY, so its recipe runs on every `make`
# invocation that needs it, but GNU Make builds any one target at most once
# per invocation even when several rules depend on it, so under `make -j`
# it still runs exactly once and (being an ordinary, if order-only,
# dependency-graph edge) completes before any dependent rule starts:
# race-free. See audiolib's Makefile for the full design writeup (same
# mechanism, applied here) -- write config.manifest first time, verify
# (hard error on mismatch = CFG_SIG collision or payload coverage gap) every
# time after, atomically (temp file + mv).
.PHONY: _cfg_guard
_cfg_guard:
	@test -n "$(CC_TARGET_SIG)" && test "$(CC_TARGET_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine compiler target; CC preprocessing with the configured flags must succeed." >&2; \
	  exit 1; }
	@if [ "$(TOOLCHAIN_CHECK)" = "1" ]; then \
	  cc_m="$$($(CC) -dumpmachine 2>/dev/null)"; cxx_m="$$($(CXX) -dumpmachine 2>/dev/null)"; \
	  if [ -z "$$cc_m" ] || [ "$$cc_m" != "$$cxx_m" ]; then \
	    echo "FATAL: CC/CXX compile for different targets (CC='$(CC)' -> '$$cc_m', CXX='$(CXX)' -> '$$cxx_m')." >&2; \
	    echo "  The audiolib NE10 archive links a C++ TU; pass the FULL toolchain:" >&2; \
	    echo "    make CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ AR=aarch64-linux-gnu-ar RANLIB=aarch64-linux-gnu-ranlib" >&2; \
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

# -MD -MP: emits $(OBJ_DIR)/<name>.d per object (header
# dependency tracking) -- included near the bottom of this file via
# `-include $(wildcard $(OBJ_DIR)/*.d)`.
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/%.o: $(EXAMPLE_DIR)/%.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_aec_res.o: $(TEST_DIR)/test_aec_res.c | _cfg_guard
	$(CC) $(CFLAGS) -I. -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_zero_heap_aec.o: $(TEST_DIR)/test_zero_heap_aec.c | _cfg_guard
	$(CC) $(CFLAGS) -fno-builtin -I$(COMMON_DIR)/test -MD -MP -c -o $@ $<

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

.PHONY: lib
lib: $(LIB_TARGET)

# `lib` needs only THIS repo's own objects -- no audiolib archive is
# linked into libaec.a -- so, per the two-phase design above, it's a real
# rule living entirely outside the COMMON_LIB dispatch (never shells out to
# audiolib, works even if audiolib isn't present at all).
# Fresh-archive discipline: always built from scratch
# into a temp file and renamed into place -- never `ar rc` onto an existing
# archive (which only adds/replaces members and would keep a removed source's
# stale member forever). One chained shell line with a PID-suffixed temp:
# two same-config processes racing this rule each write
# their OWN temp, final mv is atomic last-writer-wins. See audiolib's
# Makefile for the full writeup.
$(LIB_TARGET): $(OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rc "$$t" $(OBJS) && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "aec [$(BACKEND)] -> $@"

# Header-only AEC kernel self-test; -I$(COMMON_DIR) resolves
# utility/simd_aec.h and -I$(COMMON_DIR)/include resolves ac_complex.h.
.PHONY: selftest
selftest: $(SELFTEST_TARGET)
	"$(abspath $(SELFTEST_TARGET))"

$(SELFTEST_TARGET): $(SELFTEST_SRC) | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -MF $(OBJ_DIR)/simd_selftest_aec.d -o $@ $< -lm

-include $(OBJ_DIR)/simd_selftest_aec.d

# Regenerates c_impl/include/aec_balanced_config.h (python/diag/
# gen_aec_balanced_config_h.py) into a throwaway copy of python/ and diffs
# it against the checked-in header -- a FAIL means the header was hand-edited
# or left stale; regenerate after any change to the balanced config or the
# generator. Self-contained (no audiolib archive), so -- like
# `lib`/`selftest`/`clean` -- it lives outside the COMMON_LIB dispatch above.
# PYTHON ?= python3 is declared further down; forward reference is fine,
# make expands recipe variables when the recipe runs, not in file order.
.PHONY: test-header-fresh
test-header-fresh:
	@set -e; \
	d="$$(mktemp -d "$${TMPDIR:-/tmp}/aec_header_fresh.XXXXXX")"; \
	trap 'rm -rf "$$d"' EXIT INT TERM; \
	mkdir -p "$$d/c_impl/include"; \
	cp -R "$(CURDIR)/../python" "$$d/python"; \
	find "$$d/python" \( -name __pycache__ -prune -o -name '*.pyc' \) -exec rm -rf {} +; \
	"$(PYTHON)" "$$d/python/diag/gen_aec_balanced_config_h.py" >/dev/null; \
	if cmp -s "$$d/c_impl/include/aec_balanced_config.h" "$(CURDIR)/include/aec_balanced_config.h"; then \
	  echo "test-header-fresh: include/aec_balanced_config.h is fresh against its generator"; \
	else \
	  echo "FAIL: include/aec_balanced_config.h is STALE against python/diag/gen_aec_balanced_config_h.py -- re-run the generator:" >&2; \
	  diff -u "$(CURDIR)/include/aec_balanced_config.h" "$$d/c_impl/include/aec_balanced_config.h" >&2 || true; \
	  exit 1; \
	fi

# Builds AND runs (like `selftest`/`test-counter-saturation`, not like
# `bench` which only builds) -- the permanent delay_aec_reset() regression
# test. Run from c_impl/ (same cwd convention as every other test target
# here); the test itself is self-contained (synthetic signals, no fixture
# file) so cwd only matters for consistency, not correctness.
.PHONY: test-delay-reset
test-delay-reset: $(TEST_DR_TARGET)
	"$(abspath $(TEST_DR_TARGET))"

$(TEST_DR_TARGET): $(TEST_DR_OBJS) | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(TEST_DR_OBJS) -lm

# Builds AND runs, same shape as test-delay-reset: the permanent regression
# for the delay-SELECTION seam. Self-contained (a synthetic dual-path signal,
# no fixture file), and it asserts its own premise -- that the dominant and
# pre-echo aggregators really did produce different candidates -- before
# asserting which one was reported.
.PHONY: test-delay-dominant-selection
test-delay-dominant-selection: $(TEST_DDS_TARGET)
	"$(abspath $(TEST_DDS_TARGET))"

$(TEST_DDS_TARGET): $(TEST_DDS_OBJS) | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(TEST_DDS_OBJS) -lm

# Builds AND runs. NEON-vs-scalar equivalence sweeps for delay_aec.c's two
# local kernels (argmax tie-order semantics + decimator carried IIR state);
# the argmax half is the ONLY check in the repo that can catch a
# last-match-wins regression -- see the test file's header.
.PHONY: test-delay-kernels
test-delay-kernels: $(TEST_DKN_TARGET)
	"$(abspath $(TEST_DKN_TARGET))"

$(TEST_DKN_OBJ): $(TEST_DIR)/test_delay_kernels.c | _cfg_guard
	$(CC) $(CFLAGS) -I. -MD -MP -c -o $@ $<

$(TEST_DKN_TARGET): $(TEST_DKN_OBJ) $(OBJ_DIR)/aec_scale.o | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(TEST_DKN_OBJ) $(OBJ_DIR)/aec_scale.o -lm

# Builds AND runs. Bit-exact contract of the matched-filter kernels (dot, NLMS
# update, fused update + next dot) and of da_matched_filter_core against a
# portable C model of the AArch64 NEON dataflow; on a build without the NEON
# body it checks the scalar kernels against the scalar model. Like
# test_delay_kernels it #includes src/delay_aec.c, so it links only
# aec_scale.o and needs the same per-object -I. / -MD rule.
.PHONY: test-delay-matched-neon
test-delay-matched-neon: $(TEST_DMN_TARGET)
	"$(abspath $(TEST_DMN_TARGET))"

$(TEST_DMN_OBJ): $(TEST_DIR)/test_delay_matched_neon.c | _cfg_guard
	$(CC) $(CFLAGS) -I. -MD -MP -c -o $@ $<

$(TEST_DMN_TARGET): $(TEST_DMN_OBJ) $(OBJ_DIR)/aec_scale.o | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(TEST_DMN_OBJ) $(OBJ_DIR)/aec_scale.o -lm

# Builds AND runs. NEON-vs-scalar twins of two kernels
# (first-maximum tie order; onset re-arm boundary). Includes the
# shipped source files, so it links only aec_scale.o, like test-delay-kernels.
.PHONY: test-neon-twins
test-neon-twins: $(TEST_NT_TARGET)
	"$(abspath $(TEST_NT_TARGET))"

$(TEST_NT_OBJ): $(TEST_DIR)/test_neon_twins.c | _cfg_guard
	$(CC) $(CFLAGS) -I. -MD -MP -c -o $@ $<

$(TEST_NT_TARGET): $(TEST_NT_OBJ) $(OBJ_DIR)/aec_scale.o | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(TEST_NT_OBJ) $(OBJ_DIR)/aec_scale.o -lm

# =============================================================================
# Phase 2 real rules: only defined once COMMON_LIB is known (see the dispatch
# block above). $(COMMON_LIB): FORCE always re-invokes audiolib's own build
# (a no-op if it's already up to date -- audiolib's own Makefile decides
# that), but only actually touches $(COMMON_LIB)'s mtime if audiolib really
# rebuilt something, so $(TARGET)/$(BENCH_TARGET) only relink when it did.
# =============================================================================
ifneq ($(COMMON_LIB),)
.PHONY: all bench FORCE test-counter-saturation test-rate-structural _run-rate-structural test-config-validation test-process-context test-avg-reverb-slots test-linear-context test-stream-sim test-res-nearend-gate test-runtime-preset test-delay-num-filters test-duty-census test-external-realign test-cli-delay-flags test-shared-far-spec test-spectral-reuse test-heap-alignment test-shared-fft-handle test-static-aec test-memory-layout test-reset-parity test-zero-heap test-aec-res test-detectors-parity test-detectors-parity-all test-parity-e2e pyc-parity-driver
all: $(TARGET)
bench: $(BENCH_TARGET)

# Builds AND runs (like `selftest` below, not like `bench` which only
# builds) -- this is the permanent regression test, meant to be run every
# time, not just built. Must be invoked from c_impl/ (the delay_aec
# section reads a checked-in fixture at a path relative to that cwd; see
# test/test_counter_saturation.c's own top comment) -- same cwd convention
# `make` recipes already run under.
test-counter-saturation: $(TEST_CS_TARGET)
	"$(abspath $(TEST_CS_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above.
# Re-invokes make with AEC_TESTING=1: check (d6) drives an otherwise-static
# internal through test/aec_test_hooks.h, and the recursion is what gives that
# build its own CFG_SIG-keyed obj/bin tree instead of contaminating the
# production objects. `_run-rate-structural` is the inner half; invoke
# `test-rate-structural`.
test-rate-structural:
	+@$(MAKE) --no-print-directory AEC_TESTING=1 _run-rate-structural

_run-rate-structural: $(TEST_RS_TARGET)
	"$(abspath $(TEST_RS_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above.
test-config-validation: $(TEST_CV_TARGET)
	"$(abspath $(TEST_CV_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above.
test-process-context: $(TEST_PC_TARGET)
	"$(abspath $(TEST_PC_TARGET))"

test-avg-reverb-slots: $(TEST_ARS_TARGET)
	"$(abspath $(TEST_ARS_TARGET))"

test-aec-res: $(TEST_AR_TARGET)
	"$(abspath $(TEST_AR_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above.
test-linear-context: $(TEST_LC_TARGET)
	"$(abspath $(TEST_LC_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above.
test-stream-sim: $(TEST_SS_TARGET)
	"$(abspath $(TEST_SS_TARGET))"

# Python<->C parity: checker + driver link rules, and the end-to-end gate recipe.
$(TEST_PE_TARGET): $(OBJS) $(TEST_PE_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "parity_aec_e2e [$(BACKEND)] -> $@"

$(PYC_RUN_TARGET): $(OBJS) $(PYC_RUN_OBJ) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "pyc_aec_run [$(BACKEND)] -> $@"

pyc-parity-driver: $(PYC_RUN_TARGET)

# Builds AND runs. Exit 0 = 16/48 kHz pass and the 8 kHz row is still the known red.
test-parity-e2e: $(TEST_PE_TARGET)
	@PYTHON="$(PYTHON)" bash "$(PYC_DIR)/run_parity_e2e.sh" "$(abspath $(TEST_PE_TARGET))" "$(abspath $(CURDIR)/..)" "$(abspath $(BIN_DIR))/parity_e2e"

test-res-nearend-gate: $(TEST_RNG_TARGET)
	"$(abspath $(TEST_RNG_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above.
test-runtime-preset: $(TEST_RP_TARGET)
	"$(abspath $(TEST_RP_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above.
test-delay-num-filters: $(TEST_DNF_TARGET)
	"$(abspath $(TEST_DNF_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above.
test-duty-census: $(TEST_DUC_TARGET)
	"$(abspath $(TEST_DUC_TARGET))"

test-external-realign: $(TEST_XRA_TARGET)
	"$(abspath $(TEST_XRA_TARGET))"

# CLI plumbing for --delay-mode / --delay-num-filters / --fixed-delay /
# --print-mem-size. Depends on the CLI binary itself (not a test_*.c), so it
# hangs off $(TARGET); the script re-derives the same bin dir via
# `make print-bin-dir`.
test-cli-delay-flags: $(TARGET) $(TEST_PMR_TARGET)
	"$(TEST_DIR)/cli_delay_flags.sh"

# Builds AND runs, same convention as test-counter-saturation above.
test-shared-far-spec: $(TEST_SFS_TARGET)
	"$(abspath $(TEST_SFS_TARGET))"

test-spectral-reuse: $(TEST_SRU_TARGET)
	"$(abspath $(TEST_SRU_TARGET))"

test-heap-alignment: $(TEST_HAL_TARGET)
	"$(abspath $(TEST_HAL_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above.
test-shared-fft-handle: $(TEST_SFH_TARGET)
	"$(abspath $(TEST_SFH_TARGET))"

test-memory-layout: $(OBJS) $(COMMON_LIB) | _cfg_guard
	$(CC) $(CFLAGS) -c -o "$(OBJ_DIR)/test_memory_layout.o" "$(TEST_DIR)/test_memory_layout.c"
	$(LINK) $(CFLAGS) -o "$(BIN_DIR)/test_memory_layout" "$(OBJ_DIR)/test_memory_layout.o" $(OBJS) $(COMMON_LIB) $(LDFLAGS)
	"$(abspath $(BIN_DIR)/test_memory_layout)"

test-static-aec: $(TEST_STATIC_TARGET)
	@test -f "$(TEST_STATIC_MIC)" || { echo "FATAL: TEST_STATIC_MIC not found: $(TEST_STATIC_MIC)" >&2; exit 1; }
	@test -f "$(TEST_STATIC_REF)" || { echo "FATAL: TEST_STATIC_REF not found: $(TEST_STATIC_REF)" >&2; exit 1; }
	"$(abspath $(TEST_STATIC_TARGET))" "$(TEST_STATIC_MIC)" "$(TEST_STATIC_REF)" "$(TEST_STATIC_SR)"

# Builds AND runs, same convention as test-counter-saturation above.
test-reset-parity: $(TEST_RPY_TARGET)
	"$(abspath $(TEST_RPY_TARGET))"

# Regenerates the golden from the Python reference every run, so the gate
# compares the CURRENT Python against the CURRENT C rather than a stale blob.
test-detectors-parity: $(TEST_DET_TARGET)
	python3 ../python/diag/gen_detectors_golden.py "$(TEST_DET_GOLDEN)" $(TEST_DET_HOP) $(TEST_DET_SR)
	"$(abspath $(TEST_DET_TARGET))" "$(TEST_DET_GOLDEN)"

# Permanent four-grid gate: legacy 8 kHz plus the three product grids.
test-detectors-parity-all: $(TEST_DET_TARGET)
	python3 ../python/diag/gen_detectors_golden.py "$(TEST_DET_GOLDEN)" 128 8000
	"$(abspath $(TEST_DET_TARGET))" "$(TEST_DET_GOLDEN)"
	python3 ../python/diag/gen_detectors_golden.py "$(TEST_DET_GOLDEN)" 128 16000
	"$(abspath $(TEST_DET_TARGET))" "$(TEST_DET_GOLDEN)"
	python3 ../python/diag/gen_detectors_golden.py "$(TEST_DET_GOLDEN)" 256 16000
	"$(abspath $(TEST_DET_TARGET))" "$(TEST_DET_GOLDEN)"
	python3 ../python/diag/gen_detectors_golden.py "$(TEST_DET_GOLDEN)" 512 48000
	"$(abspath $(TEST_DET_TARGET))" "$(TEST_DET_GOLDEN)"

test-zero-heap: $(TEST_ZH_TARGET)
	+$(MAKE) -C $(COMMON_DIR) $(COMMON_FWD) test_zero_heap
	@cd $(COMMON_DIR) && "$(abspath $(TEST_ZH_TARGET))"

FORCE:

$(COMMON_LIB): FORCE
	+$(MAKE) -C $(COMMON_DIR) $(COMMON_FWD) lib
	@test -f '$@' || { echo "ERROR: audiolib did not produce $@ (query/build divergence -- supported forwarded controls are BACKEND/SIMD/WERROR/CC/CXX/EXTRA_CFLAGS/NO_STDIO/TOOLCHAIN_CHECK)"; exit 1; }

# $(LINK), not $(CC): the C++ driver supplies the C++ runtime the NE10
# archive's one C++ TU needs. Same pattern as NR's/pipelines' link recipes.
$(TARGET): $(OBJS) $(EXAMPLE_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "aec_wav [$(BACKEND)] -> $@"

$(BENCH_TARGET): $(OBJS) $(BENCH_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "bench_rtf [$(BACKEND)] -> $@"

$(TEST_CS_TARGET): $(OBJS) $(TEST_CS_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_counter_saturation [$(BACKEND)] -> $@"

$(TEST_RS_TARGET): $(OBJS) $(TEST_RS_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_rate_structural [$(BACKEND)] -> $@"

$(TEST_CV_TARGET): $(OBJS) $(TEST_CV_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_config_validation [$(BACKEND)] -> $@"

$(TEST_PC_TARGET): $(OBJS) $(TEST_PC_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_process_context [$(BACKEND)] -> $@"

$(TEST_ARS_TARGET): $(OBJS) $(TEST_ARS_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_avg_reverb_slots [$(BACKEND)] -> $@"

$(TEST_AR_TARGET): $(filter-out $(OBJ_DIR)/aec_res.o,$(OBJS)) $(TEST_AR_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_aec_res [$(BACKEND)] -> $@"

$(TEST_LC_TARGET): $(OBJS) $(TEST_LC_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_linear_context [$(BACKEND)] -> $@"

$(TEST_SS_TARGET): $(OBJS) $(TEST_SS_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "stream_sim [$(BACKEND)] -> $@"

$(TEST_RNG_TARGET): $(OBJS) $(TEST_RNG_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_res_nearend_gate [$(BACKEND)] -> $@"

$(TEST_RP_TARGET): $(OBJS) $(TEST_RP_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_runtime_preset [$(BACKEND)] -> $@"

$(TEST_DNF_TARGET): $(OBJS) $(TEST_DNF_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_delay_num_filters [$(BACKEND)] -> $@"

$(TEST_DUC_TARGET): $(OBJS) $(TEST_DUC_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_duty_census [$(BACKEND)] -> $@"

$(TEST_XRA_TARGET): $(OBJS) $(TEST_XRA_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_external_realign [$(BACKEND)] -> $@"

$(TEST_PMR_TARGET): $(OBJS) $(TEST_PMR_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "print_mem_size_ref [$(BACKEND)] -> $@"

$(TEST_SFS_TARGET): $(OBJS) $(TEST_SFS_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_shared_far_spec [$(BACKEND)] -> $@"

$(TEST_SRU_TARGET): $(OBJS) $(TEST_SRU_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_spectral_reuse [$(BACKEND)] -> $@"

# The test includes the actual aec.c with allocation hooks.
$(TEST_HAL_TARGET): $(filter-out $(OBJ_DIR)/aec.o,$(OBJS)) $(TEST_HAL_OBJ) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST_SFH_TARGET): $(OBJS) $(TEST_SFH_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_shared_fft_handle [$(BACKEND)] -> $@"

$(TEST_DET_TARGET): $(TEST_DET_SRC) $(SRC_DIR)/detectors.c $(SRC_DIR)/aec_scale.c include/detectors.h | _cfg_guard
	@mkdir -p $(BIN_DIR)
#	$(CC), not $(LINK): LINK is $(CXX) so that NE10's C++ archive links, but
#	this target pulls in no archive at all -- feeding C sources with
#	-std=gnu99 to a C++ driver just fails.
	$(CC) $(CFLAGS) -o $@ $(TEST_DET_SRC) $(SRC_DIR)/detectors.c $(SRC_DIR)/aec_scale.c -lm
	@echo "parity_detectors [$(BACKEND)] -> $@"

$(TEST_STATIC_TARGET): $(OBJS) $(TEST_STATIC_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_static_aec [$(BACKEND)] -> $@"

$(TEST_RPY_TARGET): $(OBJS) $(TEST_RPY_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_reset_parity [$(BACKEND)] -> $@"

$(TEST_ZH_TARGET): $(OBJS) $(TEST_ZH_OBJS) $(COMMON_LIB) | _cfg_guard
	+$(MAKE) -C $(COMMON_DIR) $(COMMON_FWD) test_zero_heap
ifeq ($(shell uname -s),Darwin)
	@ac_bin="$$($(MAKE) -s -C $(COMMON_DIR) $(COMMON_FWD) print-bin-dir)"; \
	$(LINK) $(CFLAGS) -fno-builtin -o $@ $(OBJS) $(TEST_ZH_OBJS) $(COMMON_LIB) \
		-L"$$ac_bin" -lzero_heap_hook -Wl,-rpath,"$$ac_bin" $(LDFLAGS)
else
	@ac_obj="$$($(MAKE) -s -C $(COMMON_DIR) $(COMMON_FWD) print-obj-dir)"; \
	$(LINK) $(CFLAGS) -fno-builtin -o $@ $(OBJS) $(TEST_ZH_OBJS) \
		"$$ac_obj/zero_heap_hook.o" $(COMMON_LIB) $(LDFLAGS) \
		-Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free
endif
	@echo "test_zero_heap_aec [$(BACKEND)] -> $@"
endif

# --- Query targets -------------------------------------------------------
# print-obj-dir / print-lib-path report THIS invocation's own build identity
# (AEC's own flags; empty COMMON_PRODUCER_SIG unless COMMON_LIB was passed in) -- they
# never shell out to audiolib, matching where a bare `make lib` /
# `make selftest` actually places its artifacts.
# print-bin-dir is different: the binaries (`aec_wav`, `bench_rtf`) only ever
# exist in the RESOLVED-producer directory `make all`/`make bench` deliver
# into, so a bare `make -s print-bin-dir` dispatches through the same phase-1
# resolution as `all`/`bench` and prints that exact directory. (Passing
# COMMON_LIB= explicitly skips the extra query sub-make, same protocol as
# phase 1.)
.PHONY: print-bin-dir print-obj-dir print-lib-path
ifeq ($(COMMON_LIB),)
print-bin-dir:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audiolib archive path (COMMON_DIR=$(COMMON_DIR))" >&2; exit 1; }; \
	 $(MAKE) -s --no-print-directory print-bin-dir COMMON_LIB="$$ac"
else
print-bin-dir:
	@echo $(abspath $(BIN_DIR))
endif
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))
print-lib-path:
	@echo $(abspath $(LIB_TARGET))

# =============================================================================
# publish v4: immutable, content-addressed release under $(DIST_ROOT) --
# same design as audiolib's publish; see that Makefile for the full
# layout/semantics writeup (deterministic MANIFEST + byte-verified
# idempotent republish incl. MANIFEST tamper detection, rename(2)-atomic
# `current` swap via the swapln helper built with $(HOSTCC)). Design notes
# (mirrored from audiolib verbatim, with one AEC-specific extension
# noted below):
#   - Dry-run guard: -n/-q/-t never touch $(DIST_ROOT), the publish lock, or
#     any stage/helper/MANIFEST/ATTEST write (see the MAKEFLAGS word-scan in
#     the `publish` recipe below).
#   - ATTEST is one-event-one-file: every publish event (including an
#     idempotent republish) writes exactly one NEW
#     attest-<utc>-<commit>[-dirty]-<seq>.txt, installed via the swapln
#     helper's `--excl-install` mode (write-temp + link(2), the atomic
#     no-clobber equivalent of O_CREAT|O_EXCL -- link(2) fails EEXIST if the
#     name is already taken). A same-second name collision regenerates the
#     event id with the next <seq> and retries; an existing attestation is
#     NEVER overwritten (a prior version used `mv -f`, so same-second
#     republishes could silently clobber the prior record). `git_commit` is
#     always the FULL 40-hex object id (a prior version used
#     `git rev-parse --short`).
#   - AEC-specific: the attestation ALSO records the audiolib PRODUCER's
#     own commit/dirty/untracked state (`audiolib_git_commit`/
#     `audiolib_git_dirty`[`/audiolib_dirty_diff_sha256`]/
#     `audiolib_git_untracked`[`/audiolib_untracked_tree_sha256`]) --
#     this repo's release links against audiolib's archive, so "which
#     source state produced this release" spans both repos, not just this
#     one. $(COMMON_DIR) is guarded (`[ -n ... ] && [ -d ... ]`) before use: its
#     wildcard default (see COMMON_DIR above) can resolve to the EMPTY string,
#     and `git -C '' ...` silently operates on the CURRENT (i.e. THIS) repo
#     instead of failing -- which would misreport audiolib's identity
#     as this repo's own. The guard makes an unresolved COMMON_DIR report
#     audiolib_git_commit=unknown (and audiolib_git_untracked=unknown)
#     instead.
#   - Dirty/untracked policy (tracked and untracked changes are checked as
#     two separate dimensions, with identity refusal as its own
#     unconditional gate): publish FATALs by default when EITHER this
#     checkout OR the resolved audiolib PRODUCER checkout (a) has
#     uncommitted TRACKED changes (`git status --porcelain -uno` non-empty),
#     (b) contains ANY untracked, non-ignored file (`git ls-files --others
#     --exclude-standard` non-empty; gitignored build output is excluded by
#     design and never part of this provenance), or (c) has no git identity
#     at all -- the FATAL names exactly which repo(s) and which dimension
#     triggered the refusal. (c) is checked, and FATALs, BEFORE (a)/(b) are
#     even computed, and is UNCONDITIONAL for BOTH repos -- no escape hatch
#     admits an identity-less checkout on either side, because neither the
#     commit nor the untracked enumeration can be named for it; the identity
#     check is its own unconditional gate and no knob, in particular not
#     ALLOW_DIRTY_PUBLISH=1, reaches it). The two escape hatches for (a)/(b)
#     remain separate and
#     orthogonal: ALLOW_DIRTY_PUBLISH=1 admits tracked changes and records
#     dirty_diff_sha256 (sha256 of `git diff --binary HEAD`) for whichever
#     repo(s) are dirty; ALLOW_UNTRACKED_PUBLISH=1 admits untracked files
#     and records untracked_tree_sha256 (this repo) /
#     audiolib_untracked_tree_sha256 (the producer) -- sha256 over
#     sorted FIXED-FIELD records (see the knob comment up top for the exact
#     encoding and its fail-closed I/O semantics) -- the tracked diff hash
#     says nothing about untracked bytes, so two different untracked source
#     states must not share a provenance record. An untracked path that is
#     neither a regular file nor a symlink (an embedded git checkout, a
#     fifo, ...) cannot be hashed deterministically and is always refused,
#     by name, naming which repo it came from.
#   - `-t` tightening: `make -t publish` prints a one-line no-op note and
#     exits 0 WITHOUT recursing into `_publish_impl` (recursing under -t
#     would let GNU make's standard touch semantics bump the mtimes of
#     libaec.a/aec_wav/bench_rtf -- and, via the COMMON_LIB dispatch,
#     audiolib's own archive -- real writes for a phony ACTION target
#     that produces no build product of its own). -n and -q are unchanged.
#   - Attestations remain UNSIGNED: they provide traceability, not
#     authenticity -- anyone with filesystem access could forge one; do not
#     treat ATTEST/ as tamper-proof under an attacker model (MANIFEST
#     byte-verification is the integrity check; ATTEST is the provenance
#     log).
# `publish` is the locked driver; _publish_impl is INTERNAL (direct
# invocation skips the lock) and goes through the same two-phase COMMON_LIB
# dispatch as `all`/`bench`.
# =============================================================================
PUBLISH_ARTIFACTS = $(LIB_TARGET) $(TARGET) $(BENCH_TARGET)
DIST_BACKEND_DIR  = $(DIST_ROOT)/$(BACKEND)
DIST_LOCK         = $(DIST_ROOT)/.lock.$(BACKEND)

.PHONY: publish
# Dry-run guard: this recipe mentions $(MAKE), so GNU make executes it for
# real under -n/-q/-t (all three share that rule) -- which, unguarded, would
# mkdir $(DIST_ROOT) persistently and take/release the real publish lock on
# every
# dry run. The word-scan below detects those flags in MAKEFLAGS and
# branches:
#   -t  prints a one-line note and exits 0 WITHOUT recursing (recursing
#       would let standard touch semantics bump the mtimes of
#       libaec.a/aec_wav/bench_rtf (and, via the COMMON_LIB dispatch,
#       audiolib's own archive) -- real writes. publish is a phony
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
# the explicit skips). Normal invocations keep the lock-FIRST discipline
# unchanged. See audiolib's Makefile for the full writeup (identical
# mechanism).
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

ifeq ($(COMMON_LIB),)
.PHONY: _publish_impl
_publish_impl:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audiolib archive path" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory _publish_impl COMMON_LIB="$$ac"
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
	  echo "  content, so NO escape hatch admits this (ALLOW_DIRTY_PUBLISH used to). Publish" >&2; \
	  echo "  from a real checkout or a scratch clone with a disposable commit instead." >&2; \
	  exit 1; \
	fi; \
	dirty=0; [ -n "$$(git status --porcelain -uno 2>/dev/null)" ] && dirty=1; \
	unt_list="$$(git ls-files --others --exclude-standard 2>/dev/null)"; \
	untracked=0; [ -n "$$unt_list" ] && untracked=1; \
	if [ -n "$(COMMON_DIR)" ] && [ -d "$(COMMON_DIR)" ]; then \
	  ac_commit="$$(git -C $(COMMON_DIR) rev-parse HEAD 2>/dev/null || echo unknown)"; \
	else \
	  ac_commit=unknown; \
	fi; \
	if [ "$$ac_commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- audiolib has no git identity here (not a git checkout, or COMMON_DIR did not resolve to a real directory: COMMON_DIR=$(COMMON_DIR))." >&2; \
	  echo "  Without an identity the attestation cannot name the source state OR enumerate untracked" >&2; \
	  echo "  content, so NO escape hatch admits this (ALLOW_DIRTY_PUBLISH used to). Publish" >&2; \
	  echo "  against a real audiolib checkout, or point COMMON_DIR at a scratch clone with a" >&2; \
	  echo "  disposable commit, instead." >&2; \
	  exit 1; \
	fi; \
	ac_dirty=0; [ -n "$$(git -C $(COMMON_DIR) status --porcelain -uno 2>/dev/null)" ] && ac_dirty=1; \
	ac_unt_list="$$(git -C $(COMMON_DIR) ls-files --others --exclude-standard 2>/dev/null)"; \
	ac_untracked=0; [ -n "$$ac_unt_list" ] && ac_untracked=1; \
	adp=0; [ "$(ALLOW_DIRTY_PUBLISH)" = "1" ] && adp=1; \
	aup=0; [ "$(ALLOW_UNTRACKED_PUBLISH)" = "1" ] && aup=1; \
	violation=0; [ "$$dirty" = "1" ] && violation=1; [ "$$ac_dirty" = "1" ] && violation=1; \
	if [ "$$adp" != "1" ] && [ "$$violation" = "1" ]; then \
	  [ "$$dirty" = "1" ] && echo "FATAL: publish refused -- aec working tree has uncommitted TRACKED changes." >&2; \
	  [ "$$ac_dirty" = "1" ] && echo "FATAL: publish refused -- audiolib working tree has uncommitted TRACKED changes." >&2; \
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
	    echo "FATAL: publish refused -- audiolib contains UNTRACKED files (they can change the artifact without a trace in the tracked diff):" >&2; \
	    printf '%s\n' "$$ac_unt_list" | head -5 | sed 's/^/    /' >&2; \
	  fi; \
	  echo "  Track/remove them for a board deliverable, or pass ALLOW_UNTRACKED_PUBLISH=1 -- the escape" >&2; \
	  echo "  hatch is recorded in the attestation with untracked_tree_sha256 over their exact contents." >&2; \
	  exit 1; \
	fi; \
	ddiff=""; [ "$$dirty" = "1" ] && ddiff="$$(git diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	ac_ddiff=""; [ "$$ac_dirty" = "1" ] && ac_ddiff="$$(git -C $(COMMON_DIR) diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
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
	  printf '%s\n' "$$ac_unt_list" | hash_untracked "$(COMMON_DIR)/" > "$$work/ac_untracked.records"; \
	  if grep -q '^X ' "$$work/ac_untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in audiolib cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/ac_untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  ac_uhash="$$(shasum -a 256 < "$$work/ac_untracked.records" | cut -d' ' -f1)"; \
	fi; \
	dateutc="$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	stamp='$(ATTEST_STAMP)'; [ -n "$$stamp" ] || stamp="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	$(HOSTCC) -O2 -o "$$work/swapln" $(COMMON_DIR)/tools/atomic_symlink_swap.c || { echo "FATAL: could not build the atomic symlink-swap helper with HOSTCC=$(HOSTCC) (pass HOSTCC=<host compiler>)" >&2; exit 1; }; \
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
	  echo "producer_cfg_sig=$(COMMON_PRODUCER_SIG)"; \
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
	    echo "audiolib_git_commit=$$ac_commit"; echo "audiolib_git_dirty=$$ac_dirty"; \
	    [ -n "$$ac_ddiff" ] && echo "audiolib_dirty_diff_sha256=$$ac_ddiff"; \
	    echo "audiolib_git_untracked=$$ac_untracked"; \
	    [ -n "$$ac_uhash" ] && echo "audiolib_untracked_tree_sha256=$$ac_uhash"; \
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
# audit-no-stdio: verifies a NO_STDIO=1 libaec.a is genuinely stdio-free.
# Self-contained recipe-time sub-makes (same pattern as
# `publish`'s COMMON_LIB-unset branch above) -- builds its own NO_STDIO=1 `lib`
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
#      stdio references under NO_STDIO=1.
#   4. minimal-main ELF gate -- link test/no_stdio_main.c (itself stdio-free:
#      no <stdio.h>, no printf) against the NO_STDIO=1 libaec.a + the
#      resolved audiolib archive, `nm -A` the EXECUTABLE, then run it.
#      The executable must be FULLY stdio-free -- zero hits, same strict set
#      as the archive check (the NE10 backend carries no stdio references).
#   5. Run the linked executable; exit code must be 0 (aec_create +
#      one hop of aec_process + aec_destroy all succeeded with no stdio
#      call of its own).
# =============================================================================
AUDIT_STRICT_ALL  = fprintf|vfprintf|printf|puts|fputs|fputc|fwrite|stderr|__stderrp
AUDIT_STRICT_OWN  = fprintf|vfprintf|printf|puts|fputs
AUDIT_BIN_DIR     = $(BIN_ROOT)/audit-no-stdio/$(BACKEND)

# Test-only symbols must not ship. Checked in BOTH directions: a one-way check
# passes just as happily against a library that stopped defining them at all,
# which would silently disable check (d6) rather than fail it. That reverse
# check earned its keep immediately -- the first pattern here was ' aec_testing_'
# and matched nothing on Darwin, where Mach-O nm emits a leading underscore, so
# the production-side check could not have failed on a real leak either. The
# ' _?' makes it work on both Mach-O and ELF.
.PHONY: test-no-testing-symbols
test-no-testing-symbols:
	@echo "== test-no-testing-symbols: BACKEND=$(BACKEND) CC=$(CC) =="
	+@$(MAKE) --no-print-directory AEC_TESTING=0 BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' lib
	+@$(MAKE) --no-print-directory AEC_TESTING=1 BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' lib
	@prod="$$($(MAKE) -s --no-print-directory AEC_TESTING=0 BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' print-lib-path)"; \
	 test_="$$($(MAKE) -s --no-print-directory AEC_TESTING=1 BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' print-lib-path)"; \
	 echo "-- production lib:  $$prod"; \
	 echo "-- AEC_TESTING lib: $$test_"; \
	 test "$$prod" != "$$test_" || { echo "FATAL: AEC_TESTING=1/0 resolved to the SAME path -- CFG_SIG key-isolation broken, the two builds would share objects" >&2; exit 1; }; \
	 echo "PASS: key-isolation"; \
	 hits="$$(nm -g "$$prod" 2>/dev/null | grep -E ' _?aec_testing_' || true)"; \
	 if [ -n "$$hits" ]; then \
	   echo "FATAL: test-only symbols exported by the PRODUCTION library:" >&2; echo "$$hits" >&2; exit 1; \
	 fi; \
	 echo "PASS: no aec_testing_* symbol in the production archive"; \
	 want="$$(nm -g "$$test_" 2>/dev/null | grep -cE ' _?aec_testing_' || true)"; \
	 if [ "$$want" -lt 1 ]; then \
	   echo "FATAL: AEC_TESTING=1 archive exports NO aec_testing_* symbol -- the check above cannot fail, and the hooks it guards are gone" >&2; exit 1; \
	 fi; \
	 echo "PASS: AEC_TESTING archive exports $$want test-only symbol(s)"

.PHONY: audit-profile
# audit-profile -- the PROFILE knob keys its own build AND actually changes
# what gets linked, asserted in BOTH directions so the check cannot pass
# vacuously. The failure mode it guards is silent by construction: a build
# that quietly loses the flag reports a full table of zeros, which reads as
# "these stages cost nothing" rather than as "nobody measured".
audit-profile:
	@echo "== audit-profile: BACKEND=$(BACKEND) SIMD=$(SIMD) CC=$(CC) =="
	@stub='-DAEC_NOW_US=aec_test_stub_now_us -include $(TEST_DIR)/profile_timer_stub.h'; \
	 p0="$$($(MAKE) -s --no-print-directory PROFILE=0 print-lib-path)"; \
	 p1="$$($(MAKE) -s --no-print-directory PROFILE=1 print-lib-path)"; \
	 ps="$$($(MAKE) -s --no-print-directory PROFILE=1 EXTRA_CFLAGS="$$stub" print-lib-path)"; \
	 px="$$($(MAKE) -s --no-print-directory PROFILE=1 EXTRA_CFLAGS=-DAEC_AUDIT_PROFILE_UNRELATED print-lib-path)"; \
	 test "$$p0" != "$$p1" || { echo "FATAL: PROFILE=0/1 resolved to the SAME path -- CFG_SIG key-isolation broken" >&2; exit 1; }; \
	 test "$$p1" != "$$ps" || { echo "FATAL: substituting the stamp did not re-key the build" >&2; exit 1; }; \
	 echo "PASS: key-isolation (PROFILE=0, PROFILE=1 and stamp-substituted all differ)"; \
	 $(MAKE) --no-print-directory PROFILE=0 lib >/dev/null; \
	 $(MAKE) --no-print-directory PROFILE=1 lib >/dev/null; \
	 $(MAKE) --no-print-directory PROFILE=1 EXTRA_CFLAGS="$$stub" lib >/dev/null; \
	 $(MAKE) --no-print-directory PROFILE=1 EXTRA_CFLAGS=-DAEC_AUDIT_PROFILE_UNRELATED lib >/dev/null; \
	 n0="$$(nm -u "$$p0" 2>/dev/null | grep -c clock_gettime || true)"; \
	 n1="$$(nm -u "$$p1" 2>/dev/null | grep -c clock_gettime || true)"; \
	 ns="$$(nm -u "$$ps" 2>/dev/null | grep -c clock_gettime || true)"; \
	 nx="$$(nm -u "$$px" 2>/dev/null | grep -c clock_gettime || true)"; \
	 echo "-- clock_gettime references: PROFILE=0 $$n0 | PROFILE=1 $$n1 | stamp substituted $$ns | PROFILE=1 with an unrelated EXTRA_CFLAGS $$nx"; \
	 test "$$n0" -eq 0 || { echo "FATAL: PROFILE=0 archive links a clock" >&2; exit 1; }; \
	 test "$$n1" -gt 0 || { echo "FATAL: PROFILE=1 archive links NO clock -- the flag did not reach the compile, so every other assertion here is vacuous" >&2; exit 1; }; \
	 test "$$ns" -eq 0 || { echo "FATAL: the substituted stamp still links CLOCK_MONOTONIC" >&2; exit 1; }; \
	 test "$$nx" -gt 0 || { echo "FATAL: a command-line EXTRA_CFLAGS dropped PROFILE=1 -- the knob is being silently overridden" >&2; exit 1; }; \
	 echo "PASS: PROFILE=0 links no clock, PROFILE=1 does, substitution removes it, and a command-line EXTRA_CFLAGS does not defeat it"
	@# Value-level half: WHERE the stamps sit, not just whether a clock is
	@# linked. A counting stamp turns each window into the number of stamps it
	@# spans, so a stage whose opening stamp went missing reports a different
	@# integer instead of the 0 that is indistinguishable from "cost nothing".
	@cnt='-DAEC_NOW_US=aec_test_count_now_us -include $(TEST_DIR)/profile_timer_count.h'; \
	 pc="$$($(MAKE) -s --no-print-directory PROFILE=1 EXTRA_CFLAGS="$$cnt" print-lib-path)"; \
	 ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audiolib archive path (COMMON_DIR=$(COMMON_DIR))" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory PROFILE=1 EXTRA_CFLAGS="$$cnt" lib >/dev/null; \
	 $(MAKE) --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) lib >/dev/null; \
	 mkdir -p "$(BIN_DIR)"; \
	 $(CC) $(CFLAGS) -c -o "$(BIN_DIR)/profile_stamp_topology.o" \
	     $(TEST_DIR)/profile_stamp_topology.c || exit 1; \
	 $(LINK) $(CFLAGS) -o "$(BIN_DIR)/profile_stamp_topology" \
	     "$(BIN_DIR)/profile_stamp_topology.o" "$$pc" "$$ac" -lm || exit 1; \
	 "$(BIN_DIR)/profile_stamp_topology"

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
	 $(MAKE) --no-print-directory -C $(COMMON_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 lib; \
	 ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 print-lib-path)"; \
	 test -n "$$ac" && test -f "$$ac" || { echo "FATAL: could not resolve/build audiolib archive (COMMON_DIR=$(COMMON_DIR))" >&2; exit 1; }; \
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
	   echo "FATAL: stdio-family symbols found in linked executable (audiolib or toolchain side):" >&2; echo "$$exe_hits" >&2; exit 1; \
	 fi; \
	 echo "PASS: linked executable is FULLY stdio-free (zero fprintf/vfprintf/printf/puts/fputs/fputc/fwrite/stderr/__stderrp, defined or undefined)"; \
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

# debug delegates to `all` with -g -DAEC_DEBUG folded into EXTRA_CFLAGS:
# the old `debug: CFLAGS += -g -DAEC_DEBUG` mutated
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
	@echo "  test-delay-reset  Build + run the permanent delay_aec_reset() consistency"
	@echo "                  regression test (test_delay_reset) -- run from c_impl/"
	@echo "  test-rate-structural  Build + run the permanent per-rate structural"
	@echo "                  regression test (test_rate_structural) -- run from c_impl/"
	@echo "  test-avg-reverb-slots  Build + run the average-render-reverb ring-slot test"
	@echo "  test-process-context  Build + run the permanent context-only-processing"
	@echo "                  regression test (test_process_context) -- run from c_impl/"
	@echo "  test-delay-num-filters  Build + run the permanent matched-filter-bank-size"
	@echo "                  regression test (test_delay_num_filters) -- run from c_impl/"
	@echo "  test-duty-census  Build + run the permanent matched-filter duty-cycle"
	@echo "                  engagement census test (test_duty_census) -- run from c_impl/"
	@echo "  test-cli-delay-flags  Run the aec_wav --delay-mode/--delay-num-filters/"
	@echo "                  --fixed-delay/--print-mem-size CLI plumbing test"
	@echo "                  (test/cli_delay_flags.sh) -- run from c_impl/"
	@echo "  test-shared-far-spec  Build + run the permanent shared-far-end-FFT"
	@echo "                  regression test (test_shared_far_spec) -- run from c_impl/"
	@echo "  test-spectral-reuse  Build + run spectrum-power reuse and fallback regression"
	@echo "  test-heap-alignment  Check aligned arena allocation and failure cleanup"
	@echo "  test-shared-fft-handle  Build + run the permanent shared-FFT-handle"
	@echo "                  regression test (test_shared_fft_handle) -- run from c_impl/"
	@echo "  test-static-aec  Build + run heap-vs-caller-pool byte-equality on a real WAV pair"
	@echo "                  (override TEST_STATIC_MIC/TEST_STATIC_REF/TEST_STATIC_SR as needed)"
	@echo "  test-reset-parity  Build + run the permanent aec_reset()-equals-fresh-instance"
	@echo "                    byte-identity gate (3 grids x full-res/ctx-only)"
	@echo "  test-zero-heap  Prove caller-pool init/process/reset/destroy performs no allocation"
	@echo "  test-parity-e2e  Python<->C end-to-end gate (parity_aec_e2e) on goldens regenerated from the pinned"
	@echo "                  doubletalk wavs; 16k/48k must pass, 8k is the known EXPECTED_RED(S-19) (16k/512 not covered)"
	@echo "  test-pyc-parity  Python<->C effect parity, two layers (std-math C vs Python, ship vs std-math C); PYC_ARGS=--quick"
	@echo "  pyc-parity-driver  Build test/pyc_parity/aec_run.c (the C side of test-pyc-parity) for this configuration"
	@echo "  test-neon-twins  NEON-vs-scalar twins: rsa_argmax_first tie order, onset re-arm boundary"
	@echo "  test-delay-kernels  NEON-vs-scalar sweeps of delay_aec.c's kernels: argmax tie order, decimator and"
	@echo "                  48 kHz anti-alias pair (the only gate that sees the pair kernel; scalar vs itself with SIMD=0)"
	@echo "  test-delay-matched-neon  Bit-exact matched-filter kernels (dot, NLMS update, fused update + next dot, core)"
	@echo "                  vs a portable model of the AArch64 NEON dataflow; scalar kernels vs the scalar model otherwise"
	@echo "  test-stream-sim  Streaming API: FIFO capacity rule, lockstep == aec_process, exact-count overrun,"
	@echo "                  underrun recovery, consumer catch-up (all grids)"
	@echo "  test-aec-res    Standalone RES stage (aec_res.h): lane-equal config, API contract, render scale, initial phase"
	@echo "  test-header-fresh  Regenerate include/aec_balanced_config.h and diff it against the checked-in copy"
	@echo "  test-detectors-parity  Gate detectors.c against the Python reference (regenerates the golden)"
	@echo "  test-detectors-parity-all  Run detector parity on legacy 8 kHz and all three product grids"
	@echo "  debug   Build with -g -DAEC_DEBUG"
	@echo "  publish Copy this build's artifacts to dist/<backend>/<config-hash>/ + update dist/<backend>/current"
	@echo "  audit-profile   Assert PROFILE keys its own build and actually changes what"
	@echo "                  is linked, in both directions -- run from c_impl/"
	@echo "  audit-no-stdio  Build lib with NO_STDIO=1 and verify it (and a minimal linked"
	@echo "                  binary) carry no fprintf/vfprintf/stderr/etc -- see Makefile comment"
	@echo "  clean   Remove obj/ and bin/ (dist/ survives -- publish is a deliberate release, not a build cache)"
	@echo ""
	@echo "Usage:"
	@echo "  make print-bin-dir   # -> this invocation's bin/<backend>-<config-hash>/"
	@echo "  make print-lib-path  # -> this invocation's libaec.a path"
	@echo "  (to get the path 'make all' itself will produce, first resolve"
	@echo "   COMMON_LIB the same way phase 1 does -- see the two-phase dispatch"
	@echo "   comment in this Makefile -- and pass it: make print-bin-dir COMMON_LIB=\"...\")"
	@echo "  ./bin/<backend>-<config-hash>/aec_wav <mic.wav> <ref.wav> <out.wav> [options]"
	@echo "  ./bin/<backend>-<config-hash>/bench_rtf [--seconds N] [--repeat M] [--preset 0|1|2|3|mild|balanced|aggressive|no_floor] [--sr HZ] [--fft N] [--stage full|linear|both] [--scene noise|bursts]"
	@echo "  make COMMON_DIR=<path>       (override audiolib location)"
	@echo "  make WERROR=1            (opt-in: -Werror for this repo's own sources)"
	@echo "  make lib NO_STDIO=1      (opt-in: stdio-free libaec.a for board/no-stdio builds --"
	@echo "                            only the 'lib' goal; CLI/bench are not built under this flag)"
	@echo ""
	@echo "Objects/binaries live in obj|bin/<backend>-<config-hash>/ -- switching"
	@echo "BACKEND, SIMD, PROFILE, EXTRA_CFLAGS, or WERROR always lands in a fresh directory"
	@echo "automatically; no manual clean needed, and two differently-configured"
	@echo "builds can run concurrently in this same tree."

dirs: _cfg_guard

.PHONY: all dirs clean debug help

# --- Header dependency tracking ------------------------------------------
# See audiolib's Makefile for the full rationale (-MD not -MMD, -MP
# guards against deleted-header stale-.d errors). $(wildcard ...) makes this
# evaluate to nothing (not an error) before any objects exist yet for this
# CFG_SIG.
-include $(wildcard $(OBJ_DIR)/*.d)
