# AEC C Implementation — Makefile

CC      ?= gcc
CXX     ?= c++
AR      ?= ar
RANLIB  ?= ranlib

# CXXFLAGS and CPPFLAGS are consumed by no recipe here (no C++ TU) and DEBUG gates nothing (`make debug` is its own
# recipe); they stay in the CFG_SIG payload so its shape matches audiolib's and NR's. CXXFLAGS/CPPFLAGS use `+=` so an
# environment value still flips their origin to "environment override" under `make -e`, which the override check below
# relies on. NO_STDIO=1 adds -DAEC_NO_STDIO and drops src/aec_debug.c (see the NO_STDIO block).
CXXFLAGS      +=
CPPFLAGS      +=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

# TOOLCHAIN_CHECK=0 skips the CC/CXX -dumpmachine coherence check in _cfg_guard. It is part of CFG_SIG.
TOOLCHAIN_CHECK ?= 1

# DIST_ROOT / HOSTCC / OBJ_ROOT / BIN_ROOT: where `make publish` writes releases, the host compiler for the symlink-swap
# helper (never $(CC), which may be a cross compiler), and where the keyed obj/bin trees live. None is in CFG_SIG.
# `make publish` refuses a checkout (this one or audiolib's, via COMMON_DIR) with no git identity (unconditional), with tracked
# changes (ALLOW_DIRTY_PUBLISH=1 admits and records a sha256 of `git diff --binary HEAD`) or with untracked non-ignored
# files (ALLOW_UNTRACKED_PUBLISH=1 admits and records a sha256 over sorted per-file records). An untracked path that cannot
# be hashed is always refused. ATTEST_STAMP overrides the UTC stamp in the attestation file name (tests only).
DIST_ROOT ?= dist
HOSTCC    ?= cc
ALLOW_DIRTY_PUBLISH ?= 0
ALLOW_UNTRACKED_PUBLISH ?= 0
ATTEST_STAMP ?=

# PROFILE=1 enables the per-hop stage timing (aec_get_last_timing(), aec.h). It is off by default so a release build
# takes no clock reads, and it is part of CFG_SIG so profile and release builds use separate obj/bin directories. The
# spelling matches the pipeline Makefiles. A libc without POSIX CLOCK_MONOTONIC combines it with -DAEC_NOW_US=<fn> (see
# src/aec.c).
PROFILE ?= 0
ifneq ($(words $(PROFILE)),1)
$(error PROFILE must be exactly 0 or 1, got '$(PROFILE)')
endif
ifneq ($(filter-out 0 1,$(PROFILE)),)
$(error PROFILE must be exactly 0 or 1, got '$(PROFILE)')
endif
# Passed through PROFILE_CFLAGS, like SIMD below, not EXTRA_CFLAGS: EXTRA_CFLAGS is forwarded to the audiolib sub-makes and
# keyed into CFG_SIG, so the -D would reach them twice and re-key producers that never read the macro.
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

# The shared DSP layer (FFT backend, fast_math) lives in audiolib. COMMON_DIR finds it next to this repo (SE/audiolib) or
# four levels up from a nested checkout (e.g. an Audio_ALG submodule).
COMMON_DIR ?= $(firstword $(wildcard ../../audiolib/common ../../../../audiolib/common))

# FFT backend: NE10 R2C/C2R is the only backend. It computes in float32 (about 1e-5 vs numpy's fp64 np.fft); the non-FFT
# C logic is bit-exact to Python. SIMD=0 routes NE10 through its scalar C kernels. BACKEND stays a variable because it
# names the keyed obj/bin directories.
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): the KISS FFT backend has been removed)
endif

# Print the banner except for print-* goals, whose stdout callers capture with `$(make -s ... print-lib-path)`.
ifeq ($(filter print-%,$(MAKECMDGOALS)),)
$(info AEC: FFT backend = $(BACKEND))
endif

CFLAGS  += -I$(COMMON_DIR)/include -I$(COMMON_DIR)

# The audiolib archive carries one C++ TU (NE10's generic-radix kernel), so binaries link with the C++ driver, which
# supplies the C++ runtime. Part of the CFG_SIG payload.
LINK = $(CXX)

# WERROR=1 adds -Werror (default off). Every object built here is this repo's own code; vendored FFT code lives in
# audiolib.
WERROR ?= 0
ifeq ($(WERROR),1)
  CFLAGS += -Werror
endif

# AEC_TESTING=1 compiles the test-only hooks declared in test/aec_test_hooks.h. They reach internals the public API cannot
# observe and must not ship. It is appended to CFLAGS before CFG_SIG is computed, so a testing build never shares objects
# with a production build; `make test-no-testing-symbols` checks the key isolation and the symbols' absence from, and
# presence in, each archive.
AEC_TESTING ?= 0
ifeq ($(AEC_TESTING),1)
  CFLAGS += -DAEC_TESTING
endif

# NO_STDIO=1 builds a stdio-free libaec.a: src/aec_debug.c (the only stdio TU) is dropped and the per-frame trace call in
# src/aec.c is compiled out under AEC_NO_STDIO. Set before CFG_SIG is computed so it keys its own obj/bin directory. Only the
# library (`make lib`) is built this way: the CLI (example/aec_wav.c) uses the FILE*-based debug API. See aec_debug.h and
# the `audit-no-stdio` target.
ifeq ($(NO_STDIO),1)
  CFLAGS += -DAEC_NO_STDIO
endif

# AEC_NO_ARRAY=1 builds the single-channel library: -DAEC_NO_ARRAY, no src/aec_array.c and no src/aec_res.c (the four-microphone
# array instance and its gain stage), and AecConfig.n_mics other than 1 is rejected by aec_resolve_valid_config() / aec_init() /
# aec_create(). aec_get_array(), the aec_array_*() diagnostics and the lane entry points are then not defined. The array-only test
# targets (test-aec-array, test-aec-array-2, test-aec-array-flow, test-aec-res) are skipped with a message. Set before CFG_SIG is
# computed so it keys its own obj/bin directory.
AEC_NO_ARRAY ?= 0
ifneq ($(words $(AEC_NO_ARRAY)),1)
$(error AEC_NO_ARRAY must be exactly 0 or 1, got '$(AEC_NO_ARRAY)')
endif
ifneq ($(filter-out 0 1,$(AEC_NO_ARRAY)),)
$(error AEC_NO_ARRAY must be exactly 0 or 1, got '$(AEC_NO_ARRAY)')
endif
ifeq ($(AEC_NO_ARRAY),1)
  CFLAGS += -DAEC_NO_ARRAY
endif

# FP-contraction policy, shared by all four repos: FP_POLICY is appended to CFLAGS last, after every other append and
# before CFG_SIG is computed, so an EXTRA_CFLAGS=-ffp-contract=fast (or -Ofast/-ffast-math) cannot win. It keeps scalar and
# NEON operation order stable and lets audiolib's guarded AArch64 sqrtf lower to FSQRT without an errno-checking libm
# fallback. It is a dependency-free `:=` literal defined here so its origin is final before the override-rejection foreach
# below queries it (needed for `make -e`). See audiolib/common/scripts/audit_fp_contract.sh for the disassembly check.
FP_POLICY := -ffp-contract=off -fno-math-errno

# A bare `:=` literal for the same `make -e` reason as FP_POLICY; the allow-list below uses it.
FP_ALLOWED_CHARS_RE := ^[A-Za-z0-9_./=,+ -]*$$

# A command-line CFLAGS=/CXXFLAGS=/CPPFLAGS=/LDFLAGS=/FP_POLICY=/FP_ALLOWED_CHARS_RE= would silently drop this file's own
# appends (-ffp-contract=off, -DAEC_NO_STDIO, the -I paths, -lm) or widen the allow-list below, so it is rejected.
# EXTRA_CFLAGS / EXTRA_LDFLAGS are the supported hooks.
$(foreach v,CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FP_POLICY FP_ALLOWED_CHARS_RE,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy, NO_STDIO, include paths); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead)))

# Reject -Ofast, -ffast-math and -ffp-contract=<x> in CFLAGS/CXXFLAGS/CPPFLAGS (EXTRA_CFLAGS and command-line values are
# already folded in). This runs before FP_POLICY is appended and before CFG_SIG is computed, so a rejected build creates
# no obj/bin directory. $(filter) matches whole words, so a token that merely contains the text (-DBUILD_NOTE=-Ofastness)
# is not rejected.
FP_INPUT_FLAGS := $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS)

# The character-safety checks below also cover LDFLAGS (and so EXTRA_LDFLAGS), because LDFLAGS is embedded in every link
# recipe. The FP-semantic check stays on FP_INPUT_FLAGS: those flags mean nothing at link time.
SHELL_SAFE_INPUT_FLAGS := $(FP_INPUT_FLAGS) $(LDFLAGS)

# $(filter) cannot see shell quoting: EXTRA_CFLAGS="'-Ofast'" would pass it yet reach the compiler as -Ofast, and an
# @response-file token can inject arbitrary flags. Only plain, unquoted, whitespace-separated argv tokens are supported;
# anything else is rejected by the allow-list below, before the conflict check runs.
# It is an allow-list rather than a deny-list because a deny-list misses globs, ~, redirection (>, <) and process
# substitution, none of which Make resolves itself. The check shells out to `grep -E` with the text embedded
# single-quoted, where only a single quote is live, so that one character is rejected first with a pure $(findstring),
# before any shell is involved.
ifneq ($(findstring ',$(SHELL_SAFE_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a single-quote character; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from the allowed character set (see the allow-list below) -- a single quote specifically must be rejected by this Make-native check BEFORE the allow-list's own $(shell) call can safely single-quote-embed the rest of the text)
endif

# Allowed characters: letters, digits and _ . / = , + - plus space (the token separator). That covers every flag this
# Makefile uses (-O2, -Wall, -std=gnu99, the -I paths, -DAEC_NO_STDIO) with headroom for pass-through flags such as
# -Wl,-rpath,dir. Everything else, including every character a deny-list would target, is rejected by construction;
# test/run_fp_flag_policy_test.sh carries the negative cases. LC_ALL=C pins the bracket ranges to ASCII.
# FP_ALLOWED_CHARS_RE is defined near the top of this file.
SHELL_SAFE_ALLOWLIST_RC := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C grep -Eq '$(FP_ALLOWED_CHARS_RE)'; echo $$?)
ifneq ($(SHELL_SAFE_ALLOWLIST_RC),0)
# Diagnostic only, on the error path: delete the allowed characters so the error names the offending ones.
SHELL_SAFE_DISALLOWED_CHARS := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C tr -d 'A-Za-z0-9_./=,+ -' | fold -w1 | LC_ALL=C sort -u | tr -d '\n')
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a character outside the allowed set [A-Za-z0-9_./=,+- ] (letters/digits/underscore/dot/slash/equals/comma/plus/minus, plus space as the token separator) -- disallowed character(s) found: "$(SHELL_SAFE_DISALLOWED_CHARS)"; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from that set -- remove/replace the disallowed character(s) shown above in CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS)
endif

FP_CONFLICT_FLAGS := $(filter -Ofast -ffast-math -ffp-contract=%,$(FP_INPUT_FLAGS))
ifneq ($(FP_CONFLICT_FLAGS),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains $(FP_CONFLICT_FLAGS); this repo pins -ffp-contract=off; remove $(FP_CONFLICT_FLAGS) from CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS)
endif

# These four internal variables are computed late from the assembled flags, so they cannot join the early foreach above. A
# command-line override of any of them would turn the checks above into no-ops while the real flags stay unvalidated, so
# they are rejected here, after their own assignments (when $(origin) has flipped). SHELL_SAFE_DISALLOWED_CHARS is left
# out: it only feeds the error text.
$(foreach v,FP_INPUT_FLAGS SHELL_SAFE_INPUT_FLAGS SHELL_SAFE_ALLOWLIST_RC FP_CONFLICT_FLAGS,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently defeat the FP-policy/shell-safety allow-list/conflict checks above, leaving the real CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS content unvalidated; this variable is internal to those checks and has no supported external use)))

CFLAGS    += $(FP_POLICY)

.DEFAULT_GOAL := all

# --- Consumer two-phase audiolib resolution ---------------------------------
# audiolib's archive path is keyed by its own CFG_SIG, so it cannot be hardcoded. Goals that need the archive work in two phases.
#   Phase 1 (COMMON_LIB unset): ask audiolib's `print-lib-path` (forwarding COMMON_FWD) and re-invoke make with COMMON_LIB set.
#   Phase 2 (COMMON_LIB known): the real rules, with $(COMMON_LIB) as a normal prerequisite so targets relink iff the archive
#   changed. The producer's config identity also enters this repo's CFG_SIG.
# `lib`, `selftest`, `clean` and the print-*/help goals never need audiolib: their rules are further down, outside the dispatch.
# Under `make -n` the dispatch recipe still runs its sub-makes (it mentions $(MAKE)); they inherit -n, so no file is created.
COMMON_FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK)

ifeq ($(COMMON_LIB),)
.PHONY: all bench test test-full test-counter-saturation test-rate-structural test-config-validation test-process-context test-split-entry test-split-entry-2 test-split-entry-3 test-aec-array test-aec-array-2 test-aec-array-flow test-avg-reverb-slots test-linear-context test-guard-unify test-res-nearend-gate test-runtime-preset test-delay-num-filters test-duty-census test-external-realign test-cli-delay-flags test-shared-far-spec test-spectral-reuse test-heap-alignment test-shared-fft-handle test-synthesis-window test-static-aec test-memory-layout test-handle-ready test-reset-parity test-zero-heap test-aec-res test-detectors-parity test-detectors-parity-all test-parity-e2e pyc-parity-driver
all bench test test-full test-counter-saturation test-rate-structural test-config-validation test-process-context test-split-entry test-split-entry-2 test-split-entry-3 test-aec-array test-aec-array-2 test-aec-array-flow test-avg-reverb-slots test-linear-context test-guard-unify test-res-nearend-gate test-runtime-preset test-delay-num-filters test-duty-census test-external-realign test-cli-delay-flags test-shared-far-spec test-spectral-reuse test-heap-alignment test-shared-fft-handle test-synthesis-window test-static-aec test-memory-layout test-handle-ready test-reset-parity test-zero-heap test-aec-res test-detectors-parity test-detectors-parity-all test-parity-e2e pyc-parity-driver:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audiolib archive path (COMMON_DIR=$(COMMON_DIR) -- is audiolib present as a sibling of this repo?)" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory $@ COMMON_LIB="$$ac"
else
# Last path component of $(COMMON_LIB)'s directory (e.g. ne10-2139547423), so a consumer built against a differently
# configured audiolib keys its own obj/bin directory. The absolute path stays out of the payload and manifest.
COMMON_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(COMMON_LIB))))
endif

# Library sources are auto-detected from src/. Declared before CFG_SIG_PAYLOAD, which folds in the sorted list, so adding or
# removing a source keys a fresh obj/bin directory. The FFT wrapper sources are in the audiolib archive, not here.
SRCS = $(wildcard $(SRC_DIR)/*.c)

# NO_STDIO=1: also filter aec_debug.c out of the sources, although it is already empty under -DAEC_NO_STDIO; `audit-no-stdio`
# checks that the archive does not list aec_debug.o.
ifeq ($(NO_STDIO),1)
  SRCS := $(filter-out $(SRC_DIR)/aec_debug.c,$(SRCS))
endif

# AEC_NO_ARRAY=1: the array instance and its gain stage (aec_res.c is used by nothing else) are not built or archived.
ifeq ($(AEC_NO_ARRAY),1)
  SRCS := $(filter-out $(SRC_DIR)/aec_array.c $(SRC_DIR)/aec_res.c,$(SRCS))
endif

# --- Hash-keyed object/binary directories + CFG_SIG ------------------------
# obj/<backend>-<sig>/ and bin/<backend>-<sig>/ keep builds that differ in any flag from sharing objects or delivering each
# other's binaries. The payload covers the tools (CC CXX AR RANLIB LINK), CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS and their EXTRA_*
# hooks, BACKEND, SIMD, PROFILE, WERROR, AEC_TESTING, NO_STDIO, AEC_NO_ARRAY, DEBUG, TOOLCHAIN_CHECK, the sorted SRCS and
# COMMON_PRODUCER_SIG (empty for goals that never touch audiolib). CC_TARGET_SIG adds the effective ABI/FP/ISA predefines,
# since the same CC command can select a different SDK through PATH. The hash is POSIX cksum of the unstripped payload.
# Define it after CFLAGS/LDFLAGS are complete and before the first rule that uses $(OBJ_DIR)/$(BIN_DIR): Make expands
# prerequisite lists at parse time, so an earlier use would see an empty CFG_SIG.
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) LDFLAGS=$(LDFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) SIMD=$(SIMD) PROFILE=$(PROFILE) WERROR=$(WERROR) AEC_TESTING=$(AEC_TESTING) NO_STDIO=$(NO_STDIO) AEC_NO_ARRAY=$(AEC_NO_ARRAY) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) COMMON_PRODUCER_SIG=$(COMMON_PRODUCER_SIG) SRCS=$(sort $(SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

# OBJ_ROOT / BIN_ROOT place the keyed trees (default obj/ and bin/ here). Placement only, so not in CFG_SIG; the isolation
# tests point them at a scratch directory.
OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR  = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR  = $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

OBJS  = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# The AecRes gain stage (src/aec_res.c) is not part of an AEC_NO_ARRAY library; the single-channel tests that use it as their
# reference stage link it as test support.
ifeq ($(AEC_NO_ARRAY),1)
TEST_RES_OBJ = $(OBJ_DIR)/aec_res.o
else
TEST_RES_OBJ =
endif

EXAMPLE_SRCS = $(EXAMPLE_DIR)/aec_wav.c
EXAMPLE_OBJS = $(patsubst $(EXAMPLE_DIR)/%.c,$(OBJ_DIR)/%.o,$(EXAMPLE_SRCS))

TARGET     = $(BIN_DIR)/aec_wav
LIB_TARGET = $(BIN_DIR)/libaec.a
BENCH_SRCS = $(TEST_DIR)/bench_rtf.c
BENCH_OBJS = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(BENCH_SRCS))
BENCH_TARGET = $(BIN_DIR)/bench_rtf

# Counter-saturation test (counter-overflow behaviour; also run under UBSan by test/run_counter_saturation_ubsan.sh). It
# needs the full Aec type, so it sits in the COMMON_LIB-dependent part of this Makefile like $(TARGET)/$(BENCH_TARGET).
TEST_CS_SRCS   = $(TEST_DIR)/test_counter_saturation.c
TEST_CS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_CS_SRCS))
TEST_CS_TARGET = $(BIN_DIR)/test_counter_saturation

# aec_get_linear() seam: aligned-far alias and content, generation bumps, UNLOCKED reporting. Needs the full Aec
# type.
TEST_LC_SRCS   = $(TEST_DIR)/test_linear_context.c
TEST_LC_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_LC_SRCS))
TEST_LC_TARGET = $(BIN_DIR)/test_linear_context

# Guard / reset / comfort-noise unification: aec_process() against context + public getters + aec_cng_fill + ac_istft over
# synthetic streams and the blind-set clips of test/guard_unify_clips.tsv (wav root: ../wav), bit for bit, plus the analysis
# pairing probe, the guard threshold fixture and the aec_cng_fill reference. Needs the full Aec type.
TEST_GU_SRCS   = $(TEST_DIR)/test_guard_unify.c
TEST_GU_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_GU_SRCS))
TEST_GU_TARGET = $(BIN_DIR)/test_guard_unify

# RES double-talk floor: the energy-based near-recent latch, its exported per-hop decision and the stationarity policy
# (test/test_res_nearend_gate.c). Needs the full Aec type.
TEST_RNG_SRCS   = $(TEST_DIR)/test_res_nearend_gate.c
TEST_RNG_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_RNG_SRCS))
TEST_RNG_TARGET = $(BIN_DIR)/test_res_nearend_gate

# Runtime strength retarget (suppression_gain_set_split_floor_far_active_db, aec_set_level): immediate retarget, ramp timing,
# retarget mid-ramp, reset keeps the target, rejection writes nothing, lanes below AEC_RESIDUAL_FULL never ramp. Needs the full
# Aec and SuppressionGain types.
TEST_RP_SRCS   = $(TEST_DIR)/test_runtime_preset.c
TEST_RP_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_RP_SRCS))
TEST_RP_TARGET = $(BIN_DIR)/test_runtime_preset

# Matched-filter bank size (AecConfig.delay_num_filters -> delay_aec_get_mem_size/delay_aec_init): the default is the
# 5-filter geometry, a shorter bank loses reach, the pool scales with n, and out-of-range values are rejected at every layer
# (aec_resolve_valid_config and the pool-first delay_aec_get_mem_size/delay_aec_init, which do not clamp). Needs the full Aec
# type.
TEST_DNF_SRCS   = $(TEST_DIR)/test_delay_num_filters.c
TEST_DNF_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_DNF_SRCS))
TEST_DNF_TARGET = $(BIN_DIR)/test_delay_num_filters

# Matched-filter duty-cycle census: the hops the duty machine actually ran vs skipped on a real run
# (Aec::duty_hops_total/duty_hops_run, surfaced via AecDebugStatus).
TEST_DUC_SRCS   = $(TEST_DIR)/test_duty_census.c
TEST_DUC_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_DUC_SRCS))
TEST_DUC_TARGET = $(BIN_DIR)/test_duty_census

TEST_XRA_SRCS   = $(TEST_DIR)/test_external_realign.c
TEST_XRA_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_XRA_SRCS))
TEST_XRA_TARGET = $(BIN_DIR)/test_external_realign

# Independent reference for cli_delay_flags.sh's print-mem-size cross-check: it calls aec_get_mem_breakdown() from its own
# source, not aec_wav.c. Not a `make test-*` target; only a prerequisite of test-cli-delay-flags.
TEST_PMR_SRCS   = $(TEST_DIR)/print_mem_size_ref.c
TEST_PMR_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_PMR_SRCS))
TEST_PMR_TARGET = $(BIN_DIR)/print_mem_size_ref

# Per-rate structural test: gates the supported grids, including the hop-authored constant retiming (check (d)). Needs the
# full Aec type and reuses the built $(OBJS).
TEST_RS_SRCS   = $(TEST_DIR)/test_rate_structural.c
TEST_RS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_RS_SRCS))
TEST_RS_TARGET = $(BIN_DIR)/test_rate_structural

# Config-validation test. Its test_direct_init_rejects_bad_rate section gates the pbfdaf_init_static/pbfdkf_init_static/
# saturation_init sample_rate preconditions.
TEST_CV_SRCS   = $(TEST_DIR)/test_config_validation.c
TEST_CV_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_CV_SRCS))
TEST_CV_TARGET = $(BIN_DIR)/test_config_validation

# The split pair aec_process_linear() + aec_process_residual(): it may be interleaved freely with aec_process() on one instance,
# and every AecLinearView / AecResidualView field a caller reads is byte-identical between them. Needs the full Aec type.
TEST_PC_SRCS   = $(TEST_DIR)/test_process_context.c
TEST_PC_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_PC_SRCS))
TEST_PC_TARGET = $(BIN_DIR)/test_process_context

# The three entry points (aec_process / aec_process_linear / aec_process_residual), their views and the gain / comfort-noise
# helpers on three grids, with the order / argument checks. Needs the full Aec type.
TEST_SE_SRCS   = $(TEST_DIR)/test_split_entry.c
TEST_SE_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SE_SRCS))
TEST_SE_TARGET = $(BIN_DIR)/test_split_entry

# The second single-channel scene set (saturation, double talk, delay shifts and the bad-relock revert, drift, reset, FIXED and
# EXTERNAL_ALIGNED with realign, levels in flight, switched-off blocks, live observer, duty cycle, 8 kHz) and the carve-end /
# alternation checks. Cases and runner: test/baseline2_mono.h.
TEST_SE2_SRCS   = $(TEST_DIR)/test_split_entry2.c
TEST_SE2_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SE2_SRCS))
TEST_SE2_TARGET = $(BIN_DIR)/test_split_entry2

# The third single-channel scene set: bad-relock judgments (revert / no revert, either side of the 1 dB margin) that free-running
# two-path scenes produce by themselves. Case table: test/baseline3_mono.h.
TEST_SE3_SRCS   = $(TEST_DIR)/test_split_entry3.c
TEST_SE3_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SE3_SRCS))
TEST_SE3_TARGET = $(BIN_DIR)/test_split_entry3

# The four-microphone array instance (n_mics = 4) driven through its residual modes, with its contract checks; and the
# shared-alignment / quarantine / duty-cycle /
# lane-contract state-machine checks ported from the flow's own suite (those reach the duty-cycle hooks of
# test/aec_test_hooks.h, so they run in an AEC_TESTING build).
TEST_AA_SRCS   = $(TEST_DIR)/test_aec_array.c
TEST_AA_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_AA_SRCS))
TEST_AA_TARGET = $(BIN_DIR)/test_aec_array
# The second array scene set (clipped / silent microphone, double talk, proxy channel, caller-supplied beamformed error, clock
# drift, levels, shadow off, admission band, quarantine, duty cycle; STATE / ESTIMATE / FULL), test/baseline2_array.h.
TEST_AA2_SRCS   = $(TEST_DIR)/test_aec_array2.c
TEST_AA2_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_AA2_SRCS))
TEST_AA2_TARGET = $(BIN_DIR)/test_aec_array2
TEST_AF_SRCS   = $(TEST_DIR)/test_aec_array_flow.c
TEST_AF_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_AF_SRCS))
TEST_AF_TARGET = $(BIN_DIR)/test_aec_array_flow

# Average-render-reverb ring-slot selection (aec_post_avg_reverb_slots): the
# delayed and the one-older partition must never alias the newest one.
TEST_ARS_SRCS   = $(TEST_DIR)/test_avg_reverb_slots.c
TEST_ARS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_ARS_SRCS))
TEST_ARS_TARGET = $(BIN_DIR)/test_avg_reverb_slots

# aec_process_linear_shared_far_from(): a borrowed far-end spectrum gives byte-identical views, is genuinely consumed,
# aec_reset() clears a pending borrow, and the real-FFT-compute counter reflects borrow vs compute.
TEST_SFS_SRCS   = $(TEST_DIR)/test_shared_far_spec.c
TEST_SFS_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SFS_SRCS))
TEST_SFS_TARGET = $(BIN_DIR)/test_shared_far_spec

TEST_SRU_SRCS   = $(TEST_DIR)/test_spectral_reuse.c
TEST_SRU_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SRU_SRCS))
TEST_SRU_TARGET = $(BIN_DIR)/test_spectral_reuse

TEST_HAL_TARGET = $(BIN_DIR)/test_heap_alignment
TEST_HAL_OBJ = $(OBJ_DIR)/test_heap_alignment.o

# Shared FftHandle: main filter, shadow filter and the post-filter block borrow the same handle (pointer identity) on the
# aec_create and aec_init paths, across a process() hop and aec_reset().
TEST_SFH_SRCS   = $(TEST_DIR)/test_shared_fft_handle.c
TEST_SFH_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SFH_SRCS))
TEST_SFH_TARGET = $(BIN_DIR)/test_shared_fft_handle
TEST_SSW_SRCS   = $(TEST_DIR)/test_synthesis_window.c
TEST_SSW_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_SSW_SRCS))
TEST_SSW_TARGET = $(BIN_DIR)/test_synthesis_window

# Heap-vs-caller-pool byte-equality test on a complete real recording.
TEST_STATIC_SRCS   = $(TEST_DIR)/test_static_aec.c
TEST_STATIC_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_STATIC_SRCS))
TEST_STATIC_TARGET = $(BIN_DIR)/test_static_aec
TEST_STATIC_MIC ?= ../wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk_mic.wav
TEST_STATIC_REF ?= ../wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_doubletalk_lpb.wav
TEST_STATIC_SR  ?= 16000

# aec_reset() equals a fresh instance: a never-warmed and a warmed-then-reset instance on the same config produce
# byte-identical out[hop], AecResidualView.formed_hop and AecResidualView.error_spec over the whole post-reset run, on the three
# product grids and both consumption modes.
TEST_RPY_SRCS   = $(TEST_DIR)/test_reset_parity.c
TEST_RPY_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_RPY_SRCS))
TEST_RPY_TARGET = $(BIN_DIR)/test_reset_parity

# Python/C parity gate for the three detectors (RenderActivity / FilterConvergence / DoubleTalk). detectors.c needs only
# detectors.h and <math.h> (plus aec_scale.c for the retiming helpers), so no $(OBJS) or audiolib archive is linked. The
# golden is generated from the Python reference on demand, not checked in.
TEST_DET_SRC    = $(TEST_DIR)/modules/parity_detectors.c
TEST_DET_TARGET = $(BIN_DIR)/parity_detectors
TEST_DET_HOP   ?= 128
TEST_DET_SR    ?= 16000
TEST_DET_GOLDEN = $(BIN_DIR)/det_golden.bin

# Allocator-hook acceptance test for the complete caller-pool lifecycle.
TEST_ZH_SRCS   = $(TEST_DIR)/test_zero_heap_aec.c
TEST_ZH_OBJS   = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(TEST_ZH_SRCS))
TEST_ZH_TARGET = $(BIN_DIR)/test_zero_heap_aec

# RES gain stage (src/aec_res_internal.h, internal): configuration equality with an Aec lane,
# gain and comfort-noise equality with the multi-lane pipelines' stage, API
# contract. The test includes src/aec_res.c for the struct, so it links the
# library objects without aec_res.o.
TEST_AR_OBJS    = $(OBJ_DIR)/test_aec_res.o
TEST_AR_TARGET  = $(BIN_DIR)/test_aec_res

# --- Python<->C parity harness (test/pyc_parity_aec.sh, test/pyc_parity/) ---
# test-parity-e2e : end-to-end gate (test/parity_aec_e2e.c) against goldens regenerated by python/diag/gen_aec_e2e_golden.py
#                   into $(BIN_DIR) (never into git); wavs pinned by test/parity_e2e_inputs.sha256. 16/48 kHz must pass;
#                   the 8 kHz row is the expected red. Part of `make test-full` only: the lean pyc suite catches the same
#                   planted regressions at a fraction of the cost.
# pyc-parity-driver : builds test/pyc_parity/aec_run.c (the C side) for this configuration; test/pyc_parity_aec.sh calls it
#                   once per build (std-math, ship).
# test-pyc-parity : the Python<->C parity suite (two layers; see test/pyc_parity_aec.sh -h). Runs the lean suite by default;
#                   PYC_ARGS="--full" runs every scenario (what `make test-full` does).
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

# The Python test suite. -p no:cacheprovider and PYTHONDONTWRITEBYTECODE keep the working tree clean; PYTHON needs numpy
# and pytest.
.PHONY: test-python
test-python:
	cd .. && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 "$(PYTHON)" -m pytest -q -p no:cacheprovider python/tests

# Header-only AEC kernel correctness/microbench harness (utility/simd_aec.h).
# Links with libm, without the audiolib archive or two-phase dispatch.
SELFTEST_SRC    = $(TEST_DIR)/simd_selftest_aec.c
SELFTEST_TARGET = $(BIN_DIR)/simd_selftest_aec

# delay_aec_reset() consistency test. It needs only delay_aec.c and aec_scale.c, not the full Aec type or the audiolib archive,
# so like `selftest` it sits outside the COMMON_LIB dispatch and shares objects with a full build of the same CFG_SIG.
TEST_DR_SRCS   = $(SRC_DIR)/delay_aec.c $(SRC_DIR)/aec_scale.c $(TEST_DIR)/test_delay_reset.c
TEST_DR_OBJS   = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(filter $(SRC_DIR)/%.c,$(TEST_DR_SRCS))) \
                  $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(filter $(TEST_DIR)/%.c,$(TEST_DR_SRCS)))
TEST_DR_TARGET = $(BIN_DIR)/test_delay_reset

TEST_DDS_SRCS   = $(SRC_DIR)/delay_aec.c $(SRC_DIR)/aec_scale.c $(TEST_DIR)/test_delay_dominant_selection.c
TEST_DDS_OBJS   = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(filter $(SRC_DIR)/%.c,$(TEST_DDS_SRCS))) \
                  $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(filter $(TEST_DIR)/%.c,$(TEST_DDS_SRCS)))
TEST_DDS_TARGET = $(BIN_DIR)/test_delay_dominant_selection

# test_delay_kernels #includes src/delay_aec.c so its static kernels are testable; it must not also link delay_aec.o, only
# aec_scale.o. Its object has its own rule: -I. resolves the quoted include from the c_impl root, and a per-object -MD makes
# an edit to the included delay_aec.c rebuild it.
TEST_DKN_OBJ    = $(OBJ_DIR)/test_delay_kernels.o
TEST_DKN_TARGET = $(BIN_DIR)/test_delay_kernels
TEST_DMN_OBJ    = $(OBJ_DIR)/test_delay_matched_neon.o
TEST_DMN_TARGET = $(BIN_DIR)/test_delay_matched_neon
TEST_NT_OBJ     = $(OBJ_DIR)/test_neon_twins.o
TEST_NT_TARGET  = $(BIN_DIR)/test_neon_twins

# --- CFG_SIG collision guard + directory creation ---------------------------
# The single order-only prerequisite of every compile/link/archive rule. It is .PHONY, but Make builds a target once per
# invocation, so under -j it runs once and completes before any dependent rule. It writes config.manifest the first time and
# verifies it afterwards (a mismatch is a CFG_SIG collision or a payload coverage gap), atomically via temp file + mv.
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

# -MD -MP writes $(OBJ_DIR)/<name>.d per object; included at the bottom of this file.
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

# `lib` needs only this repo's objects (the audiolib archive is not merged into libaec.a), so it sits outside the COMMON_LIB
# dispatch and works without audiolib. The archive is always built fresh into a PID-suffixed temp file and renamed into place:
# `ar rc` onto an existing archive would keep a removed source's stale member, and the rename makes racing same-config builds
# last-writer-wins.
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

# Regenerates include/aec_balanced_config.h (python/diag/gen_aec_balanced_config_h.py) into a throwaway copy of python/ and
# diffs it against the checked-in header; a FAIL means the header was hand-edited or is stale. It needs no audiolib archive,
# so it sits outside the COMMON_LIB dispatch. PYTHON is declared further down; recipe variables expand when the recipe runs.
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

# Compile-only C++ check of include/aec.h and include/aec_internal.h (the library itself has no C++ TU): the AEC_LEVEL_*
# enumerators must be accepted by aec_config_from_level(), aec_set_level() and the level lookups. It needs no audiolib
# archive, so it sits outside the COMMON_LIB dispatch like test-header-fresh.
.PHONY: test-header-cpp
test-header-cpp:
	$(CXX) -std=c++11 -fsyntax-only -Wall -Wextra -Werror -I./include -I$(COMMON_DIR)/include -I$(COMMON_DIR) $(TEST_DIR)/test_header_cpp.cpp
	@echo "test-header-cpp: aec.h and aec_internal.h compile as C++11 and the AEC_LEVEL_* enumerators convert"

# Builds AND runs (unlike `bench`, which only builds) the delay_aec_reset() test. It is self-contained (synthetic signals,
# no fixture file).
.PHONY: test-delay-reset
test-delay-reset: $(TEST_DR_TARGET)
	"$(abspath $(TEST_DR_TARGET))"

$(TEST_DR_TARGET): $(TEST_DR_OBJS) | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(TEST_DR_OBJS) -lm

# Builds AND runs, same shape as test-delay-reset: the delay-selection test. It is self-contained (a synthetic dual-path
# signal) and asserts its own premise, that the early path alone is detectable, before asserting which path was reported.
.PHONY: test-delay-dominant-selection
test-delay-dominant-selection: $(TEST_DDS_TARGET)
	"$(abspath $(TEST_DDS_TARGET))"

$(TEST_DDS_TARGET): $(TEST_DDS_OBJS) | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(TEST_DDS_OBJS) -lm

# Builds AND runs. NEON-vs-scalar equivalence sweeps for delay_aec.c's two local kernels (argmax tie order and the
# decimator's carried IIR state); the argmax half is the only check in the repo that tells first-match from last-match
# tie order (see the test file's header).
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
# Phase 2 real rules: only defined once COMMON_LIB is known (see the dispatch block above). $(COMMON_LIB): FORCE re-invokes
# audiolib's build on every run (a no-op when up to date), and its mtime only advances if audiolib rebuilt something, so
# $(TARGET)/$(BENCH_TARGET) relink only then.
# =============================================================================
ifneq ($(COMMON_LIB),)
.PHONY: all bench FORCE test test-full test-counter-saturation test-rate-structural _run-rate-structural test-config-validation test-process-context test-split-entry test-split-entry-2 test-split-entry-3 test-aec-array test-aec-array-2 test-aec-array-flow test-avg-reverb-slots test-linear-context test-guard-unify test-res-nearend-gate test-runtime-preset test-delay-num-filters test-duty-census test-external-realign test-cli-delay-flags test-shared-far-spec test-spectral-reuse test-heap-alignment test-shared-fft-handle test-synthesis-window test-static-aec test-memory-layout test-handle-ready test-reset-parity test-zero-heap test-aec-res test-detectors-parity test-detectors-parity-all test-parity-e2e pyc-parity-driver
all: $(TARGET)
bench: $(BENCH_TARGET)

# Builds AND runs (unlike `bench`, which only builds). Run from c_impl/: the delay_aec section reads a checked-in fixture
# at a relative path.
test-counter-saturation: $(TEST_CS_TARGET)
	"$(abspath $(TEST_CS_TARGET))"

# Builds AND runs, same convention as test-counter-saturation above. Re-invokes make with AEC_TESTING=1 because check (d6)
# drives an otherwise-static internal through test/aec_test_hooks.h, and the recursion gives that build its own
# CFG_SIG-keyed obj/bin tree. `_run-rate-structural` is the inner half; invoke `test-rate-structural`.
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

# Builds AND runs.
test-split-entry: $(TEST_SE_TARGET)
	"$(abspath $(TEST_SE_TARGET))"

# Builds AND runs.
test-split-entry-2: $(TEST_SE2_TARGET)
	"$(abspath $(TEST_SE2_TARGET))"

# Builds AND runs.
test-split-entry-3: $(TEST_SE3_TARGET)
	"$(abspath $(TEST_SE3_TARGET))"

test-avg-reverb-slots: $(TEST_ARS_TARGET)
	"$(abspath $(TEST_ARS_TARGET))"

# The array-only targets (the four-microphone instance and its gain stage) are skipped, with a message, when AEC_NO_ARRAY=1.
ifeq ($(AEC_NO_ARRAY),1)
test-aec-array test-aec-array-2 test-aec-array-flow test-aec-res:
	@echo "$@: SKIPPED -- array-only test; AEC_NO_ARRAY=1 builds the single-channel library"
else
# Build AND run. Re-invoke make with AEC_TESTING=1: both read the admission candidate through aec_test_hooks.h.
test-aec-array:
	+@$(MAKE) --no-print-directory AEC_TESTING=1 _run-aec-array

_run-aec-array: $(TEST_AA_TARGET)
	"$(abspath $(TEST_AA_TARGET))"

test-aec-array-2:
	+@$(MAKE) --no-print-directory AEC_TESTING=1 _run-aec-array-2

_run-aec-array-2: $(TEST_AA2_TARGET)
	"$(abspath $(TEST_AA2_TARGET))"

# Builds AND runs. Re-invokes make with AEC_TESTING=1 for the duty-cycle hooks, like test-rate-structural.
test-aec-array-flow:
	+@$(MAKE) --no-print-directory AEC_TESTING=1 _run-aec-array-flow

_run-aec-array-flow: $(TEST_AF_TARGET)
	"$(abspath $(TEST_AF_TARGET))"

test-aec-res: $(TEST_AR_TARGET)
	"$(abspath $(TEST_AR_TARGET))"
endif

# Builds AND runs, same convention as test-counter-saturation above.
test-linear-context: $(TEST_LC_TARGET)
	"$(abspath $(TEST_LC_TARGET))"

# Builds AND runs from c_impl/ (the clip list and wav root are relative to it).
test-guard-unify: $(TEST_GU_TARGET)
	"$(abspath $(TEST_GU_TARGET))" "$(abspath $(CURDIR)/../wav)" "$(abspath $(TEST_DIR)/guard_unify_clips.tsv)"

# Python<->C parity: checker + driver link rules, and the end-to-end gate recipe.
$(TEST_PE_TARGET): $(OBJS) $(TEST_PE_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "parity_aec_e2e [$(BACKEND)] -> $@"

$(PYC_RUN_TARGET): $(OBJS) $(PYC_RUN_OBJ) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "pyc_aec_run [$(BACKEND)] -> $@"

pyc-parity-driver: $(PYC_RUN_TARGET)

# Builds AND runs. Exit 0 = 16/48 kHz pass and the 8 kHz row is the expected red.
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

# Builds AND runs, same convention as test-counter-saturation above.
test-synthesis-window: $(TEST_SSW_TARGET)
	"$(abspath $(TEST_SSW_TARGET))"

test-memory-layout: $(OBJS) $(COMMON_LIB) | _cfg_guard
	$(CC) $(CFLAGS) -c -o "$(OBJ_DIR)/test_memory_layout.o" "$(TEST_DIR)/test_memory_layout.c"
	$(LINK) $(CFLAGS) -o "$(BIN_DIR)/test_memory_layout" "$(OBJ_DIR)/test_memory_layout.o" $(OBJS) $(COMMON_LIB) $(LDFLAGS)
	"$(abspath $(BIN_DIR)/test_memory_layout)"

# Builds AND runs. check_handle_entries.py first compares the entries the test calls with the Aec-handle prototypes of aec.h and aec_internal.h.
test-handle-ready: $(OBJS) $(COMMON_LIB) | _cfg_guard
	$(PYTHON) -I "$(TEST_DIR)/check_handle_entries.py" include/aec.h "$(TEST_DIR)/test_handle_ready.c" include/aec_internal.h
	$(CC) $(CFLAGS) -c -o "$(OBJ_DIR)/test_handle_ready.o" "$(TEST_DIR)/test_handle_ready.c"
	$(LINK) $(CFLAGS) -o "$(BIN_DIR)/test_handle_ready" "$(OBJ_DIR)/test_handle_ready.o" $(OBJS) $(COMMON_LIB) $(LDFLAGS)
	"$(abspath $(BIN_DIR)/test_handle_ready)"

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

# Four-grid gate: 8 kHz (not a product grid) plus the three product grids.
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

# --- Aggregate tiers: `make test` (default) and `make test-full` ----------------------------------------------------------
# Members run one after another (some share a scratch file or one audiolib archive) and every failure is listed at the end
# instead of stopping at the first. The dispatch above resolves audiolib once; the members inherit COMMON_LIB. The members
# that build their own configurations get COMMON_LIB= so they resolve it themselves, as they do when run alone, and the
# shell wrappers run without MAKEFLAGS for the same reason.
#   SIMD-dependent C targets and the build gates run under whichever SIMD is set. A SIMD-independent C target (its result
#   cannot differ between SIMD=0 and SIMD=1) runs in `make test` only under SIMD=1, so the default tier is `make test` with
#   SIMD=1 plus `make test` with SIMD=0. The pytest suite, the lean pyc suite and the UBSan wrappers take no SIMD input
#   (they build their own configurations) and also run only under SIMD=1.
#   `make test-full` adds what is slow or changes rarely: the SIMD-independent targets under SIMD=0 as well, the end-to-end
#   gate, the full pyc matrix instead of the lean one, the UBSan --use-standard-math variants, the Makefile flag-policy
#   script and the signal-handling script that tests the UBSan wrappers' traps.
TEST_SIMD_DEP   = selftest test-process-context test-split-entry test-split-entry-2 test-split-entry-3 test-aec-array test-aec-array-2 test-aec-array-flow test-linear-context test-guard-unify test-aec-res test-res-nearend-gate test-runtime-preset \
                  test-delay-num-filters test-duty-census test-external-realign test-reset-parity test-delay-reset \
                  test-delay-dominant-selection test-delay-kernels test-delay-matched-neon test-neon-twins \
                  test-rate-structural test-counter-saturation
TEST_SIMD_INDEP = test-heap-alignment test-memory-layout test-handle-ready test-config-validation test-avg-reverb-slots \
                  test-shared-far-spec test-spectral-reuse test-shared-fft-handle test-synthesis-window test-static-aec test-zero-heap \
                  test-header-fresh test-header-cpp test-detectors-parity-all test-cli-delay-flags
TEST_BUILD_GATES = test-no-testing-symbols audit-no-stdio
CLEAN_ENV       = env -u MAKEFLAGS -u MFLAGS -u MAKELEVEL

test: TIER_FULL =
test-full: TIER_FULL = 1
test test-full:
	+@fails=""; \
	 run() { l=$$(echo "$$*" | sed 's#^$(MAKE) --no-print-directory ##'); echo "== $$l"; "$$@" || fails="$$fails\n  $$l"; }; \
	 for t in $(TEST_SIMD_DEP); do run $(MAKE) --no-print-directory $$t; done; \
	 for t in $(TEST_BUILD_GATES); do run $(MAKE) --no-print-directory $$t COMMON_LIB=; done; \
	 $(if $(or $(filter 1,$(SIMD)),$(TIER_FULL)),\
	  for t in $(TEST_SIMD_INDEP); do run $(MAKE) --no-print-directory $$t; done; \
	  run $(MAKE) --no-print-directory audit-profile COMMON_LIB=;,\
	  echo "== SIMD-independent targets skipped under SIMD=0: the SIMD=1 run covers them";) \
	 $(if $(TIER_FULL),run $(MAKE) --no-print-directory test-parity-e2e;) \
	 $(if $(filter 1,$(SIMD)),\
	  run $(MAKE) --no-print-directory test-python COMMON_LIB=; \
	  run $(MAKE) --no-print-directory test-pyc-parity COMMON_LIB= $(if $(TIER_FULL),PYC_ARGS=--full); \
	  run $(CLEAN_ENV) sh $(TEST_DIR)/run_counter_saturation_ubsan.sh; \
	  run $(CLEAN_ENV) sh $(TEST_DIR)/run_selftest_ubsan.sh; \
	  $(if $(TIER_FULL),\
	   run $(CLEAN_ENV) sh $(TEST_DIR)/run_counter_saturation_ubsan.sh --use-standard-math; \
	   run $(CLEAN_ENV) sh $(TEST_DIR)/run_selftest_ubsan.sh --use-standard-math; \
	   run $(CLEAN_ENV) sh $(TEST_DIR)/run_fp_flag_policy_test.sh; \
	   run $(CLEAN_ENV) bash $(TEST_DIR)/run_ubsan_signal_handling_test.sh;),\
	  echo "== pytest and the pyc and sanitizer suites skipped under SIMD=0: the SIMD=1 run covers them";) \
	 if [ -n "$$fails" ]; then printf '$@ SIMD=$(SIMD): FAILED:%b\n' "$$fails" >&2; exit 1; fi; \
	 echo "$@ SIMD=$(SIMD): ALL PASS"

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

$(TEST_SE_TARGET): $(OBJS) $(TEST_RES_OBJ) $(TEST_SE_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_split_entry [$(BACKEND)] -> $@"

$(TEST_AA_TARGET): $(OBJS) $(TEST_AA_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_aec_array [$(BACKEND)] -> $@"

$(TEST_SE2_TARGET): $(OBJS) $(TEST_RES_OBJ) $(TEST_SE2_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_split_entry2 [$(BACKEND)] -> $@"

$(TEST_SE3_TARGET): $(OBJS) $(TEST_RES_OBJ) $(TEST_SE3_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_split_entry3 [$(BACKEND)] -> $@"

$(TEST_AA2_TARGET): $(OBJS) $(TEST_AA2_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_aec_array2 [$(BACKEND)] -> $@"

$(TEST_AF_TARGET): $(OBJS) $(TEST_AF_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_aec_array_flow [$(BACKEND)] -> $@"

$(TEST_ARS_TARGET): $(OBJS) $(TEST_ARS_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_avg_reverb_slots [$(BACKEND)] -> $@"

$(TEST_AR_TARGET): $(filter-out $(OBJ_DIR)/aec_res.o,$(OBJS)) $(TEST_AR_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_aec_res [$(BACKEND)] -> $@"

$(TEST_LC_TARGET): $(OBJS) $(TEST_LC_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_linear_context [$(BACKEND)] -> $@"

$(TEST_GU_TARGET): $(OBJS) $(TEST_GU_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_guard_unify [$(BACKEND)] -> $@"

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

$(TEST_SSW_TARGET): $(OBJS) $(TEST_SSW_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_synthesis_window [$(BACKEND)] -> $@"

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

$(TEST_ZH_TARGET): $(OBJS) $(TEST_RES_OBJ) $(TEST_ZH_OBJS) $(COMMON_LIB) | _cfg_guard
	+$(MAKE) -C $(COMMON_DIR) $(COMMON_FWD) test_zero_heap
ifeq ($(shell uname -s),Darwin)
	@ac_bin="$$($(MAKE) -s -C $(COMMON_DIR) $(COMMON_FWD) print-bin-dir)"; \
	$(LINK) $(CFLAGS) -fno-builtin -o $@ $(OBJS) $(TEST_RES_OBJ) $(TEST_ZH_OBJS) $(COMMON_LIB) \
		-L"$$ac_bin" -lzero_heap_hook -Wl,-rpath,"$$ac_bin" $(LDFLAGS)
else
	@ac_obj="$$($(MAKE) -s -C $(COMMON_DIR) $(COMMON_FWD) print-obj-dir)"; \
	$(LINK) $(CFLAGS) -fno-builtin -o $@ $(OBJS) $(TEST_RES_OBJ) $(TEST_ZH_OBJS) \
		"$$ac_obj/zero_heap_hook.o" $(COMMON_LIB) $(LDFLAGS) \
		-Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc -Wl,--wrap=free
endif
	@echo "test_zero_heap_aec [$(BACKEND)] -> $@"
endif

# --- Query targets -------------------------------------------------------
# print-obj-dir / print-lib-path report this invocation's own build identity and never shell out to audiolib. print-bin-dir
# dispatches through phase 1 like `all`/`bench`, because the binaries only exist in the directory of the resolved producer;
# passing COMMON_LIB= explicitly skips the extra query sub-make.
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

# publish: immutable, content-addressed release under $(DIST_ROOT), same design as audiolib's publish (deterministic MANIFEST,
# byte-verified idempotent republish, `current` repointed by an atomic rename via the swapln helper built with $(HOSTCC)).
#   - -n/-q/-t never touch $(DIST_ROOT), the lock or any stage/MANIFEST/ATTEST file (guard in the recipe).
#   - Each publish, also an idempotent republish, installs one new attest-<utc>-<commit>[-dirty]-<seq>.txt with swapln
#     --excl-install (never clobbers; a name collision retries with the next <seq>) and records audiolib's git state too.
#     $(COMMON_DIR) is guarded first: `git -C ''` would silently report this repo's state as audiolib's.
#   - Refusal policy: see the knob comment near the top. Attestations are unsigned; the MANIFEST byte check is the integrity check.
# `publish` is the locked driver; _publish_impl is internal (calling it directly skips the lock).
PUBLISH_ARTIFACTS = $(LIB_TARGET) $(TARGET) $(BENCH_TARGET)
DIST_BACKEND_DIR  = $(DIST_ROOT)/$(BACKEND)
DIST_LOCK         = $(DIST_ROOT)/.lock.$(BACKEND)

.PHONY: publish
# Dry-run guard. The recipe mentions $(MAKE), so GNU make runs it for real under -n/-q/-t; unguarded it would create
# $(DIST_ROOT) and take the real lock. The MAKEFLAGS scan below branches:
#   -t  prints a note and exits 0 without recursing (recursing would let touch semantics bump the mtimes of the artifacts and
#       of audiolib's archive). Checked first: in `make -nt publish` any t wins over n.
#   -q  exits 1 (publish is phony, so it always "needs updating") with no output.
#   -n  recurses only so the child, which inherits -n, prints _publish_impl's commands.
# The scan accepts bundled no-dash short flags ("n", "sn") as the first word or dashed words ("-n"); long options start with
# "--"; a literal "--" ends the flags, since variable overrides (BACKEND=ne10) can contain n/q/t.
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

# audit-no-stdio: verifies that a NO_STDIO=1 libaec.a is stdio-free. It builds its own NO_STDIO=1 `lib` with recipe-time
# sub-makes, whatever NO_STDIO this invocation used. Checks, in order:
#   1. key isolation: the NO_STDIO=1 print-lib-path differs from the NO_STDIO=0 one.
#   2. `ar -t`: aec_debug.o is absent from the archive.
#   3. `nm -A` on the archive: no fprintf/vfprintf/printf/puts/fputs/fputc/fwrite/stderr/__stderrp, defined or undefined.
#   4. link test/no_stdio_main.c (itself stdio-free) with the archive and audiolib's NO_STDIO=1 archive, and `nm -A` the
#      executable against the same set (the NE10 backend carries no stdio references).
#   5. run it; exit code 0 means aec_create, one aec_process hop and aec_destroy needed no stdio.
AUDIT_STRICT_ALL  = fprintf|vfprintf|printf|puts|fputs|fputc|fwrite|stderr|__stderrp
AUDIT_STRICT_OWN  = fprintf|vfprintf|printf|puts|fputs
AUDIT_BIN_DIR     = $(BIN_ROOT)/audit-no-stdio/$(BACKEND)

# Test-only symbols must not ship. Checked in both directions: a one-way check also passes on a library that stopped defining
# them, which would silently disable check (d6). The ' _?' in the pattern covers the leading underscore of Mach-O nm output
# as well as ELF.
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
# audit-profile: the PROFILE knob keys its own build and changes what is linked, asserted in both directions so the check
# cannot pass vacuously (a build that quietly loses the flag would report a table of zeros, which reads as "free").
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

# `debug` delegates to `all` with -g -DAEC_DEBUG added through EXTRA_CFLAGS, which folds into CFLAGS before CFG_SIG is
# computed, so a debug build gets its own obj/bin directory (a target-specific CFLAGS += would reach only the recipe, after
# the key was computed).
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
	@echo "  test    Default test tier: every test-* target above plus the build gates, pytest, the lean pyc suite and"
	@echo "                  the UBSan wrappers. SIMD-independent members run only under SIMD=1: run it with SIMD=1 and SIMD=0"
	@echo "  test-full  The default tier plus the full pyc matrix, test-parity-e2e, the SIMD-independent targets under"
	@echo "                  SIMD=0, the UBSan --use-standard-math variants, run_fp_flag_policy_test.sh and"
	@echo "                  run_ubsan_signal_handling_test.sh"
	@echo "  test-counter-saturation  Build + run the permanent counter-saturation"
	@echo "                  regression test (test_counter_saturation) -- run from c_impl/"
	@echo "  test-delay-reset  Build + run the permanent delay_aec_reset() consistency"
	@echo "                  regression test (test_delay_reset) -- run from c_impl/"
	@echo "  test-rate-structural  Build + run the permanent per-rate structural"
	@echo "                  regression test (test_rate_structural) -- run from c_impl/"
	@echo "  test-avg-reverb-slots  Build + run the average-render-reverb ring-slot test"
	@echo "  test-process-context  Build + run the permanent split-entry-point regression test"
	@echo "                  (test_process_context: aec_process vs the split pair) -- run from c_impl/"
	@echo "  test-split-entry  Build + run the three-entry-point test (test_split_entry)"
	@echo "  test-split-entry-2  Build + run the second single-channel scene set (saturation, double talk, delay shifts,"
	@echo "                  drift, reset, realign, levels, toggles, 8 kHz), the alternation and carve-end checks"
	@echo "                  (test_split_entry2)"
	@echo "  test-split-entry-3  Build + run the third single-channel scene set (bad-relock judgments that two-path scenes"
	@echo "                  produce by themselves: natural reverts and non-reverts either side of the margin)"
	@echo "                  (test_split_entry3)"
	@echo "  test-aec-array  Build + run the array-instance test and its contract checks (test_aec_array)"
	@echo "  test-aec-array-2  Build + run the second array scene set (clipped / silent microphone, proxy channel,"
	@echo "                  caller beamformed error, drift, levels, quarantine) (test_aec_array2)"
	@echo "                  AEC_TEST_BASELINE_WRITE=<dir> / AEC_TEST_BASELINE=<dir> capture / compare the line stream of"
	@echo "                  the five test-split-entry* / test-aec-array* tests (none shipped; run each from c_impl/)"
	@echo "  test-aec-array-flow  Build + run the array shared-alignment / quarantine / duty-cycle /"
	@echo "                  lane-contract state-machine tests (test_aec_array_flow, AEC_TESTING build)"
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
	@echo "  test-handle-ready    Every aec.h entry refuses a zero-filled, failed-create or destroyed handle"
	@echo "  test-shared-fft-handle  Build + run the permanent shared-FFT-handle"
	@echo "                  regression test (test_shared_fft_handle) -- run from c_impl/"
	@echo "  test-synthesis-window  Build + run the aec_synthesis_window() accessor test (3 grids, refusals)"
	@echo "  test-static-aec  Build + run heap-vs-caller-pool byte-equality on a real WAV pair"
	@echo "                  (override TEST_STATIC_MIC/TEST_STATIC_REF/TEST_STATIC_SR as needed)"
	@echo "  test-reset-parity  Build + run the permanent aec_reset()-equals-fresh-instance"
	@echo "                    byte-identity gate (3 grids x full-res/ctx-only)"
	@echo "  test-zero-heap  Prove caller-pool init/process/reset/destroy performs no allocation"
	@echo "  test-parity-e2e  Python<->C end-to-end gate (parity_aec_e2e) on goldens regenerated from the pinned"
	@echo "                  doubletalk wavs; 16k/48k must pass, 8k is the known EXPECTED_RED(S-19) (16k/512 not covered);"
	@echo "                  run by test-full only"
	@echo "  test-pyc-parity  Python<->C effect parity, two layers (std-math C vs Python, ship vs std-math C); lean suite"
	@echo "                  by default, PYC_ARGS=--full for every scenario"
	@echo "  test-python     The pytest suite (python/tests)"
	@echo "  pyc-parity-driver  Build test/pyc_parity/aec_run.c (the C side of test-pyc-parity) for this configuration"
	@echo "  test-neon-twins  NEON-vs-scalar twins: rsa_argmax_first tie order, onset re-arm boundary"
	@echo "  test-delay-kernels  NEON-vs-scalar sweeps of delay_aec.c's kernels: argmax tie order, decimator and"
	@echo "                  48 kHz anti-alias pair (the only gate that sees the pair kernel; scalar vs itself with SIMD=0)"
	@echo "  test-delay-matched-neon  Bit-exact matched-filter kernels (dot, NLMS update, fused update + next dot, core)"
	@echo "                  vs a portable model of the AArch64 NEON dataflow; scalar kernels vs the scalar model otherwise"
	@echo "  test-aec-res    RES gain stage (src/aec_res_internal.h): lane-equal config, API contract, render scale, initial phase"
	@echo "  test-header-fresh  Regenerate include/aec_balanced_config.h and diff it against the checked-in copy"
	@echo "  test-header-cpp  Compile-only C++11 check that aec.h and aec_internal.h compile as C++ with the AEC_LEVEL_* names"
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
	@echo "  ./bin/<backend>-<config-hash>/bench_rtf [--seconds N] [--repeat M] [--level 0|1|2|3] [--sr HZ] [--fft N] [--stage full|linear|both] [--scene noise|bursts]"
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

# Header dependency tracking: -MD (not -MMD) with -MP, so a deleted header does not break a stale .d. The wildcard is empty
# before any object exists for this CFG_SIG.
-include $(wildcard $(OBJ_DIR)/*.d)
