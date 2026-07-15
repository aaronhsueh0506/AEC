# AEC C Implementation — Makefile

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -ffp-contract=off -std=gnu99
CFLAGS += -I./include -I./example
CFLAGS += $(EXTRA_CFLAGS)
LDFLAGS = -lm

SRC_DIR     = src
EXAMPLE_DIR = example
# Hash-keyed object dir: obj/<backend>-<cfg-sig>/ (CFG_SIG is computed
# further below, once BACKEND/WERROR/CFLAGS are all resolved -- see that
# comment for why OBJ_DIR must not be referenced by any rule before then).
# Every distinct (BACKEND, CFLAGS, EXTRA_CFLAGS, WERROR) combination gets
# its own directory automatically, so nothing is ever wiped to "switch"
# between them and two differently-configured builds can run concurrently
# in this same tree. OBJ_ROOT is the whole obj/ tree (every config) -- used
# by `clean` to nuke everything, not just the current config's subdirectory.
# BIN_DIR stays flat (not per-backend/config): the final binary is a
# last-write-wins whole, same as before.
OBJ_ROOT    = obj
OBJ_DIR     = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR     = bin

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
LDFLAGS += $(AC_DIR)/bin/$(BACKEND)/libaudio_common.a
ifeq ($(BACKEND),ne10)
  # The ne10 archive carries one C++ TU (NE10's dsp .cpp kernels); linking a
  # plain C driver against it needs the C++ runtime.
  LDFLAGS += -lc++
endif

# -Werror opt-in for this repo's OWN sources (default off; CI can flip it on
# with `make WERROR=1`). AEC compiles no vendored TUs itself (the FFT vendor
# code lives in audio_common), so every object built by this Makefile is
# "our code" -- no separate warning isolation is needed here.
WERROR ?= 0
ifeq ($(WERROR),1)
  CFLAGS += -Werror
endif

# --- Hash-keyed object directory (build hygiene) ----------------------------
# obj/ used to be a single flat directory shared across every BACKEND/
# EXTRA_CFLAGS combination, so switching either on an already-built tree
# used to silently keep stale .o files around (they don't encode which
# flags produced them). CFG_SIG hashes the exact compiler invocation that
# will produce every object in this build, and OBJ_DIR (see above) embeds
# CFG_SIG directly in its path -- every distinct config gets its own
# directory, automatically.
#
# This MUST be defined here, before any rule below mentions $(OBJ_DIR) (or
# anything derived from it, e.g. $(OBJS)/$(EXAMPLE_OBJS)/$(BENCH_OBJS) used
# in a prerequisite list): GNU Make expands a rule's target/prerequisite
# list IMMEDIATELY when that line is parsed, not deferred like a recipe
# body, so a forward reference would silently resolve using CFG_SIG's value
# at THAT point in the file (empty, if this line hadn't run yet) -- verified
# empirically with a minimal reproduction.
#
# There is no parse-time comparison and nothing is ever deleted to "detect"
# a config change (the previous scheme did a $(shell rm -rf ...)/$(shell
# mkdir ...)/$(shell echo ...) unconditionally while the Makefile was being
# read, which mutated the filesystem even under `make -n` and raced two
# concurrent builds with different flags against each other's shared obj/
# directory). Under this scheme `make -n` touches nothing on disk (CFG_SIG
# is a pure string hash) and two differently-configured builds (e.g.
# `make BACKEND=kiss -j4 & make BACKEND=ne10 -j4 & wait`) can run
# concurrently in this same tree: each lands in its own obj/<backend>-<sig>/
# directory and never touches the other's objects. OBJ_DIR's own creation
# happens lazily via the order-only prerequisite on the pattern rules below
# ($(OBJ_DIR)/%.o: ... | $(OBJ_DIR)), not at Makefile-parse time.
#
# cksum (POSIX -- present on macOS/BSD/GNU/busybox, unlike shasum which is a
# GNU coreutils extension and may be absent from slim embedded build images)
# computes the hash. On stdin it prints "<checksum> <byte-count>" with no
# filename field (verified on macOS's BSD cksum), so `cut -d' ' -f1` isolates
# just the checksum.
CFG_SIG := $(shell printf '%s' "$(CC) $(CFLAGS) $(EXTRA_CFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR)" | cksum | cut -d' ' -f1)

# Algorithm sources (auto-detected — drop a new file in src/ to add it).
# fft_wrapper*.c no longer live here; they're part of the audio_common archive
# linked above.
SRCS = $(wildcard $(SRC_DIR)/*.c)

OBJS  = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

EXAMPLE_SRCS = $(EXAMPLE_DIR)/aec_wav.c
EXAMPLE_OBJS = $(patsubst $(EXAMPLE_DIR)/%.c,$(OBJ_DIR)/%.o,$(EXAMPLE_SRCS))

TEST_DIR = test

TARGET     = $(BIN_DIR)/aec_wav
LIB_TARGET = $(BIN_DIR)/libaec.a
BENCH_SRCS = $(TEST_DIR)/bench_rtf.c
BENCH_OBJS = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(BENCH_SRCS))
BENCH_TARGET = $(BIN_DIR)/bench_rtf

# Standalone bit-exactness + microbench harness for the AEC-only kernel split
# (aec_simd_kernels.h). Header-only + <math.h> -- no audio_common archive
# link needed (unlike bench_rtf/aec_wav), so this target does not depend on
# audio_common_lib.
SELFTEST_SRC    = $(TEST_DIR)/simd_selftest_aec.c
SELFTEST_TARGET = $(BIN_DIR)/simd_selftest_aec

all: $(TARGET)

# $(OBJ_DIR)/$(BIN_DIR) creation is an order-only prerequisite on the actual
# compile/link rules below (not a $(shell mkdir) at parse time, and not a
# plain prerequisite that could race a parallel compile under `make -j`) --
# every rule that writes into one of these directories depends on it
# existing first, and mkdir -p is idempotent if two rules' order-only
# prereqs resolve concurrently. `dirs` is kept as a convenience alias for
# any caller used to invoking it directly.
$(OBJ_DIR) $(BIN_DIR):
	@mkdir -p $@

dirs: $(OBJ_DIR) $(BIN_DIR)

# Order-only prereq: make sure the shared archive is built (for the selected
# BACKEND) before anything links against it.
audio_common_lib:
	$(MAKE) -C $(AC_DIR) BACKEND=$(BACKEND) lib

$(TARGET): $(OBJS) $(EXAMPLE_OBJS) | audio_common_lib $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(EXAMPLE_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

lib: $(OBJS) | audio_common_lib $(BIN_DIR)
	ar rcs $(LIB_TARGET) $(OBJS)

# Per-module RTF (real-time-factor) timing harness — no Python/corpus needed.
# Linked the same way as bin/aec_wav (same algorithm objects + audio_common
# archive), so it respects BACKEND and EXTRA_CFLAGS identically.
bench: $(BENCH_TARGET)

$(BENCH_TARGET): $(OBJS) $(BENCH_OBJS) | audio_common_lib $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Bit-exactness + microbench self-test for the AEC-only kernel split
# (include/aec_simd_kernels.h). Unlike bench/aec_wav this links with -lm
# only (no audio_common archive, no audio_common_lib prereq) -- the header
# is self-contained aside from transitively pulling in simd_kernels.h via
# CFLAGS' -I$(AC_DIR)/include. Builds then runs immediately.
selftest: $(SELFTEST_TARGET)
	./$(SELFTEST_TARGET)

$(SELFTEST_TARGET): $(SELFTEST_SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< -lm

# Build Configuration Flags (pass via EXTRA_CFLAGS)
#
# USE_STANDARD_MATH - Use standard math library functions (for debugging/parity check)
#   Default: fast_math.h approximations for expf/sqrtf in hot paths
#   Example: make EXTRA_CFLAGS="-DUSE_STANDARD_MATH"

debug: CFLAGS += -g -DAEC_DEBUG
debug: $(TARGET)

clean:
	rm -rf $(OBJ_ROOT) $(BIN_DIR)

help:
	@echo "Targets:"
	@echo "  all     Build CLI binary (bin/aec_wav)  [default]"
	@echo "  lib     Build static library (bin/libaec.a)"
	@echo "  bench   Build RTF timing harness (bin/bench_rtf)"
	@echo "  selftest Build + run the AEC-only kernel-split self-test (bin/simd_selftest_aec)"
	@echo "  debug   Build with -g -DAEC_DEBUG"
	@echo "  clean   Remove obj/ and bin/"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/aec_wav <mic.wav> <ref.wav> <out.wav> [options]"
	@echo "  ./bin/bench_rtf [--seconds N] [--repeat M] [--preset mild|balanced|aggressive]"
	@echo "  make BACKEND=kiss|ne10   (default: kiss; ne10 is the embedded deliverable)"
	@echo "  make AC_DIR=<path>       (override audio_common location)"
	@echo "  make WERROR=1            (opt-in: -Werror for this repo's own sources)"
	@echo ""
	@echo "Objects live in obj/<backend>-<config-hash>/ -- switching BACKEND,"
	@echo "EXTRA_CFLAGS, or WERROR always lands in a fresh directory automatically;"
	@echo "no manual clean needed, and two differently-configured builds can run"
	@echo "concurrently in this same tree."

.PHONY: all dirs clean lib bench selftest debug help audio_common_lib
