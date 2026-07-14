# AEC C Implementation — Makefile

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -ffp-contract=off -std=gnu99
CFLAGS += -I./include -I./example
CFLAGS += $(EXTRA_CFLAGS)
LDFLAGS = -lm

SRC_DIR     = src
EXAMPLE_DIR = example
OBJ_DIR     = obj
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

# --- Build-configuration stamp (F12 build hygiene) --------------------------
# obj/ is a single flat directory shared across every BACKEND/EXTRA_CFLAGS
# combination, so switching either on an already-built tree used to silently
# keep stale .o files around (they don't encode which flags produced them).
# CFG_SIG hashes the exact compiler invocation that will produce every object
# in this build.
#
# This check runs at PARSE time (via $(shell ...), not as a build-rule
# recipe): tried an order-only-prerequisite version first (a stamp FILE that
# a rule creates, referenced via `obj/%.o: %.c | $(CFG_STAMP)`), but that is
# racy -- verified empirically. When N objects share one order-only
# prerequisite whose recipe wipes the directory they all live in, GNU Make
# decides "is object #1 out of date?" using the prerequisite tree it already
# resolved, and for at least one sibling object that decision is finalized
# before the shared prerequisite's wipe recipe actually runs, so that object
# silently keeps its (about to be deleted) stale .o and is never recompiled.
# Doing the compare-and-wipe unconditionally while the Makefile is being
# read sidesteps the whole issue: by the time `make` starts asking "is any
# target out of date", obj/ has already been wiped (if needed) or left
# alone (if not) -- no build-graph race is possible.
CFG_SIG        := $(shell printf '%s' "$(CC) $(CFLAGS) $(EXTRA_CFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR)" | shasum -a 256 | cut -c1-12)
CFG_STAMP_FILE := $(OBJ_DIR)/.cfg_stamp
CFG_LAST_SIG   := $(shell test -f $(CFG_STAMP_FILE) && cat $(CFG_STAMP_FILE))
ifneq ($(CFG_SIG),$(CFG_LAST_SIG))
$(info AEC c_impl: build config changed -- wiping $(OBJ_DIR) for a full rebuild)
$(shell rm -rf $(OBJ_DIR))
$(shell mkdir -p $(OBJ_DIR))
$(shell echo $(CFG_SIG) > $(CFG_STAMP_FILE))
endif

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

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Order-only prereq: make sure the shared archive is built (for the selected
# BACKEND) before anything links against it.
audio_common_lib:
	$(MAKE) -C $(AC_DIR) BACKEND=$(BACKEND) lib

$(TARGET): $(OBJS) $(EXAMPLE_OBJS) | audio_common_lib
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(EXAMPLE_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

lib: dirs $(OBJS) | audio_common_lib
	ar rcs $(LIB_TARGET) $(OBJS)

# Per-module RTF (real-time-factor) timing harness — no Python/corpus needed.
# Linked the same way as bin/aec_wav (same algorithm objects + audio_common
# archive), so it respects BACKEND and EXTRA_CFLAGS identically.
bench: dirs $(BENCH_TARGET)

$(BENCH_TARGET): $(OBJS) $(BENCH_OBJS) | audio_common_lib
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Bit-exactness + microbench self-test for the AEC-only kernel split
# (include/aec_simd_kernels.h). Unlike bench/aec_wav this links with -lm
# only (no audio_common archive, no audio_common_lib prereq) -- the header
# is self-contained aside from transitively pulling in simd_kernels.h via
# CFLAGS' -I$(AC_DIR)/include. Builds then runs immediately.
selftest: dirs $(SELFTEST_TARGET)
	./$(SELFTEST_TARGET)

$(SELFTEST_TARGET): $(SELFTEST_SRC)
	$(CC) $(CFLAGS) -o $@ $< -lm

# Build Configuration Flags (pass via EXTRA_CFLAGS)
#
# USE_STANDARD_MATH - Use standard math library functions (for debugging/parity check)
#   Default: fast_math.h approximations for expf/sqrtf in hot paths
#   Example: make EXTRA_CFLAGS="-DUSE_STANDARD_MATH"

debug: CFLAGS += -g -DAEC_DEBUG
debug: dirs $(TARGET)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

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

.PHONY: all dirs clean lib bench selftest debug help audio_common_lib
