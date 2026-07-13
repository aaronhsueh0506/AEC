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

# Algorithm sources (auto-detected — drop a new file in src/ to add it).
# fft_wrapper*.c no longer live here; they're part of the audio_common archive
# linked above.
SRCS = $(wildcard $(SRC_DIR)/*.c)

OBJS  = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

EXAMPLE_SRCS = $(EXAMPLE_DIR)/aec_wav.c
EXAMPLE_OBJS = $(patsubst $(EXAMPLE_DIR)/%.c,$(OBJ_DIR)/%.o,$(EXAMPLE_SRCS))

TARGET     = $(BIN_DIR)/aec_wav
LIB_TARGET = $(BIN_DIR)/libaec.a

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

lib: dirs $(OBJS) | audio_common_lib
	ar rcs $(LIB_TARGET) $(OBJS)

# Build Configuration Flags (pass via EXTRA_CFLAGS)
#
# USE_STANDARD_MATH - Use standard math library functions (for debugging/parity check)
#   Default: fast_math.h approximations for expf/sqrtf in hot paths
#   Example: make EXTRA_CFLAGS="-DUSE_STANDARD_MATH"
#
# AEC_FAST_MATH - Opt-in fast matched-filter arithmetic (default: OFF).
#   Switches the delay estimator's two 512-tap dot products to float32
#   accumulation (what WebRTC's NEON matched filter does) and slides x2_sum
#   across the 16 sub-block iterations (O(1)/iter instead of a 512-tap
#   re-sum); uses NEON fmla intrinsics for the dots + NLMS update on
#   __ARM_NEON targets (scalar float fallback elsewhere).
#   ⚠ DIVERGES from the bit-exact Python reference (default builds stay
#   byte-identical). The integer delay output can differ on near-tie votes.
#   Quality-gated: run the 800-case bench before shipping with this on.
#   Example: make EXTRA_CFLAGS="-DAEC_FAST_MATH"

debug: CFLAGS += -g -DAEC_DEBUG
debug: dirs $(TARGET)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

help:
	@echo "Targets:"
	@echo "  all     Build CLI binary (bin/aec_wav)  [default]"
	@echo "  lib     Build static library (bin/libaec.a)"
	@echo "  debug   Build with -g -DDEBUG"
	@echo "  clean   Remove obj/ and bin/"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/aec_wav <mic.wav> <ref.wav> <out.wav> [options]"
	@echo "  make BACKEND=kiss|ne10   (default: kiss; ne10 is the embedded deliverable)"
	@echo "  make AC_DIR=<path>       (override audio_common location)"

.PHONY: all dirs clean lib debug help audio_common_lib
