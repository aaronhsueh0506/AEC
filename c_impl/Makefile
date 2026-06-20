# AEC C Implementation — Makefile

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -ffp-contract=off -std=c99
CFLAGS += -I./include -I./example
CFLAGS += $(EXTRA_CFLAGS)
LDFLAGS = -lm

SRC_DIR     = src
LIB_DIR     = lib
EXAMPLE_DIR = example
OBJ_DIR     = obj
BIN_DIR     = bin

# Algorithm sources (auto-detected — drop a new file in src/ to add it).
# The FFT backend is selected at build time (below) and excluded from the
# wildcard so exactly one backend compiles.
ALL_SRCS = $(wildcard $(SRC_DIR)/*.c)
SRCS = $(filter-out $(SRC_DIR)/fft_wrapper.c $(SRC_DIR)/fft_wrapper_ne10.c,$(ALL_SRCS))

# FFT backend selection. KISS/NE10 compute in float32 -> ~1e-5 vs numpy's fp64
# np.fft (a documented tolerance; the non-FFT C logic stays bit-exact to Python).
#   default          -> KISS FFT  (portable, vendored lib/kiss_fft)
#   make NE10_DIR=... -> NE10 R2C/C2R (ARM NEON; caller supplies the Ne10 lib)
ifdef NE10_DIR
  SRCS    += $(SRC_DIR)/fft_wrapper_ne10.c
  CFLAGS  += -I$(NE10_DIR)/inc
  LDFLAGS += -L$(NE10_DIR)/lib -lNE10
  LIB_OBJS =
else
  SRCS    += $(SRC_DIR)/fft_wrapper.c
  CFLAGS  += -I./lib/kiss_fft
  LIB_OBJS = $(OBJ_DIR)/kiss_fft.o
endif

OBJS  = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
OBJS += $(LIB_OBJS)

EXAMPLE_SRCS = $(EXAMPLE_DIR)/aec_wav.c
EXAMPLE_OBJS = $(patsubst $(EXAMPLE_DIR)/%.c,$(OBJ_DIR)/%.o,$(EXAMPLE_SRCS))

TARGET     = $(BIN_DIR)/aec_wav
LIB_TARGET = $(BIN_DIR)/libaec.a

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

$(TARGET): $(OBJS) $(EXAMPLE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/kiss_fft.o: $(LIB_DIR)/kiss_fft/kiss_fft.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(EXAMPLE_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

lib: dirs $(OBJS)
	ar rcs $(LIB_TARGET) $(OBJS)

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
	@echo "  debug   Build with -g -DDEBUG"
	@echo "  clean   Remove obj/ and bin/"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/aec_wav <mic.wav> <ref.wav> <out.wav> [options]"

.PHONY: all dirs clean lib debug help
