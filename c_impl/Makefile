# AEC C Implementation — Makefile

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -ffp-contract=off -std=c99
CFLAGS += -I./include -I./example -I./lib/pocketfft
CFLAGS += $(EXTRA_CFLAGS)
LDFLAGS = -lm

SRC_DIR     = src
LIB_DIR     = lib
EXAMPLE_DIR = example
OBJ_DIR     = obj
BIN_DIR     = bin

# All algorithm sources (auto-detected — drop a new file in src/ to add it).
# fft_pocketfft.c is the FFT backend; it links the vendored numpy 1.26.4
# pocketfft core (lib/pocketfft/pocketfft.c) for BIT-EXACT np.fft parity.
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# Vendored third-party FFT core (kept out of src/ so it is not auto-detected
# as an algorithm module; built explicitly with its own rule).
LIB_SRCS = $(LIB_DIR)/pocketfft/pocketfft.c
LIB_OBJS = $(OBJ_DIR)/pocketfft.o
OBJS    += $(LIB_OBJS)

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

$(OBJ_DIR)/pocketfft.o: $(LIB_DIR)/pocketfft/pocketfft.c
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
