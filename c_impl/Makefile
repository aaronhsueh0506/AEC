# AEC (Acoustic Echo Cancellation) Makefile
# PBFDKF + Shadow + WOLA RES + Preset

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
CFLAGS += -I./include -I./lib/kiss_fft -I./example
CFLAGS += $(EXTRA_CFLAGS)

# Debug flags (uncomment for debugging)
# CFLAGS += -g -DDEBUG

LDFLAGS = -lm

# Directories
SRC_DIR = src
LIB_DIR = lib/kiss_fft
EXAMPLE_DIR = example
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRCS = $(SRC_DIR)/aec.c \
       $(SRC_DIR)/pbfdkf.c \
       $(SRC_DIR)/res_filter.c \
       $(SRC_DIR)/fft_wrapper.c \
       $(SRC_DIR)/multi_erle.c \
       $(SRC_DIR)/render_activity.c \
       $(SRC_DIR)/convergence.c \
       $(SRC_DIR)/dt_analyzer.c \
       $(SRC_DIR)/epc_detector.c \
       $(SRC_DIR)/shadow_copy.c

LIB_SRCS = $(LIB_DIR)/kiss_fft.c

EXAMPLE_SRCS = $(EXAMPLE_DIR)/main.c

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
LIB_OBJS = $(patsubst $(LIB_DIR)/%.c,$(OBJ_DIR)/%.o,$(LIB_SRCS))
EXAMPLE_OBJS = $(patsubst $(EXAMPLE_DIR)/%.c,$(OBJ_DIR)/%.o,$(EXAMPLE_SRCS))

# Target binary
TARGET = $(BIN_DIR)/aec_wav

# Default target
all: dirs $(TARGET)

# Create directories
dirs:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)

# Link
$(TARGET): $(OBJS) $(LIB_OBJS) $(EXAMPLE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $@"

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile library files
$(OBJ_DIR)/%.o: $(LIB_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile example files
$(OBJ_DIR)/%.o: $(EXAMPLE_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: dirs $(TARGET)
	@echo "Debug build complete: $(TARGET)"

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Clean complete"

# Build static library
LIB_TARGET = $(BIN_DIR)/libaec.a

lib: dirs $(OBJS) $(LIB_OBJS)
	ar rcs $(LIB_TARGET) $(OBJS) $(LIB_OBJS)
	@echo "Static library built: $(LIB_TARGET)"

# Phase 1 parity verifier
PARITY_PBFDKF_TARGET = $(BIN_DIR)/parity_pbfdkf_test
PARITY_PBFDKF_OBJ = $(OBJ_DIR)/parity_pbfdkf_test.o

parity_pbfdkf: dirs lib $(PARITY_PBFDKF_OBJ)
	$(CC) $(CFLAGS) -o $(PARITY_PBFDKF_TARGET) $(PARITY_PBFDKF_OBJ) \
	    $(LIB_TARGET) $(LDFLAGS)
	@echo "Parity verifier built: $(PARITY_PBFDKF_TARGET)"

# Phase 2a parity verifier
PARITY_MULTI_ERLE_TARGET = $(BIN_DIR)/parity_multi_erle_test
PARITY_MULTI_ERLE_OBJ = $(OBJ_DIR)/parity_multi_erle_test.o

parity_multi_erle: dirs lib $(PARITY_MULTI_ERLE_OBJ)
	$(CC) $(CFLAGS) -o $(PARITY_MULTI_ERLE_TARGET) $(PARITY_MULTI_ERLE_OBJ) \
	    $(LIB_TARGET) $(LDFLAGS)
	@echo "Parity verifier built: $(PARITY_MULTI_ERLE_TARGET)"

# Phase 2bcd parity verifier (RenderActivity + DT + Convergence)
PARITY_AUX_TARGET = $(BIN_DIR)/parity_aux_test
PARITY_AUX_OBJ = $(OBJ_DIR)/parity_aux_test.o

parity_aux: dirs lib $(PARITY_AUX_OBJ)
	$(CC) $(CFLAGS) -o $(PARITY_AUX_TARGET) $(PARITY_AUX_OBJ) \
	    $(LIB_TARGET) $(LDFLAGS)
	@echo "Parity verifier built: $(PARITY_AUX_TARGET)"

# Phase 4 parity (EPC + ShadowCopy)
PARITY_PHASE4_TARGET = $(BIN_DIR)/parity_phase4_test
PARITY_PHASE4_OBJ = $(OBJ_DIR)/parity_phase4_test.o

parity_phase4: dirs lib $(PARITY_PHASE4_OBJ)
	$(CC) $(CFLAGS) -o $(PARITY_PHASE4_TARGET) $(PARITY_PHASE4_OBJ) \
	    $(LIB_TARGET) $(LDFLAGS)
	@echo "Parity verifier built: $(PARITY_PHASE4_TARGET)"

# Phase 3 parity (ResFilter)
PARITY_RESFILTER_TARGET = $(BIN_DIR)/parity_resfilter_test
PARITY_RESFILTER_OBJ = $(OBJ_DIR)/parity_resfilter_test.o

parity_resfilter: dirs lib $(PARITY_RESFILTER_OBJ)
	$(CC) $(CFLAGS) -o $(PARITY_RESFILTER_TARGET) $(PARITY_RESFILTER_OBJ) \
	    $(LIB_TARGET) $(LDFLAGS)
	@echo "Parity verifier built: $(PARITY_RESFILTER_TARGET)"

# Help
help:
	@echo "AEC (Acoustic Echo Cancellation) Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all    - Build the AEC executable (default)"
	@echo "  lib    - Build static library"
	@echo "  debug  - Build with debug symbols"
	@echo "  clean  - Remove build artifacts"
	@echo "  help   - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  ./bin/aec_wav <mic.wav> <ref.wav> <output.wav> [--preset balanced|aggressive|maximum]"
	@echo ""
	@echo "Options:"
	@echo "  --preset <name>    - balanced (default), aggressive, maximum"
	@echo "  --no-res           - Disable RES post-filter"
	@echo "  --filter <samples> - Filter length in samples (default: 512)"

.PHONY: all dirs clean lib help debug
