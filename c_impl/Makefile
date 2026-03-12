# AEC (Acoustic Echo Cancellation) Makefile
# PBFDAF with error-based DTD

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
       $(SRC_DIR)/subband_nlms.c \
       $(SRC_DIR)/res_filter.c \
       $(SRC_DIR)/fft_wrapper.c

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
	@echo "  ./bin/aec_wav <mic.wav> <ref.wav> <output.wav>"
	@echo ""
	@echo "Input files:"
	@echo "  mic.wav - Microphone input (near-end + echo)"
	@echo "  ref.wav - Reference signal (far-end/loudspeaker)"
	@echo "  output.wav - Echo-cancelled output"

.PHONY: all dirs clean lib help debug
