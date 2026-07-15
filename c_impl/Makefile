# AEC C Implementation — Makefile

CC      ?= gcc
CXX     ?= c++
AR      ?= ar
RANLIB  ?= ranlib

# CFG_SIG payload-coverage placeholders (round-3 review B01): CXXFLAGS/
# CPPFLAGS/EXTRA_LDFLAGS/NO_STDIO/DEBUG are not consumed by this Makefile
# today (this repo has no C++ TUs of its own, and no NO_STDIO/DEBUG-gated
# code path), but the CFG_SIG payload below unconditionally includes all of
# them so the payload string has the same stable shape as audio_common's and
# NR's, and so a future use of any of them here keys builds correctly
# instead of silently aliasing with a build that didn't set it.
CXXFLAGS      ?=
CPPFLAGS      ?=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

CFLAGS  = -Wall -Wextra -O2 -ffp-contract=off -std=gnu99
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
AC_FWD = BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO)

ifeq ($(AC_LIB),)
.PHONY: all bench
all bench:
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
# the BACKEND-conditional -lc++ and WERROR appends above) and after the
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
CFG_SIG_PAYLOAD := CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) LDFLAGS=$(LDFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) AC_PRODUCER_SIG=$(AC_PRODUCER_SIG)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

OBJ_ROOT = obj
OBJ_DIR  = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR  = bin/$(BACKEND)-$(CFG_SIG)

# Algorithm sources (auto-detected — drop a new file in src/ to add it).
# fft_wrapper*.c no longer live here; they're part of the audio_common archive
# linked above.
SRCS = $(wildcard $(SRC_DIR)/*.c)

OBJS  = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

EXAMPLE_SRCS = $(EXAMPLE_DIR)/aec_wav.c
EXAMPLE_OBJS = $(patsubst $(EXAMPLE_DIR)/%.c,$(OBJ_DIR)/%.o,$(EXAMPLE_SRCS))

TARGET     = $(BIN_DIR)/aec_wav
LIB_TARGET = $(BIN_DIR)/libaec.a
BENCH_SRCS = $(TEST_DIR)/bench_rtf.c
BENCH_OBJS = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/%.o,$(BENCH_SRCS))
BENCH_TARGET = $(BIN_DIR)/bench_rtf

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
$(LIB_TARGET): $(OBJS) | _cfg_guard
	$(AR) rc $@ $(OBJS)
	$(RANLIB) $@
	@echo "aec [$(BACKEND)] -> $@"

# Bit-exactness + microbench self-test for the AEC-only kernel split
# (include/aec_simd_kernels.h). Links with -lm only (no audio_common
# archive, no AC_LIB dependency) -- the header is self-contained aside from
# transitively pulling in simd_kernels.h via CFLAGS' -I$(AC_DIR)/include.
.PHONY: selftest
selftest: $(SELFTEST_TARGET)
	./$(SELFTEST_TARGET)

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
.PHONY: all bench FORCE
all: $(TARGET)
bench: $(BENCH_TARGET)

FORCE:

$(AC_LIB): FORCE
	+$(MAKE) -C $(AC_DIR) $(AC_FWD) lib
	@test -f '$@' || { echo "ERROR: audio_common did not produce $@ (query/build divergence -- only BACKEND/WERROR/CC/CXX/EXTRA_CFLAGS/NO_STDIO overrides are supported)"; exit 1; }

$(TARGET): $(OBJS) $(EXAMPLE_OBJS) $(AC_LIB) | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "aec_wav [$(BACKEND)] -> $@"

$(BENCH_TARGET): $(OBJS) $(BENCH_OBJS) $(AC_LIB) | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "bench_rtf [$(BACKEND)] -> $@"
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
# publish: immutable per-config release under dist/ (round-3 review B01) --
# same design as audio_common's publish (see that Makefile for the full
# writeup): dist/<backend>/<cfg_sig>/ + dist/<backend>/current symlink, a
# per-backend mkdir-lock serialising concurrent publishes, MANIFEST.txt with
# relative paths + per-file sha256 + this build's resolved AC_PRODUCER_SIG
# (the audio_common config this release was linked against). PUBLISH_ARTIFACTS
# needs $(AC_LIB) to exist, so `publish` goes through the same two-phase
# dispatch as `all`/`bench` when AC_LIB isn't already known.
# =============================================================================
PUBLISH_ARTIFACTS = $(LIB_TARGET) $(TARGET) $(BENCH_TARGET)
DIST_BACKEND_DIR  = dist/$(BACKEND)
DIST_REL_DIR      = $(DIST_BACKEND_DIR)/$(CFG_SIG)
DIST_LOCK         = dist/.lock.$(BACKEND)

ifeq ($(AC_LIB),)
.PHONY: publish
publish:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) $(AC_FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audio_common archive path" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory publish AC_LIB="$$ac"
else
.PHONY: publish
publish: $(PUBLISH_ARTIFACTS)
	@mkdir -p dist; \
	if ! mkdir "$(DIST_LOCK)" 2>/dev/null; then \
	  echo "FATAL: publish lock $(DIST_LOCK) already held (concurrent 'make publish BACKEND=$(BACKEND)'?)" >&2; \
	  exit 1; \
	fi; \
	trap 'rmdir "$(DIST_LOCK)" 2>/dev/null' EXIT INT TERM; \
	set -e; \
	commit="$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"; \
	dateutc="$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	mkdir -p "$(DIST_BACKEND_DIR)"; \
	tmp="$(DIST_BACKEND_DIR)/$(CFG_SIG).tmp"; \
	rm -rf "$$tmp"; mkdir -p "$$tmp"; \
	cp $(PUBLISH_ARTIFACTS) "$$tmp/"; \
	shas="$$( cd "$$tmp" && for f in *; do shasum -a 256 "$$f"; done )"; \
	{ echo "repo=aec"; \
	  echo "backend=$(BACKEND)"; \
	  echo "cfg_sig=$(CFG_SIG)"; \
	  echo "producer_cfg_sig=$(AC_PRODUCER_SIG)"; \
	  echo "cc=$(CC)"; echo "cxx=$(CXX)"; \
	  echo "cflags=$(CFLAGS)"; \
	  echo "extra_cflags=$(EXTRA_CFLAGS)"; \
	  echo "werror=$(WERROR)"; echo "no_stdio=$(NO_STDIO)"; echo "debug=$(DEBUG)"; \
	  echo "git_commit=$$commit"; echo "date_utc=$$dateutc"; \
	  echo "$$shas"; \
	} > "$$tmp/MANIFEST.txt"; \
	rm -rf "$(DIST_REL_DIR)"; \
	mv "$$tmp" "$(DIST_REL_DIR)"; \
	ln -s "$(CFG_SIG)" "$(DIST_BACKEND_DIR)/.current.tmp"; \
	mv -f "$(DIST_BACKEND_DIR)/.current.tmp" "$(DIST_BACKEND_DIR)/current"; \
	echo "aec: published $(BACKEND)/$(CFG_SIG) -> $(DIST_BACKEND_DIR)/current"
endif

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
	rm -rf $(OBJ_ROOT) bin

help:
	@echo "Targets:"
	@echo "  all     Build CLI binary (aec_wav)  [default]"
	@echo "  lib     Build static library (libaec.a)"
	@echo "  bench   Build RTF timing harness (bench_rtf)"
	@echo "  selftest Build + run the AEC-only kernel-split self-test (simd_selftest_aec)"
	@echo "  debug   Build with -g -DAEC_DEBUG"
	@echo "  publish Copy this build's artifacts to dist/<backend>/<config-hash>/ + update dist/<backend>/current"
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
