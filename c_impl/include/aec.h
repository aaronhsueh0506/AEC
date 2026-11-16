/* Public C API for the PBFDKF AEC: a single-channel instance or a four-microphone array instance (AecConfig.n_mics), heap
 * or caller-pool construction, one- or two-stage lockstep processing, and the residual / linear views for an external
 * post-filter.
 *
 * Single-channel build. The library built with -DAEC_NO_ARRAY (make AEC_NO_ARRAY=1) has no array instance: AecConfig.n_mics
 * other than 1 is refused (aec_get_mem_size() 0, aec_init() NULL, aec_create() NULL), and aec_get_array() and the aec_array_*()
 * diagnostics are not defined (a call is a link error). Every other function behaves as for a single-channel instance of the
 * full library, byte for byte. The header is the same in both builds. */
#ifndef AEC_AEC_H
#define AEC_AEC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "fft_wrapper.h"

/* The instance is opaque: create it with aec_create() or place it in a caller pool with aec_init(). */
typedef struct Aec Aec;

/* Residual-echo strength level, ordered 0..3 (default BALANCED = 1): the far-active minimum-gain floor is -20 / -28 / -38 dB
 * for MILD / BALANCED / AGGRESSIVE; NO_FLOOR switches the split floor off. AecConfig.level is a plain number in 0..3; a value
 * outside it is refused by create / init / the size query. */
typedef enum {
    AEC_LEVEL_MILD       = 0,
    AEC_LEVEL_BALANCED   = 1,
    AEC_LEVEL_AGGRESSIVE = 2,
    AEC_LEVEL_NO_FLOOR   = 3
} AecLevel;

/* "mild", "balanced", "aggressive" or "no_floor"; any value outside the enum reads "balanced", as aec_config_from_level() treats
 * it. Never NULL; the string is static. */
const char* aec_level_name(AecLevel level);

/* Delay mode (init-time, immutable): how the far-end is aligned to the mic.
 *   MATCHED           a matched-filter bank estimates the delay online and a
 *                     reference ring applies it. Needs delay_num_filters in
 *                     [1, 5] and fixed_delay_samples < 0.
 *   FIXED             the caller-measured delay is applied by the ring. Needs
 *                     fixed_delay_samples >= 0.
 *   EXTERNAL_ALIGNED  no estimator, no ring; ref is already aligned to mic.
 *                     Needs fixed_delay_samples < 0.
 * Illegal mode/field combinations are rejected by create / init / size query. */
typedef enum AecDelayMode {
    AEC_DELAY_MATCHED = 0,
    AEC_DELAY_FIXED = 1,
    AEC_DELAY_EXTERNAL_ALIGNED = 2
} AecDelayMode;

/* What the residual stage (aec_process_residual) produces. Ordered; each level adds to the one before.
 *   OFF       linear only. There is no residual stage; aec_process_residual() returns AEC_ERR_MODE.
 *   STATE     the selected linear residual E(f) with its echo and near spectra, the rendering base, usable_linear
 *             and the status flags.
 *   ESTIMATE  + the residual-echo PSD r2 and the comfort-noise PSD (a lane whose gain is computed elsewhere).
 *   FULL      + the residual suppression gain G_res (the default; aec_process() applies it). */
typedef enum AecResidualMode {
    AEC_RESIDUAL_OFF      = 0,
    AEC_RESIDUAL_STATE    = 1,
    AEC_RESIDUAL_ESTIMATE = 2,
    AEC_RESIDUAL_FULL     = 3
} AecResidualMode;

#define AEC_ARRAY_MICS  4

/* Result of every aec_process_* / aec_apply_* call, of the view getters (aec_get_residual / aec_get_linear / aec_get_array) and
 * of aec_resolve_signal_grid() / aec_get_mem_breakdown(): 0, or a negative reason. A refused call leaves the instance untouched,
 * except AEC_ERR_DSP below; a view getter or breakdown call zeroes its `out` record as each function states. aec_get_mem_requirements, aec_set_level and aec_synthesis_window
 * follow the same sign (0 = success, -1 = failure). aec_apply_external_realign() returns an outcome (1 / 0 / -1), and
 * aec_is_valid_sample_rate() is a predicate (1 / 0). */
typedef enum AecStatus {
    AEC_OK           =  0,
    AEC_ERR_ARG      = -1,   /* NULL, a short struct_size, or an argument out of range */
    AEC_ERR_MODE     = -2,   /* the call needs a higher AecConfig.residual than the instance has */
    AEC_ERR_SEQUENCE = -3,   /* a call out of order (residual before linear, linear twice, ...) */
    AEC_ERR_DSP      = -4    /* the hop was refused (finite input samples whose products overflow, or an array stage's own
                              * check); the instance has been reset. See "Refused hop". A non-finite INPUT sample is
                              * AEC_ERR_ARG and changes nothing. */
} AecStatus;

/* Delay estimator geometry. The matched-filter bank holds at most AEC_DELAY_NUM_FILTERS_MAX filters (AecConfig.delay_num_filters
 * is in [1, AEC_DELAY_NUM_FILTERS_MAX]). AEC_DELAY_FILTER_SIZE and AEC_DELAY_FILTER_INTRA_SHIFT count decimated samples: one
 * estimator sample is AEC_DELAY_DOWN_SAMPLING_FACTOR samples of the (at most 16 kHz) feed. */
#define AEC_DELAY_NUM_FILTERS_MAX       5
#define AEC_DELAY_FILTER_SIZE           512
#define AEC_DELAY_FILTER_INTRA_SHIFT    384
#define AEC_DELAY_DOWN_SAMPLING_FACTOR  4

/* Seed of the comfort-noise generator of a fresh or hard-reset instance (see aec_post_reset_serial()). */
#define AEC_CNG_SEED 42u

/* int16^2 power scale of AecResidualView.r2 / comfort_noise: audio-scale power = value * AEC_PSD_SCALE_INV_F. Both are exact in
 * float (powers of two). */
#define AEC_PSD_SCALE_F      (32768.0f * 32768.0f)
#define AEC_PSD_SCALE_INV_F  (1.0f / AEC_PSD_SCALE_F)

typedef struct AecConfig {
    int    sample_rate;        /* Hz; 8000, 16000 or 48000 */
    int    fft_size;           /* samples; 256 or 512 @16k, 1024 @48k; 0 is rejected */
    /* Microphone array: 1 (default) is the single-channel instance this header describes; AEC_ARRAY_MICS (4) builds an
     * array instance: four linear lanes, one shared delay estimator and aligned far, and the fusion of the lanes by the
     * caller's beamformer weights (see "Array instance" below). On an array instance delay_mode / delay_num_filters /
     * fixed_delay_samples describe the SHARED alignment (each lane then takes the aligned far, EXTERNAL_ALIGNED) and
     * `residual` must be STATE, ESTIMATE or FULL. */
    int    n_mics;                              /* 1; 4 only in a library built without AEC_NO_ARRAY */
    /* Residual-echo strength; the only strength setting (aec_set_level() changes it on a running instance). */
    AecLevel level;                             /* AEC_LEVEL_BALANCED */
    /* How far the residual stage goes: AecResidualMode. aec_process_residual() and the getters follow it.
     * The pool size of a single-channel instance does not depend on it; an array's grows with it (STATE: lanes only,
     * ESTIMATE: + the fusion buffers, FULL: + the gain stage). */
    AecResidualMode residual;      /* AEC_RESIDUAL_FULL */
    /* Alignment mode; init-time immutable (the pool layout depends on it). */
    AecDelayMode delay_mode;       /* AEC_DELAY_MATCHED */
    /* FIXED only: measured delay in samples (>= 0), also sizes the reference ring; must be -1 in the other modes. */
    int    fixed_delay_samples;    /* -1 */
    /* MATCHED only: matched-filter bank size in [1, 5] (part of the pool layout, init-time immutable); n = 1..5 gives a
     * reliable reach of 125/221/317/413/509 ms. A non-default value outside MATCHED is rejected. */
    int    delay_num_filters;      /* 5 */
    int    enable_cng;             /* comfort-noise generation; 1 */
    int    capture_proxy_channel;               /* array: the shared estimator's capture input, [0, n_mics-1]; 0 */
    /* array, MATCHED only, default off: an estimate earlier than the delay in force is held for
     * delay_backward_quarantine_s while the capture-proxy lane still cancels, then adopted; lost cancellation adopts it
     * at once. Forward moves and the first acquisition are never held. */
    int    delay_backward_quarantine_enabled;   /* 0 */
    float  delay_backward_quarantine_s;         /* [0, 3600]; 1.0 */
} AecConfig;

/* Fills the public configuration for a rate: the rate's product fft_size and level `level` (an out-of-enum level reads as
 * AEC_LEVEL_BALANCED), every other field at its default. For a rate with no grid (anything but 8000/16000/48000, including 0
 * and negative values) fft_size is 0, which aec_get_mem_size / aec_init / aec_create refuse. Everything the instance tunes
 * beyond AecConfig follows from (sample_rate, fft_size, n_mics, level) and is not caller-settable. A NULL cfg is ignored. */
void aec_config_from_level(AecConfig* cfg, AecLevel level, int sample_rate);

/* Signal grid: (sample_rate, fft_size) fixes the geometry; frame_size = fft_size
 * and hop_size = fft_size / 2. fft_size == 0 is rejected; the config factories fill
 * a concrete value.
 *   16000 Hz  256 (default) or 512   product
 *   48000 Hz  1024                   product
 *    8000 Hz  256                    supported, not a product grid (is_legacy = 1) */
typedef struct AecSignalGrid {
    int sample_rate;
    int fft_size;     /* == frame_size */
    int frame_size;
    int hop_size;     /* == fft_size / 2 */
    int n_freqs;      /* == fft_size / 2 + 1 */
    int is_legacy;    /* 1 for 8 kHz: supported, not a product grid */
} AecSignalGrid;

/* Resolves (sample_rate, fft_size) into *out. Returns AEC_OK on a supported pair; AEC_ERR_ARG otherwise (out untouched,
 * including fft_size == 0). out may be NULL. */
int aec_resolve_signal_grid(int sample_rate, int fft_size, AecSignalGrid* out);

/* Returns 1 if some grid exists at this rate (8000/16000/48000); the (rate,
 * fft_size) pair must still resolve. */
int aec_is_valid_sample_rate(int sample_rate);

/* Product-default fft_size for a rate (16k: 256, 48k: 1024, 8k: 256); 0 if
 * the rate is unsupported. */
int aec_default_fft_size(int sample_rate);

/* Per-hop wall-clock cost (microseconds) of the five largest stages in
 * aec_process*(). Diagnostic only. delay_us: delay estimation, ring write and
 * compensated read (0 under EXTERNAL_ALIGNED). frontend_us: entry to main
 * filter excluding delay_us. linear_us: main adaptive filter. steering_us:
 * stationarity, coarse-error publish, double-talk analyzer, EPV, shadow_rise,
 * misadjustment estimator. res_us: post-filter/RES block. The five do not sum
 * to the call; fields are cleared at the start of each call. On by default; built with
 * -DAEC_STAGE_TIMING=0 every field reads 0. The stamp is CLOCK_MONOTONIC, or
 * -DAEC_NOW_US=<fn> returning microseconds. */
/* Always defined so consumers can use #if AEC_STAGE_TIMING under -Wundef. */
#ifndef AEC_STAGE_TIMING
#define AEC_STAGE_TIMING 1
#endif

typedef struct AecStageTiming {
    uint32_t delay_us;
    uint32_t frontend_us;
    uint32_t linear_us;
    uint32_t res_us;
    uint32_t steering_us;
} AecStageTiming;

/* Copies the last hop's stage timings into out. A NULL instance zeroes out;
 * out must not be NULL. */
void aec_get_last_timing(const Aec* a, AecStageTiming* out);

/* Heap construction: one aligned allocation of aec_get_mem_size(cfg) bytes; NULL for a NULL or invalid cfg or a failed
 * allocation. aec_destroy() frees it (NULL-safe); aec_reset() restores the initial state. A handle that is not ready (NULL,
 * zero-filled, never built or destroyed) is refused by every entry point; a freed handle or its views must not be used. */
Aec* aec_create(const AecConfig* cfg);
void aec_destroy(Aec* a);
void aec_reset(Aec* a);

/* Realigns an AEC_DELAY_EXTERNAL_ALIGNED instance after a caller-side far alignment change (delta_samples = new - old): a
 * cancelling filter keeps its IR shifted by delta when it fits the tap span, else only the filter resets. Returns 1 for the
 * warm transfer, 0 for a filter reset or delta 0, -1 on NULL or another delay mode; bumps the alignment generation. */
int aec_apply_external_realign(Aec* a, int delta_samples);

/* Retargets the residual-echo strength of a running instance and stores it as the instance's level; filter, delay
 * state and histories carry on. The far-active floor walks to its new value
 * linearly in dB over a fixed 100 ms (a change during a ramp restarts from
 * the live value); the DT floor (aggressive differs) and the split-floor flag (NO_FLOOR clears it) step immediately.
 * Inert on an instance whose residual is below AEC_RESIDUAL_FULL. Call between
 * hops, serialised with aec_process(). Returns 0, or -1 (nothing written) on
 * NULL or an out-of-enum level. */
int aec_set_level(Aec* a, AecLevel level);

/* Static-memory construction: aec_get_mem_size() returns the pool bytes (0 if
 * cfg is invalid). aec_init() places the Aec at mem[0] without malloc and
 * returns it, or NULL for a NULL argument, invalid cfg, a pool not 16-byte
 * aligned or smaller than aec_get_mem_size(). */
size_t aec_get_mem_size(const AecConfig* cfg);
Aec*   aec_init(void* mem, size_t mem_size, const AecConfig* cfg);

/* The caller's beamformer output for an array instance (see "Array instance"). */
typedef struct AecBeam {
    const Complex* weights;           /* channel-major Complex[n_mics][n_freqs]; fused = sum(w * lane), no conjugation */
    const Complex* beamformed_error;  /* the spectrum of the same beamformer call (caller-owned, must outlive the hop), or
                                       * NULL to rebuild it from the weights */
} AecBeam;

/* One hop is hop_size = fft_size / 2 samples of the microphone and of the reference, supplied together
 * (lockstep). Three entry points on one instance; the residual stage's reach is AecConfig.residual.
 *
 *   aec_process           linear + residual and, with AEC_RESIDUAL_FULL, the gain, the comfort noise and the library's
 *                         own synthesis: `out` is the finished hop. Below FULL, `out` is
 *                         AecLinearView.linear_hop (the refined linear output, before FORM).
 *   aec_process_linear    stage 1: front end, delay, shadow and main filter. With AEC_RESIDUAL_OFF it also ends the hop.
 *   aec_process_residual  stage 2, directly after aec_process_linear of the same hop: FORM, the echo-state classifier, r2,
 *                         comfort noise and G_res, as far as AecConfig.residual reaches. It applies nothing and synthesizes nothing:
 *                         read aec_get_residual() and finish with aec_apply_gain() / aec_apply_cng() and the caller's
 *                         own inverse transform. `beam` is for an array instance and must be NULL on a single-channel one.
 *
 * Within a stream the two forms may alternate hop by hop; the library's own overlap-add tail and comfort-noise
 * generator advance only in aec_process() and aec_apply_cng().
 * The linear output depends on AecConfig.residual: OFF uses the simpler step-size rule, STATE / ESTIMATE / FULL share one
 * linear path, so compare a linear part with the full chain only on STATE or above.
 * Returns AEC_OK, or AEC_ERR_ARG / AEC_ERR_MODE / AEC_ERR_SEQUENCE with the instance untouched, or AEC_ERR_DSP (below).
 *
 * Refused hop (AEC_ERR_DSP; an array per lane): a call whose results overflow resets the instance as aec_reset(), moves
 * aec_post_reset_serial(), zeroes `out` and the linear view and leaves no hop open. The caller renders that hop as silence,
 * clears its overlap-add tail and comfort-noise state, restarts stages fed from here and goes on (docs/reference/lifecycle.md). */
int aec_process(Aec* a, const float* mic, const float* ref, float* out);
int aec_process_linear(Aec* a, const float* mic, const float* ref);
int aec_process_residual(Aec* a, const AecBeam* beam);

/* Cumulative count (since construction or aec_reset(), saturating at
 * LONG_MAX) of hops where this instance ran its own far-end FFT. */
long aec_far_fft_real_compute_count(const Aec* a);

/* 1 if the linear filter is demonstrably cancelling: last hop's windowed ERLE
 * above 2.5 dB, or the recent inst-ERLE peak above
 * the internal inst-ERLE threshold (4 dB). 0 means no evidence, not a fault (also 0
 * before the ERLE machinery has run). */
int aec_linear_is_cancelling(const Aec* a);

/* Samples per process call (fft_size / 2); 0 for a handle that is not ready. */
int  aec_hop_size(const Aec* a);

/* Everything the residual stage knows about the last hop; pointers alias internal buffers (valid until the next process or reset
 * call) and follow AecConfig.residual: STATE has no r2 / comfort_noise / res_gain, ESTIMATE no res_gain. r2 and comfort_noise are
 * int16^2 power; render base_spec, not error_spec (output guard). struct_size is an input; AEC_OK, AEC_ERR_ARG, AEC_ERR_MODE. */
typedef struct AecResidualView {
    size_t         struct_size;
    int            n_freqs;
    int            hop_size;
    const float*   formed_hop;
    const Complex* error_spec;
    const Complex* echo_spec;
    const Complex* near_spec;
    const Complex* far_spec;
    const Complex* base_spec;
    int            guard_active;
    const float*   res_gain;         /* G_res(f) in [0,1] */
    const float*   r2;               /* residual-echo PSD R^2(f), int16^2 */
    const float*   comfort_noise;    /* CNG N^2(f), int16^2 */
    int            usable_linear;      /* usable-linear-estimate verdict; 0: r2 is the conservative nonlinear estimate */
    uint32_t       post_reset_serial;  /* see aec_post_reset_serial() */
    /* What the residual gain stage was fed this hop (a single-channel instance: from its own linear stage; an array
     * instance: over the lanes with a nonzero weight). */
    int            all_converged;      /* every contributing linear filter has converged */
    float          max_dt;             /* worst double-talk indicator; above 0.2 the DT gain floor applies */
    float          max_saturation;     /* worst saturation level; above 0.5 the echo is treated as saturated */
    int            clock_drift;        /* far / near clocks drift (MATCHED alignment only) */
    unsigned int   guard_lanes;        /* bit ch: contributing lane ch's output guard fired (single channel: bit 0 = guard_active) */
    const float*   far_hop;            /* the aligned far-end samples of this hop */
} AecResidualView;

int aec_get_residual(const Aec* a, AecResidualView* out);

#define AEC_CNG_NR_GAIN_FLOOR 0.31622777f   /* -10 dB: comfort noise follows a denoiser gain but never below this */

/* out = base_spec * min(G_res, extra_gain), over n_freqs bins. extra_gain (a denoiser's gain, any [0,1] vector of n_freqs)
 * may be NULL: out = base_spec * G_res, the library's own output stage. out_spec must not overlap extra_gain or the library's
 * own buffers; the one overlap that is allowed is out_spec == base_spec, i.e. in place on a base the caller owns (an array
 * instance's caller-supplied beamformed_error when no lane's output guard fired): the result is bit-identical to the out-of-
 * place one. The view's error_spec / base_spec, which alias that buffer, then hold the gained data, so a second in-place call
 * would apply the gain twice. Needs AEC_RESIDUAL_FULL and a processed hop (AEC_ERR_MODE / AEC_ERR_SEQUENCE otherwise). */
int aec_apply_gain(const Aec* a, const float* extra_gain, Complex* out_spec);

/* Adds comfort noise to the bins the gain cut: per bin 1..n-2, sqrt(cn/32768^2) * sqrt(1 - G_res^2) * max(extra_gain,
 * AEC_CNG_NR_GAIN_FLOOR) (the extra_gain factor is omitted when it is NULL), with G_res NOT min'ed with extra_gain, from
 * the instance's own generator (seed AEC_CNG_SEED, reseeded by every hard post-chain reset). Pass the same extra_gain
 * as to aec_apply_gain(). AEC_ENABLE_CNG is the caller's decision: skip the call to leave the noise out. */
int aec_apply_cng(Aec* a, const float* extra_gain, Complex* spec);

/* The same two operations on explicit arrays, for a caller that holds its own G_res / base / comfort noise and generator
 * state; aec_apply_gain() / aec_apply_cng() are thin wrappers over these. */
void aec_apply_gain_spec(Complex* out, const Complex* base, const float* g_res, const float* g_extra, int n);
void aec_apply_cng_spec(uint32_t* rng, Complex* spec, const float* cn, const float* g_res, const float* g_extra, int n);

/* Output guard: aec_get_residual().base_spec is the rendering base; aec_apply_gain() / aec_apply_cng() render it for you. An
 * application that renders the AecResidualView spectra itself must render base_spec instead of error_spec, handle
 * aec_post_reset_serial() for its own overlap-add tail, borrow aec_synthesis_window() and, if it draws comfort noise itself,
 * start from AEC_CNG_SEED. Nothing in the API reveals that an old renderer of error_spec has lost the guard; the migration
 * steps are in docs/sdk_migration_output_guard.md. */

/* Counter of hard post-chain resets (wrapping, 0 on a fresh instance); compare it only for inequality (a single channel moves it
 * by one per event, an array by four). Events: aec_reset(), a hard delay-acquisition or delay-shift reset inside a call, and
 * AEC_ERR_DSP. At each the library clears its OLA tail and reseeds its CNG; a caller that synthesizes must too. Returns 0 for NULL. */
uint32_t aec_post_reset_serial(const Aec* a);

/* The synthesis window aec_process() applies before overlap-add (fft_size floats in the instance's pool, valid until
 * aec_destroy()). An external synthesizer that must match aec_process() bit for bit borrows it instead of audiolib's sqrt-Hann.
 * Returns 0, or -1 for a NULL instance, window or length (outputs untouched); setup: docs/sdk_migration_output_guard.md. */
int aec_synthesis_window(const Aec* a, const float** window, int* length);

/* Read-only seam for an external post-filter: aligned_far_hop aliases the aligned far hop (raw while UNLOCKED), valid until the
 * next process or reset call; generation increments (saturating) on every ring-offset change and CHANGED marks the hop that
 * bumped it. Per-mode delay_samples / delay_confidence and the bulk-delay policy: docs/reference/integration.md. */
typedef enum AecLinearDelayState {
    AEC_LINEAR_DELAY_UNLOCKED = 0,  /* far_hop is raw / not compensated */
    AEC_LINEAR_DELAY_LOCKED   = 1,  /* compensated with a stable offset */
    AEC_LINEAR_DELAY_CHANGED  = 2   /* compensated, offset changed during THIS hop (flush far caches) */
} AecLinearDelayState;

/* Read-only view of the linear stage after aec_process_linear() (and, with residual == OFF, the whole hop). Pointers
 * alias per-hop buffers, valid until the next process or reset call. struct_size as for AecResidualView. Returns AEC_OK
 * or AEC_ERR_ARG. The two views carry what callers outside the library read; the library's own state is not exposed. */
typedef struct AecLinearView {
    size_t       struct_size;
    int          hop_size;
    const float* linear_hop;         /* refined linear-filter hop, before FORM selection */
    const float* aligned_far_hop;    /* aliases the aligned far hop */
    int          delay_samples;      /* applied ring offset; -1 = none */
    float        delay_confidence;   /* 0.0 / 0.5 / 1.0 */
    AecLinearDelayState delay_state;
    /* Status of the last COMPLETED hop (the previous one while the residual stage is still pending), kept by the
     * linear stage whatever AecConfig.residual is. */
    float        far_power;
    float        dt_indicator;       /* double-talk indicator */
    float        saturation_level;
    int          filter_converged;       /* convergence latch */
} AecLinearView;

int aec_get_linear(const Aec* a, AecLinearView* out);

/* ---- Array instance (AecConfig.n_mics == AEC_ARRAY_MICS): four lanes, a shared delay estimator and the fusion in one pool; the
 * caller supplies the beamformer. Per hop: aec_process_linear(mics, ref), aec_get_array(), aec_process_residual(&beam), then
 * aec_get_residual() / aec_apply_gain() / aec_apply_cng(). Errors, views, alignment, fused gain: docs/reference. ---- */

/* Shared alignment of an array instance (a single-channel instance fills it from its own alignment).
 * `changed`: a new usable alignment generation begins on this hop; a consumer holding history derived from the aligned far
 * flushes it before the frame it produces for this hop. `solid`: a usable accepted alignment exists. */
typedef struct AecDelayState {
    int      delay_samples;
    float    confidence;
    int      solid;
    int      changed;
    uint64_t estimator_calls;
    int      estimator_updates;
} AecDelayState;

/* The lanes after aec_process_linear() (valid until the next process or reset; n_mics == 1: lane 0 only, available
 * once the residual stage ran). error_spec is the ANALYSIS spectrum per lane (the beamformer's input); base_spec the lane's
 * RENDERING base (error_spec, or the lane's capture spectrum where its output guard fired, lane_guard_active). */
typedef struct AecArrayView {
    size_t         struct_size;
    int            n_freqs;
    const Complex* lane_error_spec[AEC_ARRAY_MICS];
    const Complex* lane_base_spec[AEC_ARRAY_MICS];
    const float*   lane_formed_hop[AEC_ARRAY_MICS];
    int            lane_guard_active[AEC_ARRAY_MICS];
    const float*   aligned_ref;             /* the far fed to every lane: the caller's far shifted by the shared delay */
    AecDelayState  delay;
    uint32_t       post_reset_serial;       /* the lanes' reset counters summed modulo 2^32 */
} AecArrayView;

int aec_get_array(const Aec* a, AecArrayView* out);

/* ---- Pool descriptor ------------------------------------------------------------------------------------------------
 * A 32-byte description of the pool a build needs, to refuse a stale pool against a newer library: aec_get_mem_requirements()
 * fills it for a config, and aec_init_ex() with a non-NULL `expected` requires descriptor_version, layout_version, backend_id,
 * build_flags_hash and alignment to equal the current build's, reserved == 0, and expected->bytes and mem_size >= the current
 * bytes. aec_init(mem, size, cfg) is aec_init_ex(..., NULL). Query a fresh descriptor for every build; never persist or
 * synthesize one. */
#define AEC_DESCRIPTOR_VERSION 1u
#define AEC_LAYOUT_VERSION     5u
#define AEC_BACKEND_NE10       2u   /* backend_id 1 is never returned */
typedef struct AecMemReq {
    uint32_t descriptor_version;
    uint32_t layout_version;
    uint32_t backend_id;
    uint32_t build_flags_hash;      /* FNV-1a-32 over the carve structure, not tunable values */
    uint32_t alignment;
    uint32_t reserved;
    uint64_t bytes;
} AecMemReq;

/* Return 0 and fill *out for a valid cfg (single-channel or array); -1 (out zeroed) for a NULL argument or an invalid cfg.
 * out->bytes == aec_get_mem_size(cfg). */
int  aec_get_mem_requirements(const AecConfig* cfg, AecMemReq* out);
Aec* aec_init_ex(void* mem, size_t mem_size, const AecConfig* cfg, const AecMemReq* expected);

/* Read-only status snapshot for polling (e.g. once per second); fills a
 * struct from fields the engine already maintains, with no hot-path cost. A handle that is not ready zeroes *out. */
typedef struct AecDebugStatus {
    /* delay estimation */
    int    delay_samples;      /* applied delay in samples; -1 = not yet acquired */
    float  delay_confidence;   /* MATCHED: estimator's 0.0/0.5/1.0; FIXED, EXTERNAL_ALIGNED: 1.0 */
    int    delay_updates;      /* matched-filter update count; 0 outside MATCHED */

    /* linear filter health */
    float  erle_windowed_db;   /* windowed ERLE (dB) at the last post-chain hop; 0 before it first ran */
    int    usable_linear;      /* usable-linear-estimate gate state */
    int    filter_converged;   /* convergence latch (AecLinearView.filter_converged) */

    /* levels: EMAs (alpha = 0.95) of power, not last-hop means. */
    float  near_power;         /* EMA of mic power */
    float  out_power;          /* EMA of linear (pre-RES-gain) output power */

    /* Duty-cycle census, cumulative since create/init/reset; run / total is
     * the measured matched-filter engagement. Both 0 unless delay_mode ==
     * MATCHED. */
    unsigned long long duty_hops_total;
    unsigned long long duty_hops_run;

    /* Geometry and retiming resolved at construction (constant for the instance's life). */
    int    n_partitions;       /* partitions of the main filter (an array: lane 0's) */
    int    delay_period_hops;  /* hops per matched-filter analysis once the duty cycle settles; 1 = every hop */
} AecDebugStatus;

void aec_debug_status(const Aec* a, AecDebugStatus* out);

/* Labels two subsets of aec_get_mem_size(cfg): estimator_bytes
 * (matched-filter estimator; 0 outside MATCHED) and ring_bytes (alignment
 * ring; 0 for EXTERNAL_ALIGNED). total_bytes == aec_get_mem_size(cfg).
 * Returns AEC_OK and fills *out for a valid cfg; AEC_ERR_ARG and zeroes *out (NULL out: nothing) when cfg is NULL or
 * invalid. */
typedef struct AecMemBreakdown {
    size_t total_bytes;
    size_t estimator_bytes;   /* array: the shared delay estimator's pool (MATCHED) */
    size_t ring_bytes;        /* array: the shared aligned-far ring (MATCHED, FIXED) */
    size_t lane_bytes;        /* array: the four lanes' pools together; 0 for a single-channel instance */
    size_t fusion_bytes;      /* array: the rest of the array pool beyond the Aec header, the estimator and the ring (control
                               * block, mic/aligned-far/fusion buffers, the residual gain stage, the lanes' shared FFT handle
                               * and per-call buffers; the lanes' pools are NOT included); 0 for a single-channel instance */
} AecMemBreakdown;

int aec_get_mem_breakdown(const AecConfig* cfg, AecMemBreakdown* out);

/* ---- Array diagnostics: read-only counters since construction or aec_reset(); a single channel answers 0 / -1, a handle that is
 * not ready 0 / -1 / a zeroed record. Far FFT count: the four lanes summed (+1 per hop; +4 when the lanes could not borrow lane
 * 0's FFT). aec_array_realign_*_lane_count: lane outcomes, 4 per `changed` hop (also 4 soft for delta 0), saturating. ---- */
typedef struct AecArrayTiming {
    uint32_t delay_us;       /* shared delay estimator */
    uint32_t frontend_us;    /* four lanes: entry to the main filter */
    uint32_t linear_us;      /* four lanes: main adaptive filter incl. the far FFT */
    uint32_t lane_res_us;    /* four lanes: the lanes' residual stage */
    uint32_t fuse_us;        /* aec_process_residual(): the fusion and the gain */
} AecArrayTiming;

long aec_array_realign_warm_lane_count(const Aec* a);
long aec_array_realign_soft_lane_count(const Aec* a);
int  aec_array_matched_filter_count(const Aec* a);                    /* 1 in MATCHED, else 0 */
int  aec_array_linear_aec_count(const Aec* a);                        /* 4; single-channel: 1 */
/* Wall clock of the most recent hop in microseconds, diagnostic only; all 0 when built with -DAEC_STAGE_TIMING=0.
 * aec_get_last_timing() reports the same record without fuse_us (res_us = lane_res_us). A call refused before the hop starts
 * (NULL argument, handle not ready, non-finite sample) leaves the record untouched; any other aec_process_linear() first
 * zeroes the four linear-stage fields;
 * fuse_us is zeroed by an aec_process_residual() that passes its argument, sequence and beam checks. out must not be NULL. */
void aec_array_get_timing(const Aec* a, AecArrayTiming* out);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_H */
