/* Public C API for the PBFDKF AEC: a single-channel instance or a four-microphone array instance (AecConfig.n_mics), heap
 * or caller-pool construction, one- or two-stage lockstep processing, and the residual / linear views for an external
 * post-filter.
 *
 * Single-channel build. The library built with -DAEC_NO_ARRAY (make AEC_NO_ARRAY=1) has no array instance: AecConfig.n_mics
 * other than 1 is refused (aec_get_mem_size() 0, aec_init() NULL, aec_create() -1), and aec_get_array() and the aec_array_*()
 * diagnostics are not defined (a call is a link error). Every other function behaves as for a single-channel instance of the
 * full library, byte for byte. The header is the same in both builds. */
#ifndef AEC_AEC_H
#define AEC_AEC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "fft_wrapper.h"
#include "hpf.h"
#include "saturation.h"
#include "delay_aec.h"
#include "detectors.h"
#include "pbfdkf.h"
#include "epc_shadow.h"
#include "render_signal_analyzer.h"
#include "aec_post.h"   /* AecPost and its sub-modules */

/* Residual-echo strength level, ordered 0..3 (default BALANCED = 1):
 * min_gain_floor_far_active_db = -20 / -28 / -38 dB for MILD / BALANCED /
 * AGGRESSIVE. NO_FLOOR sets min_gain_split_floor_enabled = 0 (dB fields unused). */
typedef enum {
    AEC_LEVEL_MILD       = 0,
    AEC_LEVEL_BALANCED   = 1,
    AEC_LEVEL_AGGRESSIVE = 2,
    AEC_LEVEL_NO_FLOOR   = 3
} AecLevel;

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
 * of aec_resolve_signal_grid() / aec_get_mem_breakdown(): 0, or a negative reason. A refused call writes nothing, except
 * AEC_ERR_DSP below. aec_create, aec_get_mem_requirements, aec_set_level and aec_synthesis_window
 * follow the same sign (0 = success, -1 = failure). aec_apply_external_realign() returns an outcome (1 / 0 / -1), and
 * aec_is_valid_sample_rate() is a predicate (1 / 0). */
typedef enum AecStatus {
    AEC_OK           =  0,
    AEC_ERR_ARG      = -1,   /* NULL, a short struct_size, or an argument out of range */
    AEC_ERR_MODE     = -2,   /* the call needs a higher AecConfig.residual than the instance has */
    AEC_ERR_SEQUENCE = -3,   /* a call out of order (residual before linear, linear twice, ...) */
    AEC_ERR_DSP      = -4    /* an array instance: a stage refused the hop (non-finite or overflowing data); the instance
                              * has been reset. A non-finite INPUT sample is AEC_ERR_ARG and changes nothing. */
} AecStatus;

/* Bad-relock judgment window (hops): a shifted delay is reverted if it
 * cancels worse after this long. */
#define AEC_DELAY_SHIFT_JUDGMENT_HOPS 100
/* Capacity of Aec::delay_reject_values. */
#define AEC_DELAY_REJECT_CAP 8

typedef struct AecConfig {
    int    sample_rate;        /* Hz; 8000, 16000 or 48000 */
    int    fft_size;           /* samples; 256 or 512 @16k, 1024 @48k; 0 is rejected */
    int    filter_length;      /* samples; 52 ms (64 ms at >= 44.1 kHz) */
    int    n_partitions;       /* 0 = derived from filter_length */
    float  mu;                 /* main filter step size; 0.3 */
    float  delta;              /* division guard; 1e-8 */

    int    enable_cng;             /* comfort-noise generation; 1 */
    int    cng_n2_windowed_psd;    /* CNG noise-tracker windowed-PSD coordinate; 0 */
    int    enable_highpass;        /* mic-path high-pass; 1 */
    int    enable_saturation;      /* saturation limiter; 1 */
    int    enable_shadow;          /* shadow filter; 1 */
    int    saturation_softclip_ref;/* soft-clip the reference; 1 */
    /* How far the residual stage goes: AecResidualMode. aec_process_residual() and the getters follow it.
     * The pool size of a single-channel instance does not depend on it; an array's grows with it (STATE: lanes only,
     * ESTIMATE: + the fusion buffers, FULL: + the gain stage). */
    AecResidualMode residual;      /* AEC_RESIDUAL_FULL */

    /* Microphone array: 1 (default) is the single-channel instance this header describes; AEC_ARRAY_MICS (4) builds an
     * array instance: four linear lanes, one shared delay estimator and aligned far, and the fusion of the lanes by the
     * caller's beamformer weights (see "Array instance" below). On an array instance delay_mode / delay_num_filters /
     * fixed_delay_samples / delay_est_period_s / max_delay_ms describe the SHARED alignment (each lane then takes the
     * aligned far, EXTERNAL_ALIGNED) and `residual` must be STATE, ESTIMATE or FULL. */
    int    n_mics;                              /* 1; 4 only in a library built without AEC_NO_ARRAY */
    int    capture_proxy_channel;               /* array: the shared estimator's capture input, [0, n_mics-1]; 0 */
    /* array, MATCHED only, default off: an estimate earlier than the delay in force is held for
     * delay_backward_quarantine_s while the capture-proxy lane still cancels, then adopted; lost cancellation adopts it
     * at once. Forward moves and the first acquisition are never held. */
    int    delay_backward_quarantine_enabled;   /* 0 */
    float  delay_backward_quarantine_s;         /* [0, 3600]; 1.0 */

    /* Tunables in hops are authored on the 10 ms / 16 kHz grid and retimed at
     * create time. */
    float  shadow_err_alpha;       /* shadow error EMA alpha; 0.80 @ 10 ms hop */
    float  shadow_mu_min;          /* 0.5  */
    float  shadow_mu_nlms;         /* 0.5  */
    int    warmup_frames;          /* hops; 100 = 1000 ms @ 10 ms hop */
    int    epc_hangover;           /* hops; 20 = 200 ms @ 10 ms hop */
    float  epc_total_rise;         /* 1.5  */
    float  epc_delta_threshold;    /* 0.3  */
    float  epc_mu_floor;           /* 0.5  */
    /* MATCHED only: size the reference search ring, max(delay_buffer_ms,
     * max_delay_ms + 4096 samples). */
    float  max_delay_ms;           /* ms; 1024 */
    float  delay_buffer_ms;        /* ms; 2048 */
    float  delay_est_init_s;       /* s; hold time before duty-cycling; 0.3 */
    float  delay_est_period_s;     /* s; 0 = analyse every hop */
    /* MATCHED only: matched-filter bank size in [1, 5] (part of the pool
     * layout, init-time immutable); n = 1..5 gives a reliable reach of
     * 125/221/317/413/509 ms. A non-default value outside MATCHED is
     * rejected. */
    int    delay_num_filters;      /* 5 */
    float  highpass_cutoff_hz;     /* Hz; 80 */
    float  saturation_threshold;   /* clipping amplitude threshold (full scale 1.0); 0.95 */
    float  kalman_q_high;          /* model uncertainty while unconverged; 1e-3 */
    float  kalman_q_low;           /* model uncertainty after convergence; 1e-6 */
    int    delay_acquire_protect_converged;   /* 1 */
    /* Warm tap-transfer on first delay acquisition: shift the learned IR
     * instead of zeroing it (needs inst-ERLE peak >
     * delay_acquire_inst_erle_db and delay < tap reach). */
    int    delay_acquire_warm_transfer;       /* 1 */
    float  delay_acquire_inst_erle_db;        /* dB; 4.0 */

    /* DT-deg recovery: with the held near-end gate armed in double-talk,
     * dt_aware_recovery_soft keeps realign recovery non-destructive and
     * dt_aware_res_floor_enabled applies the DT min-gain floor. */
    int    dt_aware_recovery_soft;            /* 1 */
    int    dt_aware_res_floor_enabled;        /* 1 */
    float  min_gain_floor_dt_db;              /* dB; -20.0 */
    /* Near-end-recent gate: energy threshold, hold and sustain (hops). */
    float  ne_recent_threshold;               /* 0.3 */
    int    ne_recent_hold;                    /* hops; 150 = 1500 ms @ 10 ms hop */
    int    ne_recent_sustain;                 /* hops; 3 */

    /* Far-active min-gain floor, dB (-20 / -28 / -38 for mild / balanced /
     * aggressive). */
    float  min_gain_floor_far_active_db;

    /* Filter-scale tracker: stable / hangover (hops, retimed) and scale
     * bounds. */
    int    filter_misadjustment_stable_frames;   /* hops; 30 = 300 ms @ 10 ms hop */
    int    filter_misadjustment_hangover_frames;  /* hops; 100 = 1000 ms @ 10 ms hop */
    float  filter_misadjustment_scale_min;        /* 0.5 */
    float  filter_misadjustment_scale_max;        /* 2.0 */

    /* Alignment mode; init-time immutable (the pool layout depends on it). */
    AecDelayMode delay_mode;       /* AEC_DELAY_MATCHED */
    /* FIXED only: measured delay in samples (>= 0), also sizes the reference
     * ring; must be -1 in the other modes. */
    int    fixed_delay_samples;    /* -1 */

    /* Live observer: a second PBFDAF reading the far-end ring earlier by
     * live_observer_shift_ms; adds R^2 evidence only while it corroborates
     * the main filter. Default off. */
    int    live_observer_enabled;           /* 0 */
    float  live_observer_shift_ms;          /* ms; 24.0 */
    float  live_observer_max_hz;            /* Hz; 1000.0 */
    float  live_observer_mu;                /* observer filter step size; 0.3 */
    float  live_observer_corrob_ms;         /* ms; 100.0 */
    float  live_observer_corrob_margin_db;  /* dB; 0.0 */

    /* Gates all three min-gain floor variants; NO_FLOOR clears it. Default 1. */
    int    min_gain_split_floor_enabled;    /* 1 */
} AecConfig;

/* Both factories fill fft_size and filter_length from the rate. For a rate with
 * no grid (anything but 8000/16000/48000, including 0 and negative values) both
 * are 0, which aec_get_mem_size / aec_init / aec_create refuse. */
void aec_config_defaults(AecConfig* cfg, int sample_rate);
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

/* Matched-filter duty-cycle gate: full-rate analysis until the estimate has been solid and unchanged for hold_hops, then one
 * hop in period_hops, back to full rate on a windowed-ERLE collapse. Internal; shared by the single-channel instance and the
 * array flow (aec_duty_* in aec_array_internal.h). */
typedef struct AecDutyGate {
    int    active;
    int    stable_hops;
    int    pos;
    int    last_delay;
    float  erle_peak;
    float  erle_leak_db;
    int    hold_hops;
    int    period_hops;
    unsigned long long hops_run;
} AecDutyGate;

/* Instance state: fields are internal, not part of the API; use the functions
 * below. */
typedef struct Aec {
    AecConfig cfg;

    int hop_size, block_size, fft_size, n_freqs, n_partitions;

    Hpf  *hp_mic;             /* NULL when highpass disabled */
    Saturation sat_ref, sat_mic;  int has_sat;

    /* delay: matched-filter state, valid iff has_delay (delay_mode ==
     * MATCHED). */
    DelayAec delay;       int has_delay;
    float* ref_ring;       int ref_ring_size, ref_ring_write, ref_ring_filled;
    int    current_delay;
    int    pending_delay;  int has_pending; int pending_delay_ttl;
    int    delay_shift_judgment_active;
    int    delay_shift_judgment_pre_delay;
    float  delay_shift_judgment_pre_erle;
    int    delay_shift_judgment_hops_left;
    Complex* delay_shift_judgment_pre_W;
    Complex* delay_shift_judgment_pre_shadow_W;
    float*   delay_shift_judgment_pre_H_error;   /* main filter H_error_per_bin [n_freqs] */
    int    delay_reject_values[AEC_DELAY_REJECT_CAP];
    int    delay_reject_count;
    unsigned int delay_generation;
    unsigned int delay_gen_hop_start;
    int    far_hop_aligned;
    AecDutyGate duty;
    unsigned long long duty_hops_total;

    PBFDKF main_filter;
    PBFDAF shadow_filter;  int has_shadow;
    PBFDAF observer_filter;  int has_observer;

    RenderActivity    render_activity;
    float*            ra_pairwise_scratch;
    FilterConvergence convergence;
    DoubleTalk        dt_analyzer;
    EpcDetector       epc;
    ShadowCopy        regime;
    RenderSignalAnalyzer rsa;   int64_t* rsa_counters;

    FftHandle*  post_fft;
    AecPost    post;
    AecState              a3_state;   AecStateStorage a3_state_st;
    ResidualEchoEstimator a3_ree;
    SuppressionGain       a3_sg;
    StationarityEstimator a3_stat;
    LinearFilterSelect    a3_lfs;
    AecPostRunScratch    a3_sc;
    int    pending_gain_change;   /* bool */
    int    pending_delay_change;
    int    stationarity_active_hops;
    int    stationarity_converge_hops;
    int    non_zero_render_seen;
    float  render_peak_floor;
    int    block_stationary_next;

    float  saturation_level;
    float  erl_estimate;
    float  main_err_smooth, shadow_err_smooth;
    long   shadow_frame_count;     /* saturates at LONG_MAX */
    int    epc_render_forced_remaining;
    float  erle_window_near, erle_window_err;
    float  erle_factor_prev;
    float  inst_erle_smooth;
    float  erle_slope_buf[64];
    int    erle_slope_cap, erle_slope_len, erle_slope_head;
    float  wn_err_baseline;
    int    stat_dt_hangover;
    int    warmup_frames_remaining;
    int    warmup_far_active;
    float  simple_mu_ratio;
    int    simple_mu_holdoff;
    float* per_bin_mu_scale;   int has_per_bin_mu;
    float  near_power, raw_error_power;
    float  alpha_pow;
    long   frame_count;            /* saturates at LONG_MAX */

    int    ne_above;
    int    ne_recent_frames;

    int    poor_coarse_counter;
    int    coarse_reset_hangover;
    int    leakage_div_sustained_counter;
    int    poor_coarse_threshold_hops;
    int    coarse_reset_hangover_hops;
    int    leakage_div_sustain_hops;
    int    stat_dt_hangover_hops;

    float  misadj_e2_acum, misadj_y2_acum;
    int    misadj_n_acum;
    float  misadj_inv;
    int    misadj_overhang;
    int    misadj_stable_count;
    int    misadj_hangover_remaining;

    float  last_erle_windowed;

    float  last_far_power, last_shadow_dt, last_dt_indicator;
    float  alpha_erl_tracking, alpha_erl_converged;
    int    last_is_stationary_dt;

    float* near_hop;
    float* far_hop;
    float* raw_output;
    float* shadow_out;
    float* final_out;
    float* observer_far_hop;
    int    observer_far_hop_valid;
    int    observer_shift_hops;
    float* filter_taps_full;

    float* scr_sq;
    float* scr_e2_echo;
    float* scr_e2_near;
    float* scr_rsa_mask;
    float* scr_mu_buf_pre;
    float* scr_mu_buf;
    float* scr_e2ref_arr;

    unsigned char is_static;    /* the caller owns the memory */
    unsigned char ready;        /* set by a successful create / init, cleared by aec_destroy(); a handle that is not ready is refused */

    /* Two-stage hop bookkeeping (aec_process_linear / aec_process_residual): 0 idle, 1 = stage 1 done and the residual
     * stage pending. hop_carry holds the stage-1 locals the residual stage needs. Internal. */
    int hop_phase;
    struct AecHopCarry {
        float far_hop_mean_sq;
        float erle_inst;            /* inst-ERLE of this hop (misadjustment estimator) */
        float erle_windowed;
        int   render_active;
        int   stationarity_block_for_post;
        uint32_t t_stage;           /* clock stamp that closed the steering window */
    } hop_carry;
    int residual_valid;             /* a residual stage has completed since construction or aec_reset() */

    /* Array instance (AecConfig.n_mics == AEC_ARRAY_MICS): the flow state in src/aec_array.c; NULL for a single-channel
     * instance. On an array instance every field above is unused except cfg and the dimensions. */
    struct AecArray* array;

    void*  heap_arena;

    struct AecStageTiming {
        uint32_t delay_us;
        uint32_t frontend_us;
        uint32_t linear_us;
        uint32_t res_us;
        uint32_t steering_us;
    } last_timing;
} Aec;

/* Per-hop wall-clock cost (microseconds) of the five largest stages in
 * aec_process*(). Diagnostic only. delay_us: delay estimation, ring write and
 * compensated read (0 under EXTERNAL_ALIGNED). frontend_us: entry to main
 * filter excluding delay_us. linear_us: main adaptive filter. steering_us:
 * stationarity, coarse-error publish, double-talk analyzer, EPV, shadow_rise,
 * misadjustment estimator. res_us: post-filter/RES block. The five do not sum
 * to the call; fields are cleared at the start of each call. Active only with
 * -DAEC_STAGE_TIMING=1 (otherwise every field reads 0); the stamp is
 * CLOCK_MONOTONIC, or -DAEC_NOW_US=<fn> returning microseconds. */
/* Always defined so consumers can use #if AEC_STAGE_TIMING under -Wundef. */
#ifndef AEC_STAGE_TIMING
#define AEC_STAGE_TIMING 0
#endif

typedef struct AecStageTiming AecStageTiming;

/* Copies the last hop's stage timings into out. A NULL instance zeroes out;
 * out must not be NULL. */
void aec_get_last_timing(const Aec* a, AecStageTiming* out);

/* Heap construction: returns 0 on success, -1 on a NULL argument, an invalid
 * cfg or allocation failure. aec_destroy releases the instance (NULL-safe, repeatable); aec_reset restores the initial state
 * (NULL-safe; a handle that is not ready is left as it is).
 * A handle that is not ready: NULL, zero-filled or never built, left by a failed aec_create(), or destroyed. Every entry point
 * refuses it (AEC_ERR_ARG, -1, 0, NULL or a zeroed record, as each function states) until aec_create() / aec_init() builds it;
 * aec_hop_size() reads 0. A destroyed handle is refused for the Aec an aec_create() filled and for an aec_init() instance whose
 * pool has not been freed or overwritten. Using a handle whose Aec storage or pool has been freed or reused is undefined, as for
 * any C object; so is the pointer of a view obtained before the destroy. */
int  aec_create(Aec* a, const AecConfig* cfg);
void aec_destroy(Aec* a);
void aec_reset(Aec* a);

/* Realigns an AEC_DELAY_EXTERNAL_ALIGNED instance after a caller-side change
 * of the far alignment, without a full reset; delta_samples = new_alignment -
 * old_alignment. If the filter is cancelling (aec_linear_is_cancelling()) and
 * the shift fits the tap span, the learned IR is shifted (warm transfer);
 * otherwise the filter alone is reset and the output stays continuous. A
 * negative delta exposes echo for up to about ceil((|delta| + tap offset) /
 * hop_size) + 1 hops. Bumps the alignment generation. Returns 1 for warm
 * transfer, 0 for filter reset (or delta_samples == 0), -1 on NULL or another
 * delay mode. */
int aec_apply_external_realign(Aec* a, int delta_samples);

/* Retargets the residual-echo strength of a running instance; filter, delay
 * state and histories carry on. Updates min_gain_floor_far_active_db and, for
 * aggressive, min_gain_floor_dt_db; NO_FLOOR clears
 * min_gain_split_floor_enabled. The far-active floor walks to its new value
 * linearly in dB over a fixed 100 ms (a change during a ramp restarts from
 * the live value); the DT floor and the split-floor flag step immediately.
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
 *                         own synthesis: `out` is the finished hop. Below FULL, `out` is the linear residual.
 *   aec_process_linear    stage 1: front end, delay, shadow and main filter. With AEC_RESIDUAL_OFF it also ends the hop.
 *   aec_process_residual  stage 2, directly after aec_process_linear of the same hop: FORM, AecState, r2, comfort noise
 *                         and G_res, as far as AecConfig.residual reaches. It applies nothing and synthesizes nothing:
 *                         read aec_get_residual() and finish with aec_apply_gain() / aec_apply_cng() and the caller's
 *                         own inverse transform. `beam` is for an array instance and must be NULL on a single-channel one.
 *
 * Within a stream the two forms may alternate hop by hop; the library's own overlap-add tail and comfort-noise
 * generator advance only in aec_process() and aec_apply_cng().
 * The linear output depends on AecConfig.residual: OFF uses the simpler step-size rule, STATE / ESTIMATE / FULL share one
 * linear path, so compare a linear part with the full chain only on STATE or above.
 * Returns AEC_OK, or AEC_ERR_ARG / AEC_ERR_MODE / AEC_ERR_SEQUENCE with the instance untouched. An array instance can also
 * answer AEC_ERR_DSP, and then the instance has been reset (see "Array instance"). */
int aec_process(Aec* a, const float* mic, const float* ref, float* out);
int aec_process_linear(Aec* a, const float* mic, const float* ref);
int aec_process_residual(Aec* a, const AecBeam* beam);

/* Cumulative count (since construction or aec_reset(), saturating at
 * LONG_MAX) of hops where this instance ran its own far-end FFT. */
long aec_far_fft_real_compute_count(const Aec* a);

/* 1 if the linear filter is demonstrably cancelling: last hop's windowed ERLE
 * above 2.5 dB, or the recent inst-ERLE peak above
 * cfg.delay_acquire_inst_erle_db. 0 means no evidence, not a fault (also 0
 * before the ERLE machinery has run). */
int aec_linear_is_cancelling(const Aec* a);

/* Samples per process call (fft_size / 2); 0 for a handle that is not ready. */
int  aec_hop_size(const Aec* a);

/* Everything the residual stage knows about the last hop, kept inside the instance and read through this view. The
 * pointers alias internal buffers, are valid until the next process or reset call, and follow AecConfig.residual:
 *   STATE     formed_hop, error_spec, echo_spec, near_spec, base_spec and the flags; r2, comfort_noise, res_gain NULL.
 *   ESTIMATE  + r2, comfort_noise.            FULL  + res_gain.
 * ANALYSIS spectra (error_spec, echo_spec, near_spec, formed_hop): the formed linear residual E(f) (50% overlap,
 * periodic sqrt-Hann STFT of [previous formed hop | current formed hop], audio scale), its echo estimate, and
 * near = E + echo. The output guard never alters them. r2 and comfort_noise are int16^2 power: divide by 32768^2 for
 * audio power. RENDERING base_spec: what a gain is multiplied on. The guard evaluates once per hop, after the usable-
 * linear verdict is final: guard_active is 1 when the estimate is unusable and sum|E|^2 > sum|E + echo|^2 (strict,
 * windowed spectra, float32 pairwise reduction); then base_spec is the capture spectrum E + echo, otherwise E. base_spec
 * is not the transform of any formed hop; use formed_hop / error_spec for analysis and base_spec only for rendering.
 * far_spec is the filter's own rectangular-block far spectrum (not sqrt-Hann): not a chain-standard spectrum.
 * The linear stage's status (convergence, double-talk, saturation, ...) is in AecLinearView, not here.
 * struct_size is an input (sizeof(AecResidualView)); a shorter one is refused untouched, a larger one gets the known
 * prefix and struct_size is set to sizeof(AecResidualView). Returns AEC_OK, or AEC_ERR_ARG (NULL, short struct_size) /
 * AEC_ERR_MODE (residual == OFF: the view is zeroed apart from struct_size). */
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
    int            res_floor_protect;  /* this hop's near-recent DT-floor decision */
    int            usable_linear;      /* usable-linear-estimate verdict; 0: r2 is the conservative nonlinear estimate */
    uint32_t       post_reset_serial;  /* see aec_post_reset_serial() */
    /* What the residual gain stage was fed this hop (a single-channel instance: from its own linear stage; an array
     * instance: over the lanes with a nonzero weight). */
    int            all_converged;      /* every contributing linear filter has converged */
    float          max_dt;             /* worst double-talk indicator; above 0.2 the DT gain floor applies */
    float          max_saturation;     /* worst saturation level; above 0.5 the echo is treated as saturated */
    int            clock_drift;        /* far / near clocks drift (MATCHED alignment only) */
    int            delay_changed;      /* this hop starts a new alignment generation (array: the shared one) */
    int            delay_locked;       /* a usable alignment is in force */
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

/* Output-guard API version (2: aec_get_residual().base_spec is the rendering base; aec_apply_gain() /
 * aec_apply_cng() render it for you). An application that renders the AecResidualView spectra itself must make this
 * macro part of its build, e.g. #if !defined(AEC_OUTPUT_GUARD_API_VERSION) || AEC_OUTPUT_GUARD_API_VERSION < 2 / #error /
 * #endif, and must render base_spec instead of error_spec, handle aec_post_reset_serial() for its own overlap-add tail,
 * borrow aec_synthesis_window() and, if it draws comfort noise itself, start from AEC_CNG_SEED. Nothing else in the API
 * reveals that an old renderer of error_spec has lost the guard; the migration steps are in
 * docs/sdk_migration_output_guard.md. */
#define AEC_OUTPUT_GUARD_API_VERSION 2

/* Counter of hard post-chain resets (wrapping uint32_t, 0 on a freshly created instance), read without
 * consuming it. It advances by one on each of:
 *   - aec_reset(): a separate call, not part of any processing call; the serial already reads +1 when
 *     aec_reset() returns, and the next processing call starts from the cleared state.
 *   - a hard reset inside a processing call (aec_process / aec_process_linear), done before that hop's
 *     synthesis: the hard branch of a first delay acquisition (Path A: no warm tap transfer and no
 *     DT-aware soft acquisition) and the hard branch of a confirmed delay shift (Path B: no DT-aware soft
 *     realign).
 * It never moves on the bad-relock revert (a soft revert to the pre-shift delay and its tap snapshot), on
 * warm tap transfers or DT-aware soft acquisitions / realignments, on aec_apply_external_realign(), or
 * when only AecLinearView.generation changes. At each counted reset the library clears its own OLA
 * tail and reseeds its CNG to AEC_CNG_SEED.
 * An application that synthesizes outside the library keeps its own last-handled serial; when the value
 * after a processing call (or after aec_reset()) differs from it, the application must, BEFORE its first
 * CNG draw and BEFORE the inverse-transform/overlap-add push of the hop it renders from that processing
 * call (the hop just processed; after aec_reset(), the first hop processed afterwards), clear its OLA
 * tail (e.g. ac_istft_reset) and reseed its comfort-noise state to AEC_CNG_SEED, then store the new
 * serial. It must not reset the AEC in response, and NR/NN state is not touched by the library:
 * following the reset there is application policy. Returns 0 for NULL. */
uint32_t aec_post_reset_serial(const Aec* a);

/* The synthesis window the library itself applies before overlap-add: a block_size table (block_size ==
 * fft_size, the AecConfig grid the instance was created with). An external synthesizer that must reproduce
 * aec_process() bit for bit has to use this table, not the audiolib default sqrt-Hann: the two differ in the
 * last float bit.
 *
 * With audiolib (audio_stft.h):
 *     const float* w; int n;
 *     AcStftConfig sc;
 *     aec_synthesis_window(aec, &w, &n);              // n == the fft_size of the AcIstft below
 *     ac_stft_config_default(&sc, n);
 *     sc.window = w;                                  // borrowed table; AcIstft carves none
 *     size_t sz = ac_istft_get_mem_size(&sc);         // then ac_istft_init(mem, sz, &sc) as usual
 * The AcIstft is configured with fft_size = the returned length. The pointer is read-only, owned by the
 * library, and valid for the lifetime of the instance (it does not change across processing or reset calls);
 * the AcIstft borrows it, so it must not outlive the Aec. *window and *length are set only on success.
 * Returns 0, or -1 for a NULL instance, window or length (outputs untouched). */
int aec_synthesis_window(const Aec* a, const float** window, int* length);

/* Linear-AEC / NN post-filter seam: read-only view for an external post-filter.
 *   - aligned_far_hop aliases a->far_hop, valid until the next process/reset call.
 *   - UNLOCKED: aligned_far_hop is the raw far (not delay-compensated).
 *   - generation increments (saturating) on every ring-offset change (first
 *     acquisition, confirmed shift, reset); poll it to invalidate cross-hop caches.
 *     CHANGED is reported on the hop whose processing bumped it.
 *   - An out-of-range bulk delay is not detectable here; the product owns that policy.
 * Per mode (cfg.delay_mode):
 *   MATCHED           delay_samples = applied ring offset (-1 before acquisition);
 *                     delay_confidence = the estimator's 0.0 / 0.5 / 1.0.
 *   FIXED             delay_samples = fixed_delay_samples; confidence 1.0;
 *                     generation moves only on aec_reset(); UNLOCKED until the ring
 *                     holds fixed_delay_samples + hop_size samples.
 *   EXTERNAL_ALIGNED  aligned_far_hop is the caller's far hop; delay_samples 0,
 *                     LOCKED, confidence 1.0; generation moves only via
 *                     aec_apply_external_realign() and aec_reset(). */
typedef enum AecLinearDelayState {
    AEC_LINEAR_DELAY_UNLOCKED = 0,  /* far_hop is raw / not compensated */
    AEC_LINEAR_DELAY_LOCKED   = 1,  /* compensated with a stable offset */
    AEC_LINEAR_DELAY_CHANGED  = 2   /* compensated, offset changed during THIS hop (flush far caches) */
} AecLinearDelayState;

/* Read-only view of the linear stage after aec_process_linear() (and, with residual == OFF, the whole hop). Pointers
 * alias per-hop buffers, valid until the next process or reset call. struct_size as for AecResidualView. Returns AEC_OK
 * or AEC_ERR_ARG. */
typedef struct AecLinearView {
    size_t       struct_size;
    int          hop_size;
    int          n_freqs;
    const float* linear_hop;         /* refined PBFDKF hop, before FORM selection */
    const Complex* echo_spec;        /* main filter echo / near / far spectra (pre-FORM; the far one is rectangular-block) */
    const Complex* near_spec;
    const Complex* far_spec;
    const float* aligned_far_hop;    /* aliases the aligned far hop */
    int          delay_samples;      /* applied ring offset; -1 = none */
    float        delay_confidence;   /* 0.0 / 0.5 / 1.0 */
    AecLinearDelayState delay_state;
    unsigned int generation;
    /* Status of the last COMPLETED hop (the previous one while the residual stage is still pending), kept by the
     * linear stage whatever AecConfig.residual is. */
    float        far_power;
    float        erle_factor;
    float        dt_indicator;       /* double-talk indicator */
    float        divergence;
    float        saturation_level;
    float        erl_estimate;
    float        shadow_dt;
    int          is_stationary_dt;
    int          filter_converged;       /* convergence latch */
    int          filter_once_converged;
    int          epc_active;             /* echo-path-change regime */
} AecLinearView;

int aec_get_linear(const Aec* a, AecLinearView* out);

/* ---- Array instance (AecConfig.n_mics == AEC_ARRAY_MICS) ---------------------------------------------------------
 * One aec_init() builds the whole four-microphone linear flow: four lanes, the shared delay estimator, the aligned far
 * and the fusion buffers, in one caller pool. The beamformer between the lanes and the fusion is the CALLER's (SRP-PHAT,
 * BF, VAD and the like are not part of this library). Per hop:
 *
 *     aec_process_linear(a, mics, ref)        mics = interleaved float [hop][n_mics], ref = float [hop].
 *                                             Shared delay + the lanes' linear and per-lane residual stages.
 *     aec_get_array(a, &av)                   the lanes' error spectra: what the caller's beamformer reads.
 *     (caller: weights W[n_mics][n_freqs], beamformed spectrum E)
 *     aec_process_residual(a, &beam)          fuse the lanes (R^2 coherently, comfort noise as a power sum, near, far,
 *                                             rendering base) and, with AEC_RESIDUAL_FULL, compute the one G_res on the
 *                                             fused spectra.
 *     aec_get_residual(a, &rv)                the fused E, base, r2, comfort noise, flags (AecResidualView).
 *     aec_apply_gain / aec_apply_cng          called as for a single channel (same arguments, same
 *                                             out = base * min(G_res, extra_gain)); the gain itself is not the single-channel
 *                                             one, see "Fused gain" below.
 *
 * residual selects what the lanes and the fusion do: STATE = lanes only (no fusion, aec_process_residual refused with
 * AEC_ERR_MODE: the beamformed spectrum is rendered by the caller); ESTIMATE = fusion without a gain; FULL = fusion with
 * the gain. aec_process() needs one microphone and returns AEC_ERR_MODE on an array instance.
 *
 * Errors. A non-finite input sample (mics / ref) is AEC_ERR_ARG with the state untouched. aec_process_residual() validates the
 * beam up front: weights that are NULL, non-finite or all zero, or a beamformed_error with a non-finite bin, are AEC_ERR_ARG
 * with the instance untouched (no reset; the hop stays open and the previous views stay readable, so the caller can fix the
 * beam and call again). A lane or the fusion refusing a hop after that (finite inputs whose products overflow, an internal
 * consistency check) is AEC_ERR_DSP and the whole instance has been reset, as by aec_reset().
 * Call order. A second aec_process_linear() before the residual stage replaces the pending hop (the four filters have already
 * adapted on the first). A second aec_process_residual() on the same hop is AEC_ERR_SEQUENCE with AEC_RESIDUAL_FULL (the gain
 * stage advances once per hop) and allowed with ESTIMATE (a stateless re-fusion, e.g. of another weight set). aec_set_level()
 * retargets the fused gain stage; the lanes' own level has no effect.
 * Views. In an array view formed_hop and echo_spec of AecResidualView are NULL (the lanes' hops are in AecArrayView), and
 * AecLinearView is lane 0's with the shared alignment. res_floor_protect is max_dt > 0.2 (a single channel: the near-recent
 * latch), and all_converged / max_dt / max_saturation vote over the lanes with a nonzero weight. Lifetimes: the lane pointers
 * of AecArrayView are valid until the next aec_process_linear() or aec_reset() (a reset does not clear the lanes' buffers, so
 * an old pointer reads stale values, not a fault, but no longer describes a hop); AecResidualView.error_spec and, when no
 * lane's output guard fired, base_spec ARE the caller's beamformed_error buffer when one was passed (it must outlive the
 * reads, and an in-place aec_apply_gain() on it changes what they read), otherwise internal buffers valid until the next
 * process or reset call.
 * Shared alignment. The delay estimator (MATCHED) is fed by ONE microphone, capture_proxy_channel; the other three are never
 * seen by it, so a dead or blocked proxy microphone leaves the array without an acquired delay (all lanes run on the raw
 * far end). FIXED and EXTERNAL_ALIGNED use no estimator.
 * Fields with no effect on an array: delay_est_init_s (the duty-cycle hold is a fixed 0.3 s), delay_buffer_ms (the shared
 * ring is max_delay_ms + 2 hops + 1 samples), dt_aware_res_floor_enabled and ne_recent_* for the fused gain (its DT floor
 * applies whenever the worst lane double-talk indicator is above 0.2; the lanes still use ne_recent_* for their own
 * echo-path-change recovery). Limits that differ from a single channel: 8 kHz is refused and max_delay_ms is capped at 4096
 * (a single channel allows 60000).
 * Fused gain. With AEC_RESIDUAL_FULL the gain comes from a stand-alone gain stage fed with the fused spectra: near power is
 * |near|^2, limited by |E|^2 once every contributing lane has converged; the fused r2 serves as both the bounded and the
 * unbounded residual-echo estimate; the comfort-noise PSD is the weighted power sum; the far level is the aligned far hop's.
 * A single channel runs the instance's own post chain, which keeps separate bounded / unbounded estimates, so the array gain
 * is not guaranteed to equal a single channel's on the same lane. */

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
    int            n_channels;
    int            n_freqs;
    int            hop_size;
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
#define AEC_LAYOUT_VERSION     1u
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
_Static_assert(sizeof(AecMemReq) == 32, "AecMemReq must be exactly 32 bytes");

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
                               * block, mic/aligned-far/fusion buffers, the residual gain stage, the lanes' shared FFT handle;
                               * the lanes' pools are NOT included); 0 for a single-channel instance */
} AecMemBreakdown;

int aec_get_mem_breakdown(const AecConfig* cfg, AecMemBreakdown* out);

/* ---- Array diagnostics ----------------------------------------------------------------------------------------------
 * Read-only instrumentation of an array instance (cumulative since construction or aec_reset()). On a single-channel
 * instance they answer with the neutral value noted. A handle that is not ready answers 0 / -1 / a zeroed record.
 *   aec_far_fft_real_compute_count()   (above) on an array: the four lanes' counts summed; lane 0 runs the far FFT and lanes
 *                                      1-3 borrow it, so +1 per hop (+4 on a hop whose lanes could not borrow).
 * aec_array_realign_*_lane_count: aec_apply_external_realign() outcomes of the lanes after a shared-alignment change; a
 *   `changed` hop adds 4 to the pair (warm: IR shifted, soft: filter-only reset); fewer means a lane rejected the call. A
 *   `changed` hop whose delta is 0 (e.g. a first acquisition at delay 0) also adds 4 soft although no filter was reset.
 *   Saturate at LONG_MAX. Single-channel: 0. */
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
/* Wall clock of the most recent hop in microseconds, diagnostic only; all 0 unless built with -DAEC_STAGE_TIMING=1.
 * aec_get_last_timing() reports the same record without fuse_us (res_us = lane_res_us). A call refused before the hop starts
 * (NULL argument, handle not ready, non-finite sample) leaves the record untouched; any other aec_process_linear() first
 * zeroes the four linear-stage fields;
 * fuse_us is zeroed by an aec_process_residual() that passes its argument, sequence and beam checks. out must not be NULL. */
void aec_array_get_timing(const Aec* a, AecArrayTiming* out);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_H */
