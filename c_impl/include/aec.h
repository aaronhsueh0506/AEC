/* Public C API for the single-channel PBFDKF AEC: heap or caller-pool
 * construction, lockstep processing, and the external RES /
 * linear-context getters. */
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
    AEC_LEVEL_NO_FLOOR   = 3,

    /* Legacy names of the four levels above (same values; the Python twins call this ladder a
     * preset and their tests read these lines). Keep the literal digits. */
    AEC_PRESET_MILD       = 0,
    AEC_PRESET_BALANCED   = 1,
    AEC_PRESET_AGGRESSIVE = 2,
    AEC_PRESET_NO_FLOOR   = 3
} AecLevel;

/* Legacy type name of AecLevel. */
typedef AecLevel AecPreset;

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

/* AecConfig.spatial_linear_context levels. 0: full context. 1: this lane's
 * own G_res is not computed. 2: linear half of the context only. */
#define AEC_CONTEXT_NO_GAIN     1
#define AEC_CONTEXT_LINEAR_ONLY 2

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
    int    enable_delay_est;       /* 1; deprecated mirror of delay_mode (see aec_config_resolve_delay) */
    int    enable_highpass;        /* mic-path high-pass; 1 */
    int    enable_saturation;      /* saturation limiter; 1 */
    int    enable_shadow;          /* shadow filter; 1 */
    int    enable_res;             /* post-filter suppression (RES); 1 */
    int    saturation_softclip_ref;/* soft-clip the reference; 1 */
    /* Run the post-filter block and expose AecResContext; with enable_res ==
     * 0 its suppression is not applied and the output stays the linear
     * residual. */
    int    return_res_context;     /* 0 */
    /* AEC_CONTEXT_*: 1 skips this lane's G_res; 2 stops after the AecState
     * update (linear half of the context only; res_gain, r2, comfort_noise
     * NULL). Both need enable_res == 0 and return_res_context == 1. */
    int    spatial_linear_context; /* 0 */

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

/* Folds the deprecated enable_delay_est into delay_mode in place (idempotent;
 * called by validate / get_mem_size / create / init): enable_delay_est = 0
 * with delay_mode MATCHED becomes FIXED if fixed_delay_samples >= 0, else
 * EXTERNAL_ALIGNED; the mirror is then rewritten. Returns 0, or -1 for NULL
 * or an invalid delay_mode. */
int aec_config_resolve_delay(AecConfig* cfg);

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

/* Resolves (sample_rate, fft_size) into *out. Returns 1 on a supported pair;
 * 0 otherwise (out untouched, including fft_size == 0). out may be NULL. */
int aec_resolve_signal_grid(int sample_rate, int fft_size, AecSignalGrid* out);

/* Returns 1 if some grid exists at this rate (8000/16000/48000); the (rate,
 * fft_size) pair must still resolve. */
int aec_is_valid_sample_rate(int sample_rate);

/* Product-default fft_size for a rate (16k: 256, 48k: 1024, 8k: 256); 0 if
 * the rate is unsupported. */
int aec_default_fft_size(int sample_rate);

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
    int    duty_active;
    int    duty_stable_hops;
    int    duty_pos;
    int    duty_last_delay;
    float  duty_erle_peak;
    float  duty_erle_leak_db;
    unsigned long long duty_hops_total;
    unsigned long long duty_hops_run;

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

    int    duty_hold_hops;
    int    duty_period_hops;

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
    float* scr_e2coa_pre;
    float* scr_mu_buf;
    float* scr_e2ref_arr;
    float* scr_erl_arr;

    int is_static;

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
 * cfg or allocation failure. aec_destroy releases the instance (NULL-safe,
 * repeatable); aec_reset restores the initial state. */
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

/* Far-active min-gain floor (dB) for a level: mild -20, balanced -28,
 * aggressive -38 (no_floor: -28, an inert placeholder). Public so a product
 * that owns a SuppressionGain without an Aec can use the same table. Returns
 * 0 and writes *db_out for an in-enum level, -1 otherwise; aec_config_from_level
 * instead falls back to balanced. */
int aec_level_floor_db(AecLevel level, float* db_out);

/* DT-lifted floor (min_gain_floor_dt_db) for a level; only aggressive differs
 * from the shared -20 dB (no_floor: inert -20). Returns 0 and writes *db_out
 * for an in-enum level, -1 (nothing written) otherwise. */
int aec_level_dt_floor_db(AecLevel level, float* db_out);

/* min_gain_split_floor_enabled for a level. Returns 0 and writes
 * *enabled_out for an in-enum level, -1 (nothing written) otherwise. */
int aec_level_split_floor_enabled(AecLevel level, int* enabled_out);

/* Retargets the residual-echo strength of a running instance; filter, delay
 * state and histories carry on. Updates min_gain_floor_far_active_db and, for
 * aggressive, min_gain_floor_dt_db; NO_FLOOR clears
 * min_gain_split_floor_enabled. ramp_ms in [0, 60000] walks the far-active
 * floor linearly in dB (0 = next hop); the DT floor and the split-floor flag
 * step immediately. Inert on a spatial_linear_context lane. Call between
 * hops, serialised with aec_process(). Returns 0, or -1 (nothing written) on
 * NULL, an out-of-enum level or an out-of-range ramp_ms. */
int aec_set_level(Aec* a, AecLevel level, float ramp_ms);

/* Static-memory construction: aec_get_mem_size() returns the pool bytes (0 if
 * cfg is invalid). aec_init() places the Aec at mem[0] without malloc and
 * returns it, or NULL for a NULL argument, invalid cfg, a pool not 16-byte
 * aligned or smaller than aec_get_mem_size(). */
size_t aec_get_mem_size(const AecConfig* cfg);
Aec*   aec_init(void* mem, size_t mem_size, const AecConfig* cfg);

/* Processes exactly hop_size samples (lockstep: render and capture supplied
 * together). */
void aec_process(Aec* a, const float* mic, const float* ref, float* out);

/* As aec_process() but writes no output; for callers that only read
 * aec_get_res_context(). aec_process, aec_process_context and
 * aec_process_context_shared_far_from advance identical state and may be
 * interleaved freely on one instance. */
void aec_process_context(Aec* a, const float* mic, const float* ref);

/* Shares the far spectrum and its squared magnitudes from far_owner, which
 * must have completed this same hop. Both instances must consume identical
 * ref samples with synchronized far-buffer histories. No owner pointer is
 * retained; a NULL, self or incompatible-grid owner falls back to a local
 * computation. */
void aec_process_context_shared_far_from(
        Aec* a, const float* mic, const float* ref, const Aec* far_owner);

/* Cumulative count (since construction or aec_reset(), saturating at
 * LONG_MAX) of hops where this instance ran its own far-end FFT. */
long aec_far_fft_real_compute_count(const Aec* a);

/* 1 if the linear filter is demonstrably cancelling: last hop's windowed ERLE
 * above 2.5 dB, or the recent inst-ERLE peak above
 * cfg.delay_acquire_inst_erle_db. 0 means no evidence, not a fault (also 0
 * before the ERLE machinery has run). */
int aec_linear_is_cancelling(const Aec* a);

/* Samples per process call (fft_size / 2). */
int  aec_hop_size(const Aec* a);

/* Linear-AEC + external-RES context. */
typedef struct AecResContext {
    int            n_freqs;
    int            hop_size;
    const float*   linear_hop;       /* refined PBFDKF hop, before selection */
    const float*   formed_hop;       /* current hop underlying error_spec */
    const Complex* echo_spec;
    const Complex* far_spec;
    const Complex* near_spec;
    /* Frequency-domain seam for an external post-NR suppressor; valid when
     * enable_res or return_res_context is set, else NULL. These are the ANALYSIS
     * spectra: the output guard never alters them (the rendering base is
     * aec_get_output_base()). error_spec is the
     * formed linear E(f) (50% overlap, periodic sqrt-Hann STFT, audio scale);
     * echo_spec the matching echo estimate; near_spec = error_spec +
     * echo_spec. res_gain is G_res(f) in [0, 1]. r2 (residual-echo PSD) and
     * comfort_noise (CNG) are int16^2-scaled: divide by 32768^2 for audio
     * power. All have n_freqs entries. With spatial_linear_context ==
     * AEC_CONTEXT_LINEAR_ONLY, res_gain, r2 and comfort_noise are NULL and
     * res_floor_protect is 0. Pointers alias internal buffers and are valid
     * only until the next processing call on the instance. */
    const Complex* error_spec;     /* formed linear E(f), audio scale */
    const float*   res_gain;       /* G_res(f) in [0,1]; NULL when spatial_linear_context is set */
    const float*   r2;             /* residual-echo PSD R^2(f), int16^2-scaled */
    const float*   comfort_noise;  /* CNG N^2(f), int16^2-scaled */
    float          far_power;
    float          erle_factor;
    float          dt_indicator;
    float          divergence;
    float          saturation_level;
    float          erl_estimate;
    float          shadow_dt;
    int            is_stationary_dt;
    int            filter_converged;
    int            filter_once_converged;
    int            epc_active;
    int            res_floor_protect;  /* this hop's near-recent DT-floor decision */
    int            usable_linear;      /* usable-linear-estimate verdict; 0: r2 is the conservative nonlinear estimate */
} AecResContext;

void aec_get_res_context(const Aec* a, AecResContext* ctx);

/* Rendering base of the last processed hop (valid when enable_res or return_res_context is set; any
 * spatial_linear_context mode, any enable_res/return_res_context combination).
 *
 * AecResContext.error_spec / echo_spec / near_spec / formed_hop are the ANALYSIS seam: the selected linear
 * residual (refined or coarse) and its matched echo estimate. They are never altered by the output guard
 * (cfg output_capture_when_linear_unusable, always on in the shipped levels), and error_spec is the STFT of
 * [previous formed hop | current formed hop], so a unity inverse transform with overlap-add reproduces the
 * previous formed hop.
 *
 * The RENDERING base is what an external synthesizer must multiply by its gain(s), add comfort noise to,
 * inverse-transform and overlap-add, exactly as aec_process() does internally. It is evaluated once per hop,
 * after this hop's usable_linear verdict is final and before any context exit, identically in every mode:
 * guard_active == 1 when the linear estimate is unusable and sum|error_spec|^2 > sum|error_spec + echo_spec|^2
 * (strict, windowed spectra, float32 pairwise reduction); then base_spec = error_spec + echo_spec (the
 * capture spectrum, equal to near_spec), otherwise base_spec = error_spec. base_spec does NOT pair with
 * formed_hop (nor with the previous/next formed hop): on a guard hop it is a spectral switch to the capture
 * and unity synthesis of it does not reproduce any formed hop. Use formed_hop / error_spec for analysis and
 * base_spec only for rendering.
 *
 * Caveats for consumers:
 *   - Time-domain seam consumers (AecResContext.formed_hop, AecLinearContext.formed_linear_hop) are NOT
 *     guarded: those streams are the analysis residual on every hop, guard hops included, and there is no
 *     time-domain rendering base. A consumer that plays or re-synthesizes them gets no output guard.
 *   - A frequency-domain consumer that still renders error_spec (instead of base_spec) loses the guard:
 *     on guard hops its output carries the residual louder than the capture. This is a breaking change
 *     for renderers written against error_spec before aec_get_output_base() existed.
 *
 * struct_size is an input: the caller's declared size of *out (normally sizeof(AecOutputBase)). It is
 * read first, and nothing is written when it is smaller than sizeof(AecOutputBase) (-1). A larger value
 * (a newer caller struct) is accepted; only the sizeof(AecOutputBase) prefix is written, and on return
 * struct_size holds sizeof(AecOutputBase). base_spec has n_freqs entries and aliases a dedicated internal
 * buffer (or error_spec when the guard is off); it is valid until the next processing or reset call on
 * the instance and is not touched by comfort noise or any other internal scratch. Returns 0, or -1 for
 * a NULL out or a short struct_size (nothing written), or for a NULL instance or an instance with
 * neither enable_res nor return_res_context (the prefix is zeroed apart from struct_size). */
typedef struct AecOutputBase {
    size_t         struct_size;   /* in: sizeof(AecOutputBase) */
    int            n_freqs;
    int            guard_active;  /* this hop's output guard fired */
    const Complex* base_spec;     /* rendering base, n_freqs entries */
} AecOutputBase;

int aec_get_output_base(const Aec* a, AecOutputBase* out);

/* AEC_CNG_SEED (aec_post.h, 42u) is the comfort-noise seed after construction and after every hard
 * post-chain reset, and the seed aec_cng_fill() callers must (re)start from. */

/* Counter of hard post-chain resets (wrapping uint32_t, 0 on a freshly created instance), read without
 * consuming it. It advances by one on each of:
 *   - aec_reset(): a separate call, not part of any processing call; the serial already reads +1 when
 *     aec_reset() returns, and the next processing call starts from the cleared state.
 *   - a hard reset inside a processing call (aec_process / aec_process_context), done before that hop's
 *     synthesis: the hard branch of a first delay acquisition (Path A: no warm tap transfer and no
 *     DT-aware soft acquisition) and the hard branch of a confirmed delay shift (Path B: no DT-aware soft
 *     realign).
 * It never moves on the bad-relock revert (a soft revert to the pre-shift delay and its tap snapshot), on
 * warm tap transfers or DT-aware soft acquisitions / realignments, on aec_apply_external_realign(), or
 * when only AecLinearContext.generation changes. At each counted reset the library clears its own OLA
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

typedef struct AecLinearContext {
    int hop_size;
    const float* formed_linear_hop;  /* formed linear error (post chain active) or raw linear output */
    const float* aligned_far_hop;    /* aliases a->far_hop */
    int   delay_samples;             /* applied ring offset; -1 = none */
    float delay_confidence;          /* 0.0 / 0.5 / 1.0 */
    AecLinearDelayState delay_state;
    unsigned int generation;
} AecLinearContext;

void aec_get_linear_context(const Aec* a, AecLinearContext* ctx);

/* Read-only status snapshot for polling (e.g. once per second); fills a
 * struct from fields the engine already maintains, with no hot-path cost. */
typedef struct AecDebugStatus {
    /* delay estimation */
    int    delay_samples;      /* applied delay in samples; -1 = not yet acquired */
    float  delay_confidence;   /* MATCHED: estimator's 0.0/0.5/1.0; FIXED, EXTERNAL_ALIGNED: 1.0 */
    int    delay_updates;      /* matched-filter update count; 0 outside MATCHED */

    /* linear filter health */
    float  erle_windowed_db;   /* windowed ERLE (dB) at the last post-chain hop; 0 before it first ran */
    int    usable_linear;      /* usable-linear-estimate gate state */
    int    filter_converged;   /* convergence latch (AecResContext.filter_converged) */

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
 * Returns 0 and zeroes *out when cfg is NULL or invalid. */
typedef struct AecMemBreakdown {
    size_t total_bytes;
    size_t estimator_bytes;
    size_t ring_bytes;
} AecMemBreakdown;

int aec_get_mem_breakdown(const AecConfig* cfg, AecMemBreakdown* out);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_H */
