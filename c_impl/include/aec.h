/* aec.h — top-level AEC orchestration for v2 rewrite.
 *
 * Wires together every Wave-1/2/3 module into one frame-level process()
 * that mirrors python/aec.py AEC.process for the BALANCED preset
 * (FDAF mode, enable_dtd=False, enable_delay_est=True, enable_cng per CLI).
 *
 * Out of scope (Plan §取捨):
 *   - AecMode != FDAF (NLMS, buffered FDAF queue)
 *   - DTD detectors (BALANCED has enable_dtd=False)
 *   - get_stats / debug logger / capture_stages
 *   - return_res_context
 */
#ifndef AEC_AEC_H
#define AEC_AEC_H

#include "hpf.h"
#include "saturation.h"
#include "delay_est.h"
#include "erle.h"
#include "detectors.h"
#include "pbfdkf.h"
#include "epc_shadow.h"
#include "residual_echo.h"
#include "res_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { AEC_PRESET_MILD = 0, AEC_PRESET_SOFT, AEC_PRESET_BALANCED,
               AEC_PRESET_AGGRESSIVE, AEC_PRESET_MAXIMUM } AecPreset;

typedef struct AecConfig {
    int          sample_rate;       /* 16000 default */
    float        filter_length_ms;  /* 52.0 */
    int          n_partitions;      /* derived if 0 */
    float        mu;                /* 0.3 */
    float        delta;             /* 1e-8 */
    /* Toggles */
    int          enable_cng;        /* default 0 (User Fix 2) */
    int          enable_delay_est;  /* default 1 */
    int          enable_highpass;   /* default 1 */
    int          enable_saturation; /* default 1 */
    int          enable_shadow_filter; /* default 1 */
    int          enable_residual_filter; /* default 1 */
    int          saturation_softclip_ref; /* default 1 */
    /* Preset-tunable */
    float        res_g_min_db;
    float        res_over_sub_base;
    float        res_over_sub_scale;
    float        res_dt_reduction;
    float        res_spectral_floor_db;
    float        res_ne_protect_db;
    float        res_alpha_echo_psd;
    float        res_alpha_error_psd;
    float        res_enr_scale;
    float        res_reverb_decay;          /* User Fix 1: 0.85 BALANCED */
    float        res_reverb_gain;           /* User Fix 1: 1.6 BALANCED */
    float        saturation_over_sub_boost; /* 3.0 */
    float        startup_dt_mu_min;         /* 0.0 default */
    float        shadow_mu_min;             /* 0.6 BALANCED */
    float        shadow_q_ratio;            /* 3.5 BALANCED */
    float        shadow_copy_threshold;     /* 0.65 */
    int          shadow_copy_hysteresis;    /* 3 */
    float        shadow_dtd_offset;         /* 1.5 */
    float        shadow_dtd_advantage_scale;/* 3.0 */
    float        shadow_err_alpha;          /* 0.80 */
    int          warmup_frames;             /* 80 */
    int          epc_hangover;              /* 20 */
    float        epc_total_rise;            /* 1.5 */
    float        epc_delta_threshold;       /* 0.3 */
    float        max_delay_ms;              /* v3.10.4: 1024 (was 250 → 512 → 1024) */
    float        delay_buffer_ms;           /* v3.10.4: 2048 — render ring buffer cap, separate from max_delay_ms */
    float        delay_est_init_s;          /* 0.3 */
    float        delay_est_period_s;        /* 0.5 */
    float        highpass_cutoff_hz;        /* 80 */
    float        saturation_threshold;      /* 0.95 */
    float        kalman_q_high;             /* 1e-3 BALANCED */
    float        kalman_q_low;              /* 1e-6 (matches Python default) */
    /* Debug */
    int          debug_level;
    const char*  debug_log_path;
} AecConfig;

void aec_config_defaults(AecConfig* cfg, int sample_rate);
void aec_config_from_preset(AecConfig* cfg, AecPreset preset, int sample_rate);

/* Opaque-ish context. Members exposed for parity dump but stable layout
 * not part of public ABI. */
typedef struct Aec {
    AecConfig cfg;

    int hop_size;
    int block_size;
    int fft_size;
    int n_freqs;

    /* Modules */
    Hpf hp_mic, hp_ref;
    int   has_hp;
    Saturation sat_ref, sat_mic;
    int   has_sat;
    DelayEst delay_est;
    int   has_delay_est;
    PBFDKF main_filter;
    PBFDKF shadow_filter;
    int   has_shadow;
    RenderActivity     render_activity;
    FilterConvergence  convergence;
    DoubleTalk         dt_analyzer;
    EpcDetector        epc;
    ShadowCopy         shadow_copy;
    ResFilter          res;
    int   has_res;
    /* v3.10.0 — filter plateau detector (detects bad-startup taps trap). */
    FilterPlateauDetector plateau_detector;

    /* Delay-compensation ring buffer (only when delay_est on) */
    float* ref_ring;
    int    ref_ring_size;
    int    ref_ring_write;
    int    ref_ring_filled;
    int    current_delay;       /* -1 until first acquisition */
    int    pending_delay;       /* -1 until first 32+ shift candidate */
    int    has_pending;
    /* v3.10.3 — TTL on pending shift candidate; ages out per estimation cycle. */
    int    pending_delay_ttl;

    /* Per-frame scalars (mirrors Python AEC instance fields) */
    double saturation_level;
    double erl_estimate;        /* 0.1 init */
    double main_err_smooth;
    double shadow_err_smooth;
    int    shadow_frame_count;
    int    epc_render_forced_remaining;
    double erle_window_near;
    double erle_window_err;
    double erle_factor_prev;
    double inst_erle_smooth;
    double wn_err_baseline;
    int    stat_dt_hangover;
    int    warmup_frames_remaining;
    int    warmup_far_active;
    double simple_mu_ratio;
    int    simple_mu_holdoff;
    float* per_bin_mu_scale;     /* NULL or [n_freqs] */
    int    has_per_bin_mu;
    double limiter_gain;
    /* Power EMAs (sample-by-sample, alpha=0.95) */
    double near_power;
    double raw_error_power;
    double final_error_power;
    double alpha_pow;            /* 0.95 */
    double near_power_sum;
    double raw_error_power_sum;
    double final_error_power_sum;
    int    frame_count;
    /* Diagnostic fields (last frame's value, mirrors Python aec._diag) */
    double last_dt_indicator;
    double last_mu_scale;
    /* Stashed values from last aec_process(), exposed via aec_get_res_context */
    double last_far_power;
    double last_shadow_dt;
    int    last_is_stationary_dt;
    /* Hop scratch buffers */
    float* near_hop;             /* [hop] */
    float* far_hop;              /* [hop] */
    float* raw_output;           /* [hop] */
    float* res_output;           /* [hop] */

    int    is_static;            /* 1 = state placed in caller buffer */
} Aec;

int  aec_create(Aec* a, const AecConfig* cfg);
void aec_destroy(Aec* a);
void aec_reset(Aec* a);

/**
 * Static-memory companions to aec_create (heap-free init for embedded
 * targets). Pre-allocate one buffer of size aec_get_mem_size(cfg) bytes,
 * then pass it into aec_init. Memory must be 16-byte aligned.
 *
 *   size_t pool_bytes = aec_get_mem_size(&cfg);
 *   void*  pool       = your_static_pool_alloc(pool_bytes);
 *   Aec    aec_inst;
 *   aec_init(&aec_inst, pool, pool_bytes, &cfg);
 *   ...
 *   aec_destroy(&aec_inst);  // no-op when initialised via aec_init
 */
size_t aec_get_mem_size(const AecConfig* cfg);
int    aec_init(Aec* a, void* mem, size_t mem_size, const AecConfig* cfg);

/* Process exactly hop_size samples. */
void aec_process(Aec* a, const float* mic, const float* ref, float* out);

/* Accessors mirroring old C diag API for parity dump tools */
int  aec_hop_size(const Aec* a);

/**
 * Linear-AEC + external-RES integration hook.
 *
 * Snapshot of internal AEC state taken from the most recently processed
 * frame. Lets a downstream component (NN post-filter, classical RES,
 * pipeline that runs NR before RES) consume the same context that the
 * built-in ResFilter would have seen.
 *
 * Typical usage:
 *
 *     AecConfig cfg;
 *     aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
 *     cfg.enable_residual_filter = 0;        // skip built-in RES
 *
 *     Aec aec; aec_create(&aec, &cfg);
 *     aec_process(&aec, mic_hop, ref_hop, linear_out);
 *
 *     AecResContext ctx;
 *     aec_get_res_context(&aec, &ctx);       // valid until next aec_process
 *
 *     // ... downstream RES / NN model uses ctx ...
 *
 * The Complex* spec pointers reference internal storage and remain
 * valid until the next aec_process call. Scalars are copied.
 */
typedef struct AecResContext {
    int            n_freqs;
    int            hop_size;
    /* Time-domain linear AEC output (mic - filter echo, before built-in RES).
     * [hop_size] samples; valid until next aec_process. */
    const float*   linear_hop;
    /* Frequency-domain spectra (internal storage; valid until next aec_process). */
    const Complex* echo_spec;       /* filter echo estimate, [n_freqs] */
    const Complex* far_spec;        /* far-end FFT, [n_freqs] */
    const Complex* near_spec;       /* mic FFT, [n_freqs] */
    /* Scalar context (copied). */
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
} AecResContext;

void aec_get_res_context(const Aec* a, AecResContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_H */
