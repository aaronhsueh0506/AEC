/* aec_v2.h — top-level AEC orchestration for v2 rewrite.
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
#ifndef AEC_V2_AEC_H
#define AEC_V2_AEC_H

#include "hpf_v2.h"
#include "saturation_v2.h"
#include "delay_est_v2.h"
#include "erle_v2.h"
#include "detectors_v2.h"
#include "pbfdkf_v2.h"
#include "epc_shadow_v2.h"
#include "residual_echo_v2.h"
#include "res_filter_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { AECV2_PRESET_MILD = 0, AECV2_PRESET_BALANCED,
               AECV2_PRESET_AGGRESSIVE, AECV2_PRESET_MAXIMUM } AecV2Preset;

typedef struct AecV2Config {
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
    float        max_delay_ms;              /* 250 */
    float        delay_est_init_s;          /* 0.3 */
    float        delay_est_period_s;        /* 0.5 */
    float        highpass_cutoff_hz;        /* 80 */
    float        saturation_threshold;      /* 0.95 */
    float        kalman_q_high;             /* 1e-3 BALANCED */
    float        kalman_q_low;              /* 1e-6 (matches Python default) */
    /* Debug */
    int          debug_level;
    const char*  debug_log_path;
} AecV2Config;

void aec_v2_config_defaults(AecV2Config* cfg, int sample_rate);
void aec_v2_config_from_preset(AecV2Config* cfg, AecV2Preset preset, int sample_rate);

/* Opaque-ish context. Members exposed for parity dump but stable layout
 * not part of public ABI. */
typedef struct AecV2 {
    AecV2Config cfg;

    int hop_size;
    int block_size;
    int fft_size;
    int n_freqs;

    /* Modules */
    HpfV2 hp_mic, hp_ref;
    int   has_hp;
    SaturationV2 sat_ref, sat_mic;
    int   has_sat;
    DelayEstV2 delay_est;
    int   has_delay_est;
    PBFDKFV2 main_filter;
    PBFDKFV2 shadow_filter;
    int   has_shadow;
    RenderActivityV2     render_activity;
    FilterConvergenceV2  convergence;
    DoubleTalkV2         dt_analyzer;
    EpcDetectorV2        epc;
    ShadowCopyV2         shadow_copy;
    ResFilterV2          res;
    int   has_res;

    /* Delay-compensation ring buffer (only when delay_est on) */
    float* ref_ring;
    int    ref_ring_size;
    int    ref_ring_write;
    int    ref_ring_filled;
    int    current_delay;       /* -1 until first acquisition */
    int    pending_delay;       /* -1 until first 32+ shift candidate */
    int    has_pending;

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
    /* Hop scratch buffers */
    float* near_hop;             /* [hop] */
    float* far_hop;              /* [hop] */
    float* raw_output;           /* [hop] */
    float* res_output;           /* [hop] */
} AecV2;

int  aec_v2_create(AecV2* a, const AecV2Config* cfg);
void aec_v2_destroy(AecV2* a);
void aec_v2_reset(AecV2* a);

/* Process exactly hop_size samples. */
void aec_v2_process(AecV2* a, const float* mic, const float* ref, float* out);

/* Accessors mirroring old C diag API for parity dump tools */
int  aec_v2_hop_size(const AecV2* a);

#ifdef __cplusplus
}
#endif

#endif /* AEC_V2_AEC_H */
