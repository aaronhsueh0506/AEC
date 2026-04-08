/**
 * aec_types.h - AEC Configuration Types and Presets
 *
 * Matches Python v1.28.1 (PBFDKF mode, Kalman always on).
 * Four presets: MILD / BALANCED / AGGRESSIVE / MAXIMUM
 */

#ifndef AEC_TYPES_H
#define AEC_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Preset enum --- */
typedef enum {
    AEC_PRESET_MILD = 0,
    AEC_PRESET_BALANCED,
    AEC_PRESET_AGGRESSIVE,
    AEC_PRESET_MAXIMUM
} AecPreset;

/* --- RES echo estimation method --- */
typedef enum {
    RES_ECHO_COHERENCE = 0,
    RES_ECHO_DIRECT
} ResEchoMethod;

/* --- RES gain computation type --- */
typedef enum {
    RES_GAIN_SPECTRAL_SUB = 0,
    RES_GAIN_WIENER,
    RES_GAIN_ENR
} ResGainType;

/* --- Main configuration --- */
typedef struct {
    int sample_rate;            /* 8000 / 16000 / 48000 */
    int frame_size;             /* 20ms: 160@8k, 320@16k, 960@48k */
    int hop_size;               /* 10ms: 80@8k, 160@16k, 480@48k */
    int filter_length;          /* 32ms: 256@8k, 512@16k, 1536@48k */
    float delta;                /* Regularization: 1e-8 */

    /* RES parameters */
    int enable_res;             /* 1 */
    float res_g_min_db;         /* -55 (BALANCED) */
    float res_over_sub_base;    /* 5.0 */
    float res_over_sub_scale;   /* 9.0 */
    float res_dt_reduction;     /* 2.5 */
    float res_spectral_floor_db; /* -38.0 */
    float res_ne_protect_db;    /* -16.0 */
    float res_max_drop_db_per_frame; /* 6.0 */
    float res_max_rise_db_per_frame; /* 6.0 */
    int   enable_cng;           /* 1 */

    /* RES v2 */
    int   res_echo_method;      /* RES_ECHO_DIRECT */
    int   res_gain_type;        /* RES_GAIN_ENR */
    float res_enr_scale;        /* 0.85 (BALANCED) */
    int   res_enable_reverb;    /* 1 */
    float res_reverb_decay;     /* 0.5 */
    float res_reverb_gain;      /* 0.5 */
    float res_alpha_echo_psd;   /* 0.5 */
    float res_alpha_error_psd;  /* 0.6 */

    /* Shadow filter */
    float shadow_q_ratio;       /* 3.5 */
    float shadow_copy_threshold; /* 0.7 */
    float shadow_err_alpha;     /* 0.85 */
    float shadow_mu_min;        /* 0.6 */
    int   shadow_copy_hysteresis; /* 5 */

    /* FDKF Kalman */
    float kalman_q_high;        /* 1e-4 */
    float kalman_q_low;         /* 1e-5 */
    int   warmup_frames;        /* 150 */

    /* Echo path change detection */
    float epc_delta_threshold;  /* 0.3 */
    float epc_total_rise;       /* 1.5 */
    int   epc_hangover;         /* 20 */
    float epc_mu_floor;         /* 0.5 */

    /* Delay estimation (GCC-PHAT) */
    int enable_delay_est;        /* 0 = disabled (default for backward compat) */
    float delay_est_max_ms;      /* 250.0 — max delay to search (ms) */
    float delay_est_init_s;      /* 0.3 — initial accumulation before first estimate */
    float delay_est_period_s;    /* 0.5 — re-estimation period (seconds) */

    /* Derived (computed by factory) */
    int fft_size;               /* next pow2 >= frame_size: 256@8k, 512@16k, 1024@48k */
    int n_freqs;                /* fft_size/2 + 1: 129@8k, 257@16k, 513@48k */
    int n_partitions;           /* ceil(filter_length / hop_size): 10@8k/16k, 10@48k */
} AecConfig;

/* --- AEC context for external RES (linear pipeline) --- */
typedef struct {
    float*   echo_spec_re;      /* [n_freqs] echo estimate (real) */
    float*   echo_spec_im;      /* [n_freqs] echo estimate (imag) */
    float*   far_spec_re;       /* [n_freqs] far-end spectrum (real) */
    float*   far_spec_im;       /* [n_freqs] far-end spectrum (imag) */
    float*   near_spec_re;      /* [n_freqs] mic spectrum (real) */
    float*   near_spec_im;      /* [n_freqs] mic spectrum (imag) */
    float    far_power;         /* mean(far²) */
    int      filter_converged;  /* 0 or 1 */
    float    erle_factor;       /* [0, 1] convergence metric */
    float    dt_indicator;      /* [0, 0.8] double-talk confidence */
    float    divergence;        /* [0, 1] divergence indicator */
    float    over_sub;          /* dynamic over_sub value */
    int      n_freqs;           /* number of frequency bins */
    int      is_stationary_dt;  /* 1 = stationary far-end DT detected */
} AecResContext;

/* --- Factory functions --- */

/**
 * Create default config (BALANCED preset) for given sample rate
 */
static inline AecConfig aec_default_config(int sample_rate) {
    AecConfig c;
    c.sample_rate = sample_rate;
    c.frame_size = sample_rate * 20 / 1000;   /* 20ms: 160@8k, 320@16k, 960@48k */
    c.hop_size = c.frame_size / 2;            /* 10ms: 80@8k, 160@16k, 480@48k */
    c.filter_length = sample_rate * 32 / 1000;  /* 32ms: 256@8k, 512@16k, 1536@48k */
    c.delta = 1e-8f;

    c.enable_res = 1;
    c.res_g_min_db = -55.0f;
    c.res_over_sub_base = 5.0f;
    c.res_over_sub_scale = 9.0f;
    c.res_dt_reduction = 2.5f;
    c.res_spectral_floor_db = -38.0f;
    c.res_ne_protect_db = -16.0f;
    c.res_max_drop_db_per_frame = 6.0f;
    c.res_max_rise_db_per_frame = 6.0f;
    c.enable_cng = 0;

    /* RES v2 */
    c.res_echo_method = RES_ECHO_DIRECT;
    c.res_gain_type = RES_GAIN_ENR;
    c.res_enr_scale = 0.85f;
    c.res_enable_reverb = 1;
    c.res_reverb_decay = 0.65f;
    c.res_reverb_gain = 1.4f;
    c.res_alpha_echo_psd = 0.4f;
    c.res_alpha_error_psd = 0.5f;

    c.shadow_q_ratio = 3.5f;
    c.shadow_copy_threshold = 0.7f;
    c.shadow_err_alpha = 0.85f;
    c.shadow_mu_min = 0.6f;
    c.shadow_copy_hysteresis = 5;

    c.kalman_q_high = 1e-3f;
    c.kalman_q_low = 1e-5f;
    c.warmup_frames = 80;

    c.epc_delta_threshold = 0.3f;
    c.epc_total_rise = 1.5f;
    c.epc_hangover = 20;
    c.epc_mu_floor = 0.5f;

    /* Delay estimation (disabled by default for backward compat) */
    c.enable_delay_est = 0;
    c.delay_est_max_ms = 250.0f;
    c.delay_est_init_s = 0.3f;
    c.delay_est_period_s = 0.5f;

    /* Derived */
    c.fft_size = 256;
    while (c.fft_size < c.frame_size) c.fft_size *= 2;  /* next pow2 >= frame_size */
    c.n_freqs = c.fft_size / 2 + 1;
    c.n_partitions = (c.filter_length + c.hop_size - 1) / c.hop_size;

    return c;
}

/**
 * Create config from preset
 */
static inline AecConfig aec_config_from_preset(AecPreset preset, int sample_rate) {
    AecConfig c = aec_default_config(sample_rate);

    /* Enable delay estimation for all presets */
    c.enable_delay_est = 1;

    switch (preset) {
    case AEC_PRESET_MILD:
        c.res_g_min_db = -35.0f;
        c.res_over_sub_base = 2.5f;
        c.res_over_sub_scale = 4.0f;
        c.res_dt_reduction = 3.5f;
        c.res_spectral_floor_db = -25.0f;
        c.res_ne_protect_db = -10.0f;
        c.res_enr_scale = 1.0f;
        c.res_enable_reverb = 1;
        c.res_reverb_decay = 0.6f;
        c.res_reverb_gain = 0.8f;
        c.res_alpha_echo_psd = 0.5f;
        c.res_alpha_error_psd = 0.6f;
        c.shadow_q_ratio = 3.0f;
        c.shadow_mu_min = 0.5f;
        c.warmup_frames = 80;
        c.kalman_q_high = 1.5e-3f;
        break;

    case AEC_PRESET_BALANCED:
        /* Already set by default */
        break;

    case AEC_PRESET_AGGRESSIVE:
        c.res_g_min_db = -65.0f;
        c.res_over_sub_base = 7.0f;
        c.res_over_sub_scale = 12.0f;
        c.res_dt_reduction = 1.5f;
        c.res_spectral_floor_db = -45.0f;
        c.res_ne_protect_db = -22.0f;
        c.res_enr_scale = 0.7f;
        c.res_enable_reverb = 1;
        c.res_reverb_decay = 0.7f;
        c.res_reverb_gain = 2.0f;
        c.res_alpha_echo_psd = 0.3f;
        c.res_alpha_error_psd = 0.4f;
        c.shadow_q_ratio = 4.0f;
        c.shadow_mu_min = 0.7f;
        c.warmup_frames = 80;
        c.kalman_q_high = 7e-4f;
        break;

    case AEC_PRESET_MAXIMUM:
        c.res_g_min_db = -72.0f;
        c.res_over_sub_base = 10.0f;
        c.res_over_sub_scale = 15.0f;
        c.res_dt_reduction = 0.5f;
        c.res_spectral_floor_db = -55.0f;
        c.res_ne_protect_db = -30.0f;
        c.res_enr_scale = 0.5f;
        c.res_enable_reverb = 1;
        c.res_reverb_decay = 0.8f;
        c.res_reverb_gain = 3.0f;
        c.res_alpha_echo_psd = 0.2f;
        c.res_alpha_error_psd = 0.3f;
        c.shadow_q_ratio = 5.0f;
        c.shadow_mu_min = 0.9f;
        c.warmup_frames = 100;
        c.kalman_q_high = 7e-4f;
        break;
    }

    return c;
}

/**
 * Get preset name string
 */
static inline const char* aec_preset_name(AecPreset preset) {
    switch (preset) {
    case AEC_PRESET_MILD:       return "mild";
    case AEC_PRESET_BALANCED:   return "balanced";
    case AEC_PRESET_AGGRESSIVE: return "aggressive";
    case AEC_PRESET_MAXIMUM:    return "maximum";
    default: return "unknown";
    }
}

static inline const char* res_echo_method_name(int m) {
    return m == RES_ECHO_DIRECT ? "direct" : "coherence";
}

static inline const char* res_gain_type_name(int t) {
    switch (t) {
    case RES_GAIN_ENR:          return "enr";
    case RES_GAIN_WIENER:       return "wiener";
    case RES_GAIN_SPECTRAL_SUB: return "spectral_sub";
    default: return "unknown";
    }
}

#ifdef __cplusplus
}
#endif

#endif /* AEC_TYPES_H */
