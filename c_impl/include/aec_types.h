/**
 * aec_types.h - AEC Configuration Types and Presets
 *
 * Matches Python v1.17.0 (SUBBAND PBFDKF only, Kalman always on).
 * Three presets: BALANCED / AGGRESSIVE / MAXIMUM
 */

#ifndef AEC_TYPES_H
#define AEC_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Preset enum --- */
typedef enum {
    AEC_PRESET_BALANCED = 0,
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
    int sample_rate;            /* 16000 */
    int frame_size;             /* 320 (20ms @ 16kHz) — WOLA window length */
    int hop_size;               /* 160 (10ms @ 16kHz) */
    int filter_length;          /* 1600 samples (100ms) */
    float delta;                /* Regularization: 1e-8 */

    /* HPF */
    int enable_highpass;        /* 1 */
    float highpass_cutoff_hz;   /* 80.0 */

    /* RES parameters */
    int enable_res;             /* 1 */
    float res_g_min_db;         /* -35 (BALANCED) */
    float res_over_sub_base;    /* 2.5 */
    float res_over_sub_scale;   /* 4.0 */
    float res_dt_reduction;     /* 3.5 */
    float res_spectral_floor_db; /* -25.0 */
    float res_ne_protect_db;    /* -12.0 */
    float res_max_drop_db_per_frame; /* 6.0 */
    float res_max_rise_db_per_frame; /* 6.0 */
    int   enable_cng;           /* 1 */

    /* RES v2 */
    int   res_echo_method;      /* RES_ECHO_DIRECT */
    int   res_gain_type;        /* RES_GAIN_ENR */
    float res_enr_scale;        /* 1.0 */
    int   res_enable_reverb;    /* 1 */
    float res_reverb_decay;     /* 0.5 */
    float res_reverb_gain;      /* 0.5 */
    float res_alpha_echo_psd;   /* 0.5 */
    float res_alpha_error_psd;  /* 0.6 */

    /* Shadow filter */
    float shadow_q_ratio;       /* 3.0 */
    float shadow_copy_threshold; /* 0.7 */
    float shadow_err_alpha;     /* 0.85 */
    float shadow_mu_min;        /* 0.5 */
    int   shadow_copy_hysteresis; /* 5 */

    /* FDKF Kalman */
    float kalman_q_high;        /* 1e-4 */
    float kalman_q_low;         /* 1e-7 */
    int   warmup_frames;        /* 100 */

    /* Derived (computed by factory) */
    int fft_size;               /* 512 (next pow2 >= frame_size) */
    int n_freqs;                /* 257 (fft_size/2 + 1) */
    int n_partitions;           /* 10 (ceil(filter_length / hop_size)) */
} AecConfig;

/* --- Factory functions --- */

/**
 * Create default config (BALANCED preset) for given sample rate
 */
static inline AecConfig aec_default_config(int sample_rate) {
    AecConfig c;
    c.sample_rate = sample_rate;
    c.frame_size = 320;
    c.hop_size = 160;
    c.filter_length = 1600;   /* 100ms @ 16kHz (no delay est → need longer filter) */
    c.delta = 1e-8f;

    c.enable_highpass = 1;
    c.highpass_cutoff_hz = 80.0f;

    c.enable_res = 1;
    c.res_g_min_db = -35.0f;
    c.res_over_sub_base = 2.5f;
    c.res_over_sub_scale = 4.0f;
    c.res_dt_reduction = 3.5f;
    c.res_spectral_floor_db = -25.0f;
    c.res_ne_protect_db = -12.0f;
    c.res_max_drop_db_per_frame = 6.0f;
    c.res_max_rise_db_per_frame = 6.0f;
    c.enable_cng = 1;

    /* RES v2 */
    c.res_echo_method = RES_ECHO_DIRECT;
    c.res_gain_type = RES_GAIN_ENR;
    c.res_enr_scale = 1.0f;
    c.res_enable_reverb = 1;
    c.res_reverb_decay = 0.5f;
    c.res_reverb_gain = 0.5f;
    c.res_alpha_echo_psd = 0.5f;
    c.res_alpha_error_psd = 0.6f;

    c.shadow_q_ratio = 3.0f;
    c.shadow_copy_threshold = 0.7f;
    c.shadow_err_alpha = 0.85f;
    c.shadow_mu_min = 0.5f;
    c.shadow_copy_hysteresis = 5;

    c.kalman_q_high = 1e-4f;
    c.kalman_q_low = 1e-7f;
    c.warmup_frames = 100;

    /* Derived */
    c.fft_size = 512;
    c.n_freqs = 257;
    c.n_partitions = (c.filter_length + c.hop_size - 1) / c.hop_size;

    return c;
}

/**
 * Create config from preset
 */
static inline AecConfig aec_config_from_preset(AecPreset preset, int sample_rate) {
    AecConfig c = aec_default_config(sample_rate);

    switch (preset) {
    case AEC_PRESET_BALANCED:
        /* Already set by default */
        break;

    case AEC_PRESET_AGGRESSIVE:
        c.res_g_min_db = -60.0f;
        c.res_over_sub_base = 6.0f;
        c.res_over_sub_scale = 10.0f;
        c.res_dt_reduction = 1.0f;
        c.res_spectral_floor_db = -40.0f;
        c.res_ne_protect_db = -3.0f;
        c.res_enr_scale = 0.3f;
        c.res_enable_reverb = 1;
        c.res_reverb_decay = 0.7f;
        c.res_reverb_gain = 2.0f;
        c.res_alpha_echo_psd = 0.3f;
        c.res_alpha_error_psd = 0.4f;
        c.shadow_q_ratio = 4.0f;
        c.shadow_mu_min = 0.7f;
        c.warmup_frames = 150;
        c.kalman_q_high = 1e-4f;
        break;

    case AEC_PRESET_MAXIMUM:
        c.res_g_min_db = -72.0f;
        c.res_over_sub_base = 10.0f;
        c.res_over_sub_scale = 15.0f;
        c.res_dt_reduction = 0.0f;
        c.res_spectral_floor_db = -50.0f;
        c.res_ne_protect_db = -1.0f;
        c.res_enr_scale = 0.15f;
        c.res_enable_reverb = 1;
        c.res_reverb_decay = 0.8f;
        c.res_reverb_gain = 3.0f;
        c.res_alpha_echo_psd = 0.2f;
        c.res_alpha_error_psd = 0.3f;
        c.shadow_q_ratio = 5.0f;
        c.shadow_mu_min = 0.9f;
        c.warmup_frames = 200;
        c.kalman_q_high = 1e-4f;
        break;
    }

    return c;
}

/**
 * Get preset name string
 */
static inline const char* aec_preset_name(AecPreset preset) {
    switch (preset) {
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
