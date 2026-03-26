/**
 * res_filter.h - Residual Echo Suppressor with WOLA (v2)
 *
 * WOLA (Weighted Overlap-Add) with sqrt-Hann window:
 * - Analysis: frame_size=320 × sqrt-Hann → zero-pad → FFT(512) → 257 bins
 * - Synthesis: IFFT → truncate → sqrt-Hann → OLA
 *
 * v2 features: ENR masking gain, direct echo estimation (per-bin ERLE),
 * reverb tail model, frequency postprocessing, configurable PSD alphas.
 * Legacy: coherence-based spectral subtraction still available.
 */

#ifndef RES_FILTER_H
#define RES_FILTER_H

#include "fft_wrapper.h"
#include "aec_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResFilter ResFilter;

/**
 * RES configuration
 */
typedef struct {
    int block_size;             /* FFT size (512) */
    int frame_size;             /* WOLA window length (320) */
    int hop_size;               /* Processing hop (160) */
    int n_freqs;                /* block_size/2 + 1 = 257 */
    float g_min_db;             /* Minimum gain in dB */
    float max_drop_db_per_frame; /* 6.0 */
    float max_rise_db_per_frame; /* 6.0 */
    float spectral_floor_db;    /* -25.0 */
    float ne_protect_db;        /* -12.0 */
    int enable_cng;             /* 1 */

    /* v2 configurable */
    float alpha_echo_psd;       /* PSD smoothing (0.5 balanced) */
    float alpha_error_psd;      /* PSD smoothing (0.6 balanced) */
    int echo_method;            /* RES_ECHO_COHERENCE or RES_ECHO_DIRECT */
    int gain_type;              /* RES_GAIN_SPECTRAL_SUB / _WIENER / _ENR */
    int enable_reverb;          /* reverb tail model */
    float reverb_decay;         /* 0.5 */
    float reverb_gain;          /* 0.5 */
    float enr_scale;            /* ENR threshold scale (1.0) */
} ResConfig;

/**
 * Create ResConfig from AecConfig (for standalone RES in linear pipeline)
 */
static inline ResConfig res_config_from_aec(const AecConfig* aec_cfg) {
    ResConfig rc;
    rc.block_size = aec_cfg->fft_size;
    rc.frame_size = aec_cfg->frame_size;
    rc.hop_size = aec_cfg->hop_size;
    rc.n_freqs = aec_cfg->n_freqs;
    rc.g_min_db = aec_cfg->res_g_min_db;
    rc.max_drop_db_per_frame = aec_cfg->res_max_drop_db_per_frame;
    rc.max_rise_db_per_frame = aec_cfg->res_max_rise_db_per_frame;
    rc.spectral_floor_db = aec_cfg->res_spectral_floor_db;
    rc.ne_protect_db = aec_cfg->res_ne_protect_db;
    rc.enable_cng = aec_cfg->enable_cng;
    rc.alpha_echo_psd = aec_cfg->res_alpha_echo_psd;
    rc.alpha_error_psd = aec_cfg->res_alpha_error_psd;
    rc.echo_method = aec_cfg->res_echo_method;
    rc.gain_type = aec_cfg->res_gain_type;
    rc.enable_reverb = aec_cfg->res_enable_reverb;
    rc.reverb_decay = aec_cfg->res_reverb_decay;
    rc.reverb_gain = aec_cfg->res_reverb_gain;
    rc.enr_scale = aec_cfg->res_enr_scale;
    return rc;
}

/** Create RES filter (malloc version) */
ResFilter* res_create(const ResConfig* cfg);

/** Initialize RES in pre-allocated memory (static version) */
ResFilter* res_init(void* mem, size_t mem_size, const ResConfig* cfg);

/** Get memory required for res_init() */
size_t res_get_mem_size(const ResConfig* cfg);

/** Destroy RES filter (no-op if created via res_init) */
void res_destroy(ResFilter* res);

/**
 * Reset RES state
 */
void res_reset(ResFilter* res);

/**
 * Process hop_size error samples → hop_size enhanced output
 *
 * @param res          RES handle
 * @param error_hop    Error signal from adaptive filter [hop_size]
 * @param echo_spec    Echo estimate spectrum [n_freqs] (from PBFDKF)
 * @param far_spec     Far-end spectrum [n_freqs] (for coherence)
 * @param near_spec    Near-end (mic) spectrum [n_freqs] (for direct echo est, may be NULL)
 * @param far_power    Far-end mean power (activity detection)
 * @param converged    Filter convergence flag
 * @param erle_factor  [0,1] convergence factor for EER blending
 * @param dt_indicator [0,1] double-talk indicator
 * @param over_sub     Over-subtraction factor (dynamic, set by coordinator)
 * @param output_hop   Enhanced output [hop_size]
 */
void res_process(ResFilter* res,
                 const float* error_hop,
                 const Complex* echo_spec,
                 const Complex* far_spec,
                 const Complex* near_spec,
                 float far_power,
                 int converged,
                 float erle_factor,
                 float dt_indicator,
                 float over_sub,
                 float* output_hop);

/**
 * Get echo_psd and error_psd arrays (for per-bin mu_scale)
 */
const float* res_get_echo_psd(const ResFilter* res);
const float* res_get_error_psd(const ResFilter* res);

#ifdef __cplusplus
}
#endif

#endif /* RES_FILTER_H */
