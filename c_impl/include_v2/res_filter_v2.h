/* res_filter_v2.h — port of aec.py ResFilter (1096-2077).
 *
 * Only the "direct" echo_method + "enr" gain_type path is ported (used by
 * all 4 presets). Legacy coherence / wiener / spectral_sub paths are not
 * ported — Plan §取捨.
 *
 * Float32 storage matches Python's np.float32 arrays. Scalar EMAs (Python
 * `float`) use double in C.
 */
#ifndef AEC_V2_RES_FILTER_H
#define AEC_V2_RES_FILTER_H

#include <stddef.h>
#include "fft_wrapper.h"
#include "erle_v2.h"
#include "residual_echo_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResFilterV2 {
    int    block_size;
    int    sample_rate;
    int    frame_size;
    int    hop_size;
    int    n_freqs;

    float  g_min;
    float  over_sub;
    float  alpha;
    float  ne_protect_db;
    float  alpha_echo_psd;
    float  alpha_error_psd;
    float  enr_scale;
    float  startup_dt_min_ne_scale;
    float  startup_dt_gain_floor;
    float  startup_dt_noise_floor_scale;

    /* Precomputed per-bin constants */
    float* stat_dt_mask;
    float* enr_blend;
    int    hf_cap_bin;
    int    harm_lf_start, harm_lf_end;
    int    harm_hf_start, harm_hf_end;

    /* Mode flags */
    int    enable_reverb;
    int    enable_cng;
    int    enable_spectral_floor;
    float  spectral_floor_ratio;
    float  reverb_decay;
    float  reverb_gain;

    /* Window (sqrt-Hann over frame_size) */
    float* window;

    float  alpha_coh;
    float  alpha_noise;
    float  max_drop_ratio;
    float  max_rise_ratio;
    float  alpha_envelope;
    float  render_min_ne_factor;
    float  render_dt_gain_ceil;

    /* Runtime arrays */
    float* gain_smooth;
    float* echo_psd;
    float* error_psd;
    Complex* S_fe;
    float* S_ff;
    float* S_ee;
    float* near_psd;
    float* near_psd_buf;     /* [4 * n_freqs] */
    int    near_psd_idx;
    float* reverb_psd;
    float* noise_psd;
    int    noise_initialized;
    float* coh2_smooth;
    float* error_envelope;
    float* input_buf;
    float* ola_buf;
    float* smooth_cn_gain;
    double far_activity;
    int    nonlinear_frames;
    double last_effective_dt;

    /* Sub-modules owned by ResFilter */
    FilterErleEstV2     filter_erle;
    FullbandErleEstV2   fb_erle;
    ResidualEchoV2      residual_est;

    /* Diagnostics */
    double diag_gain_mean;
    double diag_gain_min;
    double diag_effective_g_min;
    double diag_far_activity;
    double diag_echo_psd_mean;
    double diag_error_psd_mean;

    /* FFT scratch */
    FftHandle* fft;
    float*     time_scratch;
    Complex*   spec_synth;
    Complex*   enhanced_spec;

    /* RNG state for CNG (xorshift32). 0 means uninitialised → seeded at create. */
    unsigned int cng_rng_state;
} ResFilterV2;

/* Init parameters mirror Python kwargs in __init__. Use `enable_cng=0` for
 * BALANCED CLI default (User Fix 2). */
typedef struct ResFilterV2Config {
    int   block_size;
    int   n_freqs;
    int   frame_size;       /* 0 → block_size */
    int   hop_size;         /* 0 → frame_size/2 */
    int   sample_rate;
    float g_min_db;             /* default -20 */
    float over_sub;             /* 1.5 */
    float alpha;                /* 0.8 */
    int   enable_cng;           /* default 0 (User Fix 2) */
    float max_drop_db_per_frame;/* 6.0 */
    float max_rise_db_per_frame;/* 3.0 */
    int   enable_spectral_floor;/* 1 */
    float spectral_floor_db;    /* -25 */
    float ne_protect_db;        /* -10 */
    int   enable_reverb;        /* 1 (BALANCED preset) */
    float reverb_decay;         /* 0.85 (User Fix 1) */
    float reverb_gain;          /* 1.6 (User Fix 1) */
    float alpha_echo_psd;       /* 0.7 */
    float alpha_error_psd;      /* 0.8 */
    float enr_scale;            /* 1.0 */
    float startup_dt_min_ne_scale;
    float startup_dt_gain_floor;
    float startup_dt_noise_floor_scale;
    float render_min_ne_factor; /* 0.5 */
    float render_dt_gain_ceil;  /* 0.6 */
} ResFilterV2Config;

void res_filter_v2_init(ResFilterV2* r, const ResFilterV2Config* cfg);
void res_filter_v2_free(ResFilterV2* r);
void res_filter_v2_reset(ResFilterV2* r);

/* Port of ResFilter.process. Outputs hop_size samples. */
typedef struct ResProcessInputs {
    const float*   error_hop;            /* [hop_size] time-domain residual */
    const Complex* echo_spec;            /* [n_freqs] from filter */
    const Complex* far_spec;             /* [n_freqs] (NULL allowed) */
    const Complex* near_spec;            /* [n_freqs] (NULL ⇒ direct mode disabled) */
    const Complex* error_spec_from_filter; /* [n_freqs] (NULL ⇒ use OLA spec_synth) */
    double far_power;
    double erle_factor;
    double dt_indicator;
    double divergence;
    int    is_stationary_dt;
    double saturation_level;
    int    epc_active;
    double shadow_dt;
    double erl_estimate;
    int    filter_converged;
    int    filter_once_converged;
} ResProcessInputs;

void res_filter_v2_process(ResFilterV2* r, const ResProcessInputs* in,
                           float* output_hop);

#ifdef __cplusplus
}
#endif

#endif /* AEC_V2_RES_FILTER_H */
