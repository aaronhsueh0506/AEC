/* aec_tuning.h -- the library's internal tuning (library-internal). AecConfig is the caller's view; everything else the instance
 * reads is an AecTuning, built by aec_resolve() from (sample_rate, fft_size, n_mics, level). Tests replace it only through the
 * aec_testing_* hooks (test/aec_test_tuning.h). */
#ifndef AEC_AEC_TUNING_H
#define AEC_AEC_TUNING_H

#include "aec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Names match the Python AecConfig where a field exists there. Fields in hops are authored on the 10 ms / 16 kHz grid; the
 * instance copy (Aec::tun) holds them retimed to the resolved grid. */
typedef struct AecTuning {
    /* geometry (a function of the rate) */
    int    filter_length;          /* samples; 52 ms (64 ms at >= 44.1 kHz) */
    int    n_partitions;           /* 0 = ceil(filter_length / hop) */

    /* main filter */
    float  mu;                     /* step size; 0.3 */
    float  delta;                  /* division guard; 1e-8 */

    /* structure (the pool layout depends on these) */
    int    enable_highpass;        /* mic-path high-pass; 1 */
    int    enable_saturation;      /* saturation limiter; 1 */
    int    enable_shadow;          /* shadow filter; 1 */
    int    live_observer_enabled;  /* 0 */

    /* front end */
    float  highpass_cutoff_hz;     /* Hz; 80 */
    float  saturation_threshold;   /* full scale 1.0; 0.95 */
    int    saturation_softclip_ref;/* soft-clip the reference; 1 */
    int    cng_n2_windowed_psd;    /* CNG noise-tracker windowed-PSD coordinate; 0 */

    /* shadow filter, echo-path change */
    float  shadow_err_alpha;       /* 0.80 @ 10 ms hop */
    float  shadow_mu_min;          /* 0.5 */
    float  shadow_mu_nlms;         /* 0.5 */
    int    warmup_frames;          /* hops; 100 = 1000 ms @ 10 ms hop */
    int    epc_hangover;           /* hops; 20 = 200 ms @ 10 ms hop */
    float  epc_total_rise;         /* 1.5 */
    float  epc_delta_threshold;    /* 0.3 */

    /* delay estimation */
    float  delay_est_init_s;       /* s; hold time before duty-cycling; 0.3 */
    float  delay_est_period_s;     /* s; 0 = analyse every hop */
    int    delay_acquire_protect_converged;   /* 1 */
    int    delay_acquire_warm_transfer;       /* 1 */
    float  delay_acquire_inst_erle_db;        /* dB; 4.0 */

    /* double-talk recovery and near-end-recent gate */
    int    dt_aware_recovery_soft;            /* 1 */
    int    dt_aware_res_floor_enabled;        /* 1 */
    float  ne_recent_threshold;               /* 0.3 */
    int    ne_recent_hold;                    /* hops; 150 = 1500 ms @ 10 ms hop */
    int    ne_recent_sustain;                 /* hops; 3 */

    /* filter-scale tracker */
    int    filter_misadjustment_stable_frames;   /* hops; 30 */
    int    filter_misadjustment_hangover_frames; /* hops; 100 */
    float  filter_misadjustment_scale_min;       /* 0.5 */
    float  filter_misadjustment_scale_max;       /* 2.0 */

    /* the level (AecConfig.level) as the suppressor reads it */
    float  min_gain_floor_far_active_db;      /* dB; -20 / -28 / -38 */
    float  min_gain_floor_dt_db;              /* dB; -20, -30 for aggressive */
    int    min_gain_split_floor_enabled;      /* 1; 0 for no_floor */

    /* live observer: a second PBFDAF reading the far-end ring earlier by live_observer_shift_ms */
    float  live_observer_shift_ms;          /* ms; 24.0 */
    float  live_observer_max_hz;            /* Hz; 1000.0 */
    float  live_observer_mu;                /* 0.3 */
    float  live_observer_corrob_ms;         /* ms; 100.0 */
    float  live_observer_corrob_margin_db;  /* dB; 0.0 */
} AecTuning;

/* A validated configuration with the tuning it resolves to; the single input of the size query, the carve and the init. */
typedef struct AecResolved {
    AecConfig cfg;
    AecTuning tun;
} AecResolved;

/* The tuning for cfg: a pure function of cfg.sample_rate, cfg.fft_size, cfg.n_mics and cfg.level. An out-of-enum level reads as
 * balanced. */
void aec_tuning_defaults(AecTuning* tun, const AecConfig* cfg);

/* Writes `level` to cfg->level and the level's three floors to tun; nothing is written for an out-of-enum level. Returns 0, or
 * -1. The one place aec_set_level() of a single-channel and of an array instance store a level. */
int aec_apply_level(AecConfig* cfg, AecTuning* tun, AecLevel level);

/* Range checks of every tuning field (finite, generous bounds around each default) for cfg.sample_rate. 1 = acceptable. */
int aec_tuning_valid(const AecTuning* tun, const AecConfig* cfg);

/* Validates `in` and the tuning and writes both to *out. tun == NULL uses aec_tuning_defaults(in). Returns 1, or 0 when in is
 * NULL or invalid (*out untouched). */
int aec_resolve_tuned(const AecConfig* in, const AecTuning* tun, AecResolved* out);
int aec_resolve(const AecConfig* in, AecResolved* out);

/* The configuration checks behind aec_resolve_tuned(): the caller's fields, the tuning, the delay-mode and the array rules.
 * 1 = acceptable. Defined in aec.c (the array rules live with the array instance). */
int aec_resolved_valid(const AecResolved* r);

#ifdef __cplusplus
}
#endif

#endif /* AEC_AEC_TUNING_H */
