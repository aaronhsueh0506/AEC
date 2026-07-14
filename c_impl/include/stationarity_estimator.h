/* stationarity_estimator.h — C port of
 * python/modules/state/stationarity_estimator.py (which mirrors AEC3
 * stationarity_estimator.cc + the echo_audibility residual-scaling bits).
 *
 * Render-path per-bin stationarity flag:
 *   - _NoiseSpectrum : per-bin slow noise-floor EMA tracker over a window.
 *   - StationarityEstimator : history ring of render PSDs → per-bin
 *     stationarity flags + hangover + 3-bin smoothing + is_block_stationary.
 *
 * FLOAT32-BY-DESIGN (numpy 1.26 → C):
 *   The real pipeline feeds a float32 render_psd (= |far_spec|² .astype(f32))
 *   and average_reverb is always None (→ float32 zeros). Every scalar in this
 *   module — formerly kept as fp64 to mirror numpy's weak-promotion rule
 *   (Python float scalars cast to float32 at the point they combine with the
 *   float32 arrays) — is now float32 throughout, converted for uniformity as
 *   part of the f32 campaign (drift accepted). All arrays here are therefore
 *   `float`, the hangover counter is int32, and the flags are bytes (bool).
 *   Built with -ffp-contract=off so the compiler does not fuse the three-step
 *   float32 roundings.
 *
 *   window_hops / hangover_hops are derived from AEC3 block counts (13 / 12) via
 *   aec3_blocks_to_hops(); at hop=160/sr=16000 both = 5.
 */
#ifndef STATIONARITY_ESTIMATOR_H
#define STATIONARITY_ESTIMATOR_H

#include <stdint.h>

/* ── Constants mirrored from aec3_scale.py (16 kHz / hop=160 reference) ──── */
/* STATIONARITY_MIN_NOISE_POWER_FLOAT = psd_int16_to_float(10.0) = 10 / 32768² */
#define STAT_MIN_NOISE_POWER_FLOAT (10.0f / (32768.0f * 32768.0f)) /* 9.31e-9 */
#define STAT_THR_RATIO             10.0f  /* acum < 10×noise → stationary */
#define STAT_BLOCK_FRACTION        0.75f  /* >75% bands stationary → block  */
#define STAT_ALPHA                 0.004f /* long-term smoothing            */
#define STAT_ALPHA_INIT            0.04f  /* warmup smoothing               */
/* blocks_to_hops(20,160,16000)=8 ; blocks_to_hops(500,160,16000)=200 */
#define STAT_AVG_INIT_HOPS_DEFAULT       8
#define STAT_INITIAL_PHASE_HOPS_DEFAULT  200

/* ── _NoiseSpectrum ───────────────────────────────────────────────────────
 * Per-bin slow noise-floor tracker (stationarity_estimator.cc:159-242). All
 * arithmetic float32, including the scalars (1/avg_init, alpha, ...), which
 * are float32-by-design (formerly stored as double and cast to float at each
 * multiply to mirror numpy weak promotion). */
typedef struct {
    int    n_freqs;
    float  min_noise;     /* _min  (Python float)            */
    int    avg_init;      /* _avg_init = max(1, avg_init_hops) */
    int    init_phase;    /* _init_phase                     */
    float  alpha;         /* _alpha                          */
    float  alpha_init;    /* _alpha_init                     */
    float  tilt;          /* (_alpha_init - _alpha)/max(1,_init_phase) */
    float *noise;         /* owned by caller; length n_freqs */
    int    block_counter; /* block_counter                   */
} NoiseSpectrum;

/* ── StationarityEstimator ───────────────────────────────────────────────── */
typedef struct {
    int           n_freqs;
    int           window_hops;    /* blocks_to_hops(13,...)  */
    int           hangover_hops;  /* blocks_to_hops(12,...)  */
    NoiseSpectrum noise;
    int32_t      *hangovers;          /* length n_freqs           */
    unsigned char *stationarity_flags;/* length n_freqs (bool)    */
    float        *history;            /* window_hops × n_freqs    */
    int           history_filled;
    int           history_write;
} StationarityEstimator;

/* Storage-block layout the caller must provide (sizes in elements):
 *   noise_storage      : n_freqs                  (float)
 *   hangovers_storage  : n_freqs                  (int32_t)
 *   flags_storage      : n_freqs                  (unsigned char)
 *   history_storage    : window_hops × n_freqs    (float)
 * window_hops/hangover_hops are computed inside init() from hop_samples/sr. */
void stationarity_estimator_init(StationarityEstimator *s,
                                 int n_freqs, int hop_samples, int sample_rate,
                                 float *noise_storage,
                                 int32_t *hangovers_storage,
                                 unsigned char *flags_storage,
                                 float *history_storage);

void stationarity_estimator_reset(StationarityEstimator *s);

/* NoiseSpectrum::update — feed latest render PSD (float32). */
void stationarity_estimator_update_noise_estimator(StationarityEstimator *s,
                                                   const float *render_psd);

/* update_stationarity_flags — refresh flags using latest render PSD.
 * average_reverb: previous-hop avg_reverb from aec3_post (Aec3Post.avg_reverb.reverb);
 * NULL treated as zeros. */
void stationarity_estimator_update_stationarity_flags(StationarityEstimator *s,
                                                      const float *render_psd,
                                                      const float *average_reverb);

/* band_stationary_mask: writes flags[k] && hangovers[k]==0 into out (bool). */
void stationarity_estimator_band_stationary_mask(const StationarityEstimator *s,
                                                 unsigned char *out);

/* is_block_stationary: mean(band_stationary_mask) > STAT_BLOCK_FRACTION. */
int stationarity_estimator_is_block_stationary(const StationarityEstimator *s);

/* is_band_stationary(k). */
int stationarity_estimator_is_band_stationary(const StationarityEstimator *s, int k);

#endif /* STATIONARITY_ESTIMATOR_H */
