/* Render-path per-bin stationarity flags, float32.
 *  - NoiseSpectrum: per-bin slow noise-floor tracker.
 *  - StationarityEstimator: history ring of render PSDs -> per-bin flags with hangover, 3-bin
 *  smoothing and a block-level stationary decision.
 * window_hops/hangover_hops come from reference block counts (13/12) via aec_blocks_to_hops().
 * Python reference: modules/state/stationarity_estimator.py. */
#ifndef STATIONARITY_ESTIMATOR_H
#define STATIONARITY_ESTIMATOR_H

#include <stdint.h>

#define STAT_MIN_NOISE_POWER_FLOAT (10.0f / (32768.0f * 32768.0f))
#define STAT_THR_RATIO             10.0f  /* acum < 10 x noise -> stationary */
#define STAT_BLOCK_FRACTION        0.75f  /* more than 75% of bands stationary -> block stationary */
#define STAT_ALPHA                 0.004f
#define STAT_ALPHA_INIT            0.04f

typedef struct {
    int    n_freqs;
    float  min_noise;
    int    avg_init;
    int    init_phase;
    float  alpha;
    float  alpha_init;
    float  tilt;          /* (alpha_init - alpha) / max(1, init_phase) */
    float *noise;
    int    block_counter;
} NoiseSpectrum;

typedef struct {
    int           n_freqs;
    int           window_hops;
    int           hangover_hops;
    NoiseSpectrum noise;
    int32_t      *hangovers;
    unsigned char *stationarity_flags;
    float        *history;
    int           history_filled;
    int           history_write;
} StationarityEstimator;

/* Caller-provided storage: noise_storage (n_freqs floats), hangovers_storage (n_freqs int32_t),
 * flags_storage (n_freqs uchar), history_storage (window_hops x n_freqs floats). */
void stationarity_estimator_init(StationarityEstimator *s,
                                 int n_freqs, int hop_samples, int sample_rate,
                                 float *noise_storage,
                                 int32_t *hangovers_storage,
                                 unsigned char *flags_storage,
                                 float *history_storage);

void stationarity_estimator_reset(StationarityEstimator *s);

/* Feed the latest render PSD (float32) to the noise tracker. */
void stationarity_estimator_update_noise_estimator(StationarityEstimator *s,
                                                   const float *render_psd);

/* Refresh the flags from the latest render PSD. average_reverb is the previous hop's average
 * reverb (NULL: zeros). */
void stationarity_estimator_update_stationarity_flags(StationarityEstimator *s,
                                                      const float *render_psd,
                                                      const float *average_reverb);

/* Writes flags[k] && hangovers[k] == 0 into out (0/1). */
void stationarity_estimator_band_stationary_mask(const StationarityEstimator *s,
                                                 unsigned char *out);

/* mean(band_stationary_mask) > STAT_BLOCK_FRACTION. */
int stationarity_estimator_is_block_stationary(const StationarityEstimator *s);

#endif
