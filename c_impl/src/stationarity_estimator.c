/* stationarity_estimator.c -- see stationarity_estimator.h. Float32 per-bin math. */
#include "stationarity_estimator.h"

#include "aec_scale.h"
#include "utility/simd_aec.h"

#include <limits.h>
#include <math.h>
#include <string.h>

/* NoiseSpectrum */

static void noise_spectrum_init(NoiseSpectrum *n, int n_freqs,
                                float min_noise_power, int avg_init_hops,
                                int initial_phase_hops, float alpha,
                                float alpha_init, float *noise_storage) {
    int k;
    n->n_freqs = n_freqs;
    n->min_noise = min_noise_power;
    n->avg_init = avg_init_hops > 1 ? avg_init_hops : 1;
    n->init_phase = initial_phase_hops;
    n->alpha = alpha;
    n->alpha_init = alpha_init;
    n->tilt = (alpha_init - alpha) / (float)(initial_phase_hops > 1 ? initial_phase_hops : 1);
    n->noise = noise_storage;
    for (k = 0; k < n_freqs; ++k) {
        n->noise[k] = min_noise_power;
    }
    n->block_counter = 0;
}

static void noise_spectrum_reset(NoiseSpectrum *n) {
    int k;
    for (k = 0; k < n->n_freqs; ++k) {
        n->noise[k] = n->min_noise;
    }
    n->block_counter = 0;
}

/* Smoothing weight for the current block. */
static float noise_spectrum_alpha_now(const NoiseSpectrum *n) {
    if (n->block_counter > (n->init_phase + n->avg_init)) {
        return n->alpha;
    }
    return n->alpha_init - n->tilt * (float)(n->block_counter - n->avg_init);
}

static void noise_spectrum_update(NoiseSpectrum *n, const float *spectrum) {
    int k;
    /* The counter caps only at INT_MAX so the tilt arithmetic is never frozen at a warmup
     * boundary. */
    if (n->block_counter < INT_MAX) n->block_counter += 1;

    if (n->block_counter <= n->avg_init) {
        /* Warmup: running mean of the first avg_init spectra. */
        float inv = 1.0f / (float)n->avg_init;
        for (k = 0; k < n->n_freqs; ++k) {
            float tmp = inv * spectrum[k];
            n->noise[k] = n->noise[k] + tmp;
        }
        return;
    }

    {
        float alpha = noise_spectrum_alpha_now(n);
        int   apply_mask10 = (n->block_counter > n->init_phase);
        /* Branchless SIMD twin of the scalar per-bin update. */
        sk_noise_spectrum_update_f32(n->noise, spectrum, alpha, apply_mask10,
                                     n->min_noise, n->n_freqs);
    }
}

/* StationarityEstimator */

int stationarity_estimator_window_hops(int hop_samples, int sample_rate) {
    return aec_blocks_to_hops(13, hop_samples, sample_rate);
}

void stationarity_estimator_init(StationarityEstimator *s,
                                 int n_freqs, int hop_samples, int sample_rate,
                                 float *noise_storage,
                                 int32_t *hangovers_storage,
                                 unsigned char *flags_storage,
                                 float *history_storage) {
    int i;
    s->n_freqs = n_freqs;
    s->window_hops = stationarity_estimator_window_hops(hop_samples, sample_rate);
    s->hangover_hops = aec_blocks_to_hops(12, hop_samples, sample_rate);
    /* Noise-tracker constants are rescaled to the live grid like window_hops/hangover_hops. */
    noise_spectrum_init(&s->noise, n_freqs, STAT_MIN_NOISE_POWER_FLOAT,
                        aec_blocks_to_hops(20, hop_samples, sample_rate),
                        aec_blocks_to_hops(500, hop_samples, sample_rate),
                        aec_per_block_ema_alpha_to_per_hop(STAT_ALPHA, hop_samples, sample_rate),
                        aec_per_block_ema_alpha_to_per_hop(STAT_ALPHA_INIT, hop_samples, sample_rate),
                        noise_storage);
    s->hangovers = hangovers_storage;
    s->stationarity_flags = flags_storage;
    s->history = history_storage;
    for (i = 0; i < n_freqs; ++i) {
        s->hangovers[i] = 0;
        s->stationarity_flags[i] = 0;
    }
    for (i = 0; i < s->window_hops * n_freqs; ++i) {
        s->history[i] = 0.0f;
    }
    s->history_filled = 0;
    s->history_write = 0;
}

void stationarity_estimator_reset(StationarityEstimator *s) {
    int i;
    noise_spectrum_reset(&s->noise);
    for (i = 0; i < s->n_freqs; ++i) {
        s->hangovers[i] = 0;
        s->stationarity_flags[i] = 0;
    }
    for (i = 0; i < s->window_hops * s->n_freqs; ++i) {
        s->history[i] = 0.0f;
    }
    s->history_filled = 0;
    s->history_write = 0;
}

void stationarity_estimator_update_noise_estimator(StationarityEstimator *s,
                                                   const float *render_psd) {
    noise_spectrum_update(&s->noise, render_psd);
}

static void update_hangover(StationarityEstimator *s) {
    sk_stationarity_hangover_i32(s->stationarity_flags, s->hangovers,
                                  (int32_t)s->hangover_hops, s->n_freqs);
}

static void smooth_per_freq(StationarityEstimator *s) {
    sk_stationarity_smooth_u8(s->stationarity_flags, s->n_freqs);
}

void stationarity_estimator_update_stationarity_flags(StationarityEstimator *s,
                                                      const float *render_psd,
                                                      const float *average_reverb) {
    int k;
    float *slot = &s->history[s->history_write * s->n_freqs];

    /* Store the latest PSD in the history ring. */
    for (k = 0; k < s->n_freqs; ++k) slot[k] = render_psd[k];
    s->history_write = (s->history_write + 1) % s->window_hops;
    if (s->history_filled < s->window_hops) s->history_filled += 1;

    /* Stationary if sum(filled history rows) + average_reverb (zeros if NULL) < STAT_THR_RATIO * window_hops * max(noise, 1e-30).
     * The rows are added in storage order. */
    sk_stationarity_window_flags_u8(s->stationarity_flags, s->history, (size_t)s->n_freqs, s->history_filled,
                                    average_reverb, s->noise.noise, s->window_hops, STAT_THR_RATIO, s->n_freqs);

    update_hangover(s);
    smooth_per_freq(s);
}

void stationarity_estimator_band_stationary_mask(const StationarityEstimator *s,
                                                 unsigned char *out) {
    (void)sk_stationary_mask_u8(s->stationarity_flags, s->hangovers, out, s->n_freqs);
}

int stationarity_estimator_is_block_stationary(const StationarityEstimator *s) {
    int count = sk_stationary_mask_u8(s->stationarity_flags, s->hangovers, NULL, s->n_freqs);
    /* Fraction of stationary bands. */
    return ((float)count / (float)s->n_freqs) > STAT_BLOCK_FRACTION;
}
