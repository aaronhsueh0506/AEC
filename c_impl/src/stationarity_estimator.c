/* stationarity_estimator.c — C port of
 * python/modules/state/stationarity_estimator.py.
 *
 * All per-bin math is float32 to match numpy<2 weak promotion: every Python
 * float scalar is cast to float32 before combining with the float32 arrays.
 * Each (cast scalar) op is a separate float32 rounding; -ffp-contract=off keeps
 * the compiler from fusing them. fp64 only appears for the scalar pre-compute of
 * alpha / 1.0/avg_init (exactly as Python computes those scalars in double, then
 * the numpy array op casts them down to float32).
 */
#include "stationarity_estimator.h"

#include "aec3_scale.h"   /* aec3_blocks_to_hops */

#include <math.h>
#include <string.h>

/* ── _NoiseSpectrum ───────────────────────────────────────────────────────── */

static void noise_spectrum_init(NoiseSpectrum *n, int n_freqs,
                                double min_noise_power, int avg_init_hops,
                                int initial_phase_hops, double alpha,
                                double alpha_init, float *noise_storage) {
    int k;
    n->n_freqs = n_freqs;
    n->min_noise = min_noise_power;
    n->avg_init = avg_init_hops > 1 ? avg_init_hops : 1; /* max(1, avg_init_hops) */
    n->init_phase = initial_phase_hops;
    n->alpha = alpha;
    n->alpha_init = alpha_init;
    /* _tilt = (alpha_init - alpha) / max(1, init_phase) */
    n->tilt = (alpha_init - alpha) / (double)(initial_phase_hops > 1 ? initial_phase_hops : 1);
    n->noise = noise_storage;
    for (k = 0; k < n_freqs; ++k) {
        n->noise[k] = (float)min_noise_power; /* np.full(.., _min, f32) */
    }
    n->block_counter = 0;
}

static void noise_spectrum_reset(NoiseSpectrum *n) {
    int k;
    for (k = 0; k < n->n_freqs; ++k) {
        n->noise[k] = (float)n->min_noise;
    }
    n->block_counter = 0;
}

/* _alpha_now (Python returns a float64). */
static double noise_spectrum_alpha_now(const NoiseSpectrum *n) {
    if (n->block_counter > (n->init_phase + n->avg_init)) {
        return n->alpha;
    }
    return n->alpha_init - n->tilt * (double)(n->block_counter - n->avg_init);
}

static void noise_spectrum_update(NoiseSpectrum *n, const float *spectrum) {
    int k;
    n->block_counter += 1;

    if (n->block_counter <= n->avg_init) {
        /* noise += (1.0/avg_init) * spectrum.astype(f32)
         * scalar (1.0/avg_init) computed in f64, cast to f32 at the multiply. */
        float inv = (float)(1.0 / (double)n->avg_init);
        for (k = 0; k < n->n_freqs; ++k) {
            float tmp = inv * spectrum[k];          /* f32 mul */
            n->noise[k] = n->noise[k] + tmp;        /* f32 add */
        }
        return;
    }

    {
        double alpha_d = noise_spectrum_alpha_now(n);
        float  alpha_f = (float)alpha_d;             /* scalar→f32 at array op */
        int    apply_mask10 = (n->block_counter > n->init_phase);
        for (k = 0; k < n->n_freqs; ++k) {
            float pb = spectrum[k];                  /* spectrum.astype(f32) */
            float pn = n->noise[k];
            if (pb > pn) {                           /* rising */
                /* alpha_inc = alpha * (pn / max(pb, 1e-30)) */
                float denom = pb > (float)1e-30 ? pb : (float)1e-30; /* np.maximum */
                float ratio = pn / denom;            /* f32 */
                float alpha_inc = alpha_f * ratio;   /* f32 */
                if (apply_mask10) {
                    /* mask10 = (10.0 * pn) < pb */
                    float ten_pn = (float)10.0 * pn; /* scalar 10.0 → f32 */
                    if (ten_pn < pb) {
                        alpha_inc = alpha_inc * (float)0.1; /* scalar 0.1 → f32 */
                    }
                }
                /* noise = pn + alpha_inc*(pb-pn) */
                n->noise[k] = pn + alpha_inc * (pb - pn);
            } else {                                  /* falling (~rising) */
                /* noise = max(pn + alpha*(pb-pn), min) */
                float upd = pn + alpha_f * (pb - pn);
                float floor_f = (float)n->min_noise;  /* np.maximum(.., _min) */
                n->noise[k] = upd > floor_f ? upd : floor_f;
            }
        }
    }
}

/* ── StationarityEstimator ────────────────────────────────────────────────── */

void stationarity_estimator_init(StationarityEstimator *s,
                                 int n_freqs, int hop_samples, int sample_rate,
                                 float *noise_storage,
                                 int32_t *hangovers_storage,
                                 unsigned char *flags_storage,
                                 float *history_storage) {
    int i;
    s->n_freqs = n_freqs;
    s->window_hops = aec3_blocks_to_hops(13, hop_samples, sample_rate);
    s->hangover_hops = aec3_blocks_to_hops(12, hop_samples, sample_rate);
    noise_spectrum_init(&s->noise, n_freqs, STAT_MIN_NOISE_POWER_FLOAT,
                        STAT_AVG_INIT_HOPS_DEFAULT, STAT_INITIAL_PHASE_HOPS_DEFAULT,
                        STAT_ALPHA, STAT_ALPHA_INIT, noise_storage);
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

/* _update_hangover */
static void update_hangover(StationarityEstimator *s) {
    int k, all_stationary = 1;
    for (k = 0; k < s->n_freqs; ++k) {
        if (!s->stationarity_flags[k]) { all_stationary = 0; break; }
    }
    /* hangovers[not_stationary] = hangover_hops */
    for (k = 0; k < s->n_freqs; ++k) {
        if (!s->stationarity_flags[k]) {
            s->hangovers[k] = (int32_t)s->hangover_hops;
        }
    }
    if (all_stationary) {
        /* np.maximum(hangovers - 1, 0, out=hangovers) */
        for (k = 0; k < s->n_freqs; ++k) {
            int32_t v = s->hangovers[k] - 1;
            s->hangovers[k] = v > 0 ? v : 0;
        }
    }
}

/* _smooth_per_freq */
static void smooth_per_freq(StationarityEstimator *s) {
    int k, n = s->n_freqs;
    unsigned char *f = s->stationarity_flags;
    /* smoothed[1:-1] = f[:-2] & f[1:-1] & f[2:]; ends mirror neighbour.
     * Need a temp because we read f while writing smoothed. */
    if (n >= 3) {
        /* compute interior into a small local buffer pattern without malloc:
         * walk left→right keeping the previous original flag. */
        unsigned char prev = f[0];            /* f[k-1] original */
        unsigned char cur  = f[1];            /* f[k]   original */
        unsigned char first_interior = 0;
        unsigned char last_interior = 0;
        for (k = 1; k <= n - 2; ++k) {
            unsigned char next = f[k + 1];    /* f[k+1] original */
            unsigned char sm = (unsigned char)(prev && cur && next);
            if (k == 1) first_interior = sm;
            if (k == n - 2) last_interior = sm;
            f[k] = sm;                        /* safe: prev/cur/next captured */
            prev = cur;                       /* original f[k] */
            cur = next;                       /* original f[k+1] */
        }
        f[0] = first_interior;                /* smoothed[0] = smoothed[1] */
        f[n - 1] = last_interior;             /* smoothed[-1] = smoothed[-2] */
    } else {
        /* numpy: smoothed = zeros_like(f); n_freqs<3 leaves all False */
        for (k = 0; k < n; ++k) f[k] = 0;
    }
}

void stationarity_estimator_update_stationarity_flags(StationarityEstimator *s,
                                                      const float *render_psd,
                                                      const float *average_reverb) {
    int k, row;
    float *slot = &s->history[s->history_write * s->n_freqs];

    /* self._history[write] = render_psd.astype(f32) */
    for (k = 0; k < s->n_freqs; ++k) slot[k] = render_psd[k];
    s->history_write = (s->history_write + 1) % s->window_hops;
    if (s->history_filled < s->window_hops) s->history_filled += 1;

    /* acum_power = history[:filled].sum(axis=0) + rev   (sequential f32 sum
     * over storage rows 0..filled-1, matching numpy sum(axis=0) for N<=window). */
    for (k = 0; k < s->n_freqs; ++k) {
        float acc = s->history[0 * s->n_freqs + k]; /* row 0 */
        for (row = 1; row < s->history_filled; ++row) {
            acc = acc + s->history[row * s->n_freqs + k]; /* f32 add */
        }
        /* + rev (zeros if average_reverb is NULL) */
        if (average_reverb != NULL) {
            acc = acc + average_reverb[k];          /* f32 add */
        }
        /* noise = window_hops * max(self.noise.noise, 1e-30)
         * THR_RATIO * noise; stationary if acum < that. */
        {
            float nz = s->noise.noise[k];
            float ndenom = nz > (float)1e-30 ? nz : (float)1e-30; /* np.maximum */
            float noise_scaled = (float)s->window_hops * ndenom;  /* int*f32 → f32 */
            float thr = (float)STAT_THR_RATIO * noise_scaled;     /* 10.0(f64)→f32 */
            s->stationarity_flags[k] = (acc < thr) ? 1 : 0;
        }
    }

    update_hangover(s);
    smooth_per_freq(s);
}

void stationarity_estimator_band_stationary_mask(const StationarityEstimator *s,
                                                 unsigned char *out) {
    int k;
    for (k = 0; k < s->n_freqs; ++k) {
        out[k] = (unsigned char)(s->stationarity_flags[k] && s->hangovers[k] == 0);
    }
}

int stationarity_estimator_is_band_stationary(const StationarityEstimator *s, int k) {
    return s->stationarity_flags[k] && s->hangovers[k] == 0;
}

int stationarity_estimator_is_block_stationary(const StationarityEstimator *s) {
    int k, count = 0;
    for (k = 0; k < s->n_freqs; ++k) {
        if (s->stationarity_flags[k] && s->hangovers[k] == 0) count++;
    }
    /* float(np.mean(mask)) > BLOCK_FRACTION; mean = count / n_freqs in f64 */
    return ((double)count / (double)s->n_freqs) > STAT_BLOCK_FRACTION;
}
