/* stationarity_estimator.c — C port of
 * python/modules/state/stationarity_estimator.py.
 *
 * All per-bin math is float32-by-design (converted for uniformity as part of
 * the f32 campaign; formerly matched numpy<2 weak promotion by keeping Python
 * float scalars in fp64 and casting to float32 at the point they combined with
 * the float32 arrays). Each op is a separate float32 rounding; -ffp-contract=off
 * keeps the compiler from fusing them. The scalar pre-compute of alpha /
 * 1.0/avg_init now runs directly in float32.
 */
#include "stationarity_estimator.h"

#include "aec3_scale.h"        /* aec3_blocks_to_hops */
#include "aec_simd_kernels.h"  /* sk_noise_spectrum_update_f32 (kernel 23) */

#include <limits.h>   /* INT_MAX */
#include <math.h>
#include <string.h>

/* Bins per accumulator block in update_stationarity_flags' row-major walk.
 * Purely a working-set knob: every value produces identical output, so it is
 * chosen for stack cost and for coverage rather than for the last few
 * percent of speed. 128 keeps the frame at ~1 KiB (against the 4 KiB the
 * suppressor's own render scratch already puts on this hop's deepest path)
 * and, being under the smallest shipped bin count, guarantees the blocked
 * walk actually splits on every grid -- a whole-spectrum accumulator would
 * leave the base-offset arithmetic below unexercised at 129 bins. It costs
 * ~6% of the block's own runtime against a 256-bin accumulator at 129 bins,
 * against a ~4x saving over the column-major walk it replaces. */
#define STAT_ACC_BLOCK 128

/* ── _NoiseSpectrum ───────────────────────────────────────────────────────── */

static void noise_spectrum_init(NoiseSpectrum *n, int n_freqs,
                                float min_noise_power, int avg_init_hops,
                                int initial_phase_hops, float alpha,
                                float alpha_init, float *noise_storage) {
    int k;
    n->n_freqs = n_freqs;
    n->min_noise = min_noise_power;
    n->avg_init = avg_init_hops > 1 ? avg_init_hops : 1; /* max(1, avg_init_hops) */
    n->init_phase = initial_phase_hops;
    n->alpha = alpha;
    n->alpha_init = alpha_init;
    /* _tilt = (alpha_init - alpha) / max(1, init_phase) */
    n->tilt = (alpha_init - alpha) / (float)(initial_phase_hops > 1 ? initial_phase_hops : 1);
    n->noise = noise_storage;
    for (k = 0; k < n_freqs; ++k) {
        n->noise[k] = min_noise_power; /* np.full(.., _min, f32) */
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

/* _alpha_now (float32-by-design; formerly returned a widened float64). */
static float noise_spectrum_alpha_now(const NoiseSpectrum *n) {
    if (n->block_counter > (n->init_phase + n->avg_init)) {
        return n->alpha;
    }
    return n->alpha_init - n->tilt * (float)(n->block_counter - n->avg_init);
}

static void noise_spectrum_update(NoiseSpectrum *n, const float *spectrum) {
    int k;
    /* UBSan-confirmed signed-overflow fix, floored at INT_MAX: unlike a
     * pure boolean-gate counter, block_counter's raw value also feeds
     * noise_spectrum_alpha_now()'s tilt arithmetic
     * (`block_counter - avg_init`) for as long as it's <=
     * (init_phase + avg_init) -- capping at that threshold (the value the
     * `>` gates below would otherwise use) would leave block_counter
     * permanently AT the threshold instead of past it, which would keep
     * re-entering the tilt-arithmetic branch forever with a frozen input
     * instead of settling on the constant `alpha` AEC3 intends once
     * warmup completes. Not read bit-exact by any parity harness
     * (test/historical/parity_stationarity_estimator.c never reads block_counter),
     * so there's no golden-drift constraint either way -- INT_MAX is
     * simply the boundary-only fix that's trivially safe to prove
     * correct: it's a no-op for every practically-reachable block count
     * while eliminating the UB at the true overflow boundary. */
    if (n->block_counter < INT_MAX) n->block_counter += 1;

    if (n->block_counter <= n->avg_init) {
        /* noise += (1.0/avg_init) * spectrum.astype(f32); scalar (1.0/avg_init)
         * computed and applied directly in float32 (float32-by-design). */
        float inv = 1.0f / (float)n->avg_init;
        for (k = 0; k < n->n_freqs; ++k) {
            float tmp = inv * spectrum[k];          /* f32 mul */
            n->noise[k] = n->noise[k] + tmp;        /* f32 add */
        }
        return;
    }

    {
        float alpha = noise_spectrum_alpha_now(n); /* float32-by-design */
        int   apply_mask10 = (n->block_counter > n->init_phase);
        /* sk_noise_spectrum_update_f32: branchless NEON (compute-both +
         * vbslq_f32 select) twin of the scalar per-bin update above --
         * bit-exact by construction (aec_simd_kernels.h kernel 23). */
        sk_noise_spectrum_update_f32(n->noise, spectrum, alpha, apply_mask10,
                                     n->min_noise, n->n_freqs);
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
    /* Was STAT_AVG_INIT_HOPS_DEFAULT/STAT_INITIAL_PHASE_HOPS_DEFAULT (frozen
     * #defines, blocks_to_hops(20/500,160,16000)=8/200) and STAT_ALPHA/
     * STAT_ALPHA_INIT (frozen AEC3 per-4ms-block literals, 0.004f/0.04f) --
     * all four passed straight through unlike window_hops/hangover_hops
     * above. Rescaled live the same way for consistency. */
    noise_spectrum_init(&s->noise, n_freqs, STAT_MIN_NOISE_POWER_FLOAT,
                        aec3_blocks_to_hops(20, hop_samples, sample_rate),
                        aec3_blocks_to_hops(500, hop_samples, sample_rate),
                        aec3_per_block_ema_alpha_to_per_hop(STAT_ALPHA, hop_samples, sample_rate),
                        aec3_per_block_ema_alpha_to_per_hop(STAT_ALPHA_INIT, hop_samples, sample_rate),
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

/* _update_hangover */
static void update_hangover(StationarityEstimator *s) {
    int k = 0, all_stationary = 1;
    const int n = s->n_freqs;
    const unsigned char *flags = s->stationarity_flags;
    int32_t *hang = s->hangovers;
    const int32_t hold = (int32_t)s->hangover_hops;
#if SK_HAVE_NEON
    /* The flags are 0/1 bytes, so the whole array is stationary iff its
     * lane-wise minimum is 1. */
    {
        uint8x16_t mn = vdupq_n_u8(1);
        for (; k + 16 <= n; k += 16) mn = vminq_u8(mn, vld1q_u8(flags + k));
        if (vminvq_u8(mn) == 0) all_stationary = 0;
    }
    for (; k < n; ++k)
        if (!flags[k]) { all_stationary = 0; break; }
    /* hangovers[not_stationary] = hangover_hops, then, when every bin was
     * stationary, hangovers = max(hangovers - 1, 0). Both are per-bin. */
    k = 0;
    {
        const int32x4_t vhold = vdupq_n_s32(hold);
        const int32x4_t vzero = vdupq_n_s32(0);
        const int32x4_t vone = vdupq_n_s32(1);
        for (; k + 8 <= n; k += 8) {
            uint8x8_t f8 = vld1_u8(flags + k);
            uint16x8_t f16 = vmovl_u8(f8);
            uint32x4_t st0 = vcgtq_u32(vmovl_u16(vget_low_u16(f16)), vdupq_n_u32(0));
            uint32x4_t st1 = vcgtq_u32(vmovl_u16(vget_high_u16(f16)), vdupq_n_u32(0));
            int32x4_t h0 = vbslq_s32(st0, vld1q_s32(hang + k), vhold);
            int32x4_t h1 = vbslq_s32(st1, vld1q_s32(hang + k + 4), vhold);
            if (all_stationary) {
                h0 = vmaxq_s32(vsubq_s32(h0, vone), vzero);
                h1 = vmaxq_s32(vsubq_s32(h1, vone), vzero);
            }
            vst1q_s32(hang + k, h0);
            vst1q_s32(hang + k + 4, h1);
        }
    }
    for (; k < n; ++k) {
        if (!flags[k]) hang[k] = hold;
        if (all_stationary) {
            int32_t v = hang[k] - 1;
            hang[k] = v > 0 ? v : 0;
        }
    }
    return;
#else
    for (k = 0; k < n; ++k) {
        if (!flags[k]) { all_stationary = 0; break; }
    }
    /* hangovers[not_stationary] = hangover_hops */
    for (k = 0; k < n; ++k) {
        if (!flags[k]) hang[k] = hold;
    }
    if (all_stationary) {
        /* np.maximum(hangovers - 1, 0, out=hangovers) */
        for (k = 0; k < n; ++k) {
            int32_t v = hang[k] - 1;
            hang[k] = v > 0 ? v : 0;
        }
    }
#endif
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
        k = 1;
#if SK_HAVE_NEON
        /* Sixteen interior flags a step. `carry` holds the previous block's
         * ORIGINAL flags (its store has already overwritten the array), so
         * the left neighbours come from it and the right neighbours from the
         * not-yet-visited bytes ahead. Flags are 0/1, so & is the logical
         * and of the scalar chain. */
        {
            uint8x16_t carry = vsetq_lane_u8(f[0], vdupq_n_u8(0), 15);
            for (; k + 16 <= n - 1; k += 16) {
                uint8x16_t orig = vld1q_u8(f + k);
                uint8x16_t left = vextq_u8(carry, orig, 15);
                uint8x16_t right = vld1q_u8(f + k + 1);
                uint8x16_t sm = vandq_u8(vandq_u8(left, orig), right);
                if (k == 1) first_interior = vgetq_lane_u8(sm, 0);
                vst1q_u8(f + k, sm);
                carry = orig;
            }
            if (k > 1) {
                prev = vgetq_lane_u8(carry, 15);   /* original f[k-1] */
                cur  = f[k];                       /* original f[k]   */
                last_interior = f[k - 1];          /* smoothed[k-1]   */
            }
        }
#endif
        for (; k <= n - 2; ++k) {
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
     * over storage rows 0..filled-1, matching numpy sum(axis=0) for N<=window).
     *
     * Walked ROW-major over a block of per-bin accumulators rather than
     * column-major per bin: every bin's own add sequence is unchanged
     * (row 0, row 1, ... row filled-1, then rev, then the compare), and the
     * only thing reordered is the interleaving of the INDEPENDENT per-bin
     * chains, which no float rounding depends on -- bit-exact by
     * construction, not by measurement. What it buys is contiguous reads:
     * the column-major form strode every inner add by n_freqs, which neither
     * the prefetcher nor the vectorizer could follow.
     *
     * `acc` is a fixed STAT_ACC_BLOCK-element block and the bins are
     * processed one block at a time, so the stack cost is constant and no
     * bound on n_freqs is assumed -- the same hazard detectors.c and
     * filter_analyzer.c de-stacked their fixed `float[1024]` locals for,
     * except this one never needs to scale with the runtime dimension at
     * all. See STAT_ACC_BLOCK for how the block size was chosen. */
    {
        float acc[STAT_ACC_BLOCK];
        int base;
        for (base = 0; base < s->n_freqs; base += STAT_ACC_BLOCK) {
            int m = s->n_freqs - base;
            if (m > STAT_ACC_BLOCK) m = STAT_ACC_BLOCK;
            for (k = 0; k < m; ++k) {
                acc[k] = s->history[0 * s->n_freqs + base + k]; /* row 0 */
            }
            for (row = 1; row < s->history_filled; ++row) {
                const float *hrow = s->history + (size_t)row * s->n_freqs + base;
                for (k = 0; k < m; ++k) {
                    acc[k] = acc[k] + hrow[k];      /* f32 add */
                }
            }
            /* + rev (zeros if average_reverb is NULL) */
            if (average_reverb != NULL) {
                const float *rev = average_reverb + base;
                for (k = 0; k < m; ++k) {
                    acc[k] = acc[k] + rev[k];       /* f32 add */
                }
            }
            /* noise = window_hops * max(self.noise.noise, 1e-30)
             * THR_RATIO * noise; stationary if acum < that. */
            k = 0;
#if SK_HAVE_NEON
            {
                const float32x4_t tiny = vdupq_n_f32(1e-30f);
                const float32x4_t wh = vdupq_n_f32((float)s->window_hops);
                const float32x4_t ratio = vdupq_n_f32(STAT_THR_RATIO);
                const uint8x8_t one = vdup_n_u8(1);
                for (; k + 8 <= m; k += 8) {
                    uint32x4_t lt[2];
                    int g;
                    for (g = 0; g < 2; ++g) {
                        float32x4_t nz = vld1q_f32(s->noise.noise + base + k + 4 * g);
                        float32x4_t nd = vbslq_f32(vcgtq_f32(nz, tiny), nz, tiny);
                        float32x4_t thr = vmulq_f32(ratio, vmulq_f32(wh, nd));
                        lt[g] = vcltq_f32(vld1q_f32(acc + k + 4 * g), thr);
                    }
                    vst1_u8(s->stationarity_flags + base + k,
                            vand_u8(vmovn_u16(vcombine_u16(vmovn_u32(lt[0]),
                                                           vmovn_u32(lt[1]))), one));
                }
            }
#endif
            for (; k < m; ++k) {
                float nz = s->noise.noise[base + k];
                float ndenom = nz > 1e-30f ? nz : 1e-30f;              /* np.maximum */
                float noise_scaled = (float)s->window_hops * ndenom;   /* int*f32 → f32 */
                float thr = STAT_THR_RATIO * noise_scaled;             /* f32 */
                s->stationarity_flags[base + k] = (acc[k] < thr) ? 1 : 0;
            }
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
    /* float(np.mean(mask)) > BLOCK_FRACTION; mean = count / n_freqs (float32-by-design) */
    return ((float)count / (float)s->n_freqs) > STAT_BLOCK_FRACTION;
}
