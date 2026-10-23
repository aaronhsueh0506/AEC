/* delay_aec.c — C port of python/modules/delay/. See delay_aec.h for the
 * chain overview and parity notes. The whole delay chain (decimator biquads,
 * matched-filter dot products / NLMS update, error-sum / anchor scalars, the
 * confidence getter) runs float32 unconditionally (see the "matched-filter
 * arithmetic" note below) -- an intentional divergence from the Python
 * float64 reference, so this file is not bit-exact to Python.
 *
 * Standalone build (does NOT link aec.c): see the header of
 * test/test_delay_reset.c.
 */
#include "delay_aec.h"
#include "aec_scale.h"
#include "utility/mem_align.h"   /* ALIGN16 / MEM_IS_ALIGNED16 / ck_* checked sizes */
#include "utility/simd_kernels.h" /* sk_argmax_sq_f32(_scalar), sk_argmax_i32 */

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* float-domain equivalents of the int16 thresholds (matched_filter.py) */
#define DA_EXCITATION_LIMIT   (150.0f   / 32768.0f)   /* 0.00457763671875 */
#define DA_SATURATION_LIMIT   (32000.0f / 32768.0f)   /* 0.9765625 */
#define DA_SMOOTHING_FAST     0.7f
#define DA_SMOOTHING_SLOW     0.1f
#define DA_MATCHING_THRESHOLD 0.3f  /* used as f32 multiplier of anchor */

/* ------------------------------------------------------------------ helpers */

/* np.dot over float32 inputs.  numpy routes f32 dot through OpenBLAS SDOT
 * (not portably bit-reproducible); we accumulate the float32 products in a
 * float32 running sum, consistent with the rest of this file (float32
 * throughout, see the file banner). The downstream integer lag /
 * DelayQuality are robust to the sub-ULP difference. */
static float delay_aec_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    int i;
    for (i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* ======================= matched-filter arithmetic =========================
 * The two 512-tap dot products accumulate in float32 (as a NEON matched
 * filter does), and x²_sum SLIDES
 * across the 16 sub-block iterations (the window advances by one sample per
 * iteration: x2 += new² − old²) instead of being re-summed.
 * The saturation gate, the `x2_sum > x2_sum_threshold` excitation gate, the
 * `error_sum < 0.3·anchor` reliability test and the NLMS alpha formula keep
 * the SAME comparisons/expressions as the Python reference -- only the
 * precision of the x2_sum / h·x values feeding them differs (f32 vs f64
 * accumulation, sub-ULP-to-1e-4 relative). This intentionally DIVERGES from
 * the Python float64 reference; the integer lag / winner selection is robust
 * to the sub-ULP difference.
 *
 * On AArch64 NEON the f32 dot and the NLMS update use fmla intrinsics
 * and the h² peak search a two-pass max/match reduction; a scalar float path
 * is kept for non-NEON builds.
 *
 * The matched-filter kernels (dot, NLMS update, fused update + next dot) also
 * run on 32-bit ARM when the core has a fused multiply-add (VFPv4 NEON):
 * same four 4-lane accumulators at stride 16, same load/store order, same
 * reduction tree, so the result is bit-identical to AArch64 for finite inputs
 * whose intermediates are normal. ARMv7 NEON flushes subnormal inputs and
 * results to zero and returns the default NaN instead of propagating a
 * payload; AArch64 does neither, so only subnormal-producing data and NaN
 * payloads can differ. vmlaq_f32 is an unfused multiply then add (different
 * rounding), so cores that lack fused FMA (VFPv3 NEON) keep the scalar path
 * below rather than get a numerically different vector one.
 * DA_DISABLE_ARM32_NEON forces the scalar path on 32-bit ARM. Only these
 * three kernels use DA_MF_NEON; every other kernel in this file stays keyed
 * on SK_HAVE_NEON. */
#if SK_HAVE_NEON
#define DA_MF_NEON 1
/* faddp (pairwise) then faddp: (l0 + l1) + (l2 + l3). */
#define DA_VADDV_F32(v) vaddvq_f32(v)
#elif defined(__arm__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && \
      !defined(SIMD_KERNELS_FORCE_SCALAR) && !defined(DA_DISABLE_ARM32_NEON)
#include <arm_neon.h>
#define DA_MF_NEON 1
#define DA_MF_NEON_ARM32 1
/* AArch32 has no across-lane add; two vpadd steps add the lanes in the order
 * AArch64 vaddvq_f32 does, (l0 + l1) + (l2 + l3). */
static inline float da_vaddv_f32(float32x4_t v) {
    float32x2_t p = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(p, p), 0);
}
#define DA_VADDV_F32(v) da_vaddv_f32(v)
#else
#define DA_MF_NEON 0
#endif

#if DA_MF_NEON

#if defined(DA_MF_NEON_ARM32) && defined(__GNUC__) && !defined(__clang__)
/* On 32-bit targets gcc's constant-propagated clone of the dot (n == 512,
 * pointers into the 527-float span) reports the dead scalar tail loop as
 * running off the array. The tail cannot execute for the shipped sizes. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

/* float32 h·x, 4 independent NEON accumulators (n must be a multiple of 16;
 * DA_FILTER_SIZE = 512). */
static float da_dot_f32(const float *a, const float *b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    int i;
    for (i = 0; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    {
        float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        float s = DA_VADDV_F32(acc);
        for (; i < n; ++i) s += a[i] * b[i];
        return s;
    }
}

/* h += alpha * x, 16 taps per iteration (4x fmla) with 4-wide and scalar
 * tails. Elementwise, so the width is invisible in the result: every tap
 * gets the same single vfmaq_f32 whatever iteration it lands in -- there is
 * no accumulator to re-associate, unlike the dot product above. Widened to
 * match that dot: at DA_FILTER_SIZE = 512 taps neither tail runs. */
static void da_nlms_update_f32(float *h, const float *x, float alpha, int n) {
    float32x4_t va = vdupq_n_f32(alpha);
    int i;
    for (i = 0; i + 16 <= n; i += 16) {
        vst1q_f32(h + i,      vfmaq_f32(vld1q_f32(h + i),      va, vld1q_f32(x + i)));
        vst1q_f32(h + i + 4,  vfmaq_f32(vld1q_f32(h + i + 4),  va, vld1q_f32(x + i + 4)));
        vst1q_f32(h + i + 8,  vfmaq_f32(vld1q_f32(h + i + 8),  va, vld1q_f32(x + i + 8)));
        vst1q_f32(h + i + 12, vfmaq_f32(vld1q_f32(h + i + 12), va, vld1q_f32(x + i + 12)));
    }
    for (; i + 4 <= n; i += 4)
        vst1q_f32(h + i, vfmaq_f32(vld1q_f32(h + i), va, vld1q_f32(x + i)));
    for (; i < n; ++i) h[i] += alpha * x[i];
}

#if (DA_FILTER_SIZE % 16) != 0
#error "The fused delay update requires a filter size divisible by 16"
#endif

/* Reuse each updated tap in the next dot; keep da_dot_f32's four sums. */
static float da_nlms_update_next_dot_f32(float *h, const float *x,
                                        const float *next_x, float alpha) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    float32x4_t va = vdupq_n_f32(alpha);
    for (int i = 0; i < DA_FILTER_SIZE; i += 16) {
        float32x4_t h0 = vfmaq_f32(vld1q_f32(h + i), va, vld1q_f32(x + i));
        float32x4_t h1 = vfmaq_f32(vld1q_f32(h + i + 4), va, vld1q_f32(x + i + 4));
        float32x4_t h2 = vfmaq_f32(vld1q_f32(h + i + 8), va, vld1q_f32(x + i + 8));
        float32x4_t h3 = vfmaq_f32(vld1q_f32(h + i + 12), va, vld1q_f32(x + i + 12));
        vst1q_f32(h + i, h0); vst1q_f32(h + i + 4, h1);
        vst1q_f32(h + i + 8, h2); vst1q_f32(h + i + 12, h3);
        acc0 = vfmaq_f32(acc0, h0, vld1q_f32(next_x + i));
        acc1 = vfmaq_f32(acc1, h1, vld1q_f32(next_x + i + 4));
        acc2 = vfmaq_f32(acc2, h2, vld1q_f32(next_x + i + 8));
        acc3 = vfmaq_f32(acc3, h3, vld1q_f32(next_x + i + 12));
    }
    return DA_VADDV_F32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}
#if defined(DA_MF_NEON_ARM32) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#else  /* scalar float fallback (no matched-filter NEON body) */
static float da_dot_f32(const float *a, const float *b, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i;
    const int grouped_end = n - n % 4;
    for (i = 0; i < grouped_end; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    {
        float s = (s0 + s1) + (s2 + s3);
        for (; i < n; ++i) s += a[i] * b[i];
        return s;
    }
}

static void da_nlms_update_f32(float *h, const float *x, float alpha, int n) {
    int i;
    for (i = 0; i < n; ++i) h[i] += alpha * x[i];
}
#endif /* DA_MF_NEON */

/* First maximum index; shared sk_argmax_sq_f32_scalar/sk_argmax_sq_f32 preserve tie order. */
static int da_max_square_peak_index(const float *h, int n) {
    return sk_argmax_sq_f32(h, n);
}

/* ------------------------------------------------------------------ biquad */

static const float DA_LP_B[3][3] = {
    {0.0180919877f, 0.00320961363f, 0.0180919877f},
    {1.0f,         -1.24550459f,    1.0f},
    {1.0f,         -1.4221681f,     1.0f},
};
static const float DA_LP_A[3][2] = {
    {-1.5183195f,   0.633165865f},
    {-1.49784254f,  0.853586692f},
    {-1.49791282f,  0.969572384f},
};
static const float DA_HP_B[1][3] = {
    {0.757076375f, -1.51415275f, 0.757076375f},
};
static const float DA_HP_A[1][2] = {
    {-1.45424359f, 0.574061915f},
};

/* 48kHz -> 16kHz pre-decimation anti-alias filter (DaResample48, decimate by
 * DA_RESAMPLE48_FACTOR=3). Elliptic lowpass, order 7 (4 SOS sections),
 * designed (not hand-tuned) via scipy.signal.ellipord/ellip for a 48kHz
 * input: passband edge 6800 Hz (1.0 dB ripple), stopband edge 8000 Hz (the
 * new 16kHz Nyquist, 60 dB attenuation target). Verified via
 * scipy.signal.sosfreqz: passband -0.99 dB @6800Hz, stopband -68.3 dB
 * @8000Hz, >=60 dB attenuation at every frequency that would alias into the
 * 0-8kHz post-decimation band (9/12/16/20/24 kHz all checked). Coefficients
 * are scipy's sosfilt convention ({b0,b1,b2,a0=1,a1,a2} per section,
 * Direct-Form-II-Transposed) -- da_biquad_process1() below implements the
 * identical recurrence, so this is a numerically-verifiable port, not a
 * from-scratch reimplementation. */
static const float DA_RESAMPLE48_B[4][3] = {
    {5.11914823e-03f, 4.05806316e-04f, 5.11914823e-03f},
    {1.0f,            1.0f,            0.0f},
    {1.0f,           -8.01570761e-01f, 1.0f},
    {1.0f,           -1.01138476e+00f, 1.0f},
};
static const float DA_RESAMPLE48_A[4][2] = {
    {-7.66104626e-01f,  0.0f},
    {-1.43464531e+00f,  6.95652982e-01f},
    {-1.29338870e+00f,  8.61962453e-01f},
    {-1.23733825e+00f,  9.63974476e-01f},
};

static void da_biquad_reset(DaBiquad *bq) {
    memset(bq->z, 0, sizeof(bq->z));
}

/* anti-alias stage: ds4 = kLowPassFilterDs4 (elliptic LP, 3 sections). */
static void da_biquad_init_antialias(DaBiquad *bq) {
    int s;
    bq->n_sections = 3;
    for (s = 0; s < 3; ++s) {
        bq->b[s][0] = DA_LP_B[s][0]; bq->b[s][1] = DA_LP_B[s][1]; bq->b[s][2] = DA_LP_B[s][2];
        bq->a[s][0] = DA_LP_A[s][0]; bq->a[s][1] = DA_LP_A[s][1];
    }
    da_biquad_reset(bq);
}

/* noise-reduction stage: ds4 = kHighPassFilter (butter HP @1 kHz). */
static void da_biquad_init_noise_reduction(DaBiquad *bq) {
    bq->n_sections = 1;
    bq->b[0][0] = DA_HP_B[0][0]; bq->b[0][1] = DA_HP_B[0][1]; bq->b[0][2] = DA_HP_B[0][2];
    bq->a[0][0] = DA_HP_A[0][0]; bq->a[0][1] = DA_HP_A[0][1];
    da_biquad_reset(bq);
}

/* Process one sample through the cascade, returning the float32 output.
 * Direct-form-II transposed, all-float32 (unlike Python's
 * _CascadedBiquad.process, which runs float64; see the file banner).
 *   out = b0*x + z0
 *   z0  = b1*x - a1*out + z1
 *   z1  = b2*x - a2*out
 */
static float da_biquad_process1(DaBiquad *bq, float xn) {
    int s;
    float yn = xn;
    for (s = 0; s < bq->n_sections; ++s) {
        float inp = yn;
        float out = bq->b[s][0] * inp + bq->z[s][0];
        bq->z[s][0] = bq->b[s][1] * inp - bq->a[s][0] * out + bq->z[s][1];
        bq->z[s][1] = bq->b[s][2] * inp - bq->a[s][1] * out;
        yn = out;
    }
    return yn;
}

/* -------------------------------------------------- 48kHz->16kHz sidechain */

static void da_biquad_init_resample48(DaBiquad *bq) {
    int s;
    bq->n_sections = 4;
    for (s = 0; s < 4; ++s) {
        bq->b[s][0] = DA_RESAMPLE48_B[s][0];
        bq->b[s][1] = DA_RESAMPLE48_B[s][1];
        bq->b[s][2] = DA_RESAMPLE48_B[s][2];
        bq->a[s][0] = DA_RESAMPLE48_A[s][0];
        bq->a[s][1] = DA_RESAMPLE48_A[s][1];
    }
    da_biquad_reset(bq);
}

static void da_resample48_init(DaResample48 *r) {
    da_biquad_init_resample48(&r->near_lp);
    da_biquad_init_resample48(&r->far_lp);
    r->phase = 0;
}

static void da_resample48_reset(DaResample48 *r) {
    da_biquad_reset(&r->near_lp);
    da_biquad_reset(&r->far_lp);
    r->phase = 0;
}

/* Filter n_in samples through bq (state carries across calls -- every input
 * sample is fed through the cascade even though only every DA_RESAMPLE48_
 * FACTOR-th post-filter sample is kept, exactly like da_decimator_decimate's
 * ds4 stage above), keep every DA_RESAMPLE48_FACTOR-th sample with phase
 * continuity across calls (mirrors the 4ch pipeline's own delay_phase /
 * SharedMatchedDelayEstimator._to_16k_pair stride-pick, just with a real
 * anti-alias filter ahead of the stride instead of a naive pick). Returns
 * the count of samples written to out (<= out_cap, which the caller sizes
 * DA_RESAMPLE48_CAP_FOR(hop) = ceil(hop/3) at the one 48kHz grid, so the cap
 * is a defence-in-depth bound rather than a live truncation). */
static int da_resample48_channel(DaBiquad *bq, const float *in, int n_in,
                                 float *out, int out_cap, int *phase) {
    int i, count = 0;
    int p = *phase;
    for (i = 0; i < n_in; ++i) {
        float filtered = da_biquad_process1(bq, in[i]);
        if (((i + p) % DA_RESAMPLE48_FACTOR) == 0 && count < out_cap) {
            out[count++] = filtered;
        }
    }
    *phase = (p + n_in) % DA_RESAMPLE48_FACTOR;
    return count;
}

/* ---------------------------------------------------------------- decimator */

static void da_decimator_init(DaDecimator *dec) {
    da_biquad_init_antialias(&dec->anti_alias);
    da_biquad_init_noise_reduction(&dec->noise_reduction);
}

/* Generic cascade walk over a 64-sample block: every sample through both
 * cascades (state carries), output kept on the FIRST sample of every
 * DA_DOWN_SAMPLING_FACTOR group. DA_BLOCK_SIZE / DA_DOWN_SAMPLING_FACTOR
 * == DA_SUB_BLOCK_SIZE by construction (64 / 4 == 16). Nested so neither
 * `i % DOWN_SAMPLING_FACTOR` nor a per-sample branch runs in the inner loop.
 * This is the shape-independent form; da_decimator_decimate below runs it
 * whenever the cascade is not the 3-LP + 1-HP one da_decimator_init builds. */
static void da_decimator_decimate_generic(DaDecimator *dec, const float *in_block,
                                          float *out_sub) {
    int o, j;
    for (o = 0; o < DA_SUB_BLOCK_SIZE; ++o) {
        int base = o * DA_DOWN_SAMPLING_FACTOR;
        float lp = da_biquad_process1(&dec->anti_alias, in_block[base]);
        float hp = da_biquad_process1(&dec->noise_reduction, lp);
        out_sub[o] = hp;
        for (j = 1; j < DA_DOWN_SAMPLING_FACTOR; ++j) {
            lp = da_biquad_process1(&dec->anti_alias, in_block[base + j]);
            da_biquad_process1(&dec->noise_reduction, lp);
        }
    }
}

/* Decimate a 64-sample block to a 16-sample sub-block.
 * Python: x = lp.process(block) [f64], x = hp.process(x) [f64], then
 * x[::4].astype(f32).  The biquad chains run on ALL 64 samples (state
 * carries), but only every 4th post-HP sample is kept & cast to f32.
 *
 * Both decimators run on EVERY hop -- analysed or merely fed (see
 * da_estimator_process_inner_block) -- so this is the whole non-engaged
 * floor of the delay stage. da_decimator_init builds exactly one shape (3
 * anti-alias sections + 1 noise-reduction section), and for that shape the
 * per-sample da_biquad_process1 call re-reads five coefficients and
 * round-trips two state words through memory per section. Specialising it
 * holds all twenty coefficients and all eight state words in locals across
 * the block; the per-sample expressions are the same products, same
 * left-associative grouping and same order da_biquad_process1 evaluates,
 * which under -ffp-contract=off (see the Makefile's FP_POLICY) is the same
 * float32 arithmetic, not merely equivalent arithmetic. Any other cascade
 * shape takes the generic walk. */
static void da_decimator_decimate(DaDecimator *dec, const float *in_block, float *out_sub) {
    DaBiquad *lp = &dec->anti_alias;
    DaBiquad *hp = &dec->noise_reduction;

    /* the shape test comes FIRST: only sections the cascade actually declares
     * have been initialised, so an off-shape cascade must reach the generic
     * walk before any b/a/z row past its own section count is read */
    if (lp->n_sections != 3 || hp->n_sections != 1) {
        da_decimator_decimate_generic(dec, in_block, out_sub);
        return;
    }
    {
        const float lb00 = lp->b[0][0], lb01 = lp->b[0][1], lb02 = lp->b[0][2];
        const float lb10 = lp->b[1][0], lb11 = lp->b[1][1], lb12 = lp->b[1][2];
        const float lb20 = lp->b[2][0], lb21 = lp->b[2][1], lb22 = lp->b[2][2];
        const float la00 = lp->a[0][0], la01 = lp->a[0][1];
        const float la10 = lp->a[1][0], la11 = lp->a[1][1];
        const float la20 = lp->a[2][0], la21 = lp->a[2][1];
        const float hb00 = hp->b[0][0], hb01 = hp->b[0][1], hb02 = hp->b[0][2];
        const float ha00 = hp->a[0][0], ha01 = hp->a[0][1];
        float lz00 = lp->z[0][0], lz01 = lp->z[0][1];
        float lz10 = lp->z[1][0], lz11 = lp->z[1][1];
        float lz20 = lp->z[2][0], lz21 = lp->z[2][1];
        float hz00 = hp->z[0][0], hz01 = hp->z[0][1];
        int o, j;
        for (o = 0; o < DA_SUB_BLOCK_SIZE; ++o) {
            const float *group = in_block + o * DA_DOWN_SAMPLING_FACTOR;
            for (j = 0; j < DA_DOWN_SAMPLING_FACTOR; ++j) {
                float x = group[j];
                float y0 = lb00 * x + lz00;
                float y1, y2, y3;
                lz00 = lb01 * x - la00 * y0 + lz01;
                lz01 = lb02 * x - la01 * y0;
                y1 = lb10 * y0 + lz10;
                lz10 = lb11 * y0 - la10 * y1 + lz11;
                lz11 = lb12 * y0 - la11 * y1;
                y2 = lb20 * y1 + lz20;
                lz20 = lb21 * y1 - la20 * y2 + lz21;
                lz21 = lb22 * y1 - la21 * y2;
                y3 = hb00 * y2 + hz00;
                hz00 = hb01 * y2 - ha00 * y3 + hz01;
                hz01 = hb02 * y2 - ha01 * y3;
                if (j == 0) out_sub[o] = y3;
            }
        }
        lp->z[0][0] = lz00; lp->z[0][1] = lz01;
        lp->z[1][0] = lz10; lp->z[1][1] = lz11;
        lp->z[2][0] = lz20; lp->z[2][1] = lz21;
        hp->z[0][0] = hz00; hp->z[0][1] = hz01;
    }
}

/* The render and capture decimators run back to back on every block, share
 * no data and (da_decimator_init) carry the same coefficients. The 3-LP + 1-HP
 * cascade is a long chain of dependent float32 multiplies and adds per
 * sample, so running both signals through it as the two lanes of one vector
 * halves the instruction count. Every lane executes exactly the scalar
 * expressions of da_decimator_decimate above -- the same products, the same
 * left-associative grouping, no fused multiply-add -- so outputs and filter
 * states are bit-identical (except the payload bits of a NaN). Off-shape
 * cascades, or two cascades whose coefficients differ, take two single calls. */
#if SK_HAVE_NEON
static int da_biquad_same_coefficients(const DaBiquad *a, const DaBiquad *b) {
    /* only the declared sections are initialised; the rows past them are not
     * part of the filter */
    return a->n_sections == b->n_sections &&
           memcmp(a->b, b->b, (size_t)a->n_sections * sizeof(a->b[0])) == 0 &&
           memcmp(a->a, b->a, (size_t)a->n_sections * sizeof(a->a[0])) == 0;
}

static int da_decimator_same_coefficients(const DaDecimator *a, const DaDecimator *b) {
    return da_biquad_same_coefficients(&a->anti_alias, &b->anti_alias) &&
           da_biquad_same_coefficients(&a->noise_reduction, &b->noise_reduction);
}

static void da_decimator_decimate_pair(DaDecimator *da, const float *in_a, float *out_a,
                                       DaDecimator *db, const float *in_b, float *out_b,
                                       int same_coefficients) {
    if (da->anti_alias.n_sections != 3 || da->noise_reduction.n_sections != 1 ||
        db->anti_alias.n_sections != 3 || db->noise_reduction.n_sections != 1 ||
        !same_coefficients) {
        da_decimator_decimate(da, in_a, out_a);
        da_decimator_decimate(db, in_b, out_b);
        return;
    }
    {
        const DaBiquad *lp = &da->anti_alias, *hp = &da->noise_reduction;
        const float32x2_t lb00 = vdup_n_f32(lp->b[0][0]), lb01 = vdup_n_f32(lp->b[0][1]), lb02 = vdup_n_f32(lp->b[0][2]);
        const float32x2_t lb10 = vdup_n_f32(lp->b[1][0]), lb11 = vdup_n_f32(lp->b[1][1]), lb12 = vdup_n_f32(lp->b[1][2]);
        const float32x2_t lb20 = vdup_n_f32(lp->b[2][0]), lb21 = vdup_n_f32(lp->b[2][1]), lb22 = vdup_n_f32(lp->b[2][2]);
        const float32x2_t la00 = vdup_n_f32(lp->a[0][0]), la01 = vdup_n_f32(lp->a[0][1]);
        const float32x2_t la10 = vdup_n_f32(lp->a[1][0]), la11 = vdup_n_f32(lp->a[1][1]);
        const float32x2_t la20 = vdup_n_f32(lp->a[2][0]), la21 = vdup_n_f32(lp->a[2][1]);
        const float32x2_t hb00 = vdup_n_f32(hp->b[0][0]), hb01 = vdup_n_f32(hp->b[0][1]), hb02 = vdup_n_f32(hp->b[0][2]);
        const float32x2_t ha00 = vdup_n_f32(hp->a[0][0]), ha01 = vdup_n_f32(hp->a[0][1]);
        float32x2_t lz00 = vset_lane_f32(db->anti_alias.z[0][0], vdup_n_f32(da->anti_alias.z[0][0]), 1);
        float32x2_t lz01 = vset_lane_f32(db->anti_alias.z[0][1], vdup_n_f32(da->anti_alias.z[0][1]), 1);
        float32x2_t lz10 = vset_lane_f32(db->anti_alias.z[1][0], vdup_n_f32(da->anti_alias.z[1][0]), 1);
        float32x2_t lz11 = vset_lane_f32(db->anti_alias.z[1][1], vdup_n_f32(da->anti_alias.z[1][1]), 1);
        float32x2_t lz20 = vset_lane_f32(db->anti_alias.z[2][0], vdup_n_f32(da->anti_alias.z[2][0]), 1);
        float32x2_t lz21 = vset_lane_f32(db->anti_alias.z[2][1], vdup_n_f32(da->anti_alias.z[2][1]), 1);
        float32x2_t hz00 = vset_lane_f32(db->noise_reduction.z[0][0], vdup_n_f32(da->noise_reduction.z[0][0]), 1);
        float32x2_t hz01 = vset_lane_f32(db->noise_reduction.z[0][1], vdup_n_f32(da->noise_reduction.z[0][1]), 1);
        int o, j;
        for (o = 0; o < DA_SUB_BLOCK_SIZE; ++o) {
            const float32x4_t ga = vld1q_f32(in_a + o * DA_DOWN_SAMPLING_FACTOR);
            const float32x4_t gb = vld1q_f32(in_b + o * DA_DOWN_SAMPLING_FACTOR);
            /* (a_j, b_j) pairs: zip1 -> j = 0,1 ; zip2 -> j = 2,3 */
            const float32x4_t z1 = vzip1q_f32(ga, gb), z2 = vzip2q_f32(ga, gb);
            for (j = 0; j < DA_DOWN_SAMPLING_FACTOR; ++j) {
                float32x2_t x = (j == 0) ? vget_low_f32(z1) : (j == 1) ? vget_high_f32(z1)
                              : (j == 2) ? vget_low_f32(z2) : vget_high_f32(z2);
                float32x2_t y0 = vadd_f32(vmul_f32(lb00, x), lz00);
                float32x2_t y1, y2, y3;
                lz00 = vadd_f32(vsub_f32(vmul_f32(lb01, x), vmul_f32(la00, y0)), lz01);
                lz01 = vsub_f32(vmul_f32(lb02, x), vmul_f32(la01, y0));
                y1 = vadd_f32(vmul_f32(lb10, y0), lz10);
                lz10 = vadd_f32(vsub_f32(vmul_f32(lb11, y0), vmul_f32(la10, y1)), lz11);
                lz11 = vsub_f32(vmul_f32(lb12, y0), vmul_f32(la11, y1));
                y2 = vadd_f32(vmul_f32(lb20, y1), lz20);
                lz20 = vadd_f32(vsub_f32(vmul_f32(lb21, y1), vmul_f32(la20, y2)), lz21);
                lz21 = vsub_f32(vmul_f32(lb22, y1), vmul_f32(la21, y2));
                y3 = vadd_f32(vmul_f32(hb00, y2), hz00);
                hz00 = vadd_f32(vsub_f32(vmul_f32(hb01, y2), vmul_f32(ha00, y3)), hz01);
                hz01 = vsub_f32(vmul_f32(hb02, y2), vmul_f32(ha01, y3));
                if (j == 0) { out_a[o] = vget_lane_f32(y3, 0); out_b[o] = vget_lane_f32(y3, 1); }
            }
        }
        da->anti_alias.z[0][0] = vget_lane_f32(lz00, 0); db->anti_alias.z[0][0] = vget_lane_f32(lz00, 1);
        da->anti_alias.z[0][1] = vget_lane_f32(lz01, 0); db->anti_alias.z[0][1] = vget_lane_f32(lz01, 1);
        da->anti_alias.z[1][0] = vget_lane_f32(lz10, 0); db->anti_alias.z[1][0] = vget_lane_f32(lz10, 1);
        da->anti_alias.z[1][1] = vget_lane_f32(lz11, 0); db->anti_alias.z[1][1] = vget_lane_f32(lz11, 1);
        da->anti_alias.z[2][0] = vget_lane_f32(lz20, 0); db->anti_alias.z[2][0] = vget_lane_f32(lz20, 1);
        da->anti_alias.z[2][1] = vget_lane_f32(lz21, 0); db->anti_alias.z[2][1] = vget_lane_f32(lz21, 1);
        da->noise_reduction.z[0][0] = vget_lane_f32(hz00, 0); db->noise_reduction.z[0][0] = vget_lane_f32(hz00, 1);
        da->noise_reduction.z[0][1] = vget_lane_f32(hz01, 0); db->noise_reduction.z[0][1] = vget_lane_f32(hz01, 1);
    }
}
#else
static void da_decimator_decimate_pair(DaDecimator *da, const float *in_a, float *out_a,
                                       DaDecimator *db, const float *in_b, float *out_b,
                                       int same_coefficients) {
    (void)same_coefficients;
    da_decimator_decimate(da, in_a, out_a);
    da_decimator_decimate(db, in_b, out_b);
}
#endif

/* The 48 kHz -> 16 kHz anti-alias cascades of the capture and render signals
 * share no data and carry the same coefficients, and each is a dependent chain
 * of float32 multiplies and adds per sample. Both are filtered together as the
 * two lanes of one vector, with the expressions of da_biquad_process1 and the
 * same left-associative grouping, no fused multiply-add, so outputs and
 * carried states are bit-identical to two da_resample48_channel() calls (except
 * the payload bits of a NaN). The kept-sample selection and the phase update
 * are those of da_resample48_channel(); both channels have one length, one cap
 * and one phase, so they produce the same count, which is returned. Cascades
 * that are not the 4-section shape, or whose coefficients differ, take two
 * single calls. */
#if SK_HAVE_NEON
static int da_resample48_pair(DaBiquad *bq_a, const float *in_a, float *out_a,
                              DaBiquad *bq_b, const float *in_b, float *out_b,
                              int n_in, int out_cap, int *phase) {
    int i, count = 0;
    const int p = *phase;
    if (bq_a->n_sections != 4 || bq_b->n_sections != 4 ||
        !da_biquad_same_coefficients(bq_a, bq_b)) {
        int pa = p, pb = p;
        count = da_resample48_channel(bq_a, in_a, n_in, out_a, out_cap, &pa);
        (void)da_resample48_channel(bq_b, in_b, n_in, out_b, out_cap, &pb);
        *phase = pa;
        return count;
    }
    {
        const DaBiquad *c = bq_a;
        const float32x2_t b00 = vdup_n_f32(c->b[0][0]), b01 = vdup_n_f32(c->b[0][1]), b02 = vdup_n_f32(c->b[0][2]);
        const float32x2_t b10 = vdup_n_f32(c->b[1][0]), b11 = vdup_n_f32(c->b[1][1]), b12 = vdup_n_f32(c->b[1][2]);
        const float32x2_t b20 = vdup_n_f32(c->b[2][0]), b21 = vdup_n_f32(c->b[2][1]), b22 = vdup_n_f32(c->b[2][2]);
        const float32x2_t b30 = vdup_n_f32(c->b[3][0]), b31 = vdup_n_f32(c->b[3][1]), b32 = vdup_n_f32(c->b[3][2]);
        const float32x2_t a00 = vdup_n_f32(c->a[0][0]), a01 = vdup_n_f32(c->a[0][1]);
        const float32x2_t a10 = vdup_n_f32(c->a[1][0]), a11 = vdup_n_f32(c->a[1][1]);
        const float32x2_t a20 = vdup_n_f32(c->a[2][0]), a21 = vdup_n_f32(c->a[2][1]);
        const float32x2_t a30 = vdup_n_f32(c->a[3][0]), a31 = vdup_n_f32(c->a[3][1]);
        float32x2_t z00 = vset_lane_f32(bq_b->z[0][0], vdup_n_f32(bq_a->z[0][0]), 1);
        float32x2_t z01 = vset_lane_f32(bq_b->z[0][1], vdup_n_f32(bq_a->z[0][1]), 1);
        float32x2_t z10 = vset_lane_f32(bq_b->z[1][0], vdup_n_f32(bq_a->z[1][0]), 1);
        float32x2_t z11 = vset_lane_f32(bq_b->z[1][1], vdup_n_f32(bq_a->z[1][1]), 1);
        float32x2_t z20 = vset_lane_f32(bq_b->z[2][0], vdup_n_f32(bq_a->z[2][0]), 1);
        float32x2_t z21 = vset_lane_f32(bq_b->z[2][1], vdup_n_f32(bq_a->z[2][1]), 1);
        float32x2_t z30 = vset_lane_f32(bq_b->z[3][0], vdup_n_f32(bq_a->z[3][0]), 1);
        float32x2_t z31 = vset_lane_f32(bq_b->z[3][1], vdup_n_f32(bq_a->z[3][1]), 1);
        for (i = 0; i < n_in; ++i) {
            float32x2_t x = vld1_lane_f32(in_b + i, vld1_dup_f32(in_a + i), 1);
            float32x2_t y0 = vadd_f32(vmul_f32(b00, x), z00);
            float32x2_t y1, y2, y3;
            z00 = vadd_f32(vsub_f32(vmul_f32(b01, x), vmul_f32(a00, y0)), z01);
            z01 = vsub_f32(vmul_f32(b02, x), vmul_f32(a01, y0));
            y1 = vadd_f32(vmul_f32(b10, y0), z10);
            z10 = vadd_f32(vsub_f32(vmul_f32(b11, y0), vmul_f32(a10, y1)), z11);
            z11 = vsub_f32(vmul_f32(b12, y0), vmul_f32(a11, y1));
            y2 = vadd_f32(vmul_f32(b20, y1), z20);
            z20 = vadd_f32(vsub_f32(vmul_f32(b21, y1), vmul_f32(a20, y2)), z21);
            z21 = vsub_f32(vmul_f32(b22, y1), vmul_f32(a21, y2));
            y3 = vadd_f32(vmul_f32(b30, y2), z30);
            z30 = vadd_f32(vsub_f32(vmul_f32(b31, y2), vmul_f32(a30, y3)), z31);
            z31 = vsub_f32(vmul_f32(b32, y2), vmul_f32(a31, y3));
            if (((i + p) % DA_RESAMPLE48_FACTOR) == 0 && count < out_cap) {
                out_a[count] = vget_lane_f32(y3, 0);
                out_b[count] = vget_lane_f32(y3, 1);
                ++count;
            }
        }
        bq_a->z[0][0] = vget_lane_f32(z00, 0); bq_b->z[0][0] = vget_lane_f32(z00, 1);
        bq_a->z[0][1] = vget_lane_f32(z01, 0); bq_b->z[0][1] = vget_lane_f32(z01, 1);
        bq_a->z[1][0] = vget_lane_f32(z10, 0); bq_b->z[1][0] = vget_lane_f32(z10, 1);
        bq_a->z[1][1] = vget_lane_f32(z11, 0); bq_b->z[1][1] = vget_lane_f32(z11, 1);
        bq_a->z[2][0] = vget_lane_f32(z20, 0); bq_b->z[2][0] = vget_lane_f32(z20, 1);
        bq_a->z[2][1] = vget_lane_f32(z21, 0); bq_b->z[2][1] = vget_lane_f32(z21, 1);
        bq_a->z[3][0] = vget_lane_f32(z30, 0); bq_b->z[3][0] = vget_lane_f32(z30, 1);
        bq_a->z[3][1] = vget_lane_f32(z31, 0); bq_b->z[3][1] = vget_lane_f32(z31, 1);
    }
    *phase = (p + n_in) % DA_RESAMPLE48_FACTOR;
    return count;
}
#else
static int da_resample48_pair(DaBiquad *bq_a, const float *in_a, float *out_a,
                              DaBiquad *bq_b, const float *in_b, float *out_b,
                              int n_in, int out_cap, int *phase) {
    int pa = *phase, pb = *phase;
    int count = da_resample48_channel(bq_a, in_a, n_in, out_a, out_cap, &pa);
    (void)da_resample48_channel(bq_b, in_b, n_in, out_b, out_cap, &pb);
    *phase = pa;
    return count;
}
#endif

/* ------------------------------------------------------------------- ring */

static void da_ring_reset(DaRing *r) {
    memset(r->buffer, 0, (size_t)r->capacity * sizeof(float));
    r->write = 0;
}

static void da_ring_push(DaRing *r, const float *sub, int n) {
    /* Values identical to a per-sample-modulo walk, but written as at most two
     * contiguous memcpy segments -- no integer division in the inner loop (n =
     * DA_SUB_BLOCK_SIZE, currently 16, samples are pushed twice per 64-sample
     * block). Same approach as da_ring_gather_back below. */
    int first = r->capacity - r->write;
    if (first >= n) {
        memcpy(r->buffer + r->write, sub, (size_t)n * sizeof(float));
        r->write += n;
        if (r->write == r->capacity) r->write = 0;
    } else {
        memcpy(r->buffer + r->write, sub, (size_t)first * sizeof(float));
        memcpy(r->buffer, sub + first, (size_t)(n - first) * sizeof(float));
        r->write = n - first;
    }
}

/* gather_back: return `length` samples in time-reversed order starting at
 * newest_index() - start_offset.  out[0] = sample start_offset ago. */
static void da_ring_gather_back(const DaRing *r, int start_offset, int length, float *out) {
    /* Values identical to the per-sample-modulo walk, but copied as at most two
     * contiguous reversed segments — no integer division in the inner loop.
     * This gather runs 80x per 64-sample block (5 filters x 16 sub-samples,
     * 512 taps each), so a per-sample `% capacity` would cost ~10M
     * integer divisions/sec at 16 kHz and dominate the matched-filter cost. */
    int cap = r->capacity;
    int newest = (r->write - 1 + cap) % cap;
    int end = (newest - start_offset + cap) % cap;
    int k = 0;
    int first = (end + 1 < length) ? end + 1 : length;
    const float *buf = r->buffer;
    for (; k < first; ++k) out[k] = buf[end - k];
    if (k < length) {
        int idx2 = cap - 1;
        int base = k;
        for (; k < length; ++k) out[k] = buf[idx2 - (k - base)];
    }
}

/* -------------------------------------------------------- matched filter core */

/* Mirrors matched_filter_core(). Returns filters_updated; writes *error_sum. */
static int da_matched_filter_core(const DaRing *ring, int alignment_shift_back,
                                  float x2_sum_threshold, float smoothing,
                                  const float *y, float *h, float *error_sum_out) {
    float error_sum = 0.0f;
    int filters_updated = 0;
    int i;
    /* All DA_SUB_BLOCK_SIZE windows of this call are subranges of ONE
     * contiguous span: window_i[k] = ring sample (alignment_shift_back +
     * (SUB_BLOCK_SIZE-1-i) + k) back = span[(SUB_BLOCK_SIZE-1-i) + k], where
     * span = gather_back(alignment_shift_back, FILTER_SIZE+SUB_BLOCK_SIZE-1).
     * One 527-sample gather replaces 16 gathers of 512 (8192 -> 527 floats
     * copied per filter per block); every window is the identical value
     * sequence the per-i gather produced (the ring is not written during the
     * whole matched-filter update), so all downstream arithmetic is
     * byte-identical. Deepest sample touched is alignment_shift_back +
     * FILTER_SIZE+SUB_BLOCK_SIZE-2, i.e. (n-1)*384 + 526 at the last filter
     * -- DA_RING_CAPACITY_FOR(n) is exactly that plus two at every bank
     * size, so this is always in range (2062 < 2064 at the n=5 default). */
    float span[DA_FILTER_SIZE + DA_SUB_BLOCK_SIZE - 1];
    da_ring_gather_back(ring, alignment_shift_back,
                        DA_FILTER_SIZE + DA_SUB_BLOCK_SIZE - 1, span);
    /* Sliding x²_sum: window i covers span[(SB-1-i) .. (SB-1-i)+511]; moving
     * i→i+1 adds span[SB-2-i] (one newer sample at the front) and drops
     * span[(SB-1-i)+512] (the oldest). Initialised once for i=0, then O(1)
     * per iteration instead of a 512-tap re-sum. f32 accumulation — see the
     * matched-filter arithmetic note above. */
    float x2_slide = da_dot_f32(span + (DA_SUB_BLOCK_SIZE - 1),
                                span + (DA_SUB_BLOCK_SIZE - 1), DA_FILTER_SIZE);
#if DA_MF_NEON
    float next_dot = da_dot_f32(h, span + (DA_SUB_BLOCK_SIZE - 1), DA_FILTER_SIZE);
#endif
    for (i = 0; i < DA_SUB_BLOCK_SIZE; ++i) {
        const float *x_window = span + (DA_SUB_BLOCK_SIZE - 1 - i);
        float x2_sum, s, e;
        int saturation;
        if (i > 0) {
            float in_v  = x_window[0];               /* newly entered sample */
            float out_v = x_window[DA_FILTER_SIZE];  /* just left the window */
            x2_slide += in_v * in_v - out_v * out_v;
        }
        x2_sum = x2_slide;
#if DA_MF_NEON
        s = next_dot;
#else
        s = da_dot_f32(h, x_window, DA_FILTER_SIZE);
#endif
        e = y[i] - s;
        saturation = (y[i] >= DA_SATURATION_LIMIT || y[i] <= -DA_SATURATION_LIMIT);
        error_sum += e * e;
        if (x2_sum > x2_sum_threshold && !saturation) {
            /* alpha = smoothing * e / x2_sum, computed directly in float32;
             * h += alpha*x runs in f32 via da_nlms_update_f32. */
            float alpha_f = smoothing * e / x2_sum;
#if DA_MF_NEON
            if (i + 1 < DA_SUB_BLOCK_SIZE) {
                next_dot = da_nlms_update_next_dot_f32(h, x_window, x_window - 1, alpha_f);
            } else {
                da_nlms_update_f32(h, x_window, alpha_f, DA_FILTER_SIZE);
            }
#else
            da_nlms_update_f32(h, x_window, alpha_f, DA_FILTER_SIZE);
#endif
            filters_updated = 1;
#if DA_MF_NEON
        } else if (i + 1 < DA_SUB_BLOCK_SIZE) {
            next_dot = da_dot_f32(h, x_window - 1, DA_FILTER_SIZE);
#endif
        }
    }
    *error_sum_out = error_sum;
    return filters_updated;
}

/* Row accessors for the two pool-carved row-major banks. Both row strides are
 * multiples of 16 bytes (2048 and 512), so every row inherits the block's
 * ALIGN16 base. */
static float *da_mf_row(const DaMatchedFilter *mf, int n) {
    return mf->filters + (size_t)n * DA_FILTER_SIZE;
}
static float *da_mf_acc(const DaMatchedFilter *mf, int n) {
    return mf->accumulated_error + (size_t)n * DA_ACC_ERR_SIZE;
}

/* num_filters: runtime bank size. The pool holds EXACTLY this many rows (the
 * bank is carved to fit), so both loops below cover the whole allocation --
 * there is no inactive tail. The heap-vs-caller-pool byte-equality gate
 * (test_static_aec) therefore compares fully-defined memory: every carved
 * byte is written here. */
static void da_matched_filter_init(DaMatchedFilter *mf, int num_filters) {
    int n, k;
    mf->num_filters = num_filters;
    memset(mf->filters, 0,
           (size_t)num_filters * DA_FILTER_SIZE * sizeof(float));
    for (n = 0; n < num_filters; ++n) {
        float *acc = da_mf_acc(mf, n);
        for (k = 0; k < DA_ACC_ERR_SIZE; ++k) acc[k] = 1.0f;
    }
    memset(mf->instantaneous_error, 0, (size_t)DA_ACC_ERR_SIZE * sizeof(float));
    mf->last_detected_best_lag_filter = -1;
    mf->number_pre_echo_updates = 0;
    mf->reported_valid = 0;
    mf->reported_lag = 0;
    mf->reported_pre_echo_lag = 0;
    mf->winner_lag_valid = 0;
    mf->winner_lag = 0;
}

static void da_matched_filter_reset(DaMatchedFilter *mf, int full_reset) {
    memset(mf->filters, 0,
           (size_t)mf->num_filters * DA_FILTER_SIZE * sizeof(float));
    mf->winner_lag_valid = 0;
    mf->reported_valid = 0;
    if (full_reset) {
        int n, k;
        for (n = 0; n < mf->num_filters; ++n) {
            float *acc = da_mf_acc(mf, n);
            for (k = 0; k < DA_ACC_ERR_SIZE; ++k) acc[k] = 1.0f;
        }
        mf->number_pre_echo_updates = 0;
    }
}

/* Mirrors MatchedFilter.update, minus the pre-echo onset search: the
 * accumulated-error/onset estimate had no consumer (see
 * da_aggregator_aggregate's PRODUCT POLICY note -- the reported delay is
 * always the dominant peak), so the per-tap-prefix error it was derived from
 * is not computed and mf->accumulated_error / mf->instantaneous_error hold
 * their initialised values. */
static void da_matched_filter_update(DaMatchedFilter *mf, const DaRing *ring,
                                     const float *capture, int use_slow_smoothing) {
    float smoothing = use_slow_smoothing ? DA_SMOOTHING_SLOW : DA_SMOOTHING_FAST;
    float x2_sum_threshold = (float)DA_FILTER_SIZE
                            * DA_EXCITATION_LIMIT * DA_EXCITATION_LIMIT;
    float error_sum_anchor = delay_aec_dot(capture, capture, DA_SUB_BLOCK_SIZE);
    float winner_error_sum = error_sum_anchor;
    int winner_index = -1;
    int alignment_shift = 0;
    int prev_lag_valid = 0, prev_lag = 0;
    int n;

    mf->winner_lag_valid = 0;
    mf->reported_valid = 0;

    /* mf->num_filters is both the loop bound AND the number of rows the pool
     * actually holds (the bank is carved to fit -- there is no tail beyond
     * it to walk off into). This loop is the entire search cost (~4.2 MMAC/s
     * per filter at the full 250 block/s rate); together with the 5,728 B/
     * filter the layout saves, a short bank is where the whole embedded
     * saving comes from. At the default 5 this is the identical loop it
     * always was. */
    for (n = 0; n < mf->num_filters; ++n) {
        float error_sum;
        float *h = da_mf_row(mf, n);
        int filters_updated, lag_estimate, reliable, lag;
        filters_updated = da_matched_filter_core(ring, alignment_shift, x2_sum_threshold,
                                                 smoothing, capture, h, &error_sum);
        lag_estimate = da_max_square_peak_index(h, DA_FILTER_SIZE);
        reliable = (lag_estimate > 2
                    && lag_estimate < (DA_FILTER_SIZE - 10)
                    && error_sum < DA_MATCHING_THRESHOLD * error_sum_anchor);
        lag = lag_estimate + alignment_shift;
        if (filters_updated && reliable && error_sum < winner_error_sum) {
            winner_error_sum = error_sum;
            winner_index = n;
            if (prev_lag_valid && prev_lag == lag) {
                mf->winner_lag = prev_lag;
                mf->winner_lag_valid = 1;
                winner_index = n - 1;
            } else {
                mf->winner_lag = lag;
                mf->winner_lag_valid = 1;
            }
        }
        prev_lag = lag;
        prev_lag_valid = 1;
        alignment_shift += DA_FILTER_INTRA_SHIFT;
    }

    if (winner_index != -1 && mf->winner_lag_valid) {
        mf->reported_valid = 1;
        mf->reported_lag = mf->winner_lag;
        /* number_pre_echo_updates counts, saturating, the blocks on which the
         * winner filter repeated over a non-degenerate anchor (the reference gate
         * is int16-scale, error_sum > 1.0 in Q15²; the float[-1,1] equivalent
         * is 1.0 / 32768² ≈ 9.3e-10, matching matched_filter.py). It is
         * zeroed only by da_matched_filter_init() (construction) and
         * da_matched_filter_reset(mf, full_reset=1) (a genuine delay/echo-
         * path-change reset) -- neither fires just because the delay estimate
         * stays locked, which is the ordinary steady state for a device that
         * hasn't moved, so without the cap the count would run for the whole
         * unbounded-lock timeframe and eventually overflow. */
        if (mf->last_detected_best_lag_filter == winner_index
            && error_sum_anchor > 1.0f / (32768.0f * 32768.0f)
            && mf->number_pre_echo_updates < DA_PRE_ECHO_UPDATES_TO_REPORT) {
            mf->number_pre_echo_updates += 1;
        }
        mf->last_detected_best_lag_filter = winner_index;
    }
}

/* ---------------------------------------------------- highest-peak aggregator */

/* da_argmax_incremental_update -- O(1)-when-provable incremental maintenance
 * of (*candidate, *candidate_valid) as the FIRST index of the maximum value
 * in `histogram[0..hist_size)`, called by da_highest_peak_aggregate. The
 * aggregator's 0-initialized ring always evicts a prior occupant, so every
 * call both decrements histogram[A] and increments histogram[B]. This
 * function performs BOTH histogram writes AND the candidate/M bookkeeping --
 * callers must NOT separately apply them, and must NOT call this when
 * `A == B && *candidate_valid` (a pure net-unchanged wash the caller must
 * fast-path BEFORE calling: see the correctness note below for why that
 * combination breaks the general derivation and has to be excluded
 * structurally, not merely as an optimization).
 *
 * Correctness:
 *
 * Invariant maintained: whenever *candidate_valid is true, `c = *candidate`
 * is provably the first index of the maximum value `M = histogram[c]`
 * (equivalent to sk_argmax_i32(histogram, hist_size)).
 *
 *  - !*candidate_valid: state isn't trustworthy (fresh reset, or a prior
 *    call already gave up) -- mutate histogram, then do the one authoritative
 *    sk_argmax_i32 rescan and mark valid. Exact by construction; this is also
 *    the path a fresh A==B call takes (see below), so A==B is never actually
 *    unsafe THROUGH this branch, only through the two branches below it.
 *
 *  - A == c (the tracked candidate itself lost a count): after
 *    the decrement c's own value drops to M-1, so (c,M) cannot be
 *    trusted -- there may be an UNTRACKED bin already tied at value M (it
 *    must be at some index > c, since c was defined as the first index
 *    reaching M, so nothing below c could already equal M). If the insert at
 *    B produces a value > the OLD M, B is now strictly greater than every
 *    other bin (every other bin was <= M, including any untracked tie), so B
 *    is unambiguously the new unique argmax -- O(1) and exact regardless of
 *    what's hidden. Otherwise (Bval0+1 <= M) we cannot rule out an untracked
 *    tie or a third bin now being the true max, so we MUST rescan -- this is
 *    exactly the case the naive "compare against the candidate's own
 *    post-decrement value" heuristic gets wrong (counterexample:
 *    H=[3,5,5,2,0], candidate=1 (M=5, untracked tie at index 2); evicting
 *    bin 1 (->4) while inserting at bin 3 (2->3) has post-decrement 4 >= 3,
 *    which LOOKS safe under that heuristic, but the true new argmax is index
 *    2, not 1 or 3 -- comparing against the OLD M=5, not the post-decrement
 *    4, correctly routes this case to the rescan fallback instead).
 *
 *  - otherwise (A != c): (c,M) is UNCHANGED by the decrement, because decrementing a
 *    non-candidate bin can only lower a value already <= M, never raise any
 *    other bin above M, and never touches c. So (c,M) remains exactly
 *    correct going into the insert. Only B's value changed (to Bval0+1,
 *    since A != B guarantees the increment is the sole modification to B
 *    this call): if Bval0+1 > M, B is a strictly new unique max (every other
 *    bin, tracked or not, was <= M); if Bval0+1 == M, B ties the max and
 *    numpy argmax's first-index rule means the new answer is min(B,c) -- but
 *    nothing below c can already hold M (c is BY DEFINITION the first such
 *    index), so the comparison reduces to "B < c" alone (no untracked bin
 *    can be at a lower index than both); if Bval0+1 < M, nothing changes.
 *    O(1) and exact in every sub-case.
 *
 * The A == B fast-path exclusion: if it were allowed through,
 * `Bval0 = histogram[B]` (read before mutation) would ALREADY equal
 * histogram[A]'s pre-mutation value (since A==B), and the decrement+increment
 * nets to zero -- but the ">M"/"==M" comparisons above assume the decrement
 * and increment are on DIFFERENT bins (the decrement's effect on `M` is
 * "none, because A!=c" or "drop to M-1, because A==c" -- either way a
 * distinct bin from B). With A==B those two assumptions collide: e.g.
 * candidate=c (M), and A(=B) is a DIFFERENT bin ALSO already at value M
 * (untracked tie, index > c) -- Bval0=M, so "Bval0+1>M" is trivially true,
 * which would incorrectly move candidate to A even though the net histogram
 * change is exactly zero and the true answer is still c. Hence: the caller
 * must intercept A==B && *candidate_valid before ever reaching
 * here (net-unchanged, zero mutation needed, candidate/M already correct by
 * the standing invariant). */
static void da_argmax_incremental_update(int *histogram, int hist_size,
                                         int A, int B,
                                         int *candidate, int *candidate_valid) {
    if (!*candidate_valid) {
        histogram[A] -= 1;
        histogram[B] += 1;
        *candidate = sk_argmax_i32(histogram, hist_size);
        *candidate_valid = 1;
        return;
    }
    {
        int c = *candidate;
        int M = histogram[c];
        int Bval0 = histogram[B];
        histogram[A] -= 1;
        histogram[B] += 1;
        if (A == c) {
            if (Bval0 + 1 > M) { *candidate = B; return; }
            *candidate = sk_argmax_i32(histogram, hist_size);
            return;
        }
        if (Bval0 + 1 > M || (Bval0 + 1 == M && B < c)) *candidate = B;
        /* else: unchanged, still c */
    }
}

static void da_highest_peak_reset(DaHighestPeak *hp) {
    memset(hp->histogram, 0, (size_t)hp->hist_size * sizeof(int));
    memset(hp->ring, 0, (size_t)hp->window * sizeof(int));
    hp->ring_index = 0;
    /* candidate is NOT reset in Python HighestPeakAggregator.reset; preserve.
     * candidate_valid IS reset (internal-only guard, no Python equivalent --
     * see the DaHighestPeak struct comment): forces the next aggregate()
     * call to take the authoritative sk_argmax_i32 rescan path instead of
     * trusting a candidate/M pair computed against the pre-reset histogram. */
    hp->candidate_valid = 0;
}

static void da_highest_peak_init(DaHighestPeak *hp, int window) {
    hp->window = window;
    da_highest_peak_reset(hp);
    hp->candidate = -1;
}

static void da_highest_peak_aggregate(DaHighestPeak *hp, int lag) {
    int old_lag = hp->ring[hp->ring_index];
    int clamped_lag = lag;
    if (clamped_lag < 0) clamped_lag = 0;
    else if (clamped_lag >= hp->hist_size) clamped_lag = hp->hist_size - 1;
    hp->ring[hp->ring_index] = clamped_lag;
    hp->ring_index = (hp->ring_index + 1) % hp->window;
    if (old_lag == clamped_lag && hp->candidate_valid) {
        /* net-unchanged: evicted bin == inserted bin, histogram provably
         * untouched, existing candidate/M pair still exactly correct. */
        return;
    }
    da_argmax_incremental_update(hp->histogram, hp->hist_size,
                                 old_lag, clamped_lag,
                                 &hp->candidate, &hp->candidate_valid);
}

/* ---------------------------------------------- pre-echo histogram (inert) */

/* Carved and reset like every other piece of estimator state so the block's
 * byte image is deterministic, but never fed: the reported delay is the
 * dominant peak (see da_aggregator_aggregate's PRODUCT POLICY note), so
 * there is no onset candidate to accumulate. */

static int da_get_ds_block_size_log2(int down_sampling_factor) {
    int ds_log2 = 0;
    int f = down_sampling_factor >> 1;
    int v;
    while (f > 0) { ds_log2 += 1; f >>= 1; }
    v = DA_BLOCK_SIZE_LOG2 - ds_log2;
    return v < 0 ? 0 : v;
}

static void da_pre_echo_reset(DaPreEcho *pe) {
    int i;
    memset(pe->histogram, 0, (size_t)pe->hist_size * sizeof(int));
    for (i = 0; i < pe->window; ++i) pe->ring[i] = -1;
    pe->ring_index = 0;
    pe->number_updates = 0;
    pe->pre_echo_candidate = 0;
    pe->argmax_idx = 0;
    pe->argmax_valid = 0;
}

static void da_pre_echo_init(DaPreEcho *pe, int window) {
    pe->window = window;
    pe->block_size_log2 = da_get_ds_block_size_log2(DA_DOWN_SAMPLING_FACTOR);
    da_pre_echo_reset(pe);
}

/* ---------------------------------------------------------- lag aggregator */

static void da_aggregator_init(DaLagAggregator *agg, int window) {
    da_highest_peak_init(&agg->highest_peak, window);
    da_pre_echo_init(&agg->pre_echo, window);
    agg->significant_candidate_found = 0;
}

static void da_aggregator_reset(DaLagAggregator *agg, int hard_reset) {
    da_highest_peak_reset(&agg->highest_peak);
    da_pre_echo_reset(&agg->pre_echo);
    if (hard_reset) agg->significant_candidate_found = 0;
}

static int da_aggregator_reliable_delay_found(const DaLagAggregator *agg) {
    return agg->significant_candidate_found;
}

static int da_aggregator_delay_at_highest_peak(const DaLagAggregator *agg) {
    return agg->highest_peak.candidate;
}

/* Mirrors MatchedFilterLagAggregator.aggregate. Returns 1 if an estimate was
 * produced (writes *quality, *delay in DOWNSAMPLED samples). */
static int da_aggregator_aggregate(DaLagAggregator *agg, int lag_estimate_valid,
                                   int lag_estimate_lag,
                                   DelayQuality *quality_out, int *delay_out) {
    int lag, candidate, hist_val;
    if (!lag_estimate_valid) return 0;
    lag = lag_estimate_lag - DA_HEADROOM;
    if (lag < 0) lag = 0;
    da_highest_peak_aggregate(&agg->highest_peak, lag);
    candidate = agg->highest_peak.candidate;
    hist_val = agg->highest_peak.histogram[candidate];
    if (agg->significant_candidate_found || hist_val > DA_THRESH_CONVERGED)
        agg->significant_candidate_found = 1;
    if (hist_val > DA_THRESH_CONVERGED ||
        (hist_val > DA_THRESH_INITIAL && !agg->significant_candidate_found)) {
        *quality_out = agg->significant_candidate_found ? DELAY_QUALITY_REFINED
                                                        : DELAY_QUALITY_COARSE;
        /* PRODUCT POLICY, and a deliberate departure from upstream at this
         * seam: the reported delay is the dominant peak. `quality` above is
         * derived from the dominant histogram, so reporting an earliest-onset
         * candidate instead would attach that confidence to a delay it does
         * not describe -- which is how a 100 ms-early onset reached a PBFDKF
         * spanning 52 ms while carrying confidence 1.0. Upstream wants the
         * earliest onset because its estimator feeds a RenderDelayController;
         * this one aligns a short filter directly, so the onset estimate has
         * no consumer here and agg->pre_echo carries no live candidate.
         * Mirrors python/modules/delay/lag_aggregator.py's aggregate(). */
        *delay_out = candidate;
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------------ clockdrift */

static void da_clockdrift_init(DaClockdrift *cd, int sample_rate) {
    cd->delay_history[0] = cd->delay_history[1] = cd->delay_history[2] = 0;
    cd->level = CLOCKDRIFT_NONE;
    cd->stability_counter = 0;
    /* 7500 blocks (~30 s at the native 4 ms/block cadence). Ticks are
     * inner DA_BLOCK_SIZE(=64)-sample blocks, so the tick period is
     * 64/sample_rate seconds -- rescale by real sample_rate, not by outer
     * hop_size (matches python/modules/delay/clockdrift_detector.py).
     * Routed through the canonical aec_ms_to_hops() (aec_scale.h) instead
     * of a hand-rolled round-half-away-from-zero formula, matching that
     * helper's own "never paste a raw ms-block count -- route it through
     * here" contract and its lrintf (round-half-to-even) rounding. */
    cd->stability_reset_hops =
        aec_ms_to_hops(30000.0f, DA_BLOCK_SIZE, sample_rate);
}

static void da_clockdrift_update(DaClockdrift *cd, int delay_estimate) {
    int d1, d2, d3, probable_up, verified_up, probable_down, verified_down;
    if (delay_estimate == cd->delay_history[0]) {
        /* Ceilinged at stability_reset_hops+1 to avoid signed overflow: the ONLY
         * consumer of stability_counter in this file/header is the
         * `> stability_reset_hops` check on the very next line. Once the counter
         * exceeds stability_reset_hops, further increments can never change that
         * comparison's outcome (already true, stays true), so gating the increment at
         * the smallest value that still satisfies `>` is observationally identical to
         * an unbounded increment for every reachable state. */
        if (cd->stability_counter <= cd->stability_reset_hops) cd->stability_counter += 1;
        if (cd->stability_counter > cd->stability_reset_hops) cd->level = CLOCKDRIFT_NONE;
        return;
    }
    cd->stability_counter = 0;
    d1 = cd->delay_history[0] - delay_estimate;
    d2 = cd->delay_history[1] - delay_estimate;
    d3 = cd->delay_history[2] - delay_estimate;
    probable_up   = (d1 == -1 && d2 == -2) || (d1 == -2 && d2 == -1);
    verified_up   = probable_up && d3 == -3;
    probable_down = (d1 == 1 && d2 == 2) || (d1 == 2 && d2 == 1);
    verified_down = probable_down && d3 == 3;
    if (verified_up || verified_down) cd->level = CLOCKDRIFT_VERIFIED;
    else if ((probable_up || probable_down) && cd->level == CLOCKDRIFT_NONE)
        cd->level = CLOCKDRIFT_PROBABLE;
    cd->delay_history[2] = cd->delay_history[1];
    cd->delay_history[1] = cd->delay_history[0];
    cd->delay_history[0] = delay_estimate;
}

/* --------------------------------------------------- EchoPathDelayEstimator */

/* Ring entries in use by both lag aggregators: 1 s of 64-sample inner
 * blocks at the feed rate, clamped to the carved capacity DA_HIST_WINDOW. */
static int da_hist_window_for_rate(int sample_rate) {
    int w = aec_ms_to_hops(1000.0f, DA_BLOCK_SIZE, sample_rate);
    if (w < 1) w = 1;
    if (w > DA_HIST_WINDOW) w = DA_HIST_WINDOW;
    return w;
}

static void da_estimator_init(DaEstimator *e, int sample_rate, int num_filters) {
    da_decimator_init(&e->capture_decimator);
    da_decimator_init(&e->render_decimator);
#if SK_HAVE_NEON
    e->decimators_same_coefficients =
        da_decimator_same_coefficients(&e->capture_decimator, &e->render_decimator);
#endif
    da_ring_reset(&e->render_ring);
    da_matched_filter_init(&e->matched_filter, num_filters);
    /* Histogram window = 1 s of inner blocks at the feed rate, the Python
     * lag_aggregator._num_blocks_per_second (round(sr / 64)): 250 at 16 kHz
     * (and at 48 kHz, which feeds this estimator its 16 kHz-equivalent
     * stream), 125 at the LEGACY native 8 kHz feed. A bare capacity
     * (250) at every rate would be a 2 s window at 8 kHz. Routed through the
     * canonical aec_ms_to_hops() like the two thresholds around it. */
    da_aggregator_init(&e->aggregator, da_hist_window_for_rate(sample_rate));
    da_clockdrift_init(&e->clockdrift, sample_rate);
    /* Half a second of blocks (125 at 4 ms/block, ~500 ms), computed live from the
     * rate (same inner-block-rate basis as clockdrift above) via
     * aec_ms_to_hops() -- see stability_reset_hops above. */
    e->consistent_estimate_threshold =
        aec_ms_to_hops(500.0f, DA_BLOCK_SIZE, sample_rate);
    e->old_agg_valid = 0;
    e->old_agg_delay = 0;
    e->old_agg_quality = DELAY_QUALITY_COARSE;
    e->consistent_estimate_counter = 0;
    e->pending_count = 0;
}

/* _reset (reset_lag_aggregator, reset_delay_confidence). */
static void da_estimator_reset_internal(DaEstimator *e, int reset_lag_aggregator,
                                        int reset_delay_confidence) {
    if (reset_lag_aggregator)
        da_aggregator_reset(&e->aggregator, reset_delay_confidence);
    da_matched_filter_reset(&e->matched_filter, reset_lag_aggregator);
    e->old_agg_valid = 0;
    e->consistent_estimate_counter = 0;
}

/* public reset(reset_delay_confidence) -> full reset of the delay-estimator
 * signal chain (matched filter + aggregator + decimators + ring buffer +
 * pending edge-chunker samples).
 *
 * The only caller of delay_aec_reset() (which calls this) is aec_reset()
 * (aec.c) / AEC.reset() on the Python side -- a top-level cold-start-style
 * reset that recreates AecState/ResidualEchoEstimator/SuppressionGain, clears
 * CNG state, zeroes the OLA buffer, and resets every other subsystem (HPF,
 * saturation detectors, filter taps, regime handler, stationarity estimator,
 * ...). It is NOT a lightweight per-echo-path-change nudge -- that is the
 * SEPARATE da_estimator_reset_internal(e, 0, 0) call from
 * da_estimator_process_inner_block (fires on the consistent-estimate
 * stability counter, every ~500 ms in normal steady state), which leaves the
 * whole signal chain (sidechain, decimators, ring, pending) untouched.
 *
 * delay_aec_reset() therefore clears the ENTIRE chain consistently: the
 * outer 48kHz resample48 sidechain (reset by delay_aec_reset() itself) AND
 * this estimator's inner decimators / ring buffer / pending samples, so stale
 * pre-reset echo-path audio cannot contaminate the matched filter's first
 * correlation windows. da_decimator_init is reused as the reset (it only
 * zeroes coefficients/state, no allocation, so calling it again is a safe,
 * idempotent reset). clockdrift is deliberately left untouched: it tracks a
 * 30s-horizon hardware clock-drift estimate, not reset-scoped audio history. */
static void da_estimator_reset(DaEstimator *e, int reset_delay_confidence) {
    da_estimator_reset_internal(e, 1, reset_delay_confidence);
    da_decimator_init(&e->capture_decimator);
    da_decimator_init(&e->render_decimator);
#if SK_HAVE_NEON
    e->decimators_same_coefficients =
        da_decimator_same_coefficients(&e->capture_decimator, &e->render_decimator);
#endif
    da_ring_reset(&e->render_ring);
    e->pending_count = 0;
}

/* _process_inner_block. Returns 1 if an estimate was produced this block
 * (writes *quality / *delay in RAW samples).
 *
 * run_matched_filter (duty-cycle support, see delay_aec_accumulate_ex):
 * when 0, the block is FED but not ANALYSED — both decimators still run
 * (their biquad cascades carry state across every sample, so skipping them
 * would corrupt all later sub-samples) and the decimated render sub-block is
 * still pushed into the ring (the matched filter's x-window must see gapless
 * audio, else the next analysed block correlates against a buffer with holes
 * and reports garbage). Only the matched-filter NLMS update + lag
 * aggregation + clockdrift + stability bookkeeping are skipped; every piece
 * of estimator DECISION state (histograms, old_agg, counters) is left
 * exactly as the previous analysed block set it. */
static int da_estimator_process_inner_block(DaEstimator *e, const float *render_block,
                                            const float *capture_block,
                                            int run_matched_filter,
                                            DelayQuality *quality_out, int *delay_out) {
    float render_sub[DA_SUB_BLOCK_SIZE];
    float capture_sub[DA_SUB_BLOCK_SIZE];
    int produced;
    DelayQuality q = DELAY_QUALITY_COARSE;
    int delay_ds = 0;
    int agg_valid;
    int agg_delay = 0;
    DelayQuality agg_quality = DELAY_QUALITY_COARSE;

    da_decimator_decimate_pair(&e->render_decimator, render_block, render_sub,
                               &e->capture_decimator, capture_block, capture_sub,
                               e->decimators_same_coefficients);
    da_ring_push(&e->render_ring, render_sub, DA_SUB_BLOCK_SIZE);
    if (!run_matched_filter) return 0;   /* fed, not analysed (see above) */
    da_matched_filter_update(&e->matched_filter, &e->render_ring, capture_sub,
                             da_aggregator_reliable_delay_found(&e->aggregator));
    produced = da_aggregator_aggregate(
        &e->aggregator,
        e->matched_filter.reported_valid,
        e->matched_filter.reported_lag,
        &q, &delay_ds);

    if (produced && q == DELAY_QUALITY_REFINED)
        da_clockdrift_update(&e->clockdrift, da_aggregator_delay_at_highest_peak(&e->aggregator));

    agg_valid = produced;
    if (produced) {
        agg_quality = q;
        agg_delay = delay_ds * DA_DOWN_SAMPLING_FACTOR;
    }

    /* stability counter */
    if (e->old_agg_valid && agg_valid && e->old_agg_delay == agg_delay)
        e->consistent_estimate_counter += 1;
    else
        e->consistent_estimate_counter = 0;
    e->old_agg_valid = agg_valid;
    e->old_agg_delay = agg_delay;
    e->old_agg_quality = agg_quality;

    if (e->consistent_estimate_counter > e->consistent_estimate_threshold)
        da_estimator_reset_internal(e, 0, 0);  /* partial reset */

    if (agg_valid) {
        *quality_out = agg_quality;
        *delay_out = agg_delay;
        return 1;
    }
    return 0;
}

/* True when the n floats at `in` share bytes with the array `pend`; such an
 * input must not go through memcpy. */
#define DA_OVERLAPS(in, n, pend)                                             \
    ((n) > 0 && (uintptr_t)(in) < (uintptr_t)(pend) + sizeof(pend) &&       \
     (uintptr_t)(pend) < (uintptr_t)(in) + (size_t)(n) * sizeof(float))

/* estimate_delay: buffer raw hop into the edge-chunker, drain 64-sample
 * blocks.  Returns 1 if ANY estimate was produced during the drain (writes
 * the LATEST such estimate into *quality / *delay). The hop is appended in
 * runs up to the next block boundary; an input that overlaps either pending
 * array is appended one sample at a time so each store still sees the earlier
 * ones. */
static int da_estimator_estimate_delay(DaEstimator *e, const float *render_hop,
                                       const float *capture_hop, int hop,
                                       int run_matched_filter,
                                       DelayQuality *quality_out, int *delay_out) {
    int produced_any = 0;
    int i = 0;
    DelayQuality q = DELAY_QUALITY_COARSE;
    int delay = 0;
    const int aliased =
        DA_OVERLAPS(render_hop, hop, e->render_pending) ||
        DA_OVERLAPS(render_hop, hop, e->capture_pending) ||
        DA_OVERLAPS(capture_hop, hop, e->render_pending) ||
        DA_OVERLAPS(capture_hop, hop, e->capture_pending);
    while (i < hop) {
        int n = aliased ? 1 : DA_BLOCK_SIZE - e->pending_count;
        if (n > hop - i) n = hop - i;
        if (aliased) {
            e->render_pending[e->pending_count] = render_hop[i];
            e->capture_pending[e->pending_count] = capture_hop[i];
        } else {
            memcpy(e->render_pending + e->pending_count, render_hop + i,
                   (size_t)n * sizeof(float));
            memcpy(e->capture_pending + e->pending_count, capture_hop + i,
                   (size_t)n * sizeof(float));
        }
        e->pending_count += n;
        i += n;
        if (e->pending_count == DA_BLOCK_SIZE) {
            DelayQuality bq;
            int bd;
            if (da_estimator_process_inner_block(e, e->render_pending, e->capture_pending,
                                                 run_matched_filter, &bq, &bd)) {
                produced_any = 1;
                q = bq;
                delay = bd;
            }
            e->pending_count = 0;
        }
    }
    if (produced_any) {
        *quality_out = q;
        *delay_out = delay;
        return 1;
    }
    return 0;
}

#undef DA_OVERLAPS

/* ====================== LegacyDelayShim (public API) ====================== */

/* ─────────────────────── pool layout ─────────────────────────────
 * ONE description of the block, shared by delay_aec_get_mem_size() and
 * delay_aec_init(): the query returns `total`, the carve walks the fields in
 * the SAME order with the SAME ALIGN16 bumps and asserts it lands exactly on
 * `total`. There is no second size expression anywhere -- that is the whole
 * point of routing both through here. */
typedef struct {
    int    num_filters;     /* validated: [1, DA_NUM_FILTERS] */
    int    ring_capacity;
    int    hp_hist_size;
    int    pe_hist_size;
    int    resample_cap;    /* 0 unless the 48 kHz sidechain is built */
    size_t total;           /* ALIGN16-sized */
} DaLayout;

/* Reject, not clamp -- see delay_aec_get_mem_size()'s header comment. Sole
 * definition, so a query and an init can never disagree about whether a
 * request is admissible. A silent clamp here would let a direct low-level
 * caller run a different bank size than requested with no error signal. */
static int da_valid_num_filters(int num_filters) {
    return num_filters >= 1 && num_filters <= DA_NUM_FILTERS;
}

/* Returns 1 on success (L fully populated), 0 on invalid input or on a size
 * computation that overflowed. Every add/multiply/align goes through
 * mem_align.h's saturating ck_* helpers, so one MEM_SIZE_INVALID() test at
 * the end catches an overflow anywhere in the walk. */
static int da_compute_layout(int sample_rate, int hop_size, int num_filters,
                             DaLayout *L) {
    size_t t = 0;
    int n;
    if (!L) return 0;
    /* hop_size is bounded well below the point where hop/3+1 could overflow
     * an int; sample_rate/hop_size <= 0 are the real caller mistakes. */
    if (sample_rate <= 0 || hop_size <= 0) return 0;
    if (!da_valid_num_filters(num_filters)) return 0;

    n = num_filters;
    L->num_filters   = n;
    L->ring_capacity = DA_RING_CAPACITY_FOR(n);
    L->hp_hist_size  = DA_HP_HIST_SIZE_FOR(n);
    L->pe_hist_size  = DA_PE_HIST_SIZE_FOR(n);
    /* 48 kHz-only sidechain scratch: nothing carved at any
     * other rate, so a 16 kHz instance does not carry two 48 kHz buffers. */
    L->resample_cap  = (sample_rate == 48000)
                     ? DA_RESAMPLE48_CAP_FOR(hop_size) : 0;

    /* ---- field walk; da_delay_carve() below MUST match this order ---- */
    t = ck_field_size(t, ck_mul_size((size_t)n, (size_t)DA_FILTER_SIZE),
                      sizeof(float));                                   /* filters */
    t = ck_field_size(t, ck_mul_size((size_t)n, (size_t)DA_ACC_ERR_SIZE),
                      sizeof(float));                                   /* accumulated_error */
    t = ck_field_size(t, (size_t)DA_ACC_ERR_SIZE, sizeof(float));       /* instantaneous_error */
    t = ck_field_size(t, (size_t)L->ring_capacity, sizeof(float));      /* render ring */
    t = ck_field_size(t, (size_t)L->hp_hist_size, sizeof(int));         /* highest-peak hist */
    t = ck_field_size(t, (size_t)DA_HIST_WINDOW, sizeof(int));          /* highest-peak ring */
    t = ck_field_size(t, (size_t)L->pe_hist_size, sizeof(int));         /* pre-echo hist */
    t = ck_field_size(t, (size_t)DA_HIST_WINDOW, sizeof(int));          /* pre-echo ring */
    if (L->resample_cap > 0) {
        t = ck_field_size(t, (size_t)L->resample_cap, sizeof(float));   /* near16 scratch */
        t = ck_field_size(t, (size_t)L->resample_cap, sizeof(float));   /* far16 scratch */
    }

    if (MEM_SIZE_INVALID(t)) return 0;
    L->total = t;
    return 1;
}

/* Bind every pool pointer + its runtime size. Pure pointer bumps; no writes
 * to the pool itself (da_estimator_init() does all the initialisation, after
 * this, through the pointers bound here). Returns the byte count consumed so
 * the caller can assert the lockstep. */
static size_t da_delay_carve(DelayAec *d, uint8_t *mem, const DaLayout *L) {
    uint8_t *p = mem;
    DaMatchedFilter *mf = &d->est.matched_filter;
    DaHighestPeak   *hp = &d->est.aggregator.highest_peak;
    DaPreEcho       *pe = &d->est.aggregator.pre_echo;

    mf->filters = (float *)p;
    p += ALIGN16((size_t)L->num_filters * DA_FILTER_SIZE * sizeof(float));
    mf->accumulated_error = (float *)p;
    p += ALIGN16((size_t)L->num_filters * DA_ACC_ERR_SIZE * sizeof(float));
    mf->instantaneous_error = (float *)p;
    p += ALIGN16((size_t)DA_ACC_ERR_SIZE * sizeof(float));

    d->est.render_ring.buffer   = (float *)p;
    d->est.render_ring.capacity = L->ring_capacity;
    p += ALIGN16((size_t)L->ring_capacity * sizeof(float));

    hp->histogram = (int *)p;
    p += ALIGN16((size_t)L->hp_hist_size * sizeof(int));
    hp->ring = (int *)p;
    p += ALIGN16((size_t)DA_HIST_WINDOW * sizeof(int));
    hp->hist_size = L->hp_hist_size;

    pe->histogram = (int *)p;
    p += ALIGN16((size_t)L->pe_hist_size * sizeof(int));
    pe->ring = (int *)p;
    p += ALIGN16((size_t)DA_HIST_WINDOW * sizeof(int));
    pe->hist_size = L->pe_hist_size;

    if (L->resample_cap > 0) {
        d->near16_scratch = (float *)p;
        p += ALIGN16((size_t)L->resample_cap * sizeof(float));
        d->far16_scratch = (float *)p;
        p += ALIGN16((size_t)L->resample_cap * sizeof(float));
    } else {
        d->near16_scratch = NULL;
        d->far16_scratch  = NULL;
    }
    d->resample_cap = L->resample_cap;

    return (size_t)(p - mem);
}

size_t delay_aec_get_mem_size(int sample_rate, int hop_size, int num_filters) {
    DaLayout L;
    if (!da_compute_layout(sample_rate, hop_size, num_filters, &L)) return 0;
    return L.total;
}

int delay_aec_init(DelayAec *d, void *mem, size_t bytes,
                   int sample_rate, int hop_size, int num_filters) {
    DaLayout L;
    int rate_factor;
    size_t used;

    if (!d || !mem) return -1;
    /* Every f32 array below is ALIGN16-bumped off this base, so a misaligned
     * base would misalign the 512-tap banks the NEON matched-filter kernels
     * load from. Reject rather than silently de-align. */
    if (!MEM_IS_ALIGNED16(mem)) return -1;
    if (!da_compute_layout(sample_rate, hop_size, num_filters, &L)) return -1;
    if (bytes < L.total) return -1;

    memset(d, 0, sizeof(*d));
    used = da_delay_carve(d, (uint8_t *)mem, &L);
    /* Lockstep assertion: the carve must land EXACTLY on the
     * byte count the query promised -- not "within", exactly. A mismatch
     * means the two field walks have drifted, which is a programming error,
     * so fail construction instead of running on a layout nobody sized. */
    if (used != L.total) return -1;

    /* DaEstimator's internal clockdrift/consistent-estimate constants
     * (da_estimator_init) are computed in real seconds FROM the sample_rate
     * argument -- they are not hardcoded, so this must be the estimator's
     * true feed rate, not always "16000". Only 48kHz is resampled down to a
     * 16kHz-equivalent stream before reaching the estimator (the sidechain
     * below); native 8kHz/16kHz feed the estimator directly at their own
     * rate. */
    rate_factor = (sample_rate == 48000) ? DA_RESAMPLE48_FACTOR : 1;
    da_estimator_init(&d->est, (rate_factor > 1) ? 16000 : sample_rate,
                      L.num_filters);
    d->rate_factor = rate_factor;
    if (d->rate_factor > 1) {
        da_resample48_init(&d->resample48);
    }
    d->latest_valid = 0;
    d->latest_delay = 0;
    d->latest_quality = DELAY_QUALITY_COARSE;
    d->estimate_count = 0;
    return 0;
}

void delay_aec_reset(DelayAec *d) {
    da_estimator_reset(&d->est, 1);
    if (d->rate_factor > 1) {
        da_resample48_reset(&d->resample48);
    }
    d->latest_valid = 0;
    d->estimate_count = 0;
}

int delay_aec_accumulate(DelayAec *d, const float *near, const float *far, int hop) {
    return delay_aec_accumulate_ex(d, near, far, hop, 1);
}

/* Duty-cycle-aware accumulate (driven unconditionally by aec.c's always-on
 * duty-cycle state machine; a DIVERGENCE from the Python reference when
 * run_matched_filter=0 — the reference always analyses).
 * run_matched_filter=1 is byte-identical to delay_aec_accumulate.
 * With 0, this hop's samples are decimated + pushed into the ring (the
 * estimator's buffers stay gapless — the correctness crux) but no
 * matched-filter/aggregator work runs, so latest_* stays put. */
int delay_aec_accumulate_ex(DelayAec *d, const float *near, const float *far,
                            int hop, int run_matched_filter) {
    DelayQuality q = DELAY_QUALITY_COARSE;
    int delay = 0;
    int produced;
    int prev_delay = d->latest_valid ? d->latest_delay : -1;
    int emitted_new = 0;
    const float *near_fed = near;
    const float *far_fed = far;
    int hop_fed = hop;
    if (d->rate_factor > 1) {
        /* Anti-alias-filter + decimate both channels to the estimator's
         * fixed 16kHz-native feed rate before it ever sees a sample (see
         * delay_aec_init()'s comment). The two channels are independent
         * phase-continuous filter chains that are fed one hop length per
         * call, so they share one phase and one output count. */
        int phase = d->resample48.phase;
        hop_fed = da_resample48_pair(&d->resample48.near_lp, near, d->near16_scratch,
                                     &d->resample48.far_lp, far, d->far16_scratch,
                                     hop, d->resample_cap, &phase);
        d->resample48.phase = phase;
        near_fed = d->near16_scratch;
        far_fed = d->far16_scratch;
    }
    /* NOTE: estimate_delay takes (render, capture) = (far, near). */
    produced = da_estimator_estimate_delay(&d->est, far_fed, near_fed, hop_fed,
                                           run_matched_filter, &q, &delay);
    if (produced) {
        d->latest_valid = 1;
        d->latest_delay = delay;
        d->latest_quality = q;
        /* estimate_count is a public diagnostic counter (delay_aec_n_updates()
         * below) with no internal threshold comparison anywhere in this file
         * -- external callers may read the raw value, so it is not safe to
         * assume a boolean-threshold cap. Saturate at INT_MAX instead: keeps
         * monotonic-increase-until-cap semantics for any reader, never
         * invokes signed-integer-overflow UB. */
        if (d->estimate_count < INT_MAX) d->estimate_count += 1;
        if (delay != prev_delay) emitted_new = 1;
    }
    return emitted_new;
}

int delay_aec_estimated_delay(const DelayAec *d) {
    if (!d->latest_valid) return -1;
    /* latest_delay is always in the estimator's own fixed-16kHz-equivalent
     * domain (see delay_aec_init()'s comment); rescale to the caller's
     * native sample domain here so every external reader gets a
     * ready-to-use value without needing to know rate_factor exists. */
    return d->latest_delay * d->rate_factor;
}

float delay_aec_confidence(const DelayAec *d) {
    if (!d->latest_valid) return 0.0f;
    if (d->latest_quality == DELAY_QUALITY_REFINED) return 1.0f;
    if (d->latest_quality == DELAY_QUALITY_COARSE) return 0.5f;
    return 0.0f;
}

int delay_aec_is_solid(const DelayAec *d) {
    return delay_aec_confidence(d) >= 1.0f;
}

int delay_aec_n_updates(const DelayAec *d) {
    return d->estimate_count;
}

int delay_aec_has_clockdrift(const DelayAec *d) {
    return d->est.clockdrift.level != CLOCKDRIFT_NONE;
}
