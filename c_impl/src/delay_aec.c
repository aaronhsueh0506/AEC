/* delay_aec.c -- see delay_aec.h. The delay chain runs in float32. */
#include "delay_aec.h"
#include "aec_scale.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* float-domain equivalents of the int16 thresholds */
#define DA_EXCITATION_LIMIT   (150.0f   / 32768.0f)
#define DA_SATURATION_LIMIT   (32000.0f / 32768.0f)
#define DA_SMOOTHING_FAST     0.7f
#define DA_SMOOTHING_SLOW     0.1f
#define DA_MATCHING_THRESHOLD 0.3f  /* used as a float32 multiplier of the anchor */

/* float32 dot product with a float32 running sum. */
static float delay_aec_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    int i;
    for (i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* Matched-filter arithmetic: both 512-tap dot products accumulate in float32 and x2_sum slides
 * across the 16 sub-block iterations (x2 += new^2 - old^2) instead of being re-summed. The NEON
 * kernels (dot, NLMS update, fused update + next dot) use fused multiply-add on AArch64 and on
 * 32-bit ARM cores with VFPv4 NEON, with the same accumulator layout and reduction tree; cores
 * without fused FMA keep the scalar path, and DA_DISABLE_ARM32_NEON forces it. Only these three
 * kernels use DA_MF_NEON; the others key on SK_HAVE_NEON. */
#if SK_HAVE_NEON
#define DA_MF_NEON 1
#define DA_VADDV_F32(v) vaddvq_f32(v)
#elif defined(__arm__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA) && \
      !defined(SIMD_KERNELS_FORCE_SCALAR) && !defined(DA_DISABLE_ARM32_NEON)
#include <arm_neon.h>
#define DA_MF_NEON 1
#define DA_MF_NEON_ARM32 1
/* AArch32 has no across-lane add; two vpadd steps add the lanes in AArch64's order. */
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
/* gcc's constant-propagated clone of the dot warns about the dead scalar tail loop; the tail
 * cannot run for the shipped sizes. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

/* float32 dot product with 4 independent NEON accumulators (n a multiple of 16). */
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

/* h += alpha * x, 16 taps per iteration with 4-wide and scalar tails (elementwise, so the width
 * does not change the result). */
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

/* Fused NLMS update and next dot: each updated tap feeds the next dot; keeps da_dot_f32's four
 * accumulators. */
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
#else
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
#endif

/* First maximum index (ties resolve to the lowest). */
static int da_max_square_peak_index(const float *h, int n) {
    return sk_argmax_sq_f32(h, n);
}

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

/* 48 kHz -> 16 kHz pre-decimation anti-alias filter (decimate by 3): elliptic lowpass, order 7
 * (4 sections), passband edge 6800 Hz (1 dB ripple), stopband edge 8000 Hz (60 dB). Coefficients
 * use the {b0, b1, b2, a1, a2} per-section convention of da_biquad_process1(). */
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

/* Anti-alias stage: elliptic LP, 3 sections. */
static void da_biquad_init_antialias(DaBiquad *bq) {
    int s;
    bq->n_sections = 3;
    for (s = 0; s < 3; ++s) {
        bq->b[s][0] = DA_LP_B[s][0]; bq->b[s][1] = DA_LP_B[s][1]; bq->b[s][2] = DA_LP_B[s][2];
        bq->a[s][0] = DA_LP_A[s][0]; bq->a[s][1] = DA_LP_A[s][1];
    }
    da_biquad_reset(bq);
}

/* Noise-reduction stage: Butterworth HP at 1 kHz. */
static void da_biquad_init_noise_reduction(DaBiquad *bq) {
    bq->n_sections = 1;
    bq->b[0][0] = DA_HP_B[0][0]; bq->b[0][1] = DA_HP_B[0][1]; bq->b[0][2] = DA_HP_B[0][2];
    bq->a[0][0] = DA_HP_A[0][0]; bq->a[0][1] = DA_HP_A[0][1];
    da_biquad_reset(bq);
}

/* Process one sample through the cascade (direct-form II transposed, float32):
 * out = b0*x + z0; z0 = b1*x - a1*out + z1; z1 = b2*x - a2*out. */
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

/* Filter n_in samples through bq (state carries across calls; every input sample is filtered)
 * and keep every third post-filter sample with phase continuity across calls. Returns the number of
 * samples written to out (<= out_cap, a defensive bound). */
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

static void da_decimator_init(DaDecimator *dec) {
    da_biquad_init_antialias(&dec->anti_alias);
    da_biquad_init_noise_reduction(&dec->noise_reduction);
}

/* Generic cascade walk over a 64-sample block: every sample goes through both cascades and the
 * output is kept on the first sample of each group of 4. Used when the cascade is not the 3-LP +
 * 1-HP shape da_decimator_init builds. */
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

/* Decimate a 64-sample block to a 16-sample sub-block: all 64 samples run through the cascades
 * (state carries) and every 4th post-HP sample is kept. For the shipped 3-LP + 1-HP shape the
 * coefficients and states are held in locals; other shapes take the generic walk. */
static void da_decimator_decimate(DaDecimator *dec, const float *in_block, float *out_sub) {
    DaBiquad *lp = &dec->anti_alias;
    DaBiquad *hp = &dec->noise_reduction;

    /* Shape test first: only declared sections are initialised. */
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

/* The render and capture decimators run back to back, share no data and carry the same
 * coefficients, so they run as the two lanes of one vector with the scalar expressions of
 * da_decimator_decimate (no fused multiply-add). Off-shape cascades or differing coefficients take
 * two single calls. */
#if SK_HAVE_NEON
static int da_biquad_same_coefficients(const DaBiquad *a, const DaBiquad *b) {
    /* Only the declared sections are part of the filter. */
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

/* The 48 kHz -> 16 kHz cascades of capture and render share coefficients and run as the two
 * lanes of one vector (no fused multiply-add); cascades of another shape or with differing
 * coefficients take two single calls. */
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

static void da_ring_reset(DaRing *r) {
    memset(r->buffer, 0, (size_t)r->capacity * sizeof(float));
    r->write = 0;
}

static void da_ring_push(DaRing *r, const float *sub, int n) {
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

/* gather_back: `length` samples in time-reversed order starting at newest - start_offset;
 * out[0] is the sample start_offset ago. */
static void da_ring_gather_back(const DaRing *r, int start_offset, int length, float *out) {
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

static int da_matched_filter_core(const DaRing *ring, int alignment_shift_back,
                                  float x2_sum_threshold, float smoothing,
                                  const float *y, float *h, float *error_sum_out) {
    float error_sum = 0.0f;
    int filters_updated = 0;
    int i;
    /* All sub-block windows of this call are subranges of one contiguous span: window_i[k] =
     * span[(SUB_BLOCK_SIZE - 1 - i) + k], with span = gather_back(alignment_shift_back, FILTER_SIZE
     * + SUB_BLOCK_SIZE - 1). The deepest sample touched is covered by DA_RING_CAPACITY_FOR(n). */
    float span[DA_FILTER_SIZE + DA_SUB_BLOCK_SIZE - 1];
    da_ring_gather_back(ring, alignment_shift_back,
                        DA_FILTER_SIZE + DA_SUB_BLOCK_SIZE - 1, span);
    /* Sliding x2_sum: moving from window i to i+1 adds span[SB-2-i] and drops span[(SB-1-i)+512];
     * initialised once for i = 0. */
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
            float in_v  = x_window[0];
            float out_v = x_window[DA_FILTER_SIZE];
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
            /* alpha = smoothing * e / x2_sum in float32; h += alpha * x. */
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

/* Row accessors for the pool-carved banks (strides 2048 B and 512 B, both 16-byte aligned). */
static float *da_mf_row(const DaMatchedFilter *mf, int n) {
    return mf->filters + (size_t)n * DA_FILTER_SIZE;
}
static float *da_mf_acc(const DaMatchedFilter *mf, int n) {
    return mf->accumulated_error + (size_t)n * DA_ACC_ERR_SIZE;
}

/* num_filters is the runtime bank size; the pool holds exactly that many rows and every carved
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

/* MatchedFilter.update without the pre-echo onset search (the reported delay is always the
 * dominant peak), so the accumulated-error and instantaneous-error buffers keep their initialised
 * values. */
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

    /* The loop covers exactly the rows the pool holds; it is the entire search cost, so a short
     * bank saves compute as well as memory. */
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
        /* number_pre_echo_updates counts, saturating at DA_PRE_ECHO_UPDATES_TO_REPORT, the blocks
         * on which the winner filter repeated over a non-degenerate anchor (1 / 32768^2 in float
         * scale). It is zeroed only by init and a full reset. */
        if (mf->last_detected_best_lag_filter == winner_index
            && error_sum_anchor > 1.0f / (32768.0f * 32768.0f)
            && mf->number_pre_echo_updates < DA_PRE_ECHO_UPDATES_TO_REPORT) {
            mf->number_pre_echo_updates += 1;
        }
        mf->last_detected_best_lag_filter = winner_index;
    }
}

/* Incremental maintenance of (*candidate, *candidate_valid) as the first index of the maximum of
 * histogram[0..hist_size). It performs both histogram writes (decrement at A, increment at B);
 * callers handle A == B && *candidate_valid (a net-zero update) themselves.
 * When *candidate_valid, c = *candidate is the first index of the maximum M = histogram[c]:
 *  - !candidate_valid: mutate, then one authoritative sk_argmax_i32 rescan.
 *  - A == c: if B's new value exceeds the old M, B is the unique argmax; otherwise rescan.
 *  - Otherwise (c, M) is unchanged; B takes over if its new value exceeds M, or ties M at a lower
 *    index. */
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
    }
}

static void da_highest_peak_reset(DaHighestPeak *hp) {
    memset(hp->histogram, 0, (size_t)hp->hist_size * sizeof(int));
    memset(hp->ring, 0, (size_t)hp->window * sizeof(int));
    hp->ring_index = 0;
    /* The candidate is kept across a reset; candidate_valid is cleared so the next aggregate() call
     * rescans instead of trusting a pre-reset candidate. */
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
        /* Evicted bin == inserted bin: net-zero, the candidate stays correct. */
        return;
    }
    da_argmax_incremental_update(hp->histogram, hp->hist_size,
                                 old_lag, clamped_lag,
                                 &hp->candidate, &hp->candidate_valid);
}

/* Carved and reset like the other state but never fed: the reported delay is the dominant peak. */

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

/* Lag aggregation. Returns 1 if an estimate is produced (writes
 * *quality and *delay in down-sampled samples). */
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
        /* The reported delay is the dominant histogram peak, so `quality` describes the delay it is
         * attached to. The earliest-onset candidate has no consumer here (it would feed a render
         * delay controller), so the pre-echo aggregator carries no live candidate. */
        *delay_out = candidate;
        return 1;
    }
    return 0;
}

static void da_clockdrift_init(DaClockdrift *cd, int sample_rate) {
    cd->delay_history[0] = cd->delay_history[1] = cd->delay_history[2] = 0;
    cd->level = CLOCKDRIFT_NONE;
    cd->stability_counter = 0;
    /* 30 s of inner 64-sample blocks (tick period 64 / sample_rate), via aec_ms_to_hops(). */
    cd->stability_reset_hops =
        aec_ms_to_hops(30000.0f, DA_BLOCK_SIZE, sample_rate);
}

static void da_clockdrift_update(DaClockdrift *cd, int delay_estimate) {
    int d1, d2, d3, probable_up, verified_up, probable_down, verified_down;
    if (delay_estimate == cd->delay_history[0]) {
        /* Capped at stability_reset_hops + 1 to avoid signed overflow; only the `>` check below
         * reads it. */
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

/* Ring entries in use by both aggregators: 1 s of 64-sample inner blocks at the feed rate,
 * clamped to the carved capacity DA_HIST_WINDOW. */
static int da_hist_window_for_rate(int sample_rate) {
    int w = aec_ms_to_hops(1000.0f, DA_BLOCK_SIZE, sample_rate);
    if (w < 1) w = 1;
    if (w > DA_HIST_WINDOW) w = DA_HIST_WINDOW;
    return w;
}

/* Zero the inner signal chain (decimators and render ring). da_decimator_init
 * only zeroes coefficients/state, so re-running it is a safe, idempotent
 * reset. */
static void da_estimator_clear_chain(DaEstimator *e) {
    da_decimator_init(&e->capture_decimator);
    da_decimator_init(&e->render_decimator);
#if SK_HAVE_NEON
    e->decimators_same_coefficients =
        da_decimator_same_coefficients(&e->capture_decimator, &e->render_decimator);
#endif
    da_ring_reset(&e->render_ring);
}

static void da_estimator_init(DaEstimator *e, int sample_rate, int num_filters) {
    da_estimator_clear_chain(e);
    da_matched_filter_init(&e->matched_filter, num_filters);
    /* Histogram window = 1 s of inner blocks at the feed rate (250 at 16 kHz and at 48 kHz, 125 at
     * 8 kHz). */
    da_aggregator_init(&e->aggregator, da_hist_window_for_rate(sample_rate));
    da_clockdrift_init(&e->clockdrift, sample_rate);
    /* Half a second of inner blocks, computed from the rate. */
    e->consistent_estimate_threshold =
        aec_ms_to_hops(500.0f, DA_BLOCK_SIZE, sample_rate);
    e->old_agg_valid = 0;
    e->old_agg_delay = 0;
    e->old_agg_quality = DELAY_QUALITY_COARSE;
    e->consistent_estimate_counter = 0;
    e->pending_count = 0;
}

static void da_estimator_reset_internal(DaEstimator *e, int reset_lag_aggregator,
                                        int reset_delay_confidence) {
    if (reset_lag_aggregator)
        da_aggregator_reset(&e->aggregator, reset_delay_confidence);
    da_matched_filter_reset(&e->matched_filter, reset_lag_aggregator);
    e->old_agg_valid = 0;
    e->consistent_estimate_counter = 0;
}

/* Public reset: clears the whole delay-estimator signal chain (matched filter, aggregator,
 * decimators, ring buffer, pending samples) so stale pre-reset audio cannot contaminate the first
 * correlation windows; clockdrift is kept (a 30 s hardware drift estimate). The periodic partial
 * reset on the consistent-estimate counter leaves the signal chain alone. */
static void da_estimator_reset(DaEstimator *e, int reset_delay_confidence) {
    da_estimator_reset_internal(e, 1, reset_delay_confidence);
    da_estimator_clear_chain(e);
    e->pending_count = 0;
}

/* Process one 64-sample inner block. Returns 1 if an estimate was produced (writes *quality and
 * *delay in raw samples). With run_matched_filter = 0 the block is fed but not analysed: both
 * decimators still run and the decimated render sub-block is still pushed into the ring (the
 * matched filter must see gapless audio); the matched-filter update, aggregation, clockdrift and
 * stability bookkeeping are skipped, leaving all decision state as the previous analysed block set
 * it. */
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
    if (!run_matched_filter) return 0;
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

    if (e->old_agg_valid && agg_valid && e->old_agg_delay == agg_delay)
        e->consistent_estimate_counter += 1;
    else
        e->consistent_estimate_counter = 0;
    e->old_agg_valid = agg_valid;
    e->old_agg_delay = agg_delay;
    e->old_agg_quality = agg_quality;

    if (e->consistent_estimate_counter > e->consistent_estimate_threshold)
        da_estimator_reset_internal(e, 0, 0);

    if (agg_valid) {
        *quality_out = agg_quality;
        *delay_out = agg_delay;
        return 1;
    }
    return 0;
}

/* True when the n floats at `in` share bytes with `pend`; such input must not go through memcpy. */
#define DA_OVERLAPS(in, n, pend)                                             \
    ((n) > 0 && (uintptr_t)(in) < (uintptr_t)(pend) + sizeof(pend) &&       \
     (uintptr_t)(pend) < (uintptr_t)(in) + (size_t)(n) * sizeof(float))

/* Buffer the raw hop into the edge-chunker and drain 64-sample blocks. Returns 1 if any estimate
 * was produced during the drain (the latest one is written to *quality and *delay). The hop is
 * appended in runs up to the next block boundary; an input overlapping a pending array is appended
 * one sample at a time. */
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

/* Pool layout: one description of the block shared by delay_aec_get_mem_size() and
 * delay_aec_init(). The carve walks the fields in the same order with the same ALIGN16 bumps and
 * must land exactly on `total`. */
typedef struct {
    int    num_filters;     /* validated: [1, DA_NUM_FILTERS] */
    int    ring_capacity;
    int    hp_hist_size;
    int    pe_hist_size;
    int    resample_cap;    /* 0 unless the 48 kHz sidechain is built */
    size_t total;
} DaLayout;

/* Out-of-range bank sizes are rejected, not clamped; the query and init share this check. */
static int da_valid_num_filters(int num_filters) {
    return num_filters >= 1 && num_filters <= DA_NUM_FILTERS;
}

/* Returns 1 with L populated, or 0 on invalid input or size overflow. All arithmetic uses the
 * saturating ck_* helpers, so one MEM_SIZE_INVALID() test at the end catches overflow. */
static int da_compute_layout(int sample_rate, int hop_size, int num_filters,
                             DaLayout *L) {
    size_t t = 0;
    int n;
    if (!L) return 0;
    if (sample_rate <= 0 || hop_size <= 0) return 0;
    if (!da_valid_num_filters(num_filters)) return 0;

    n = num_filters;
    L->num_filters   = n;
    L->ring_capacity = DA_RING_CAPACITY_FOR(n);
    L->hp_hist_size  = DA_HP_HIST_SIZE_FOR(n);
    L->pe_hist_size  = DA_PE_HIST_SIZE_FOR(n);
    L->resample_cap  = (sample_rate == 48000)
                     ? DA_RESAMPLE48_CAP_FOR(hop_size) : 0;

    t = ck_field_size(t, ck_mul_size((size_t)n, (size_t)DA_FILTER_SIZE),
                      sizeof(float));
    t = ck_field_size(t, ck_mul_size((size_t)n, (size_t)DA_ACC_ERR_SIZE),
                      sizeof(float));
    t = ck_field_size(t, (size_t)DA_ACC_ERR_SIZE, sizeof(float));
    t = ck_field_size(t, (size_t)L->ring_capacity, sizeof(float));
    t = ck_field_size(t, (size_t)L->hp_hist_size, sizeof(int));
    t = ck_field_size(t, (size_t)DA_HIST_WINDOW, sizeof(int));
    t = ck_field_size(t, (size_t)L->pe_hist_size, sizeof(int));
    t = ck_field_size(t, (size_t)DA_HIST_WINDOW, sizeof(int));
    if (L->resample_cap > 0) {
        t = ck_field_size(t, (size_t)L->resample_cap, sizeof(float));
        t = ck_field_size(t, (size_t)L->resample_cap, sizeof(float));
    }

    if (MEM_SIZE_INVALID(t)) return 0;
    L->total = t;
    return 1;
}

/* Bind every pool pointer and its runtime size (pure pointer bumps; da_estimator_init() does the
 * initialisation). Returns the byte count consumed. */
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
    /* Reject a misaligned base: the NEON matched-filter kernels load the 512-tap banks aligned. */
    if (!MEM_IS_ALIGNED16(mem)) return -1;
    if (!da_compute_layout(sample_rate, hop_size, num_filters, &L)) return -1;
    if (bytes < L.total) return -1;

    memset(d, 0, sizeof(*d));
    used = da_delay_carve(d, (uint8_t *)mem, &L);
    /* The carve must land exactly on the queried byte count; a mismatch means the two field walks
     * have drifted. */
    if (used != L.total) return -1;

    /* The estimator's clockdrift and consistent-estimate constants derive from the sample_rate it
     * is fed: 16000 for the 48 kHz sidechain, otherwise the native rate. */
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

/* Driven by aec.c's duty-cycle state machine. With run_matched_filter = 0 the hop is decimated and
 * pushed into the ring (buffers stay gapless) but no matched-filter or aggregator work runs, so
 * latest_* is unchanged. */
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
        /* Anti-alias filter and decimate both channels to the estimator's 16 kHz feed rate; the two
         * chains share one phase and one output count. */
        int phase = d->resample48.phase;
        hop_fed = da_resample48_pair(&d->resample48.near_lp, near, d->near16_scratch,
                                     &d->resample48.far_lp, far, d->far16_scratch,
                                     hop, d->resample_cap, &phase);
        d->resample48.phase = phase;
        near_fed = d->near16_scratch;
        far_fed = d->far16_scratch;
    }
    produced = da_estimator_estimate_delay(&d->est, far_fed, near_fed, hop_fed,
                                           run_matched_filter, &q, &delay);
    if (produced) {
        d->latest_valid = 1;
        d->latest_delay = delay;
        d->latest_quality = q;
        /* estimate_count is a public counter with no internal threshold, so it saturates at
         * INT_MAX. */
        if (d->estimate_count < INT_MAX) d->estimate_count += 1;
        if (delay != prev_delay) emitted_new = 1;
    }
    return emitted_new;
}

int delay_aec_estimated_delay(const DelayAec *d) {
    if (!d->latest_valid) return -1;
    /* latest_delay is in the estimator's 16 kHz-equivalent domain; rescale to the caller's rate. */
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
