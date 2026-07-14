/**
 * aec_simd_kernels.h - AEC-specific NEON/scalar bit-exact DSP micro-kernels.
 *
 * The AEC-only half of the shared micro-kernel split (generic kernels stay
 * in audio_common, AEC-specific kernels move here): kernels whose
 * doc-comments mirror one specific
 * AEC/c_impl source shape and are not used anywhere outside this repo --
 * the numpy |z|/|z|**2 magnitude family, the PBFDAF/PBFDKF per-bin
 * filter-weight updates, the AEC3-post coherence-gate/CNG per-bin
 * trackers, and the four numpy-pairwise-sum float32 reduction trees used
 * by the filter-quality/audit modules. The algorithm-agnostic kernels
 * (min/clip/gain-apply/complex-add/sq-scale/EMA/fast_sqrt) that the wider
 * Audio_ALG pipeline also consumes directly stay in
 * audio_common/include/simd_kernels.h and are pulled in transitively by
 * the #include below -- every AEC source that used to `#include
 * "simd_kernels.h"` now includes this header instead and gets both sets.
 *
 * ─────────────────────────── Bit-exactness contract ──────────────────────
 * This header inherits the FULL contract verbatim from simd_kernels.h's
 * top-of-file comment (the NEON-vs-scalar byte-identity requirement, the
 * "replicate the scalar op SEQUENCE -- which ops are fused, which are
 * separate, and in what order -- lane-for-lane" rule, the ban on estimate
 * instructions (vrsqrteq_f32/vrecpeq_f32 and friends), and the min/clip
 * compare+select-not-vminq/vmaxq signed-zero tie-break rule) -- see that
 * file for the complete text, this is a pointer to it, not a restatement.
 *
 * Restated only for the FMA-fusion call this file's own kernels make (the
 * general rule lives in simd_kernels.h; these are this file's instances of
 * it):
 *
 *   Uses explicit fmaf()/vfmaq_f32 (mirrors an explicit `fmaf(...)` call in
 *   the AEC source -- always fused, with or without -ffp-contract, since an
 *   explicit fmaf() call requests FMA directly rather than relying on
 *   contraction of separate ops):
 *     - the cabs_np/cmag2_np magnitude helper's `ratio*ratio + 1.0f` term
 *       (kernels 1, 2, 3, 5's internal magnitude computation)
 *     - sk_cmac_np_f32          (kernel 6)
 *     - sk_wupdate_nlms_f32's grad = err * conj(X) term (kernel 7)
 *     - sk_wupdate_kf_f32's final W += K_scaled * error_spec term (kernel 8)
 *
 *   Uses separate mul then add, NEVER fused (mirrors the source NOT calling
 *   fmaf for that particular step -- needs -ffp-contract=off to stay this
 *   way, the same flag AEC/c_impl's own Makefile already mandates for every
 *   TU that includes this header):
 *     - sk_ema_cmag2_f32's outer alpha*state+beta*mag2 combine (kernel 5)
 *     - sk_wupdate_nlms_f32's final W += mu_eff*grad combine   (kernel 7)
 *     - sk_wupdate_kf_f32's K = mu*conj(X) and K *= mu_scale steps
 *       (kernel 8)
 *
 * ───────────────────────────────── Style ──────────────────────────────────
 * Header-only, C99, static inline -- same convention as simd_kernels.h and
 * fast_math.h. `#include "simd_kernels.h"` first: this pulls in Complex
 * (fft_wrapper.h), <math.h>/<stdint.h>/<stddef.h>, the SK_HAVE_NEON /
 * SIMD_KERNELS_FORCE_SCALAR knobs, and the generic kernel set (sk_min_f32,
 * sk_capply_gain_f32, sk_cadd_f32, sk_sq_scale_f32, sk_ema_f32,
 * sk_fast_sqrt_f32) that Audio_ALG's own pipeline calls directly.
 * SK_HAVE_NEON is NOT redefined here -- it is inherited as-is from
 * simd_kernels.h, so both files always agree on whether NEON bodies are
 * compiled in.
 *
 * No `restrict` anywhere, same aliasing rule as simd_kernels.h: pointers
 * are assumed non-aliasing except where a kernel's own comment documents
 * an in-place case.
 */

#ifndef AEC_SIMD_KERNELS_H
#define AEC_SIMD_KERNELS_H

#include "simd_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════ shared per-element helpers ═══════════════════
 * Moved here from simd_kernels.h together with the kernels that use them
 * (kernels 1, 2, 3, 5 below): used by every scalar kernel AND by every
 * NEON kernel's scalar tail, so the tail matches the fully-scalar path
 * bit-for-bit by construction. sk__fast_sqrt_elem (used by kernel 15)
 * stays in simd_kernels.h since kernel 15 itself stays there. */

/* numpy complex64 |z| — scaled-hypot with an explicit fmaf, verbatim from
 * AEC/c_impl/src/aec3_post.c cabs_np() / pbfdkf.c cmag2_np()'s `m`. */
static inline float sk__cabs_np_elem(float re, float im) {
    float ar = re < 0.0f ? -re : re;
    float ai = im < 0.0f ? -im : im;
    float larger  = ar > ai ? ar : ai;
    float smaller = ar > ai ? ai : ar;
    if (larger == 0.0f) return 0.0f;
    {
        float ratio = smaller / larger;
        return larger * sqrtf(fmaf(ratio, ratio, 1.0f));
    }
}

/* numpy |z|**2, verbatim from pbfdkf.c cmag2_np() / aec.c cmag2_c(). */
static inline float sk__cmag2_np_elem(float re, float im) {
    float m = sk__cabs_np_elem(re, im);
    return m * m;
}

#if SK_HAVE_NEON
/* 4-lane version of sk__cabs_np_elem: `larger==0` lanes computed as 0/0 =
 * NaN through the divide/sqrt, then bslq-selected to an exact +0.0f — same
 * final bits as the scalar early-return, just reached without branching. */
static inline float32x4_t sk__cabs_np_neon4(float32x4_t re, float32x4_t im) {
    float32x4_t ar = vabsq_f32(re);
    float32x4_t ai = vabsq_f32(im);
    float32x4_t larger  = vmaxq_f32(ar, ai);   /* both operands >= +0.0f: no
                                                 * signed-zero tie is possible
                                                 * here (see header comment). */
    float32x4_t smaller = vminq_f32(ar, ai);
    float32x4_t ratio = vdivq_f32(smaller, larger);
    float32x4_t m = vmulq_f32(larger,
                       vsqrtq_f32(vfmaq_f32(vdupq_n_f32(1.0f), ratio, ratio)));
    uint32x4_t is_zero = vceqq_f32(larger, vdupq_n_f32(0.0f));
    return vbslq_f32(is_zero, vdupq_n_f32(0.0f), m);
}

static inline float32x4_t sk__cmag2_np_neon4(float32x4_t re, float32x4_t im) {
    float32x4_t m = sk__cabs_np_neon4(re, im);
    return vmulq_f32(m, m);
}
#endif /* SK_HAVE_NEON */

/* ═══════════════════════════════ kernel 1 ══════════════════════════════════
 * sk_cabs_np_f32 — out[i] = numpy |z[i]|. */

static inline void sk_cabs_np_f32_scalar(const Complex *z, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__cabs_np_elem(z[i].r, z[i].i);
}

#if SK_HAVE_NEON
static inline void sk_cabs_np_f32(const Complex *z, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t v = vld2q_f32((const float *)(z + i));
        float32x4_t r = sk__cabs_np_neon4(v.val[0], v.val[1]);
        vst1q_f32(out + i, r);
    }
    for (; i < n; ++i) out[i] = sk__cabs_np_elem(z[i].r, z[i].i);
}
#else
static inline void sk_cabs_np_f32(const Complex *z, float *out, int n) {
    sk_cabs_np_f32_scalar(z, out, n);
}
#endif

/* ═══════════════════════════════ kernel 2 ══════════════════════════════════
 * sk_cmag2_np_f32 — out[i] = numpy |z[i]|**2. */

static inline void sk_cmag2_np_f32_scalar(const Complex *z, float *out, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = sk__cmag2_np_elem(z[i].r, z[i].i);
}

#if SK_HAVE_NEON
static inline void sk_cmag2_np_f32(const Complex *z, float *out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t v = vld2q_f32((const float *)(z + i));
        float32x4_t m2 = sk__cmag2_np_neon4(v.val[0], v.val[1]);
        vst1q_f32(out + i, m2);
    }
    for (; i < n; ++i) out[i] = sk__cmag2_np_elem(z[i].r, z[i].i);
}
#else
static inline void sk_cmag2_np_f32(const Complex *z, float *out, int n) {
    sk_cmag2_np_f32_scalar(z, out, n);
}
#endif

/* ═══════════════════════════════ kernel 3 ══════════════════════════════════
 * sk_cmag2_np_acc_f32 — acc[i] += numpy |z[i]|**2 (pbfdaf.c x2_partition_sum
 * / far-power accumulation pattern: acc[k] += cmag2_np(...)). */

static inline void sk_cmag2_np_acc_f32_scalar(const Complex *z, float *acc, int n) {
    int i;
    for (i = 0; i < n; ++i) acc[i] += sk__cmag2_np_elem(z[i].r, z[i].i);
}

#if SK_HAVE_NEON
static inline void sk_cmag2_np_acc_f32(const Complex *z, float *acc, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t v = vld2q_f32((const float *)(z + i));
        float32x4_t m2 = sk__cmag2_np_neon4(v.val[0], v.val[1]);
        float32x4_t a = vld1q_f32(acc + i);
        vst1q_f32(acc + i, vaddq_f32(a, m2));
    }
    for (; i < n; ++i) acc[i] += sk__cmag2_np_elem(z[i].r, z[i].i);
}
#else
static inline void sk_cmag2_np_acc_f32(const Complex *z, float *acc, int n) {
    sk_cmag2_np_acc_f32_scalar(z, acc, n);
}
#endif

/* ═══════════════════════════════ kernel 5 ══════════════════════════════════
 * sk_ema_cmag2_f32 — state[i] = alpha*state[i] + beta*cmag2_np(z[i]).
 *
 * Cross-checked against pbfdkf.c:366-385 (the actual far-power EMA call
 * site): the source is a plain two-branch EMA — a direct assignment on cold
 * start (`power[k] = cmag2_np(...)` when sum(power) < 1e-10 and far is
 * active), else `power[k] = a*power[k] + b*cmag2_np(...)`. That cold-start
 * branch is a CALLER-level condition on the aggregate power sum, not a
 * per-element algebraic form (e.g. not a `s += a*(x-s)` delta-EMA anywhere
 * in this call site) — so the per-element kernel below is the vanilla
 * two-term EMA already covered by kernel 4's shape, just with `x[i]`
 * replaced by `cmag2_np(z[i])`; no separate "delta-form" kernel is needed.
 * The outer combine is mul/mul/add, NOT fmaf, same discipline as kernel 4. */

static inline void sk_ema_cmag2_f32_scalar(float *state, const Complex *z,
                                            float alpha, float beta, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float mag2 = sk__cmag2_np_elem(z[i].r, z[i].i);
        state[i] = alpha * state[i] + beta * mag2;
    }
}

#if SK_HAVE_NEON
static inline void sk_ema_cmag2_f32(float *state, const Complex *z,
                                     float alpha, float beta, int n) {
    int i = 0;
    float32x4_t va = vdupq_n_f32(alpha), vb = vdupq_n_f32(beta);
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t zv = vld2q_f32((const float *)(z + i));
        float32x4_t mag2 = sk__cmag2_np_neon4(zv.val[0], zv.val[1]);
        float32x4_t s = vld1q_f32(state + i);
        float32x4_t r = vaddq_f32(vmulq_f32(va, s), vmulq_f32(vb, mag2));
        vst1q_f32(state + i, r);
    }
    for (; i < n; ++i) {
        float mag2 = sk__cmag2_np_elem(z[i].r, z[i].i);
        state[i] = alpha * state[i] + beta * mag2;
    }
}
#else
static inline void sk_ema_cmag2_f32(float *state, const Complex *z,
                                     float alpha, float beta, int n) {
    sk_ema_cmag2_f32_scalar(state, z, alpha, beta, n);
}
#endif

/* ═══════════════════════════════ kernel 6 ══════════════════════════════════
 * sk_cmac_np_f32 — acc[i] += w[i] * x[i] (numpy complex64 multiply, FMA
 * form), verbatim from pbfdkf.c's echo_spec accumulation:
 *   acc[k].r += fmaf(wr, xr, -(wi * xi));
 *   acc[k].i += fmaf(wr, xi,  (wi * xr)); */

static inline void sk_cmac_np_f32_scalar(Complex *acc, const Complex *w,
                                          const Complex *x, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float wr = w[i].r, wi = w[i].i, xr = x[i].r, xi = x[i].i;
        acc[i].r += fmaf(wr, xr, -(wi * xi));
        acc[i].i += fmaf(wr, xi,  (wi * xr));
    }
}

#if SK_HAVE_NEON
static inline void sk_cmac_np_f32(Complex *acc, const Complex *w,
                                   const Complex *x, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t wv = vld2q_f32((const float *)(w + i));
        float32x4_t wr = wv.val[0], wi = wv.val[1];
        float32x4x2_t xv = vld2q_f32((const float *)(x + i));
        float32x4_t xr = xv.val[0], xi = xv.val[1];
        float32x4_t re_prod = vfmaq_f32(vnegq_f32(vmulq_f32(wi, xi)), wr, xr);
        float32x4_t im_prod = vfmaq_f32(vmulq_f32(wi, xr), wr, xi);
        float32x4x2_t av = vld2q_f32((const float *)(acc + i));
        float32x4x2_t rv;
        rv.val[0] = vaddq_f32(av.val[0], re_prod);
        rv.val[1] = vaddq_f32(av.val[1], im_prod);
        vst2q_f32((float *)(acc + i), rv);
    }
    for (; i < n; ++i) {
        float wr = w[i].r, wi = w[i].i, xr = x[i].r, xi = x[i].i;
        acc[i].r += fmaf(wr, xr, -(wi * xi));
        acc[i].i += fmaf(wr, xi,  (wi * xr));
    }
}
#else
static inline void sk_cmac_np_f32(Complex *acc, const Complex *w,
                                   const Complex *x, int n) {
    sk_cmac_np_f32_scalar(acc, w, x, n);
}
#endif

/* ═══════════════════════════════ kernel 7 ══════════════════════════════════
 * sk_wupdate_nlms_f32 — the PBFDAF NLMS coarse-filter W-update, per-bin, for
 * ONE partition (the outer loop-over-partitions stays scalar at the call
 * site; this kernel vectorizes over the K frequency bins of a single
 * partition). Verbatim from pbfdkf.c:504-519:
 *   cxr = xr, cxi = -xi;                                  // conj(X)
 *   gr = fmaf(er, cxr, -(ei * cxi));
 *   gi = fmaf(er, cxi,  (ei * cxr));                      // grad = err*conj(X)
 *   W[k].r += mu_eff[k] * gr;                             // NOT fmaf
 *   W[k].i += mu_eff[k] * gi;                             // NOT fmaf
 * The grad computation is explicit fmaf (fused); the final mu_eff*grad
 * combine is a plain separate multiply then add (needs -ffp-contract=off). */

static inline void sk_wupdate_nlms_f32_scalar(Complex *W, const Complex *X,
                                               const Complex *err,
                                               const float *mu_eff, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float er = err[i].r, ei = err[i].i;
        float xr = X[i].r, xi = X[i].i;
        float cxr = xr, cxi = -xi;
        float gr = fmaf(er, cxr, -(ei * cxi));
        float gi = fmaf(er, cxi,  (ei * cxr));
        W[i].r += mu_eff[i] * gr;
        W[i].i += mu_eff[i] * gi;
    }
}

#if SK_HAVE_NEON
static inline void sk_wupdate_nlms_f32(Complex *W, const Complex *X,
                                        const Complex *err,
                                        const float *mu_eff, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t ev = vld2q_f32((const float *)(err + i));
        float32x4_t er = ev.val[0], ei = ev.val[1];
        float32x4x2_t xv = vld2q_f32((const float *)(X + i));
        float32x4_t xr = xv.val[0], xi = xv.val[1];
        float32x4_t cxr = xr;
        float32x4_t cxi = vnegq_f32(xi);
        float32x4_t gr = vfmaq_f32(vnegq_f32(vmulq_f32(ei, cxi)), er, cxr);
        float32x4_t gi = vfmaq_f32(vmulq_f32(ei, cxr), er, cxi);
        float32x4_t mu = vld1q_f32(mu_eff + i);
        float32x4x2_t wv = vld2q_f32((const float *)(W + i));
        float32x4x2_t rv;
        rv.val[0] = vaddq_f32(wv.val[0], vmulq_f32(mu, gr));
        rv.val[1] = vaddq_f32(wv.val[1], vmulq_f32(mu, gi));
        vst2q_f32((float *)(W + i), rv);
    }
    for (; i < n; ++i) {
        float er = err[i].r, ei = err[i].i;
        float xr = X[i].r, xi = X[i].i;
        float cxr = xr, cxi = -xi;
        float gr = fmaf(er, cxr, -(ei * cxi));
        float gi = fmaf(er, cxi,  (ei * cxr));
        W[i].r += mu_eff[i] * gr;
        W[i].i += mu_eff[i] * gi;
    }
}
#else
static inline void sk_wupdate_nlms_f32(Complex *W, const Complex *X,
                                        const Complex *err,
                                        const float *mu_eff, int n) {
    sk_wupdate_nlms_f32_scalar(W, X, err, mu_eff, n);
}
#endif

/* ═══════════════════════════════ kernel 8 ══════════════════════════════════
 * sk_wupdate_kf_f32 — the PBFDKF Kalman W-update, per-bin, for ONE
 * partition. Verbatim from pbfdkf.c:861-874:
 *   kr = mu[k] * xr;             ki = -(mu[k] * xi);        // K = mu*conj(X)
 *   ksr = kr * mu_scale[k];      ksi = ki * mu_scale[k];     // K *= mu_scale
 *   W[k].r += fmaf(ksr, er, -(ksi * ei));
 *   W[k].i += fmaf(ksr, ei,  (ksi * er));                   // W += K*err
 * The K/K*=mu_scale steps are plain separate multiplies (no add involved,
 * so no fusion risk there either way); the final `W += fmaf(...)` step is
 * itself two operations — the fmaf (fused, explicit) producing the K*err
 * increment, THEN a separate plain add onto the existing W[k] (the source
 * writes `+=`, not a second fmaf) — replicated here as vfmaq_f32 followed
 * by a separate vaddq_f32. */

static inline void sk_wupdate_kf_f32_scalar(Complex *W, const Complex *X,
                                             const Complex *err,
                                             const float *mu,
                                             const float *mu_scale, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float xr = X[i].r, xi = X[i].i;
        float kr = mu[i] * xr;
        float ki = -(mu[i] * xi);
        float ksr = kr * mu_scale[i];
        float ksi = ki * mu_scale[i];
        float er = err[i].r, ei = err[i].i;
        W[i].r += fmaf(ksr, er, -(ksi * ei));
        W[i].i += fmaf(ksr, ei,  (ksi * er));
    }
}

#if SK_HAVE_NEON
static inline void sk_wupdate_kf_f32(Complex *W, const Complex *X,
                                      const Complex *err,
                                      const float *mu,
                                      const float *mu_scale, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t xv = vld2q_f32((const float *)(X + i));
        float32x4_t xr = xv.val[0], xi = xv.val[1];
        float32x4_t muv = vld1q_f32(mu + i);
        float32x4_t msv = vld1q_f32(mu_scale + i);
        float32x4_t kr = vmulq_f32(muv, xr);
        float32x4_t ki = vnegq_f32(vmulq_f32(muv, xi));
        float32x4_t ksr = vmulq_f32(kr, msv);
        float32x4_t ksi = vmulq_f32(ki, msv);
        float32x4x2_t ev = vld2q_f32((const float *)(err + i));
        float32x4_t er = ev.val[0], ei = ev.val[1];
        float32x4_t dr = vfmaq_f32(vnegq_f32(vmulq_f32(ksi, ei)), ksr, er);
        float32x4_t di = vfmaq_f32(vmulq_f32(ksi, er), ksr, ei);
        float32x4x2_t wv = vld2q_f32((const float *)(W + i));
        float32x4x2_t rv;
        rv.val[0] = vaddq_f32(wv.val[0], dr);
        rv.val[1] = vaddq_f32(wv.val[1], di);
        vst2q_f32((float *)(W + i), rv);
    }
    for (; i < n; ++i) {
        float xr = X[i].r, xi = X[i].i;
        float kr = mu[i] * xr;
        float ki = -(mu[i] * xi);
        float ksr = kr * mu_scale[i];
        float ksi = ki * mu_scale[i];
        float er = err[i].r, ei = err[i].i;
        W[i].r += fmaf(ksr, er, -(ksi * ei));
        W[i].i += fmaf(ksr, ei,  (ksi * er));
    }
}
#else
static inline void sk_wupdate_kf_f32(Complex *W, const Complex *X,
                                      const Complex *err,
                                      const float *mu,
                                      const float *mu_scale, int n) {
    sk_wupdate_kf_f32_scalar(W, X, err, mu, mu_scale, n);
}
#endif

/* ═══════════════════════════════ kernel 13 ═════════════════════════════════
 * sk_pairwise_sum_f32 — numpy-1.26-bit-exact pairwise float32 sum, verbatim
 * tree from AEC/c_impl/src/aec3_post.c pairwise_sum_f32() (n<=8 serial;
 * n<=128 8-accumulator leaf; recursive split with half -= half%8).
 *
 * NEON leaf math (verified by hand): q0 holds running column-sums for
 * acc[0..3], q1 for acc[4..7] (same 8-wide grouping as the scalar leaf).
 * `t = vpaddq_f32(q0,q1)` = [acc0+acc1, acc2+acc3, acc4+acc5, acc6+acc7].
 * `u = vpaddq_f32(t,t)` = [ (acc0+acc1)+(acc2+acc3), (acc4+acc5)+(acc6+acc7),
 * <repeat> ]. `u[0]+u[1]` therefore equals
 * `((acc0+acc1)+(acc2+acc3)) + ((acc4+acc5)+(acc6+acc7))` — the exact same
 * grouping as the scalar leaf's `r`. */

static inline float sk__pairwise_sum_leaf_scalar(const float *a, size_t n) {
    float acc[8];
    size_t i, j;
    for (j = 0; j < 8; ++j) acc[j] = a[j];
    for (i = 8; i + 8 <= n; i += 8)
        for (j = 0; j < 8; ++j) acc[j] += a[i + j];
    {
        float s = 0.0f, r;
        for (; i < n; ++i) s += a[i];
        r = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
          + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        return r + s;
    }
}

static inline float sk_pairwise_sum_f32_scalar(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) s += a[i];
        return s;
    }
    if (n <= 128) return sk__pairwise_sum_leaf_scalar(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_pairwise_sum_f32_scalar(a, half)
             + sk_pairwise_sum_f32_scalar(a + half, n - half);
    }
}

#if SK_HAVE_NEON
static inline float sk__pairwise_sum_leaf_neon(const float *a, size_t n) {
    float32x4_t q0 = vld1q_f32(a), q1 = vld1q_f32(a + 4);
    size_t i;
    for (i = 8; i + 8 <= n; i += 8) {
        q0 = vaddq_f32(q0, vld1q_f32(a + i));
        q1 = vaddq_f32(q1, vld1q_f32(a + i + 4));
    }
    {
        float s = 0.0f, r;
        for (; i < n; ++i) s += a[i];
        {
            float32x4_t t = vpaddq_f32(q0, q1);
            float32x4_t u = vpaddq_f32(t, t);
            r = vgetq_lane_f32(u, 0) + vgetq_lane_f32(u, 1);
        }
        return r + s;
    }
}

static inline float sk_pairwise_sum_f32(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) s += a[i];
        return s;
    }
    if (n <= 128) return sk__pairwise_sum_leaf_neon(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_pairwise_sum_f32(a, half) + sk_pairwise_sum_f32(a + half, n - half);
    }
}
#else
static inline float sk_pairwise_sum_f32(const float *a, size_t n) {
    return sk_pairwise_sum_f32_scalar(a, n);
}
#endif

/* ═══════════════════════════════ kernel 14 ═════════════════════════════════
 * sk_sum_sq_pairwise_f32 — same tree as kernel 13, over squared elements.
 * Verbatim from AEC/c_impl/src/aec3_post.c sum_sq_f32_pairwise() (lines
 * 394-423): identical structure/split/combine order, leaf accumulates
 * a[i+j]*a[i+j] instead of a[i+j]. */

static inline float sk__sum_sq_leaf_scalar(const float *a, size_t n) {
    float acc[8];
    size_t i, j;
    for (j = 0; j < 8; ++j) { float v = a[j]; acc[j] = v * v; }
    for (i = 8; i + 8 <= n; i += 8)
        for (j = 0; j < 8; ++j) { float v = a[i + j]; acc[j] += v * v; }
    {
        float s = 0.0f, r;
        for (; i < n; ++i) { float v = a[i]; s += v * v; }
        r = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
          + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        return r + s;
    }
}

static inline float sk_sum_sq_pairwise_f32_scalar(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) { float v = a[i]; s += v * v; }
        return s;
    }
    if (n <= 128) return sk__sum_sq_leaf_scalar(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_sum_sq_pairwise_f32_scalar(a, half)
             + sk_sum_sq_pairwise_f32_scalar(a + half, n - half);
    }
}

#if SK_HAVE_NEON
static inline float sk__sum_sq_leaf_neon(const float *a, size_t n) {
    float32x4_t a0 = vld1q_f32(a), a1 = vld1q_f32(a + 4);
    float32x4_t q0 = vmulq_f32(a0, a0), q1 = vmulq_f32(a1, a1);
    size_t i;
    for (i = 8; i + 8 <= n; i += 8) {
        float32x4_t b0 = vld1q_f32(a + i), b1 = vld1q_f32(a + i + 4);
        q0 = vaddq_f32(q0, vmulq_f32(b0, b0));
        q1 = vaddq_f32(q1, vmulq_f32(b1, b1));
    }
    {
        float s = 0.0f, r;
        for (; i < n; ++i) { float v = a[i]; s += v * v; }
        {
            float32x4_t t = vpaddq_f32(q0, q1);
            float32x4_t u = vpaddq_f32(t, t);
            r = vgetq_lane_f32(u, 0) + vgetq_lane_f32(u, 1);
        }
        return r + s;
    }
}

static inline float sk_sum_sq_pairwise_f32(const float *a, size_t n) {
    if (n <= 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) { float v = a[i]; s += v * v; }
        return s;
    }
    if (n <= 128) return sk__sum_sq_leaf_neon(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_sum_sq_pairwise_f32(a, half) + sk_sum_sq_pairwise_f32(a + half, n - half);
    }
}
#else
static inline float sk_sum_sq_pairwise_f32(const float *a, size_t n) {
    return sk_sum_sq_pairwise_f32_scalar(a, n);
}
#endif

/* ═══════════════════════════════ kernel 16 ═════════════════════════════════
 * sk_coherence_ema_gate_f32 — AEC3-post coherence Γ²(Ŷ,Y) EMA update + the
 * ERLE coh-gate threshold, FUSED into one per-bin pass. Verbatim from
 * AEC/c_impl/src/aec3_post.c aec3_post_compute_coherence(): originally 2
 * separate per-bin loops over the same [0,nb) range —
 *
 *   loop 1 (per k): pr = er*nr + ei*ni;  pi = ei*nr - er*ni;   // echo *
 *                     conj(near), plain mul/add — source does NOT call
 *                     fmaf for this cross-product.
 *                   sye_re[k] = omaf*sye_re[k] + af*pr;         // plain
 *                   sye_im[k] = omaf*sye_im[k] + af*pi;         // mul/mul/add,
 *                                                                NOT fmaf
 *                   syy[k] = (1.0f-a)*syy[k] + af*abs_echo[k]^2;  // source
 *                   see[k] = (1.0f-a)*see[k] + af*abs_near[k]^2;  // recomputes
 *                     "(1.0f-a)" inline here instead of reusing the `omaf`
 *                     local — bit-identical to omaf regardless (same `a`,
 *                     same deterministic IEEE subtract), so this kernel
 *                     computes omaf once and reuses it for all four EMAs.
 *   loop 2 (per k): sye2 = sye_re[k]^2 + sye_im[k]^2;           // plain
 *                                                                mul/mul/add
 *                   denom = syy[k]*see[k];
 *                   if (denom<1e-30f) denom=1e-30f;             // compare+
 *                     select, not vmaxq (signed-zero risk per header note)
 *                   g2 = sye2/denom;
 *                   mask[k] = (g2 >= threshold) ? 1u : 0u;
 *
 * FUSION SAFETY: loop 2 at index k reads ONLY sye_re[k]/sye_im[k]/syy[k]/
 * see[k] — the values loop 1 just wrote at that SAME index k, never a
 * different index (no cross-bin term anywhere in either loop). So merging
 * into one per-k pass (update-then-gate) is order-preserving: loop 1's
 * write at k happens-before loop 2's read at k in both the original 2-loop
 * form and this fused form. */

static inline void sk__coherence_ema_gate_elem(
    float *sye_re, float *sye_im, float *syy, float *see,
    float er, float ei, float nr, float ni,
    float abs_echo, float abs_near,
    float alpha, float omaf, float threshold,
    unsigned char *mask_out) {
    float pr = er * nr + ei * ni;
    float pi = ei * nr - er * ni;
    float echo_abs2 = abs_echo * abs_echo;
    float near_abs2 = abs_near * abs_near;
    float sye2, denom, g2;
    *sye_re = omaf * (*sye_re) + alpha * pr;
    *sye_im = omaf * (*sye_im) + alpha * pi;
    *syy    = omaf * (*syy)    + alpha * echo_abs2;
    *see    = omaf * (*see)    + alpha * near_abs2;
    sye2 = (*sye_re) * (*sye_re) + (*sye_im) * (*sye_im);
    denom = (*syy) * (*see);
    if (denom < 1.0e-30f) denom = 1.0e-30f;
    g2 = sye2 / denom;
    *mask_out = (g2 >= threshold) ? (unsigned char)1 : (unsigned char)0;
}

static inline void sk_coherence_ema_gate_f32_scalar(
    float *sye_re, float *sye_im, float *syy, float *see,
    const Complex *echo, const Complex *near_spec,
    const float *abs_echo, const float *abs_near,
    float alpha, float threshold,
    unsigned char *mask, int n) {
    int i;
    float omaf = 1.0f - alpha;
    for (i = 0; i < n; ++i) {
        sk__coherence_ema_gate_elem(&sye_re[i], &sye_im[i], &syy[i], &see[i],
                                     echo[i].r, echo[i].i,
                                     near_spec[i].r, near_spec[i].i,
                                     abs_echo[i], abs_near[i],
                                     alpha, omaf, threshold, &mask[i]);
    }
}

#if SK_HAVE_NEON
static inline void sk_coherence_ema_gate_f32(
    float *sye_re, float *sye_im, float *syy, float *see,
    const Complex *echo, const Complex *near_spec,
    const float *abs_echo, const float *abs_near,
    float alpha, float threshold,
    unsigned char *mask, int n) {
    int i = 0;
    float omaf = 1.0f - alpha;
    float32x4_t va = vdupq_n_f32(alpha), vomaf = vdupq_n_f32(omaf);
    float32x4_t vfloor = vdupq_n_f32(1.0e-30f);
    for (; i + 4 <= n; i += 4) {
        float32x4x2_t ev = vld2q_f32((const float *)(echo + i));
        float32x4_t er = ev.val[0], ei = ev.val[1];
        float32x4x2_t nv = vld2q_f32((const float *)(near_spec + i));
        float32x4_t nr = nv.val[0], ni = nv.val[1];
        float32x4_t pr = vaddq_f32(vmulq_f32(er, nr), vmulq_f32(ei, ni));
        float32x4_t pi = vsubq_f32(vmulq_f32(ei, nr), vmulq_f32(er, ni));

        float32x4_t abs_echo_v = vld1q_f32(abs_echo + i);
        float32x4_t abs_near_v = vld1q_f32(abs_near + i);
        float32x4_t echo_abs2 = vmulq_f32(abs_echo_v, abs_echo_v);
        float32x4_t near_abs2 = vmulq_f32(abs_near_v, abs_near_v);

        float32x4_t sye_re_v = vld1q_f32(sye_re + i);
        float32x4_t sye_im_v = vld1q_f32(sye_im + i);
        float32x4_t syy_v = vld1q_f32(syy + i);
        float32x4_t see_v = vld1q_f32(see + i);

        sye_re_v = vaddq_f32(vmulq_f32(vomaf, sye_re_v), vmulq_f32(va, pr));
        sye_im_v = vaddq_f32(vmulq_f32(vomaf, sye_im_v), vmulq_f32(va, pi));
        syy_v    = vaddq_f32(vmulq_f32(vomaf, syy_v), vmulq_f32(va, echo_abs2));
        see_v    = vaddq_f32(vmulq_f32(vomaf, see_v), vmulq_f32(va, near_abs2));

        vst1q_f32(sye_re + i, sye_re_v);
        vst1q_f32(sye_im + i, sye_im_v);
        vst1q_f32(syy + i, syy_v);
        vst1q_f32(see + i, see_v);

        {
            float32x4_t sye2 = vaddq_f32(vmulq_f32(sye_re_v, sye_re_v),
                                          vmulq_f32(sye_im_v, sye_im_v));
            float32x4_t denom = vmulq_f32(syy_v, see_v);
            uint32x4_t lt = vcltq_f32(denom, vfloor);
            float32x4_t g2;
            float g2_arr[4];
            int j;
            denom = vbslq_f32(lt, vfloor, denom);
            g2 = vdivq_f32(sye2, denom);
            vst1q_f32(g2_arr, g2);
            for (j = 0; j < 4; ++j)
                mask[i + j] = (g2_arr[j] >= threshold) ? (unsigned char)1
                                                        : (unsigned char)0;
        }
    }
    for (; i < n; ++i) {
        sk__coherence_ema_gate_elem(&sye_re[i], &sye_im[i], &syy[i], &see[i],
                                     echo[i].r, echo[i].i,
                                     near_spec[i].r, near_spec[i].i,
                                     abs_echo[i], abs_near[i],
                                     alpha, omaf, threshold, &mask[i]);
    }
}
#else
static inline void sk_coherence_ema_gate_f32(
    float *sye_re, float *sye_im, float *syy, float *see,
    const Complex *echo, const Complex *near_spec,
    const float *abs_echo, const float *abs_near,
    float alpha, float threshold,
    unsigned char *mask, int n) {
    sk_coherence_ema_gate_f32_scalar(sye_re, sye_im, syy, see, echo, near_spec,
                                      abs_echo, abs_near, alpha, threshold,
                                      mask, n);
}
#endif

/* ═══════════════════════════════ kernel 17 ═════════════════════════════════
 * sk_ema_delta_f32 — state[i] = state[i] + alpha*(x[i]-state[i]) (the
 * "delta-form" EMA — distinct from kernel 4's alpha*state+beta*x shape:
 * different rounding path, NOT interchangeable bit-for-bit). Verbatim from
 * AEC/c_impl/src/aec3_post.c aec3_post_compute_comfort_noise()'s
 * y2_smoothed update:
 *   p->y2_smoothed[k] = p->y2_smoothed[k]
 *                     + y2a * (p->near_psd[k] - p->y2_smoothed[k]);
 * Source computes this as a separate subtract, a separate multiply, then a
 * separate add — no fmaf call at this line — so NOT fused here either (needs
 * -ffp-contract=off to stay that way, same discipline as kernel 4). */

static inline void sk_ema_delta_f32_scalar(float *state, const float *x,
                                            float alpha, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float diff = x[i] - state[i];
        state[i] = state[i] + alpha * diff;
    }
}

#if SK_HAVE_NEON
static inline void sk_ema_delta_f32(float *state, const float *x,
                                     float alpha, int n) {
    int i = 0;
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4) {
        float32x4_t sv = vld1q_f32(state + i);
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t diff = vsubq_f32(xv, sv);
        float32x4_t r = vaddq_f32(sv, vmulq_f32(va, diff));
        vst1q_f32(state + i, r);
    }
    for (; i < n; ++i) {
        float diff = x[i] - state[i];
        state[i] = state[i] + alpha * diff;
    }
}
#else
static inline void sk_ema_delta_f32(float *state, const float *x,
                                     float alpha, int n) {
    sk_ema_delta_f32_scalar(state, x, alpha, n);
}
#endif

/* ═══════════════════════════════ kernel 18 ═════════════════════════════════
 * sk_n2_track_f32 — the CNG N2-tracking data-dependent update. Verbatim from
 * AEC/c_impl/src/aec3_post.c aec3_post_compute_comfort_noise():
 *   track = (fresh*y2s[k] + retain*n2[k]) * g_up;  // mul, mul, add, mul —
 *                                                     all plain, no fmaf
 *   up    = n2[k] * g_up;
 *   n2[k] = (y2s[k] < n2[k]) ? track : up;          // exact IEEE '<'
 *                                                       compare+select
 */

static inline void sk_n2_track_f32_scalar(float *n2, const float *y2s,
                                           float fresh, float retain,
                                           float g_up, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float track = (fresh * y2s[i] + retain * n2[i]) * g_up;
        float up = n2[i] * g_up;
        n2[i] = (y2s[i] < n2[i]) ? track : up;
    }
}

#if SK_HAVE_NEON
static inline void sk_n2_track_f32(float *n2, const float *y2s,
                                    float fresh, float retain,
                                    float g_up, int n) {
    int i = 0;
    float32x4_t vfresh = vdupq_n_f32(fresh), vretain = vdupq_n_f32(retain);
    float32x4_t vgup = vdupq_n_f32(g_up);
    for (; i + 4 <= n; i += 4) {
        float32x4_t y2v = vld1q_f32(y2s + i);
        float32x4_t n2v = vld1q_f32(n2 + i);
        float32x4_t track = vmulq_f32(
            vaddq_f32(vmulq_f32(vfresh, y2v), vmulq_f32(vretain, n2v)), vgup);
        float32x4_t up = vmulq_f32(n2v, vgup);
        uint32x4_t lt = vcltq_f32(y2v, n2v);
        float32x4_t r = vbslq_f32(lt, track, up);
        vst1q_f32(n2 + i, r);
    }
    for (; i < n; ++i) {
        float track = (fresh * y2s[i] + retain * n2[i]) * g_up;
        float up = n2[i] * g_up;
        n2[i] = (y2s[i] < n2[i]) ? track : up;
    }
}
#else
static inline void sk_n2_track_f32(float *n2, const float *y2s,
                                    float fresh, float retain,
                                    float g_up, int n) {
    sk_n2_track_f32_scalar(n2, y2s, fresh, retain, g_up, n);
}
#endif

/* ═══════════════════════════════ kernel 19 ═════════════════════════════════
 * sk_n2_initial_track_f32 — the CNG N2-initial slow-tracking data-dependent
 * update. Verbatim from AEC/c_impl/src/aec3_post.c
 * aec3_post_compute_comfort_noise():
 *   slow = n2i[k] + ia*(n2[k]-n2i[k]);        // plain sub/mul/add, no fmaf
 *   n2i[k] = (n2[k] > n2i[k]) ? slow : n2[k];  // exact IEEE '>' compare+select,
 *                                                 comparing against the
 *                                                 ORIGINAL n2i[k] (captured
 *                                                 before the overwrite)
 */

static inline void sk_n2_initial_track_f32_scalar(float *n2i, const float *n2,
                                                    float alpha, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float old = n2i[i];
        float slow = old + alpha * (n2[i] - old);
        n2i[i] = (n2[i] > old) ? slow : n2[i];
    }
}

#if SK_HAVE_NEON
static inline void sk_n2_initial_track_f32(float *n2i, const float *n2,
                                            float alpha, int n) {
    int i = 0;
    float32x4_t va = vdupq_n_f32(alpha);
    for (; i + 4 <= n; i += 4) {
        float32x4_t oldv = vld1q_f32(n2i + i);
        float32x4_t n2v = vld1q_f32(n2 + i);
        float32x4_t diff = vsubq_f32(n2v, oldv);
        float32x4_t slow = vaddq_f32(oldv, vmulq_f32(va, diff));
        uint32x4_t gt = vcgtq_f32(n2v, oldv);
        float32x4_t r = vbslq_f32(gt, slow, n2v);
        vst1q_f32(n2i + i, r);
    }
    for (; i < n; ++i) {
        float old = n2i[i];
        float slow = old + alpha * (n2[i] - old);
        n2i[i] = (n2[i] > old) ? slow : n2[i];
    }
}
#else
static inline void sk_n2_initial_track_f32(float *n2i, const float *n2,
                                            float alpha, int n) {
    sk_n2_initial_track_f32_scalar(n2i, n2, alpha, n);
}
#endif

/* ═══════════════════════════════ kernel 20 ═════════════════════════════════
 * sk_mask_zero_f32 — out[i] = mask[i] ? 0.0f : out[i] (in-place, byte mask,
 * C truthiness: any nonzero byte triggers the zero — matches a plain
 * `if (mask[k]) x[k] = 0.0f;` source loop verbatim, e.g.
 * AEC/c_impl/src/aec3_post.c's stationarity R²-zeroing step). */

static inline void sk_mask_zero_f32_scalar(float *x, const unsigned char *mask,
                                            int n) {
    int i;
    for (i = 0; i < n; ++i) if (mask[i]) x[i] = 0.0f;
}

#if SK_HAVE_NEON
static inline void sk_mask_zero_f32(float *x, const unsigned char *mask, int n) {
    int i = 0;
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        uint32_t m[4];
        int j;
        float32x4_t xv = vld1q_f32(x + i);
        uint32x4_t mv;
        for (j = 0; j < 4; ++j) m[j] = mask[i + j] ? 0xFFFFFFFFu : 0u;
        mv = vld1q_u32(m);
        vst1q_f32(x + i, vbslq_f32(mv, zero, xv));
    }
    for (; i < n; ++i) if (mask[i]) x[i] = 0.0f;
}
#else
static inline void sk_mask_zero_f32(float *x, const unsigned char *mask, int n) {
    sk_mask_zero_f32_scalar(x, mask, n);
}
#endif

/* ═══════════════════════════════ kernel 21 ═════════════════════════════════
 * sk_pairwise_sum_tailfold_f32 — numpy-style pairwise float32 sum, variant
 * "A". Verbatim tree from AEC/c_impl/src/pbfdkf.c's pw_leaf_f32() +
 * pairwise_sum_f32() (an exact, whitespace-only-diff twin also lives in
 * AEC/c_impl/src/linear_filter_output.c — diff-verified byte-identical).
 *
 * Two structural differences from kernel 13's sk_pairwise_sum_f32 (so this
 * is a genuinely different value-function, not a restyling):
 *   1. There is no outer n<=8 gate ahead of the n<=128 leaf. The leaf is
 *      entered directly for any n<=128 and branches internally on
 *      `n < 8` (strict). Consequence: at n==8 this kernel takes the
 *      8-accumulator tree-combine path, whereas kernel 13's OUTER n<=8
 *      check (inclusive) diverts n==8 to a plain sequential sum instead —
 *      different rounding, different bits.
 *   2. In the n in (8,128] leaf, the n%8 remainder tail is folded ONE
 *      ELEMENT AT A TIME straight into the already-combined `res` root
 *      (`res += a[i]` per leftover element), not accumulated into a
 *      separate running total and added once at the end the way kernel
 *      13's leaf does (`return r + s;`). Whenever there is more than one
 *      leftover element these two tail strategies round differently.
 *
 * The small-n (n<8) serial path here starts its accumulator at 0.0f and
 * adds every element, `s = 0.0f; for (i=0;i<n;++i) s += a[i];` — see
 * kernel 22 for the OTHER small-n convention (res = a[0], then fold from
 * i=1) used by the four other pairwise-sum call sites, which is a third
 * distinct value-function: it differs from this one only on signed-zero
 * inputs (e.g. n=1 with a[0]=-0.0f: this kernel returns
 * 0.0f + (-0.0f) = +0.0f; kernel 22 returns -0.0f unchanged) — confirmed by
 * direct probe, see kernel 22's comment.
 *
 * NEON leaf: identical q0/q1 accumulation + vpaddq_f32×2 combine order to
 * kernel 13's leaf (bit-identical group-of-8 sums), then the same
 * one-at-a-time tail fold applied to the scalar `res` (not to q0/q1) to
 * match the scalar tree above exactly. */

static inline float sk__pairwise_sum_tailfold_leaf_scalar(const float *a, size_t n) {
    if (n < 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) s += a[i];
        return s;
    }
    {
        float acc[8];
        size_t i, j;
        float res;
        for (j = 0; j < 8; ++j) acc[j] = a[j];
        for (i = 8; i + 8 <= n; i += 8)
            for (j = 0; j < 8; ++j) acc[j] += a[i + j];
        res = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
            + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
}

static inline float sk_pairwise_sum_tailfold_f32_scalar(const float *a, size_t n) {
    if (n <= 128) return sk__pairwise_sum_tailfold_leaf_scalar(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_pairwise_sum_tailfold_f32_scalar(a, half)
             + sk_pairwise_sum_tailfold_f32_scalar(a + half, n - half);
    }
}

#if SK_HAVE_NEON
static inline float sk__pairwise_sum_tailfold_leaf_neon(const float *a, size_t n) {
    if (n < 8) {
        float s = 0.0f;
        size_t i;
        for (i = 0; i < n; ++i) s += a[i];
        return s;
    }
    {
        float32x4_t q0 = vld1q_f32(a), q1 = vld1q_f32(a + 4);
        size_t i;
        float res;
        for (i = 8; i + 8 <= n; i += 8) {
            q0 = vaddq_f32(q0, vld1q_f32(a + i));
            q1 = vaddq_f32(q1, vld1q_f32(a + i + 4));
        }
        {
            float32x4_t t = vpaddq_f32(q0, q1);
            float32x4_t u = vpaddq_f32(t, t);
            res = vgetq_lane_f32(u, 0) + vgetq_lane_f32(u, 1);
        }
        for (; i < n; ++i) res += a[i];
        return res;
    }
}

static inline float sk_pairwise_sum_tailfold_f32(const float *a, size_t n) {
    if (n <= 128) return sk__pairwise_sum_tailfold_leaf_neon(a, n);
    {
        size_t half = n / 2;
        half -= half % 8;
        return sk_pairwise_sum_tailfold_f32(a, half)
             + sk_pairwise_sum_tailfold_f32(a + half, n - half);
    }
}
#else
static inline float sk_pairwise_sum_tailfold_f32(const float *a, size_t n) {
    return sk_pairwise_sum_tailfold_f32_scalar(a, n);
}
#endif

/* ═══════════════════════════════ kernel 22 ═════════════════════════════════
 * sk_pairwise_sum_tailfold_b_f32 — numpy-style pairwise float32 sum, variant
 * "B". Verbatim tree from AEC/c_impl/src/filter_analyzer.c's
 * fa_f32_pairwise_sum() (byte-identical, whitespace/line-wrap-only-diff
 * twins also live in AEC/c_impl/src/reverb_frequency_response.c's
 * f32_pairwise_sum(), AEC/c_impl/src/filter_state_bridge.c's
 * fsb_f32_pairwise_sum(), and AEC/c_impl/src/fullband_erle.c's
 * fb_erle_pairwise_sum() — all four diffed byte-for-byte identical modulo
 * cosmetic renaming/line-wrapping).
 *
 * Same 8<=n<=128 leaf shape, tail-fold, and n/2-rounded-down-to-a-multiple-
 * of-8 split as kernel 21 (sk_pairwise_sum_tailfold_f32) — differs ONLY in
 * the small-n (n<8) path and the explicit n==0 case:
 *   - n==0 returns 0.0f explicitly (kernel 21 also returns 0.0f for n==0,
 *     via its 0.0f-initialized accumulator with a zero-iteration loop —
 *     same result, different code path, no observable difference).
 *   - n in [1,7]: `res = a[0]; for (i=1;i<n;++i) res = res + a[i];` — the
 *     accumulator STARTS AT a[0] (a plain copy, no addition performed for
 *     the first element), rather than starting at 0.0f and adding a[0] the
 *     way kernel 21 does. These two conventions are bit-identical for
 *     every finite/normal nonzero a[0], but diverge on signed zero: with
 *     a[0] = -0.0f and n==1, this kernel returns -0.0f (copied through
 *     untouched); kernel 21 returns 0.0f + (-0.0f) = +0.0f (IEEE-754
 *     round-to-nearest: an unlike-signed zero sum rounds to +0). Verified
 *     directly with a standalone probe for n=1 and for an all -0.0f array
 *     of length 5: kernel 21 and kernel 22 disagree with each other on
 *     those inputs (each stays internally bit-identical scalar-vs-NEON),
 *     confirming two distinct kernels are required here, not one shared
 *     implementation. */

static inline float sk_pairwise_sum_tailfold_b_f32_scalar(const float *a, size_t n) {
    if (n == 0) return 0.0f;
    if (n < 8) {
        float res = a[0];
        size_t i;
        for (i = 1; i < n; ++i) res = res + a[i];
        return res;
    }
    if (n <= 128) {
        float acc[8];
        size_t i, j;
        float res;
        for (j = 0; j < 8; ++j) acc[j] = a[j];
        for (i = 8; i + 8 <= n; i += 8)
            for (j = 0; j < 8; ++j) acc[j] = acc[j] + a[i + j];
        res = ((acc[0] + acc[1]) + (acc[2] + acc[3]))
            + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; i < n; ++i) res = res + a[i];
        return res;
    }
    {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return sk_pairwise_sum_tailfold_b_f32_scalar(a, n2)
             + sk_pairwise_sum_tailfold_b_f32_scalar(a + n2, n - n2);
    }
}

#if SK_HAVE_NEON
static inline float sk_pairwise_sum_tailfold_b_f32(const float *a, size_t n) {
    if (n == 0) return 0.0f;
    if (n < 8) {
        float res = a[0];
        size_t i;
        for (i = 1; i < n; ++i) res = res + a[i];
        return res;
    }
    if (n <= 128) {
        float32x4_t q0 = vld1q_f32(a), q1 = vld1q_f32(a + 4);
        size_t i;
        float res;
        for (i = 8; i + 8 <= n; i += 8) {
            q0 = vaddq_f32(q0, vld1q_f32(a + i));
            q1 = vaddq_f32(q1, vld1q_f32(a + i + 4));
        }
        {
            float32x4_t t = vpaddq_f32(q0, q1);
            float32x4_t u = vpaddq_f32(t, t);
            res = vgetq_lane_f32(u, 0) + vgetq_lane_f32(u, 1);
        }
        for (; i < n; ++i) res = res + a[i];
        return res;
    }
    {
        size_t n2 = n / 2;
        n2 -= n2 % 8;
        return sk_pairwise_sum_tailfold_b_f32(a, n2)
             + sk_pairwise_sum_tailfold_b_f32(a + n2, n - n2);
    }
}
#else
static inline float sk_pairwise_sum_tailfold_b_f32(const float *a, size_t n) {
    return sk_pairwise_sum_tailfold_b_f32_scalar(a, n);
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* AEC_SIMD_KERNELS_H */
