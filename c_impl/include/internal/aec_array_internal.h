/* aec_array_internal.h -- private to aec.c, aec_array.c and the white-box tests (not installed): the entry points aec.c dispatches
 * to on an array instance (AecConfig.n_mics == AEC_ARRAY_MICS), the lane stages and the post-beam projection kernels. The
 * public contract is the "Array instance" section of aec.h. */
#ifndef AEC_ARRAY_INTERNAL_H
#define AEC_ARRAY_INTERNAL_H

#include "aec_impl.h"
#include "aec_view_internal.h"
#include "utility/simd_kernels.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AecArray AecArray;      /* the flow state, defined in aec_array.c */

/* ---- dispatch from aec.c (the AecResolved is validated unless noted) -------------------------------------------------------------- */

#ifndef AEC_NO_ARRAY
/* Checks the array fields of a cfg (n_mics in {1, 4}, capture_proxy_channel, the quarantine rule and, for an array, the flow's own: residual != OFF,
 * the three product grids, a sizeable estimator, a valid derived lane). Called at the end of aec_resolve_tuned(). 1 = acceptable. */
int    aec_array_impl_config_ok(const AecResolved* resolved);
size_t aec_array_impl_get_mem_size(const AecResolved* resolved);                    /* 0 on failure */
int    aec_array_impl_get_mem_breakdown(const AecResolved* resolved, AecMemBreakdown* out);   /* AecStatus as aec_get_mem_breakdown */
Aec*   aec_array_impl_init(void* mem, size_t mem_size, const AecResolved* resolved); /* Aec at mem[0]; NULL on failure */
void   aec_array_impl_destroy(Aec* a);                                            /* a live array handle; clears a->ready, frees a heap block */
void   aec_array_impl_reset(Aec* a);
int    aec_array_impl_set_level(Aec* a, AecLevel level);

int    aec_array_impl_process_linear(Aec* a, const float* mics, const float* ref);
int    aec_array_impl_process_residual(Aec* a, const AecBeam* beam);
int    aec_array_impl_get_residual(const Aec* a, AecResidualViewFull* out);
int    aec_array_impl_get_linear(const Aec* a, AecLinearViewFull* out);
int    aec_array_impl_apply_gain(const Aec* a, const float* extra_gain, Complex* out_spec);
int    aec_array_impl_apply_cng(Aec* a, const float* extra_gain, Complex* spec);
uint32_t aec_array_impl_post_reset_serial(const Aec* a);
int    aec_array_impl_synthesis_window(const Aec* a, const float** window, int* length);
void   aec_array_impl_debug_status(const Aec* a, AecDebugStatus* out);
long   aec_array_impl_far_fft_count(const Aec* a);
int    aec_array_impl_is_cancelling(const Aec* a);
Aec*   aec_array_impl_linear_lane(Aec* a);                                     /* the lane behind aec_get_linear() */
/* aec_get_array() (both kinds) and the diagnostics of aec.h are defined in aec_array.c directly. */
#else
/* -DAEC_NO_ARRAY: src/aec_array.c is not built and no instance is an array, so aec.c never reaches these; they only keep its
 * dispatch sites well-formed (and free of undefined references) at any optimisation level. */
#define aec_array_impl_get_mem_size(cfg)             ((size_t)0)
#define aec_array_impl_get_mem_breakdown(cfg, out)   AEC_ERR_ARG
#define aec_array_impl_init(mem, mem_size, cfg)      ((Aec*)NULL)
#define aec_array_impl_destroy(a)                    ((void)0)
#define aec_array_impl_reset(a)                      ((void)0)
#define aec_array_impl_set_level(a, level)           (-1)
#define aec_array_impl_process_linear(a, mics, ref)  AEC_ERR_MODE
#define aec_array_impl_process_residual(a, beam)     AEC_ERR_MODE
#define aec_array_impl_get_residual(a, out)          AEC_ERR_MODE
#define aec_array_impl_get_linear(a, out)            AEC_ERR_MODE
#define aec_array_impl_apply_gain(a, gain, out)      AEC_ERR_MODE
#define aec_array_impl_apply_cng(a, gain, spec)      AEC_ERR_MODE
#define aec_array_impl_post_reset_serial(a)          0u
#define aec_array_impl_synthesis_window(a, win, len) (-1)
#define aec_array_impl_debug_status(a, out)          ((void)0)
#define aec_array_impl_far_fft_count(a)              0L
#define aec_array_impl_is_cancelling(a)              0
#define aec_array_impl_linear_lane(a)                (a)
#endif

/* ---- lane construction (aec.c) -------------------------------------------------------------------------------------- *
 * A lane is a single-channel Aec of the array flow: n_mics 1, EXTERNAL_ALIGNED, residual STATE or ESTIMATE (any other configuration is
 * refused). Its pool omits what that contract never runs and it borrows the array's FFT handle (same fft_size, calls strictly
 * sequential across lanes). aec_lane_init() takes the mem at least aec_lane_get_mem_size() bytes, 16-byte aligned. */
/* Buffers a lane writes before it reads inside one lane call, and the constant window tables its filters read: one set per array,
 * borrowed by all four lanes (aec_lane_scratch_bind() zeroes it). Nothing kept across hops, returned by a view or handed from lane 0
 * to lanes 1-3 (far PSD, scaled far block, narrow-band mask) lives here. */
typedef struct AecLaneScratch {
    float*         e2_echo;        /* each of these n_freqs floats */
    float*         e2_near;
    float*         mu_buf_pre;
    float*         mu_buf;
    float*         e2ref;
    float*         ree;            /* NULL for a STATE lane (it has no residual-echo estimator) */
    float*         abs_near;
    float*         abs_far;
    float*         abs_sel_echo;
    float*         abs_error;
    float*         abs_echo_coh;
    float*         abs_nsw_e1;
    float*         abs_ybase;
    float*         x2_at_delay;
    float*         r2_unb;
    float*         nearend_pwr;
    float*         psd_near;       /* the AecPost power spectra of this hop */
    float*         psd_echo;
    float*         psd_error;
    float*         psd_capture_win;
    float*         x2_reverb;
    float*         nf;
    float*         shadow_out;     /* each of these hop_size floats */
    float*         final_out;
    float*         lfs_sq;
    float*         lfs_sref;
    float*         lfs_scoa;
    float*         fa_render_sq;
    float*         lfs_tin;        /* fft_size floats */
    float*         fa_abs;         /* n_partitions * hop_size floats */
    float*         taps;           /* n_partitions * hop_size floats: the tap span the filter analyzer asks for */
    float*         w_mag2;         /* n_partitions * n_freqs floats */
    Complex*       nsw_e1;         /* n_freqs bins each */
    Complex*       lfs_sel_esw;
    unsigned char* stat_mask;      /* n_freqs bytes each */
    unsigned char* coh_gate_mask;
    float*         td_window;      /* constants every lane's filters read: fft_size and block_size floats */
    float*         sqrt_hann;
    PbfdafScratch  main_kf;        /* the main filter's scratch rows */
    PbfdafScratch  shadow;         /* the shadow follower's (unused without a shadow) */
} AecLaneScratch;

#ifndef AEC_NO_ARRAY
size_t aec_lane_scratch_get_mem_size(const AecResolved* lane);
void   aec_lane_scratch_bind(AecLaneScratch* out, void* mem, const AecResolved* lane);
size_t aec_lane_get_mem_size(const AecResolved* lane);
Aec*   aec_lane_init(void* mem, size_t mem_size, const AecResolved* lane, FftHandle* shared_fft, const AecLaneScratch* scratch);
#endif

#ifndef AEC_NO_ARRAY
/* One linear-stage hop of an array lane. Donor (`publish` != 0): runs locally and fills *share. Otherwise reads *share.
 * Both lane stages answer AEC_ERR_DSP on the single-channel checks (aec.h "Refused hop"); the caller resets the array. */
int aec_lane_process_linear(Aec* lane, const float* mic, const float* ref, AecFarShare* share, int publish);
/* The residual stage of a lane that ran aec_lane_process_linear() (share: this hop's, or NULL for the donor). */
int aec_lane_process_residual(Aec* lane, const AecFarShare* share);
/* Copies the donor's analysis state (tonal counters, noise tracker) into a lane that has not maintained its own. */
void aec_lane_adopt_far_analysis(Aec* lane, const Aec* donor);
/* 1 when both lanes hold the same far history (the buffers every far-only stage derives from), else 0. */
int aec_lane_far_history_equal(const Aec* a, const Aec* b);
#endif /* !AEC_NO_ARRAY */

/* ---- saturating counters --------------------------------------------------------------------------------------------- */
static inline long aec_array_count_add_sat(long total, long add) {
    if (total < 0 || add < 0 || add > LONG_MAX - total) return LONG_MAX;
    return total + add;
}

static inline int aec_array_far_fft_delta_matches(long before, long after, long expected) {
    if (before < 0 || after < before || after == LONG_MAX) return 0;
    return after - before == expected;
}

static inline int aec_array_complex_close(Complex a, Complex b) {
    float dr = fabsf(a.r - b.r);
    float di = fabsf(a.i - b.i);
    float scale = 1.0f + fabsf(a.r) + fabsf(a.i);
    return dr <= 1e-5f * scale && di <= 1e-5f * scale;
}

/* Cross-lane far-end agreement (fusion precondition): 1 = this lane may be folded into the beam, 0 = it carries a different
 * far-end spectrum and the fusion must reject the frame. shared_far_provenance is the per-hop evidence that the four lanes
 * consumed one spectrum (aec_process_linear()); with it the answer is immediate, without it every bin is compared. */
static inline int aec_array_far_spec_agrees(int shared_far_provenance, const Complex* lane_far_spec,
                                            const Complex* reference_far_spec, int n_freqs) {
    int k;
    if (!lane_far_spec || !reference_far_spec || n_freqs < 0) return 0;
    if (shared_far_provenance) return 1;
    for (k = 0; k < n_freqs; ++k)
        if (!aec_array_complex_close(lane_far_spec[k], reference_far_spec[k])) return 0;
    return 1;
}

/* Rebuilds the residual-echo PSD as a complex vector with the echo estimate's phase and the residual's magnitude:
 * |out[i]|^2 == max(r2[i], 0). The ratio form avoids atan2/sincos, and for 0 < |echo|^2 <= phase_floor2 (e.g. the DC bin after
 * the mic HPF) the result is pinned to the +real axis. `scale` has no upper bound: r2 is int16^2-scaled while echo_spec is
 * audio-scaled, so sqrt(r2/|echo|^2) legitimately carries a factor of 32768. Overflow modes: power/|echo|^2 saturating gives
 * inf, and |echo|^2 saturating gives 0 (no residual echo reported). The exceptional path normalises the phase vector first,
 * so no intermediate leaves float32 range. */
static inline Complex aec_array_residual_sample(float er, float ei, float raw_r2) {
    const float phase_floor2 = 1.0e-12f; /* (1e-6)^2 */
    Complex out;
    float mag2 = er * er + ei * ei;
    float power = fmaxf(raw_r2, 0.0f);
    float ratio, axis, unit_r, unit_i, scale;

    if (!(mag2 > phase_floor2)) {
        out.r = sqrtf(power);
        out.i = 0.0f;
        return out;
    }

    /* mag2 > phase_floor2 > 0, so the ratio is never 0/0; only its range is in question. */
    ratio = power / mag2;
    if (isfinite(mag2) && isfinite(ratio)) {
        scale = sqrtf(ratio);
        out.r = er * scale;
        out.i = ei * scale;
        return out;
    }

    /* Exceptional path: divide by the larger component first (sum of squares in [1, 2]), so a plain sqrt suffices. */
    axis = fmaxf(fabsf(er), fabsf(ei));
    unit_r = er / axis;
    unit_i = ei / axis;
    scale = sqrtf(power) / sqrtf(unit_r * unit_r + unit_i * unit_i);
    out.r = unit_r * scale;
    out.i = unit_i * scale;
    return out;
}

static inline void aec_array_residual_vector_scalar(
    Complex* out, const Complex* echo, const float* r2, int n) {
    int i;
    for (i = 0; i < n; ++i) out[i] = aec_array_residual_sample(echo[i].r, echo[i].i, r2[i]);
}

#if SK_HAVE_NEON
static inline void aec_array_residual_vector(
    Complex* out, const Complex* echo, const float* r2, int n) {
    /* Must match aec_array_residual_sample's phase_floor2 so the SIMD and scalar forms agree for finite inputs. */
    const float phase_floor2 = 1.0e-12f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t zero = vdupq_n_f32(0.0f);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4x2_t e = sk__cquad_load(echo + i);
        float32x4_t mag2 = vaddq_f32(vmulq_f32(e.val[0], e.val[0]), vmulq_f32(e.val[1], e.val[1]));
        float32x4_t raw_power = vld1q_f32(r2 + i);
        uint32x4_t power_positive = vcgtq_f32(raw_power, zero);
        float32x4_t power = vbslq_f32(power_positive, raw_power, zero);
        uint32x4_t has_phase = vcgtq_f32(mag2, vdupq_n_f32(phase_floor2));
        float32x4_t denominator = vbslq_f32(has_phase, mag2, one);
        float32x4_t scale = vsqrtq_f32(vdivq_f32(power, denominator));
        float32x4x2_t residual;
        residual.val[0] = vbslq_f32(has_phase, vmulq_f32(e.val[0], scale), scale);
        residual.val[1] = vbslq_f32(has_phase, vmulq_f32(e.val[1], scale), zero);
        sk__cquad_store(out + i, residual);

        /* Both overflow modes are visible in values already held (mag2 saturating, scale infinite or degraded to zero), so
         * detection costs no arithmetic and keeps an in-place call legal. The test is false on real audio; when it trips,
         * all four lanes go back through the scalar form, which reproduces the stored bytes for lanes that did not overflow. */
        {
            const float32x4_t huge = vdupq_n_f32(FLT_MAX);
            uint32x4_t overflowed = vandq_u32(
                has_phase, vorrq_u32(vcgtq_f32(mag2, huge), vcgtq_f32(scale, huge)));
            if (vmaxvq_u32(overflowed)) {
                float lane_r[4], lane_i[4], lane_power[4];
                int lane;
                vst1q_f32(lane_r, e.val[0]);
                vst1q_f32(lane_i, e.val[1]);
                vst1q_f32(lane_power, raw_power);
                for (lane = 0; lane < 4; ++lane)
                    out[i + lane] = aec_array_residual_sample(lane_r[lane], lane_i[lane], lane_power[lane]);
            }
        }
    }
    aec_array_residual_vector_scalar(out + i, echo + i, r2 + i, n - i);
}
#else
static inline void aec_array_residual_vector(
    Complex* out, const Complex* echo, const float* r2, int n) {
    aec_array_residual_vector_scalar(out, echo, r2, n);
}
#endif

static inline void aec_array_comfort_accumulate_scalar(
    float* acc, const Complex* weights, const float* comfort, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float wr = weights[i].r;
        float wi = weights[i].i;
        float weight2 = wr * wr + wi * wi;
        acc[i] += weight2 * fmaxf(comfort[i], 0.0f);
    }
}

#if SK_HAVE_NEON
static inline void aec_array_comfort_accumulate(
    float* acc, const Complex* weights, const float* comfort, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t zero = vdupq_n_f32(0.0f);
        float32x4x2_t w = sk__cquad_load(weights + i);
        float32x4_t weight2 = vaddq_f32(vmulq_f32(w.val[0], w.val[0]), vmulq_f32(w.val[1], w.val[1]));
        float32x4_t raw_comfort = vld1q_f32(comfort + i);
        uint32x4_t comfort_positive = vcgtq_f32(raw_comfort, zero);
        float32x4_t nonnegative = vbslq_f32(comfort_positive, raw_comfort, zero);
        float32x4_t value = vaddq_f32(vld1q_f32(acc + i), vmulq_f32(weight2, nonnegative));
        vst1q_f32(acc + i, value);
    }
    aec_array_comfort_accumulate_scalar(acc + i, weights + i, comfort + i, n - i);
}
#else
static inline void aec_array_comfort_accumulate(
    float* acc, const Complex* weights, const float* comfort, int n) {
    aec_array_comfort_accumulate_scalar(acc, weights, comfort, n);
}
#endif

/* Weight-matrix gate: 1 when every component is finite and the sequential sum of |re| + |im| is finite and above 1e-12, else 0.
 * The vector screen decides the common case without the sum (terms <= 2^100 and fewer than 2^24 of them cannot overflow, and
 * one term above 1e-12 decides); any other input runs the sequential loop. */
static inline int aec_array_weights_valid_scalar(const Complex* weights, size_t count) {
    size_t i;
    float sum = 0.0f;
    for (i = 0; i < count; ++i) {
        if (!isfinite(weights[i].r) || !isfinite(weights[i].i)) return 0;
        sum += fabsf(weights[i].r) + fabsf(weights[i].i);
    }
    return isfinite(sum) && sum > 1e-12f;
}

#if SK_HAVE_NEON
static inline int aec_array_weights_valid(const Complex* weights, size_t count) {
    const float32x4_t bound = vdupq_n_f32(0x1p100f);
    const float32x4_t floor_value = vdupq_n_f32(1e-12f);
    uint32x4_t bounded = vdupq_n_u32(~0u);
    uint32x4_t above = vdupq_n_u32(0u);
    size_t i = 0;
    if (count >= ((size_t)1 << 24)) return aec_array_weights_valid_scalar(weights, count);
    for (; i + 4 <= count; i += 4) {
        float32x4x2_t w = sk__cquad_load(weights + i);
        float32x4_t term = vaddq_f32(vabsq_f32(w.val[0]), vabsq_f32(w.val[1]));
        bounded = vandq_u32(bounded, vcleq_f32(term, bound));
        above = vorrq_u32(above, vcgtq_f32(term, floor_value));
    }
    if (vminvq_u32(bounded) == ~0u) {
        int tail_above = 0;
        int tail_bounded = 1;
        for (; i < count; ++i) {
            float term = fabsf(weights[i].r) + fabsf(weights[i].i);
            tail_bounded &= term <= 0x1p100f;
            tail_above |= term > 1e-12f;
        }
        if (tail_bounded && (tail_above || vmaxvq_u32(above) != 0u)) return 1;
    }
    return aec_array_weights_valid_scalar(weights, count);
}
#else
static inline int aec_array_weights_valid(const Complex* weights, size_t count) {
    return aec_array_weights_valid_scalar(weights, count);
}
#endif

#ifdef __cplusplus
}
#endif
#endif /* AEC_ARRAY_INTERNAL_H */
