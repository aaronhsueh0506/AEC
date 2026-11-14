/* aec_array_internal.h -- private to aec.c, aec_array.c and the white-box tests (not installed): the entry points aec.c dispatches
 * to on an array instance (AecConfig.n_mics == AEC_ARRAY_MICS), the shared-alignment admission rule and the post-beam
 * projection kernels. The public contract is the "Array instance" section of aec.h. */
#ifndef AEC_ARRAY_INTERNAL_H
#define AEC_ARRAY_INTERNAL_H

#include "aec_impl.h"
#include "utility/simd_kernels.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Stage timing is off by default (stamps fold to 0); build with -DAEC_STAGE_TIMING=1. A 32-bit microsecond monotonic stamp; a
 * target without CLOCK_MONOTONIC defines -DAEC_NOW_US=<fn> (no-argument function returning monotonic uint32_t microseconds). */
#if AEC_STAGE_TIMING
#ifndef AEC_NOW_US
#include <time.h>
static inline uint32_t aec_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull);
}
#else
static inline uint32_t aec_now_us(void) { return AEC_NOW_US(); }
#endif
#else
#define aec_now_us() 0u
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* A NULL, never-built, failed-create or destroyed handle: every entry point refuses it. */
static inline int aec_unusable(const Aec* a) { return !a || !a->ready; }

typedef struct AecArray AecArray;      /* the flow state, defined in aec_array.c */

/* ---- dispatch from aec.c (the AecResolved is validated unless noted) -------------------------------------------------------------- */

#ifndef AEC_NO_ARRAY
/* Checks the array-related fields of a cfg (n_mics in {1, 4}, capture_proxy_channel, the quarantine-enabled rule and, for an
 * array, the flow's own rules: residual != OFF, the three product grids, max_delay_ms <= 4096, an estimator that can be sized,
 * a valid derived lane). Called at the end of aec_resolve_tuned() on the resolved configuration, after the quarantine ranges.
 * 1 = acceptable. */
int    aec_array_impl_config_ok(const AecResolved* resolved);
size_t aec_array_impl_get_mem_size(const AecResolved* resolved);                    /* 0 on failure */
int    aec_array_impl_get_mem_breakdown(const AecResolved* resolved, AecMemBreakdown* out);   /* AecStatus as aec_get_mem_breakdown */
Aec*   aec_array_impl_init(void* mem, size_t mem_size, const AecResolved* resolved); /* Aec at mem[0]; NULL on failure */
void   aec_array_impl_destroy(Aec* a);                                            /* a live array handle; clears a->ready, frees a heap block */
void   aec_array_impl_reset(Aec* a);
int    aec_array_impl_set_level(Aec* a, AecLevel level);

int    aec_array_impl_process_linear(Aec* a, const float* mics, const float* ref);
int    aec_array_impl_process_residual(Aec* a, const AecBeam* beam);
int    aec_array_impl_get_residual(const Aec* a, AecResidualView* out);
int    aec_array_impl_get_linear(const Aec* a, AecLinearView* out);
int    aec_array_impl_apply_gain(const Aec* a, const float* extra_gain, Complex* out_spec);
int    aec_array_impl_apply_cng(Aec* a, const float* extra_gain, Complex* spec);
uint32_t aec_array_impl_post_reset_serial(const Aec* a);
int    aec_array_impl_synthesis_window(const Aec* a, const float** window, int* length);
void   aec_array_impl_debug_status(const Aec* a, AecDebugStatus* out);
long   aec_array_impl_far_fft_count(const Aec* a);
int    aec_array_impl_is_cancelling(const Aec* a);
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
#endif

/* ---- lane construction (aec.c) -------------------------------------------------------------------------------------- *
 * A lane is a single-channel Aec of the array flow: n_mics 1, EXTERNAL_ALIGNED, residual STATE or ESTIMATE (any other configuration is
 * refused). Its pool omits what that contract never runs and it borrows the array's FFT handle (same fft_size, calls strictly
 * sequential across lanes). aec_lane_init() takes the mem at least aec_lane_get_mem_size() bytes, 16-byte aligned. */
#ifndef AEC_NO_ARRAY
size_t aec_lane_get_mem_size(const AecResolved* lane);
Aec*   aec_lane_init(void* mem, size_t mem_size, const AecResolved* lane, FftHandle* shared_fft);
#endif

/* ---- far-only analysis shared across lanes ------------------------------------------------------------------------- *
 * Lane 0 publishes its far-only results each hop; lanes 1-3 reuse them. hop: stages that read the far hop alone; fft: the
 * donor's far spectrum and what derives from it (needs the lanes' spectra to agree); analysis: the tonal counters and noise
 * floor, valid while the lane's far history has equalled the donor's since the last reset; post: the donor's far PSD. */
typedef struct AecFarShare {
    const Aec* donor;
    int hop, fft, analysis, post;
    float sat_ref;
    RenderActivityResult ra;
    float far_mean_sq;
    int far_floor_hit;
    int rsa_poor;
    int rsa_mask_active;
} AecFarShare;

#ifndef AEC_NO_ARRAY
/* One linear-stage hop of an array lane. Donor (`publish` != 0): runs locally and fills *share. Otherwise reads *share. */
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

/* ---- shared-delay change admission (the Path-B rule of the single-channel AEC) ---------------------------------------- *
 * A change to an alignment in force is admitted only when the movement exceeds AEC_ARRAY_DELAY_CHANGE_MIN_SAMPLES and is
 * offered again, within AEC_ARRAY_DELAY_CHANGE_CONFIRM_SAMPLES, before the held candidate ages out (TTL in hops). The
 * single-channel Path B in aec.c uses the same three figures. */
#define AEC_ARRAY_DELAY_CHANGE_MIN_SAMPLES      32
#define AEC_ARRAY_DELAY_CHANGE_CONFIRM_SAMPLES  16
#define AEC_ARRAY_DELAY_CHANGE_CANDIDATE_TTL     3

typedef struct AecArrayAdmission {
    int candidate;
    int ttl;
} AecArrayAdmission;

/* One hop of life spent, whether or not this hop had a usable estimate. Call once per hop, before offer(). */
static inline void aec_array_admission_age(AecArrayAdmission* admission) {
    if (!admission || admission->ttl <= 0) return;
    admission->ttl -= 1;
    if (admission->ttl <= 0) {
        admission->ttl = 0;
        admission->candidate = 0;
    }
}

/* Offers this hop's eligible estimate against the alignment in force. 1: the movement is admitted (the caller applies it and
 * realigns); 0: absorbed as too small, or held / replaced as a candidate. */
static inline int aec_array_admission_offer(AecArrayAdmission* admission, int accepted_delay, int estimated) {
    if (!admission) return 0;
    /* Too small to be worth four IR shifts: absorbed and left to age. */
    if (abs(estimated - accepted_delay) <= AEC_ARRAY_DELAY_CHANGE_MIN_SAMPLES) return 0;
    if (admission->ttl > 0 &&
        abs(estimated - admission->candidate) < AEC_ARRAY_DELAY_CHANGE_CONFIRM_SAMPLES) {
        admission->candidate = 0;
        admission->ttl = 0;
        return 1;
    }
    admission->candidate = estimated;
    admission->ttl = AEC_ARRAY_DELAY_CHANGE_CANDIDATE_TTL;
    return 0;
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

/* Matched-filter duty cycle, shared by the single-channel instance and the array: the analysis period is the configured period
 * divided by AEC_DUTY_PERIOD_DIVISOR, and a windowed-ERLE drop of AEC_DUTY_ERLE_COLLAPSE_DB below the leaky peak resumes full rate.
 * The caller configures the gate once (aec_duty_configure) and keeps its own census. */
#define AEC_DUTY_PERIOD_DIVISOR    5
#define AEC_DUTY_ERLE_COLLAPSE_DB  6.0f

/* Per-hop leak of the duty watchdog the grid implies: 0.001 dB at the 10 ms reference hop, scaled with the hop duration. */
float aec_duty_leak_db(int hop_size, int sample_rate);
/* Seconds -> hops (round, floor 1) against the grid: hold_hops from hold_s, period_hops from period_s / AEC_DUTY_PERIOD_DIVISOR,
 * and the grid's erle_leak_db. */
void  aec_duty_configure(AecDutyGate* g, float hold_s, float period_s, int hop_size, int sample_rate);
/* Clears the gate's state (not its hold / period / leak). */
void  aec_duty_reset(AecDutyGate* g);
/* Before the estimator is fed: advances the decimation phase and returns 1 when this hop's analysis runs (counts it). */
int   aec_duty_begin(AecDutyGate* g);
/* After the estimator is fed: the estimate (cur_delay, solid) and the windowed ERLE read at this hop arm, hold or drop the gate. */
void  aec_duty_end(AecDutyGate* g, int cur_delay, int solid, float erle);

/* Reference-ring copies of n <= ring_size samples at index `at`: at most one wrap, so one or two bounded runs. */
static inline void aec_ring_write(float* ring, int ring_size, int at, const float* src, int n) {
    int first = ring_size - at;
    if (first > n) first = n;
    memcpy(ring + at, src, (size_t)first * sizeof(float));
    if (first < n) memcpy(ring, src + first, (size_t)(n - first) * sizeof(float));
}
static inline void aec_ring_read(const float* ring, int ring_size, int at, float* dst, int n) {
    int first = ring_size - at;
    if (first > n) first = n;
    memcpy(dst, ring + at, (size_t)first * sizeof(float));
    if (first < n) memcpy(dst + first, ring, (size_t)(n - first) * sizeof(float));
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
