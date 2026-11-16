/* test_aec_array_flow.c -- the shared-alignment state machine, backward quarantine, duty cycle, realign counters, memory
 * descriptor, lane contract and error handling of the array instance. The groups that need the SRP/BF sample helper or
 * the NR / RES chain live in the 4ch sample-layer suite.
 *
 * Test-local wrappers (Af, AfFront, AfFrame, af_process_front, af_fuse, ...) sit over the entry points (aec_process_linear +
 * aec_get_array, aec_process_residual + aec_get_residual): see "short names" below. They add no behaviour.
 *
 * Built and run by `make test-aec-array-flow` (an AEC_TESTING build: the duty-cycle hooks of aec_test_hooks.h). */
#ifndef AEC_TESTING
#error "test_aec_array_flow.c requires -DAEC_TESTING (make test-aec-array-flow)"
#endif

#include "aec.h"
#include "aec_internal.h"
#include "aec_test_tuning.h"
#include "audio_stft.h"
#include "fft_wrapper.h"
#include "aec_array_internal.h"
#include "utility/simd_kernels.h"
#include "array_signal.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failures = 0;
static int passes = 0;

#define CHECK(condition, label)                                            \
    do {                                                                   \
        if (condition) {                                                   \
            printf("PASS: %s\n", label);                                   \
            passes += 1;                                                   \
        } else {                                                           \
            printf("FAIL: %s\n", label);                                   \
            failures += 1;                                                 \
        }                                                                  \
    } while (0)

/* ---- short names: test-local aliases over the array instance ---------------------------------------- */
typedef Aec Af;
#define AF_CHANNELS            AEC_ARRAY_MICS
#define AF_OK                  AEC_OK
#define AF_INVALID_ARGUMENT    AEC_ERR_ARG
#define AF_SEQUENCE_ERROR      AEC_ERR_SEQUENCE
#define AF_DSP_ERROR           AEC_ERR_DSP

typedef AecDelayState AfDelayState;
typedef AecArrayTiming AfLastTiming;
typedef AecMemReq AfMemReq;

#define af_realign_warm_lane_count   aec_array_realign_warm_lane_count
#define af_realign_soft_lane_count   aec_array_realign_soft_lane_count
#define af_pending_delay_candidate   aec_testing_array_pending_delay_candidate
static unsigned long long af_duty_hops_total(const Aec* a) { AecDebugStatus ds; aec_debug_status(a, &ds); return ds.duty_hops_total; }
static unsigned long long af_duty_hops_run(const Aec* a) { AecDebugStatus ds; aec_debug_status(a, &ds); return ds.duty_hops_run; }
#define af_matched_filter_count      aec_array_matched_filter_count
#define af_linear_aec_count          aec_array_linear_aec_count
#define af_far_fft_real_compute_count aec_far_fft_real_compute_count
#define af_get_last_timing           aec_array_get_timing
#define af_reset                     aec_reset
#define af_hop_size                  aec_hop_size
#define af_duty_erle_peak            aec_testing_array_duty_erle_peak
#define af_duty_erle_leak_db         aec_testing_array_duty_erle_leak_db
#define af_pin_duty_full_rate        aec_testing_array_pin_duty_full_rate
#define af_far_spec_provenance       aec_testing_array_far_spec_provenance
#define af_duty_leak_db_for_grid     aec_duty_leak_db
#define AfDelayAdmission             AecDelayAdmission
#define af_admission_age             aec_delay_admission_age
#define af_admission_offer           aec_delay_admission_offer
#define af_far_spec_agrees           aec_array_far_spec_agrees
#define AF_DELAY_CHANGE_MIN_SAMPLES      AEC_DELAY_CHANGE_MIN_SAMPLES
#define AF_DELAY_CHANGE_CANDIDATE_TTL    AEC_DELAY_CHANGE_CANDIDATE_TTL
#define af_projection_cmac_scalar    sk_cmac_f32_scalar
#define af_projection_cmac           sk_cmac_f32
#define af_complex_mag2_scalar(o, i, n) sk_cmag_f32_scalar((o), (i), (n), 0.0f)
#define af_complex_mag2(o, i, n)     sk_cmag_f32((o), (i), (n), 0.0f)
#define af_residual_vector_scalar    aec_array_residual_vector_scalar
#define af_residual_vector           aec_array_residual_vector
#define af_comfort_accumulate_scalar aec_array_comfort_accumulate_scalar
#define af_comfort_accumulate        aec_array_comfort_accumulate
#define af_weights_valid_scalar      aec_array_weights_valid_scalar
#define af_weights_valid             aec_array_weights_valid
#define af_count_add_sat             aec_array_count_add_sat
#define af_far_fft_delta_matches     aec_array_far_fft_delta_matches

/* A flow description; af_to_aec() turns it into the array AecConfig (enable_post = 1: the fusion without a gain,
 * AEC_RESIDUAL_ESTIMATE; enable_post = 0: AEC_RESIDUAL_STATE; with_gain = 1 adds the gain stage, AEC_RESIDUAL_FULL).
 * The lane level is the balanced default. */
typedef struct AfConfig {
    int sample_rate;
    int fft_size;                          /* 0 = rate default */
    AecDelayMode delay_mode;
    int delay_num_filters;
    int fixed_delay_samples;
    int capture_proxy_channel;
    int delay_backward_quarantine_enabled;
    float delay_backward_quarantine_s;
    float delay_est_period_s;
    int enable_post;
    int with_gain;
} AfConfig;

static AfConfig af_default_config(int sample_rate) {
    AfConfig c;
    memset(&c, 0, sizeof c);
    c.sample_rate = sample_rate;
    c.delay_mode = AEC_DELAY_MATCHED;
    c.delay_num_filters = DA_NUM_FILTERS;
    c.fixed_delay_samples = -1;
    c.delay_backward_quarantine_s = 1.0f;
    c.enable_post = 1;
    return c;
}

static void af_to_aec(const AfConfig* c, AecTestCfg* t) {
    AecConfig* a = &t->c;
    aec_tcfg_from_level(t, AEC_LEVEL_BALANCED, c->sample_rate);
    if (c->fft_size != 0) a->fft_size = c->fft_size;
    a->n_mics = AEC_ARRAY_MICS;
    a->residual = !c->enable_post ? AEC_RESIDUAL_STATE : (c->with_gain ? AEC_RESIDUAL_FULL : AEC_RESIDUAL_ESTIMATE);
    if (c->enable_post > 1) a->residual = (AecResidualMode)7;      /* a non-boolean enable: an out-of-range mode */
    a->delay_mode = c->delay_mode;
    a->delay_num_filters = c->delay_num_filters;
    a->fixed_delay_samples = c->fixed_delay_samples;
    a->capture_proxy_channel = c->capture_proxy_channel;
    a->delay_backward_quarantine_enabled = c->delay_backward_quarantine_enabled;
    a->delay_backward_quarantine_s = c->delay_backward_quarantine_s;
    t->t.delay_est_period_s = c->delay_est_period_s;
}

static int af_get_mem_requirements(const AfConfig* c, AfMemReq* out) {
    AecTestCfg a;
    af_to_aec(c, &a);
    return aec_tcfg_mem_requirements(&a, out);
}

/* The whole pool of the instance. 0, or -1 for an invalid config. */
static int af_sample_flow_get_mem_size(const AfConfig* c, size_t* bytes) {
    AecTestCfg a;
    af_to_aec(c, &a);
    *bytes = aec_tcfg_mem_size(&a);
    return *bytes ? 0 : -1;
}

/* The lane the array builds inside itself, written out here from the contract (single channel, the flow's grid, the shared
 * aligned far = EXTERNAL_ALIGNED, the residual stage at ESTIMATE / STATE, the flow's duty period): the stand-alone reference of
 * the lane-parity tests. -1 for an invalid config. */
static int lane_config_for(const AfConfig* c, AecTestCfg* lane) {
    AecTestCfg a;
    af_to_aec(c, &a);
    if (aec_tcfg_mem_size(&a) == 0) return -1;
    *lane = a;
    lane->c.n_mics = 1;
    lane->c.capture_proxy_channel = 0;
    lane->c.delay_backward_quarantine_enabled = 0;
    lane->c.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    lane->c.delay_num_filters = DA_NUM_FILTERS;
    lane->c.fixed_delay_samples = -1;
    lane->c.residual = c->enable_post ? AEC_RESIDUAL_ESTIMATE : AEC_RESIDUAL_STATE;
    return 0;
}

/* Heap instance (aec_create) for cfg; NULL for an invalid config. */
static Af* af_test_create(const AfConfig* c) {
    AecTestCfg a;
    af_to_aec(c, &a);
    return aec_tcfg_create(&a);
}

static void af_test_destroy(Af* p) {
    if (!p) return;
    aec_destroy(p);
}

static int af_n_freqs(const Af* p) { return aec_hop_size(p) + 1; }
static int af_fft_size(const Af* p) { return 2 * aec_hop_size(p); }
static int af_sample_rate(const Af* p) { return p->cfg.sample_rate; }
static int af_enable_post(const Af* p) { return p->cfg.residual != AEC_RESIDUAL_STATE; }

typedef struct AfFront {
    AfDelayState delay;
    int hop_size, n_channels, n_freqs;
    const Complex* linear_spectra[AF_CHANNELS];
    const float* formed_hop[AF_CHANNELS];
    const Complex* render_spectra[AF_CHANNELS];
    int guard_active[AF_CHANNELS];
    uint32_t post_reset_serial;
    const float* aligned_ref;
} AfFront;

typedef struct AfFrame {
    int n_freqs, hop;
    const Complex* e;
    const Complex* b;
    int guard_lanes;
    uint32_t post_reset_serial;
    const Complex* y;
    const Complex* x;
    const float* r2;
    const float* cn;
    const float* g_res_ctx;                 /* the instance's gain (FULL) or NULL */
    const float* far_hop;
    const float* formed_hop;
    float far_power, max_dt, max_sat;
    int all_converged, usable_linear, clock_drift;
} AfFrame;

static int af_process_front(Af* p, const float* mics, const float* ref, AfFront* out) {
    AecArrayView av;
    int rc, ch;
    if (!out) return AEC_ERR_ARG;
    rc = aec_process_linear(p, mics, ref);
    if (rc != AEC_OK) return rc;
    memset(&av, 0, sizeof av); av.struct_size = sizeof av;
    rc = aec_get_array(p, &av);
    if (rc != AEC_OK) return rc;
    out->delay = av.delay;
    out->hop_size = aec_hop_size(p);
    out->n_channels = AF_CHANNELS;
    out->n_freqs = av.n_freqs;
    for (ch = 0; ch < AF_CHANNELS; ++ch) {
        out->linear_spectra[ch] = av.lane_error_spec[ch];
        out->formed_hop[ch] = av.lane_formed_hop[ch];
        out->render_spectra[ch] = av.lane_base_spec[ch];
        out->guard_active[ch] = av.lane_guard_active[ch];
    }
    out->post_reset_serial = av.post_reset_serial;
    out->aligned_ref = av.aligned_ref;
    return AEC_OK;
}

static int af_fuse(Af* p, const Complex* weights, const Complex* beamformed_error, AfFrame* frame) {
    AecBeam beam;
    AecResidualView rv;
    AecLinearView lv;
    int rc;
    if (!frame) return AEC_ERR_ARG;
    beam.weights = weights;
    beam.beamformed_error = beamformed_error;
    rc = aec_process_residual(p, &beam);
    if (rc != AEC_OK) return rc;
    memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
    memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
    rc = aec_get_residual(p, &rv);
    if (rc != AEC_OK) return rc;
    (void)aec_get_linear(p, &lv);
    memset(frame, 0, sizeof *frame);
    frame->n_freqs = rv.n_freqs;
    frame->hop = rv.hop_size;
    frame->e = rv.error_spec;
    frame->b = rv.base_spec;
    frame->guard_lanes = (int)rv.guard_lanes;
    frame->post_reset_serial = rv.post_reset_serial;
    frame->y = rv.near_spec;
    frame->x = rv.far_spec;
    frame->r2 = rv.r2;
    frame->cn = rv.comfort_noise;
    frame->g_res_ctx = rv.res_gain;
    frame->far_hop = rv.far_hop;
    frame->formed_hop = rv.formed_hop;
    frame->far_power = lv.far_power;
    frame->max_dt = rv.max_dt;
    frame->max_sat = rv.max_saturation;
    frame->all_converged = rv.all_converged;
    frame->usable_linear = rv.usable_linear;
    frame->clock_drift = rv.clock_drift;
    return AEC_OK;
}

static int all_finite(const float* values, int count) {
    int i;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static void fill_inputs(float* microphones, float* ref, int hop, int sample_rate, int frame_index) {
    int i, ch;
    for (i = 0; i < hop; ++i) {
        int64_t absolute = (int64_t)frame_index * hop + i;
        float phase = 2.0f * 3.14159265358979323846f * 440.0f * (float)absolute / (float)sample_rate;
        ref[i] = 0.08f * sinf(phase);
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            float near = 0.01f * sinf(phase * (1.0f + 0.05f * (float)ch));
            microphones[i * AF_CHANNELS + ch] = ref[i] * (0.4f + 0.05f * (float)ch) + near;
        }
    }
}

/* The fusion on the linear stage just processed, for tests that only need the lanes advanced through a complete hop. */
static int fuse_only(Af* p, const Complex* weights) {
    AfFrame view;
    return af_fuse(p, weights, NULL, &view);
}

/* Production-scale seed shared by the two residual-projection blocks: an audio-domain echo estimate against an
 * int16^2-domain residual PSD. */
static void seed_residual_inputs(Complex* echo, float* r2, int n) {
    int b;
    for (b = 0; b < n; ++b) {
        echo[b].r = 0.05f * (float)(b - 7);
        echo[b].i = -0.03f * (float)(b + 1);
        r2[b] = 6.0e11f * (float)(b + 1);
    }
}

/* ---- ported groups ------------------------------------------------------------------------------------------------------ */

/* ---- projection kernels, weights gate ---- */
static void test_projection_kernels(void) {
    enum { N = 17 };
    Complex weights[N];
    Complex input[N];
    Complex echo[N];
    Complex acc_scalar[N];
    Complex acc_dispatch[N];
    Complex residual_scalar[N];
    Complex residual_dispatch[N];
    float r2[N];
    float comfort[N];
    float comfort_scalar[N];
    float comfort_dispatch[N];
    float mag2_scalar[N];
    float mag2_dispatch[N];
    int i;

    for (i = 0; i < N; ++i) {
        float x = (float)(i + 1);
        weights[i].r = 0.03125f * x;
        weights[i].i = -0.015625f * (float)(i % 5);
        input[i].r = 0.0078125f * (float)(i - 8);
        input[i].i = 0.00390625f * (float)(7 - i);
        echo[i].r = 0.002f * (float)(i - 6);
        echo[i].i = -0.001f * (float)(i + 2);
        r2[i] = 1.0e-5f * x;
        comfort[i] = (i % 4 == 0) ? -0.25f : 0.01f * x;
        acc_scalar[i].r = acc_dispatch[i].r = 0.125f * x;
        acc_scalar[i].i = acc_dispatch[i].i = -0.0625f * x;
        comfort_scalar[i] = comfort_dispatch[i] = 0.5f * x;
    }
    /* Explicitly exercise both no-phase fallback shapes and a zero R2 bin. */
    echo[0].r = echo[0].i = 0.0f;
    echo[1].r = 1.0e-21f;
    echo[1].i = -1.0e-21f;
    r2[2] = 0.0f;

    af_projection_cmac_scalar(
        acc_scalar, weights, input, N);
    af_projection_cmac(
        acc_dispatch, weights, input, N);
    CHECK(memcmp(acc_scalar, acc_dispatch, sizeof(acc_scalar)) == 0,
          "context complex-MAC SIMD matches scalar bytes");

    af_complex_mag2_scalar(mag2_scalar, input, N);
    af_complex_mag2(mag2_dispatch, input, N);
    CHECK(memcmp(mag2_scalar, mag2_dispatch, sizeof(mag2_scalar)) == 0,
          "context magnitude-squared SIMD matches scalar bytes");

    af_residual_vector_scalar(
        residual_scalar, echo, r2, N);
    af_residual_vector(
        residual_dispatch, echo, r2, N);
    CHECK(memcmp(
              residual_scalar, residual_dispatch,
              sizeof(residual_scalar)) == 0,
          "residual projection SIMD matches scalar bytes");

    /* The kernel's contract: |out|^2 == max(r2, 0), with echo supplying phase only. The
     * byte-equality check above cannot see a wrong magnitude (scalar and SIMD agree on it),
     * and the seed magnitudes (r2 ~ 1e-5, echo ~ 1e-3) are far from production, where r2 is
     * int16^2-scaled (~1e11) against an audio-scaled echo (~1e-2); so the identity is
     * asserted at the production levels. */
    {
        Complex prod_echo[N];
        float prod_r2[N];
        Complex prod_out[N];
        int b;
        seed_residual_inputs(prod_echo, prod_r2, N);
        prod_echo[0].r = prod_echo[0].i = 0.0f;   /* no-phase fallback */
        prod_r2[1] = 0.0f;                        /* zero-power bin */
        af_residual_vector(prod_out, prod_echo, prod_r2, N);
        for (b = 0; b < N; ++b) {
            float got = prod_out[b].r * prod_out[b].r
                      + prod_out[b].i * prod_out[b].i;
            float want = prod_r2[b] > 0.0f ? prod_r2[b] : 0.0f;
            float tol = want * 1.0e-5f + 1.0e-6f;
            CHECK(fabsf(got - want) <= tol,
                  "residual projection carries r2 as its squared magnitude "
                  "at production scales");
        }
    }

    /* Range safety, exercised where it actually lives. A length below four
     * falls straight through to the scalar tail, so a scalar-vs-dispatch
     * memcmp on one element compares the scalar path against itself and the
     * vectorized repair goes untested. These pathological lanes therefore sit
     * inside vector blocks (1, 5, 9) and one in the tail (N-1). */
    {
        Complex range_echo[N];
        float range_r2[N];
        Complex range_scalar[N];
        Complex range_dispatch[N];
        int b;

        seed_residual_inputs(range_echo, range_r2, N);
        /* power / mag2 overflows although the normalized result cannot. */
        range_echo[1].r = 1.1e-6f;
        range_echo[1].i = 0.2e-6f;
        range_r2[1] = FLT_MAX;
        /* mag2 itself overflows on finite components. */
        range_echo[5].r = FLT_MAX;
        range_echo[5].i = FLT_MAX / 2.0f;
        range_r2[5] = 4.0f;
        /* Below the phase floor: no usable phase, real-axis fallback. */
        range_echo[9].r = 1.0e-9f;
        range_echo[9].i = 0.0f;
        /* Same overflow in the scalar tail. */
        range_echo[N - 1].r = 2.0e-6f;
        range_echo[N - 1].i = -0.5e-6f;
        range_r2[N - 1] = FLT_MAX;

        af_residual_vector_scalar(
            range_scalar, range_echo, range_r2, N);
        af_residual_vector(
            range_dispatch, range_echo, range_r2, N);

        CHECK(memcmp(range_scalar, range_dispatch,
                     sizeof(range_scalar)) == 0,
              "range-safe residual projection SIMD matches scalar bytes");
        /* Compares |out|, not |out|^2 like the block above: at r2 == FLT_MAX
         * the squared form overflows inside the assertion itself. An inf or
         * NaN component fails this compare too, so it doubles as the
         * finiteness check. */
        for (b = 0; b < N; ++b) {
            float want = sqrtf(range_r2[b]);
            float got = hypotf(range_dispatch[b].r, range_dispatch[b].i);
            CHECK(fabsf(got - want) <= want * 4.0e-6f,
                  "range-safe residual projection preserves sqrt(r2)");
        }
    }

    af_comfort_accumulate_scalar(
        comfort_scalar, weights, comfort, N);
    af_comfort_accumulate(
        comfort_dispatch, weights, comfort, N);
    CHECK(memcmp(
              comfort_scalar, comfort_dispatch,
              sizeof(comfort_scalar)) == 0,
          "comfort projection SIMD matches scalar bytes");

    /* n=0 is part of every header kernel's boundary contract. */
    af_projection_cmac(acc_dispatch, weights, input, 0);
    af_complex_mag2(mag2_dispatch, input, 0);
    af_residual_vector(residual_dispatch, echo, r2, 0);
    af_comfort_accumulate(
        comfort_dispatch, weights, comfort, 0);
    CHECK(1, "projection kernels accept an empty range");
}

/* The weight-matrix gate against its sequential definition, written out here
 * rather than taken from the header. */
static int weights_gate_reference(const Complex* w, size_t count) {
    size_t i;
    float sum = 0.0f;
    for (i = 0; i < count; ++i) {
        if (!isfinite(w[i].r) || !isfinite(w[i].i)) return 0;
        sum += fabsf(w[i].r) + fabsf(w[i].i);
    }
    return isfinite(sum) && sum > 1e-12f;
}

static float weights_float_from_bits(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static int weights_gate_agrees(const Complex* w, size_t count) {
    int want = weights_gate_reference(w, count);
    return af_weights_valid_scalar(w, count) == want &&
           af_weights_valid(w, count) == want;
}

static void test_weights_gate(void) {
    enum { MAXC = 40 };
    Complex w[MAXC];
    const float specials[] = {
        0.0f, -0.0f, 1.0f, -1.0f, 1e-13f, 1e-12f, 1.0000001e-12f,
        FLT_MIN, FLT_MIN * 0.5f, 1.0e-40f, -1.0e-45f, 1e30f, 0x1p100f,
        0x1.000002p100f, 0x1p101f, FLT_MAX, -FLT_MAX, INFINITY, -INFINITY,
        NAN, weights_float_from_bits(0x7fc00001u),
        weights_float_from_bits(0xffa00000u)};
    const int n_special = (int)(sizeof specials / sizeof specials[0]);
    uint32_t seed = 20240611u;
    int ok = 1, ok_tiny = 1, ok_tail = 1, ok_big = 1, ok_edge = 1;
    int trial, c, i;
    size_t count;

    /* Random vectors: mostly ordinary values, a few special entries. */
    for (trial = 0; trial < 60000 && ok; ++trial) {
        seed = seed * 1664525u + 1013904223u;
        count = (size_t)(seed >> 8) % (MAXC + 1);
        for (i = 0; i < (int)count; ++i) {
            seed = seed * 1664525u + 1013904223u;
            w[i].r = (float)((int32_t)seed >> 8) / (float)(1 << 23);
            seed = seed * 1664525u + 1013904223u;
            w[i].i = (float)((int32_t)seed >> 8) / (float)(1 << 23);
            seed = seed * 1664525u + 1013904223u;
            if ((seed >> 24) < 6u) {
                w[i].r = specials[(seed >> 4) % (uint32_t)n_special];
                seed = seed * 1664525u + 1013904223u;
                if ((seed >> 24) < 128u)
                    w[i].i = specials[(seed >> 4) % (uint32_t)n_special];
            }
        }
        ok = weights_gate_agrees(w, count);
    }

    /* Every special in every position of every length, against zeros and
     * against ones. */
    for (c = 0; c <= 12 && ok_edge; ++c) {
        int s, pos, comp, fill;
        for (fill = 0; fill < 2 && ok_edge; ++fill)
            for (s = 0; s < n_special && ok_edge; ++s)
                for (pos = 0; pos < c && ok_edge; ++pos)
                    for (comp = 0; comp < 2 && ok_edge; ++comp) {
                        for (i = 0; i < c; ++i) {
                            w[i].r = fill ? 1.0f : 0.0f;
                            w[i].i = 0.0f;
                        }
                        if (comp) w[pos].i = specials[s];
                        else w[pos].r = specials[s];
                        ok_edge = weights_gate_agrees(w, (size_t)c);
                    }
    }
    CHECK(ok && ok_edge,
          "weight gate: dispatch and scalar twin equal the sequential definition");

    /* Only small terms: the sequential sum decides, and it crosses the floor
     * after enough of them. */
    for (c = 1; c <= MAXC && ok_tiny; ++c) {
        for (i = 0; i < c; ++i) { w[i].r = 3e-14f; w[i].i = 0.0f; }
        ok_tiny = weights_gate_agrees(w, (size_t)c);
    }
    for (i = 0; i < 40; ++i) { w[i].r = 3e-14f; w[i].i = 0.0f; }
    CHECK(ok_tiny && af_weights_valid(w, 40) == 1 &&
          af_weights_valid(w, 8) == 0,
          "weight gate: small terms that only add up past 1e-12 pass");

    /* The single qualifying term sits in the scalar tail. */
    for (c = 1; c <= 11 && ok_tail; ++c) {
        for (i = 0; i < c; ++i) { w[i].r = 0.0f; w[i].i = 0.0f; }
        w[c - 1].i = 2e-12f;
        ok_tail = weights_gate_agrees(w, (size_t)c) &&
                  af_weights_valid(w, (size_t)c) == 1;
        w[c - 1].i = 1e-12f;
        ok_tail = ok_tail && af_weights_valid(w, (size_t)c) == 0;
    }
    CHECK(ok_tail, "weight gate: a lone term above 1e-12 in the tail decides");

    /* Terms above the screen bound and sums that overflow. */
    for (i = 0; i < MAXC; ++i) { w[i].r = 0x1p100f; w[i].i = 0.0f; }
    ok_big = weights_gate_agrees(w, MAXC) &&
             af_weights_valid(w, MAXC) == 1;
    for (i = 0; i < MAXC; ++i) { w[i].r = 0x1p127f; w[i].i = 0.0f; }
    ok_big = ok_big && weights_gate_agrees(w, MAXC) &&
             af_weights_valid(w, MAXC) == 0;
    for (i = 0; i < MAXC; ++i) { w[i].r = FLT_MAX; w[i].i = FLT_MAX; }
    ok_big = ok_big && af_weights_valid(w, MAXC) == 0;
    for (i = 0; i < MAXC; ++i) { w[i].r = 0x1.000002p100f; w[i].i = 0.0f; }
    ok_big = ok_big && weights_gate_agrees(w, MAXC) &&
             af_weights_valid(w, MAXC) == 1;
    CHECK(ok_big, "weight gate: large terms and an overflowing sum");

    CHECK(af_weights_valid(w, 0) == 0,
          "weight gate: an empty matrix is refused");
}

/* The default lane as aec.h documents it, written out here: lib/aec's balanced level on the
 * flow's rate and grid, EXTERNAL_ALIGNED, the linear context returned without a gain of its own,
 * the context level of enable_post and the flow's duty period. -1 for a config the flow refuses. */

/* ---- far-FFT sharing ---- */
static void test_far_fft_sharing_reduces_four_to_one(
        int sample_rate, int fft_size) {
    AfConfig cfg;
    Af* p;
    AfFront pre;
    AfFrame fused;
    float* microphones;
    float* ref;
    float* out;
    Complex* weights;
    int hop;
    int n_freqs;
    int frame;
    int ch, k;
    long before, after;
    int saw_a_plus_one_increment = 0;
    int max_single_hop_increment = 0;
    char label[160];

    cfg = af_default_config(sample_rate);
    cfg.fft_size = fft_size;
    p = af_test_create(&cfg);
    CHECK(p != NULL, "far-FFT sharing test: create");
    if (!p) return;

    hop = af_hop_size(p);
    n_freqs = af_n_freqs(p);
    microphones = (float*)calloc(
        (size_t)hop * AF_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)AF_CHANNELS * n_freqs, sizeof(Complex));
    CHECK(microphones && ref && out && weights,
          "far-FFT sharing test: buffers allocate");
    if (!microphones || !ref || !out || !weights) {
        free(microphones); free(ref); free(out); free(weights);
        af_test_destroy(p);
        return;
    }
    for (ch = 0; ch < AF_CHANNELS; ++ch)
        for (k = 0; k < n_freqs; ++k)
            weights[(size_t)ch * n_freqs + k].r = 0.25f;

    snprintf(label, sizeof(label),
             "far-FFT sharing sr=%d/fft=%d: counter starts at 0",
             sample_rate, fft_size);
    CHECK(af_far_fft_real_compute_count(p) == 0, label);

    for (frame = 0; frame < 40; ++frame) {
        fill_inputs(microphones, ref, hop, sample_rate, frame);
        before = af_far_fft_real_compute_count(p);
        if (af_process_front(p, microphones, ref, &pre) !=
                AF_OK) {
            CHECK(0, "far-FFT sharing test: pre stage succeeds");
            break;
        }
        after = af_far_fft_real_compute_count(p);
        if (after - before > max_single_hop_increment)
            max_single_hop_increment = (int)(after - before);
        if (after == before + 1) saw_a_plus_one_increment = 1;
        if (af_fuse(p, weights, NULL, &fused) !=
                AF_OK) {
            CHECK(0, "far-FFT sharing test: fuse stage succeeds");
            break;
        }
    }

    snprintf(label, sizeof(label),
             "far-FFT sharing sr=%d/fft=%d: total never jumps by more than "
             "1 in a single af_process_front() call across 40 hops (max seen: %d)",
             sample_rate, fft_size, max_single_hop_increment);
    CHECK(max_single_hop_increment <= 1, label);
    snprintf(label, sizeof(label),
             "far-FFT sharing sr=%d/fft=%d: at least one hop actually shows "
             "the +1 increment (not a vacuous pass from e.g. every hop "
             "being a reset)",
             sample_rate, fft_size);
    CHECK(saw_a_plus_one_increment, label);

    af_reset(p);
    snprintf(label, sizeof(label),
             "far-FFT sharing sr=%d/fft=%d: af_reset() zeroes "
             "the counter",
             sample_rate, fft_size);
    CHECK(af_far_fft_real_compute_count(p) == 0, label);

    free(microphones);
    free(ref);
    free(out);
    free(weights);
    af_test_destroy(p);
}


/* ---- delay modes, known delay, admission, duty cycle, provenance, mixed realign ---- */
static void test_delay_modes_and_bank_sizing(void) {
    AfConfig matched5 = af_default_config(16000);
    AfConfig matched2 = matched5;
    AfConfig fixed = matched5;
    AfConfig external = matched5;
    AfConfig pre_only;
    AfMemReq r5, r2, rf, re, rp;
    size_t t5 = 0, t2 = 0, tf = 0, te = 0, tp = 0;
    Af *pf = NULL, *pe = NULL;
    AfFront pre;
    float *mic = NULL, *far = NULL;
    int hop;

    matched5.fft_size = 256;
    matched2.fft_size = 256;
    matched2.delay_num_filters = 2;
    fixed.fft_size = 256;
    fixed.delay_mode = AEC_DELAY_FIXED;
    fixed.fixed_delay_samples = 320;
    external.fft_size = 256;
    external.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    pre_only = external;
    pre_only.enable_post = 0;

    CHECK(af_get_mem_requirements(&matched5, &r5) == 0 &&
          af_get_mem_requirements(&matched2, &r2) == 0 &&
          af_get_mem_requirements(&fixed, &rf) == 0 &&
          af_get_mem_requirements(&external, &re) == 0,
          "all three delay modes and a short matched bank size cleanly");
    /* The shared estimator is inside the one pool: the bank size moves the total. */
    CHECK(af_sample_flow_get_mem_size(&matched5, &t5) == 0 &&
          af_sample_flow_get_mem_size(&matched2, &t2) == 0 &&
          af_sample_flow_get_mem_size(&fixed, &tf) == 0 &&
          af_sample_flow_get_mem_size(&external, &te) == 0 &&
          af_sample_flow_get_mem_size(&pre_only, &tp) == 0,
          "every delay mode has an application-side total");
    CHECK(t5 > t2 && t2 > te,
          "matched-filter bank size changes the 4ch total");
    CHECK(r5.bytes > r2.bytes && r5.bytes == t5 && r2.bytes == t2,
          "the bank size lives in the shared estimator, which is inside the descriptor's pool");
    CHECK(tf > te && tf < t2 && rf.bytes > re.bytes,
          "FIXED keeps only its exact reference ring; EXTERNAL has no matcher/ring");
    CHECK(af_get_mem_requirements(&pre_only, &rp) == 0 &&
          rp.bytes < re.bytes && tp < te,
          "a lanes-only (STATE) array omits the fusion buffers");

    external.delay_num_filters = 2;
    CHECK(af_get_mem_requirements(&external, &re) != 0,
          "EXTERNAL rejects an inapplicable matched-filter bank size");
    external.delay_num_filters = DA_NUM_FILTERS;
    fixed.fixed_delay_samples = -1;
    CHECK(af_get_mem_requirements(&fixed, &rf) != 0,
          "FIXED rejects an unset fixed delay");
    fixed.fixed_delay_samples = 320;

    pf = af_test_create(&fixed);
    pe = af_test_create(&pre_only);
    CHECK(pf && pe, "FIXED and EXTERNAL instances initialize");
    CHECK(af_matched_filter_count(pf) == 0 &&
          af_matched_filter_count(pe) == 0,
          "FIXED and EXTERNAL allocate no matched-filter instance");
    if (!pf || !pe) goto cleanup;
    hop = af_hop_size(pe);
    mic = (float*)calloc((size_t)hop * AF_CHANNELS,
                         sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    CHECK(mic && far, "delay-mode smoke buffers allocate");
    if (!mic || !far) goto cleanup;

    fill_inputs(mic, far, hop, 16000, 0);
    CHECK(af_process_front(pe, mic, far, &pre) ==
              AF_OK && pre.delay.solid &&
              pre.delay.delay_samples == 0,
          "EXTERNAL consumes caller-aligned far without acquisition");
    CHECK(af_process_front(pf, mic, far, &pre) ==
              AF_OK && !pre.delay.solid &&
              pre.delay.delay_samples == 320,
          "FIXED reports the configured delay and ring warm-up state");

cleanup:
    free(mic);
    free(far);
    af_test_destroy(pf);
    af_test_destroy(pe);
}

/* ============================================================================
 * Known-delay profile check (product delay gate, not an audio-quality run): a synthesised
 * echo with an exactly known bulk delay, checked against ground truth for acquisition
 * (each bank size acquires the delays it should, how fast and how accurately), coverage
 * (the reliable range per n holds at its boundary), mislock (a delay beyond the bank's
 * reach is detectable only against ground truth, not from the seam) and cost.
 * ========================================================================== */

/* Reliable bulk-delay search ceiling for a bank of n matched filters, in native 16 kHz
 * samples, from lib/aec's own constants: the ring reaches (n-1)*DA_FILTER_INTRA_SHIFT +
 * (DA_FILTER_SIZE - 11) downsampled samples (the -11 is the `lag < filter_size - 10`
 * reliability cut), and one downsampled sample is DA_DOWN_SAMPLING_FACTOR native samples
 * (0.25 ms at 16 kHz). For n = 1..5 that is 125/221/317/413/509 ms. */
#define KD_RELIABLE_SAMPLES(n) \
    (((n) - 1) * DA_FILTER_INTRA_SHIFT * DA_DOWN_SAMPLING_FACTOR + \
     (DA_FILTER_SIZE - 11) * DA_DOWN_SAMPLING_FACTOR)

/* The applied alignment must never land LATER than the true echo: the AEC
 * delay estimator deliberately reports early so PBFDKF sees a POSITIVE residual it can
 * model (a negative residual is non-causal for the filter and cannot be).
 * Measured across n=1..5 and true delays 4..509 ms, the shortfall is 64 or 80
 * samples -- one estimator headroom (32) plus pre-echo block quantisation.
 * 128 samples (8 ms) leaves margin without admitting a whole extra hop. */
#define KD_MAX_UNDERSHOOT 128

typedef struct KnownDelayRun {
    int locked;
    int lock_hop;
    int applied_delay;
    float confidence;
    int changed_events;
    int first_changed_hop;
    /* Hops on which the published `solid` disagreed with the acceptance
     * predicate recomputed from the seam's OWN estimator fields, and hops on
     * which `solid` was already 1 before any alignment generation had been
     * accepted. Both must stay 0: see the invariant note above the recompute
     * in known_delay_run(). */
    int solid_disagreements;
    int solid_before_accept;
    double us_per_hop;
} KnownDelayRun;

/* Drives `hops` hops of a two-path synthetic echo (a dominant path at
 * dominant_delay plus an optional weaker early path) through a MATCHED core
 * with the given bank size, and reports what the shared estimator did. */
static void known_delay_run(int num_filters, int hops,
                            int dominant_delay, float dominant_gain,
                            int early_delay, float early_gain,
                            KnownDelayRun* out) {
    enum { KD_HOP = 256, KD_PAD = 16384 };
    AfConfig cfg = af_default_config(16000);
    Af* p;
    float* far_hist;
    float mic[KD_HOP * AF_CHANNELS];
    float far[KD_HOP];
    uint32_t rng = 0x1234567u;
    clock_t t0, t1;
    int hop, i, ch;
    int expect_solid = 0;

    memset(out, 0, sizeof(*out));
    out->lock_hop = -1;
    out->applied_delay = -1;
    out->first_changed_hop = -1;

    cfg.fft_size = 512;             /* the ULCNet grid: hop 256 */
    cfg.delay_num_filters = num_filters;
    p = af_test_create(&cfg);
    if (!p) return;

    far_hist = (float*)malloc((size_t)(hops * KD_HOP + KD_PAD) * sizeof(float));
    if (!far_hist) { af_test_destroy(p); return; }
    for (i = 0; i < hops * KD_HOP + KD_PAD; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        far_hist[i] = 0.25f * (((float)(rng >> 8) * (1.0f / 16777216.0f)) - 0.5f);
    }

    t0 = clock();
    for (hop = 0; hop < hops; ++hop) {
        AfFront pre;
        int base = hop * KD_HOP + KD_PAD;
        for (i = 0; i < KD_HOP; ++i) {
            int t = base + i;
            float echo = dominant_gain * far_hist[t - dominant_delay] +
                         early_gain * far_hist[t - early_delay];
            far[i] = far_hist[t];
            for (ch = 0; ch < AF_CHANNELS; ++ch)
                mic[i * AF_CHANNELS + ch] = echo;
        }
        if (af_process_front(p, mic, far, &pre) !=
            AF_OK) break;
        /* The published `solid` must be exactly "a usable accepted alignment generation exists",
         * recomputed from the other published estimator fields (confidence >= 1.0 is
         * delay_aec_is_solid(), estimator_updates is delay_aec_n_updates(); the remaining term,
         * estimated >= 0, is implied by confidence >= 1.0). Sticky: a confidence dip must not
         * retract an alignment still applied.
         * MUTATIONS: publishing `solid` on COARSE confidence makes it lead the accepted delay and
         * both counters go non-zero; widening the wrapper's `n_updates >= 3` term makes it lag and
         * solid_disagreements alone goes non-zero. */
        if (pre.delay.changed) {
            out->changed_events += 1;
            if (out->first_changed_hop < 0) out->first_changed_hop = hop;
        }
        {
            int accepted_now = pre.delay.confidence >= 1.0f &&
                               pre.delay.estimator_updates >= 3;
            expect_solid = expect_solid || accepted_now;
            if (pre.delay.solid != expect_solid) out->solid_disagreements += 1;
            if (pre.delay.solid && out->first_changed_hop < 0)
                out->solid_before_accept += 1;
        }
        if (!out->locked && pre.delay.solid) {
            out->locked = 1;
            out->lock_hop = hop;
            out->applied_delay = pre.delay.delay_samples;
            out->confidence = pre.delay.confidence;
        }
    }
    t1 = clock();
    out->us_per_hop = hops > 0
        ? (double)(t1 - t0) * 1e6 / (double)CLOCKS_PER_SEC / (double)hops
        : 0.0;

    free(far_hist);
    af_test_destroy(p);
}

static void test_known_delay_acquisition_and_coverage(void) {
    /* One point just INSIDE each bank's reliable ceiling and one just
     * OUTSIDE it (the next bank's inside point, which is by construction
     * beyond this bank's reach). */
    static const int inside_ms[6] = { 0, 125, 221, 317, 413, 509 };
    int n;

    printf("known-delay acquisition (16 kHz, hop 256, synthetic echo):\n");
    for (n = 1; n <= 5; ++n) {
        KnownDelayRun in_range, out_of_range;
        int inside = inside_ms[n] * 16;             /* ms -> samples @16k */
        /* n=5 has no next bank; 9271 samples = 579.44 ms is the real
         * recording that first exposed its ceiling. */
        int outside = (n == 5) ? 9271 : inside_ms[n + 1] * 16;
        char label[192];

        known_delay_run(n, 200, inside, 0.6f, inside, 0.0f, &in_range);
        known_delay_run(n, 200, outside, 0.6f, outside, 0.0f, &out_of_range);

        printf("  n=%d ceiling %d samples (%.2f ms): "
               "in-range %d ms -> lock hop %d applied %d (short by %d); "
               "out-of-range %d ms -> %s\n",
               n, KD_RELIABLE_SAMPLES(n), KD_RELIABLE_SAMPLES(n) / 16.0,
               inside_ms[n], in_range.lock_hop, in_range.applied_delay,
               in_range.locked ? inside - in_range.applied_delay : -1,
               outside / 16,
               out_of_range.locked ? "LOCKED" : "no lock");

        snprintf(label, sizeof(label),
                 "n=%d acquires a %d ms bulk delay within 60 hops (lock hop "
                 "%d, inside its %.2f ms ceiling)", n, inside_ms[n],
                 in_range.lock_hop, KD_RELIABLE_SAMPLES(n) / 16.0);
        CHECK(in_range.locked && in_range.lock_hop >= 0 &&
              in_range.lock_hop < 60, label);

        /* The alignment contract, not a tolerance pulled from the air: the
         * applied delay may sit EARLY of the true echo (PBFDKF then models a
         * positive residual) but must never sit LATE. */
        snprintf(label, sizeof(label),
                 "n=%d applied delay is early-or-exact and short by at most "
                 "%d samples (true %d, applied %d)",
                 n, KD_MAX_UNDERSHOOT, inside, in_range.applied_delay);
        CHECK(in_range.locked &&
              inside - in_range.applied_delay >= 0 &&
              inside - in_range.applied_delay <= KD_MAX_UNDERSHOOT, label);

        snprintf(label, sizeof(label),
                 "n=%d exactly one alignment generation for a static delay "
                 "(%d)", n, in_range.changed_events);
        CHECK(in_range.changed_events == 1, label);

        /* `solid` publishes "a usable accepted alignment exists", so it can
         * never lead the accepted delay: the hop it first goes 1 is exactly
         * the hop the generation is accepted on, and it agrees with the
         * acceptance predicate on EVERY hop of the run (in range and out of
         * range alike -- the out-of-range run must stay 0/0 too, which is
         * what stops these from being vacuous). */
        snprintf(label, sizeof(label),
                 "n=%d `solid` never leads the accepted delay (first solid "
                 "hop %d, first alignment generation hop %d, %d early-solid "
                 "hops)", n, in_range.lock_hop, in_range.first_changed_hop,
                 in_range.solid_before_accept);
        CHECK(in_range.solid_before_accept == 0 &&
              in_range.first_changed_hop == in_range.lock_hop, label);
        snprintf(label, sizeof(label),
                 "n=%d published `solid` equals the sticky acceptance "
                 "predicate on every hop (%d in-range and %d out-of-range "
                 "disagreements)", n, in_range.solid_disagreements,
                 out_of_range.solid_disagreements);
        CHECK(in_range.solid_disagreements == 0 &&
              out_of_range.solid_disagreements == 0, label);

        snprintf(label, sizeof(label),
                 "n=%d does NOT acquire a %d ms bulk delay (beyond its "
                 "%.2f ms ceiling)", n, outside / 16,
                 KD_RELIABLE_SAMPLES(n) / 16.0);
        CHECK(outside > KD_RELIABLE_SAMPLES(n) && !out_of_range.locked, label);
    }
}

/* An out-of-range bulk delay is not always a clean "never locks": with any in-range echo
 * component (early reflection, second speaker path) the estimator locks onto THAT with full
 * confidence while the dominant path stays unmodelled. This reproduces a dominant path at
 * 579.44 ms with a confident lock at ~32 ms. The seam cannot see the error (solid=1,
 * confidence=1.0, as for a correct lock), so what is asserted is the applied delay against
 * an independently known ground-truth delay. */
static void test_known_delay_mislock_is_detectable(void) {
    KnownDelayRun mislock, control;
    const int dominant = 9271;      /* 579.44 ms -- beyond every bank */
    const int early = 512;          /* 32 ms -- inside every bank */
    int error_samples;
    char label[224];

    known_delay_run(5, 200, dominant, 0.6f, early, 0.5f, &mislock);
    /* Control: same two paths, but the dominant one inside the ceiling.
     * The estimator must prefer the dominant path, and the ground-truth
     * check must stay quiet -- otherwise the check below is just "always
     * flags", which would prove nothing. */
    known_delay_run(5, 200, 3536, 0.6f, early, 0.5f, &control);

    error_samples = mislock.locked ? dominant - mislock.applied_delay : -1;
    printf("known-delay mislock: dominant %d samples (%.2f ms) + early %d "
           "(%.2f ms) -> lock hop %d applied %d (%.2f ms) confidence %.2f, "
           "wrong by %d samples (%.2f ms)\n",
           dominant, dominant / 16.0, early, early / 16.0,
           mislock.lock_hop, mislock.applied_delay,
           mislock.applied_delay / 16.0, (double)mislock.confidence,
           error_samples, error_samples / 16.0);

    CHECK(mislock.locked,
          "an in-range early path makes an out-of-range bulk delay lock "
          "anyway");
    CHECK(mislock.confidence >= 1.0f,
          "the mislock is reported at FULL confidence -- the seam's own "
          "fields cannot distinguish it from a correct lock");
    snprintf(label, sizeof(label),
             "ground-truth comparison FLAGS the mislock: applied is short by "
             "%d samples (%.2f ms), far past the %d-sample alignment "
             "contract", error_samples, error_samples / 16.0,
             KD_MAX_UNDERSHOOT);
    CHECK(mislock.locked && error_samples > KD_MAX_UNDERSHOOT, label);

    snprintf(label, sizeof(label),
             "control: with the dominant path in range the SAME check stays "
             "quiet (applied %d, short by %d)",
             control.applied_delay,
             control.locked ? 3536 - control.applied_delay : -1);
    CHECK(control.locked &&
          3536 - control.applied_delay >= 0 &&
          3536 - control.applied_delay <= KD_MAX_UNDERSHOOT, label);
}

/* Report the 4ch numbers by calling the pool queries. n drives the one shared estimator; the four lanes are
 * EXTERNAL_ALIGNED and must not move with it, so the mono "11,264 B per filter per AEC instance (16 kHz)" contract is not multiplied by
 * four here. */
static void test_known_delay_memory_and_cost(void) {
    AecMemBreakdown b[6];
    AecMemReq req[6];
    size_t lanes[6], est[6], total[6], ring[6], flow[6];
    KnownDelayRun cost;
    int n;
    int ok = 1;
    char label[192];

    /* The pieces of the one pool: four lanes, the shared estimator, the ring and the flow's own buffers. */
    for (n = 1; n <= 5; ++n) {
        AfConfig cfg = af_default_config(16000);
        AecTestCfg ac;
        cfg.fft_size = 512;
        cfg.delay_num_filters = n;
        af_to_aec(&cfg, &ac);
        if (aec_get_mem_breakdown(&ac.c, &b[n]) != AEC_OK || aec_tcfg_mem_requirements(&ac, &req[n]) != 0 ||
            b[n].estimator_bytes == 0) {
            ok = 0;
            break;
        }
        lanes[n] = b[n].lane_bytes;
        est[n] = b[n].estimator_bytes;
        total[n] = b[n].total_bytes;
        ring[n] = b[n].ring_bytes;
        flow[n] = b[n].fusion_bytes;
    }
    CHECK(ok, "4ch memory breakdown and pool query answer for every n=1..5");
    if (!ok) return;

    printf("4ch pool vs matched-filter bank size (16 kHz, fft 512):\n");
    for (n = 1; n <= 5; ++n) {
        printf("  n=%d  lanes(4x AEC) %9zu  shared est %9zu  ring %9zu  flow buffers %9zu  total %9zu",
               n, lanes[n], est[n], ring[n], flow[n], total[n]);
        if (n > 1)
            printf("  (+%lld vs n=%d)", (long long)total[n] - (long long)total[n - 1], n - 1);
        printf("\n");
    }

    for (n = 2; n <= 5; ++n) {
        snprintf(label, sizeof(label),
                 "n=%d leaves the four lane AEC pools untouched (%zu bytes, same as n=1) -- n lives in the ONE shared estimator",
                 n, lanes[n]);
        CHECK(lanes[n] == lanes[1], label);

        snprintf(label, sizeof(label), "n=%d costs strictly more estimator RAM than n=%d (%zu > %zu)", n, n - 1, est[n], est[n - 1]);
        CHECK(est[n] > est[n - 1], label);

        snprintf(label, sizeof(label), "n=%d leaves the flow's own buffers untouched (%zu bytes)", n, flow[n]);
        CHECK(flow[n] == flow[1], label);

        /* The ring is the estimator's reach + hop: one more filter reaches 384 * 4 samples further. */
        snprintf(label, sizeof(label), "n=%d ring grows by one filter's reach (%zu - %zu bytes)", n, ring[n], ring[n - 1]);
        CHECK(ring[n] - ring[n - 1] == 384u * 4u * sizeof(float), label);

        snprintf(label, sizeof(label),
                 "n=%d total grows by exactly the per-filter cost of one shared estimator bank (%lld bytes, same as every other "
                 "step)", n, (long long)total[n] - (long long)total[n - 1]);
        CHECK((long long)total[n] - (long long)total[n - 1] == (long long)total[2] - (long long)total[1], label);

        snprintf(label, sizeof(label), "n=%d breakdown total agrees with the pool query (%zu == %llu)", n, b[n].total_bytes,
                 (unsigned long long)req[n].bytes);
        CHECK((unsigned long long)b[n].total_bytes == (unsigned long long)req[n].bytes, label);
    }

    /* Rough per-hop CPU, recorded rather than tuned: one hop is 256 samples
     * = 16 ms of audio at 16 kHz, so the real-time factor is
     * us_per_hop / 16000. The bound is a liveness guard (a catastrophic
     * regression), not a performance target -- this is a host measurement on
     * whatever machine ran the suite, not a board number. */
    known_delay_run(5, 400, 3536, 0.6f, 3536, 0.0f, &cost);
    printf("known-delay cost: 4 lanes + shared estimator (n=5) = "
           "%.1f us/hop, %.4f x real time (hop = 16.00 ms of audio)\n",
           cost.us_per_hop, cost.us_per_hop / 16000.0);
    snprintf(label, sizeof(label),
             "4ch core runs faster than real time on the host (%.1f us/hop "
             "vs a 16000 us budget)", cost.us_per_hop);
    CHECK(cost.us_per_hop > 0.0 && cost.us_per_hop < 16000.0, label);
}

/* Shared-delay change admission and the four-lane realign sweep. A published `changed`
 * sweeps aec_apply_external_realign() over all four lanes; the rows pin that the sweep
 * happens on exactly the hops it should, sized four, and that the applied alignment and
 * the sweep are one event (the delay never moves on a hop without `changed`). Movement is
 * absorbed at two scales: DelayAec publishes on a 16-downsampled-sample grid (64 native
 * samples at 16 kHz), measured by the WANDER row; the wrapper's admission (> 32 samples,
 * repeated within 16) holds every movement for one hop, measured by MOVING and SHIFT. */

typedef struct RealignRun {
    int acquisition_hop;        /* first `changed` (the acquisition)         */
    int later_changes;          /* `changed` hops after the acquisition      */
    int last_change_hop;
    int applied_at_end;
    int hold_hops;              /* hops holding an unconfirmed candidate     */
    int unconfirmed_changes;    /* accepted with no candidate held first     */
    int silent_delay_moves;     /* applied delay moved without a `changed`   */
    int wrong_sized_sweeps;     /* a `changed` hop that did not realign 4    */
    int sweeps_without_change;  /* lanes realigned on a hop with no `changed`*/
    long warm;
    long soft;
} RealignRun;

/* Shared synthesis for the realign/admission scenes: xorshift far history
 * and one hop of 4-lane echo at the given dominant delay (optional
 * simultaneous second path at delay_b when second_gain > 0). */
static void realign_scene_history(float* far_hist, int count) {
    uint32_t rng = 0x1234567u;
    int i;
    for (i = 0; i < count; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        far_hist[i] = 0.25f *
            (((float)(rng >> 8) * (1.0f / 16777216.0f)) - 0.5f);
    }
}

static void realign_scene_hop(const float* far_hist, int base, int hop_len,
                              int dominant, float second_gain, int delay_b,
                              float* mic, float* far) {
    int i, ch;
    for (i = 0; i < hop_len; ++i) {
        int t = base + i;
        float echo = 0.6f * far_hist[t - dominant];
        if (second_gain > 0.0f) echo += second_gain * far_hist[t - delay_b];
        far[i] = far_hist[t];
        for (ch = 0; ch < AF_CHANNELS; ++ch)
            mic[i * AF_CHANNELS + ch] = echo;
    }
}

/* Drives one 16 kHz/hop-256 MATCHED scene and reports what the shared alignment did: a
 * single echo path at `delay_a`, except that second_gain > 0 adds a simultaneous second
 * path at delay_b (an ambiguous scene whose peak moves on its own, like real jitter),
 * shift_at >= 0 moves the path to delay_b from that hop on, and wander_hops > 0 alternates
 * it between delay_a and delay_b every wander_hops hops. */
static void realign_run(int hops, int delay_a, int delay_b, float second_gain,
                        int shift_at, int wander_hops, RealignRun* out) {
    enum { RA_HOP = 256, RA_PAD = 16384 };
    AfConfig cfg = af_default_config(16000);
    Af* p;
    float* far_hist;
    float mic[RA_HOP * AF_CHANNELS];
    float far[RA_HOP];
    int hop;
    int previous_delay = 0;
    int previous_pending = -1;
    long previous_sweeps = 0;

    memset(out, 0, sizeof(*out));
    out->acquisition_hop = -1;
    out->last_change_hop = -1;
    out->applied_at_end = -1;

    cfg.fft_size = 512;             /* the ULCNet grid: hop 256 */
    p = af_test_create(&cfg);
    if (!p) return;

    far_hist = (float*)malloc((size_t)(hops * RA_HOP + RA_PAD) * sizeof(float));
    if (!far_hist) { af_test_destroy(p); return; }
    realign_scene_history(far_hist, hops * RA_HOP + RA_PAD);

    for (hop = 0; hop < hops; ++hop) {
        AfFront pre;
        int base = hop * RA_HOP + RA_PAD;
        int dominant = delay_a;
        int pending;
        long sweeps;
        if (shift_at >= 0 && hop >= shift_at) dominant = delay_b;
        if (wander_hops > 0 && ((hop / wander_hops) % 2)) dominant = delay_b;
        realign_scene_hop(far_hist, base, RA_HOP, dominant, second_gain,
                          delay_b, mic, far);
        if (af_process_front(p, mic, far, &pre) !=
            AF_OK) break;

        sweeps = af_realign_warm_lane_count(p) +
                 af_realign_soft_lane_count(p);
        pending = af_pending_delay_candidate(p);
        if (pending >= 0) out->hold_hops += 1;

        if (pre.delay.changed) {
            if (out->acquisition_hop < 0) {
                out->acquisition_hop = hop;
            } else {
                out->later_changes += 1;
                /* The candidate must have been HELD on the previous hop and
                 * must be the value applied: that pair is the whole
                 * two-step rule, and it is what disappears if acceptance
                 * happens on the first sighting. */
                if (previous_pending < 0 ||
                    abs(previous_pending - pre.delay.delay_samples) >= 16)
                    out->unconfirmed_changes += 1;
            }
            out->last_change_hop = hop;
            if (sweeps - previous_sweeps != AF_CHANNELS)
                out->wrong_sized_sweeps += 1;
        } else {
            if (hop > 0 && pre.delay.delay_samples != previous_delay)
                out->silent_delay_moves += 1;
            if (sweeps != previous_sweeps) out->sweeps_without_change += 1;
        }

        previous_delay = pre.delay.delay_samples;
        previous_pending = pending;
        previous_sweeps = sweeps;
        out->applied_at_end = pre.delay.delay_samples;
    }

    out->warm = af_realign_warm_lane_count(p);
    out->soft = af_realign_soft_lane_count(p);
    free(far_hist);
    af_test_destroy(p);
}

static void test_shared_delay_change_admission(void) {
    RealignRun wander, moving, shift;
    char label[224];

    /* WANDER: the bulk delay itself moves by 24 samples (1.5 ms) back and
     * forth, below both the wrapper's 32-sample admission and DelayAec's own
     * 64-sample output grid. Nothing may reach the lanes: one acquisition
     * sweep for the whole run and an alignment that never moves again. */
    realign_run(240, 512, 536, 0.0f, -1, 30, &wander);
    printf("shared-delay admission: wander +/-24 -> acquired hop %d applied "
           "%d, %d later generations, %ld warm + %ld soft lane realigns\n",
           wander.acquisition_hop, wander.applied_at_end, wander.later_changes,
           wander.warm, wander.soft);
    snprintf(label, sizeof(label),
             "a 24-sample bulk-delay wander starts no alignment generation "
             "after the acquisition (%d later generations)",
             wander.later_changes);
    CHECK(wander.acquisition_hop >= 0 && wander.later_changes == 0, label);
    snprintf(label, sizeof(label),
             "and realigns the four lanes exactly once, for the acquisition "
             "itself (%ld warm + %ld soft = %ld lane calls)",
             wander.warm, wander.soft, wander.warm + wander.soft);
    CHECK(wander.warm + wander.soft == AF_CHANNELS, label);

    /* MOVING: the echo path travels 512 <-> 1024 samples in 400-hop blocks, so the shared
     * estimate moves several times, each move far larger than the admission band. This row
     * makes the two-step rule falsifiable (a single-generation scene would pass the hold
     * assertions for free). Blocks and run are long because the matched filter is
     * duty-cycled: each re-lock costs a number of analysed hops. */
    realign_run(2400, 512, 1024, 0.0f, -1, 400, &moving);
    printf("shared-delay admission: 512<->1024 in 400-hop blocks -> acquired "
           "hop %d, %d later generations, %d holds, applied %d, %ld warm + "
           "%ld soft\n", moving.acquisition_hop, moving.later_changes,
           moving.hold_hops, moving.applied_at_end, moving.warm, moving.soft);
    snprintf(label, sizeof(label),
             "control: the moving-path scene really does re-lock several "
             "times (%d later generations)", moving.later_changes);
    CHECK(moving.later_changes >= 4, label);
    snprintf(label, sizeof(label),
             "every alignment change is admitted only after the same value is "
             "seen on a second eligible hop (%d accepted on a first sighting, "
             "%d hold hops for %d changes)", moving.unconfirmed_changes,
             moving.hold_hops, moving.later_changes);
    CHECK(moving.unconfirmed_changes == 0 &&
          moving.hold_hops == moving.later_changes, label);
    snprintf(label, sizeof(label),
             "the applied alignment never moves on a hop that publishes no "
             "`changed` (%d silent moves)", moving.silent_delay_moves);
    CHECK(moving.silent_delay_moves == 0, label);
    snprintf(label, sizeof(label),
             "every generation realigns exactly four lanes and no other hop "
             "realigns any (%d wrong-sized sweeps, %d sweeps without a "
             "generation)", moving.wrong_sized_sweeps,
             moving.sweeps_without_change);
    CHECK(moving.wrong_sized_sweeps == 0 &&
          moving.sweeps_without_change == 0, label);
    snprintf(label, sizeof(label),
             "the warm/soft split accounts for every lane call (%ld + %ld == "
             "4 x %d generations)", moving.warm, moving.soft,
             moving.later_changes + 1);
    CHECK(moving.warm + moving.soft ==
          (long)AF_CHANNELS * (moving.later_changes + 1),
          label);

    /* SHIFT: the echo really moves, 512 -> 1024 samples, and stays there. One
     * generation, held for one hop first, then all four lanes realigned. The
     * run outlasts the re-lock for the same reason the MOVING scene's blocks
     * are long. */
    realign_run(700, 512, 1024, 0.0f, 100, 0, &shift);
    printf("shared-delay admission: shift 512 -> 1024 at hop 100 -> acquired "
           "hop %d, adopted hop %d applied %d, %ld warm + %ld soft\n",
           shift.acquisition_hop, shift.last_change_hop, shift.applied_at_end,
           shift.warm, shift.soft);
    snprintf(label, sizeof(label),
             "a sustained 512-sample shift starts exactly one new alignment "
             "generation (%d), on hop %d", shift.later_changes,
             shift.last_change_hop);
    CHECK(shift.later_changes == 1 && shift.last_change_hop > 100, label);
    snprintf(label, sizeof(label),
             "it is held for confirmation for exactly one hop before it is "
             "applied (%d holds, %d accepted on a first sighting)",
             shift.hold_hops, shift.unconfirmed_changes);
    CHECK(shift.hold_hops == 1 && shift.unconfirmed_changes == 0, label);
    snprintf(label, sizeof(label),
             "and then realigns all four lanes on that one hop (%ld warm + "
             "%ld soft = %ld, two generations x 4 lanes)",
             shift.warm, shift.soft, shift.warm + shift.soft);
    CHECK(shift.wrong_sized_sweeps == 0 &&
          shift.sweeps_without_change == 0 &&
          shift.warm + shift.soft == 2 * AF_CHANNELS, label);
    snprintf(label, sizeof(label),
             "the alignment ends on the moved path, early-or-exact within the "
             "%d-sample contract (applied %d for a true 1024)",
             KD_MAX_UNDERSHOOT, shift.applied_at_end);
    CHECK(1024 - shift.applied_at_end >= 0 &&
          1024 - shift.applied_at_end <= KD_MAX_UNDERSHOOT, label);
}

/* ============================================================================
 * Held-candidate lifetime. The admission state machine is driven directly: through a
 * stream DelayAec re-offers a movement every hop, so the candidate is resolved on the next
 * eligible hop and its TTL never runs out. What a scene does cover (a candidate is really
 * held and spends a realign when confirmed) is asserted in
 * test_shared_delay_change_admission(); the reset row below clears a candidate that a real
 * stream produced.
 * ========================================================================== */

/* One hop, spent the way update_shared_delay() spends it: the held candidate
 * always ages, and the estimate is only offered on a hop the estimator was
 * eligible on. */
static int admission_hop(AfDelayAdmission* admission, int accepted,
                         int estimated, int eligible) {
    af_admission_age(admission);
    if (!eligible) return 0;
    return af_admission_offer(admission, accepted, estimated);
}

static void test_delay_change_candidate_ttl(void) {
    const int applied = 512;        /* the alignment in force */
    const int moved = 1024;         /* a movement far outside the band */
    AfDelayAdmission a;
    AfConfig cfg;
    Af* p;
    char label[224];
    int i;
    int held_hop = -1;
    int candidate_before_reset = -1;
    int candidate_after_reset = 0;

    /* Offered once, repeated on the very next hop: admitted, and nothing is
     * left held afterwards. */
    memset(&a, 0, sizeof(a));
    CHECK(admission_hop(&a, applied, moved, 1) == 0 &&
          a.ttl == AF_DELAY_CHANGE_CANDIDATE_TTL && a.candidate == moved,
          "a first sighting is held, not applied, with a full life");
    CHECK(admission_hop(&a, applied, moved, 1) == 1 &&
          a.ttl == 0,
          "the same movement on the next hop is admitted and releases the "
          "candidate");

    /* One hop without a usable estimate does not end it: lib/aec's rule is a
     * bounded life, not a strictly consecutive pair. */
    memset(&a, 0, sizeof(a));
    admission_hop(&a, applied, moved, 1);
    CHECK(admission_hop(&a, applied, 0, 0) == 0 && a.ttl > 0,
          "a hop with no usable estimate spends one hop of the candidate's "
          "life and keeps it");
    CHECK(admission_hop(&a, applied, moved, 1) == 1,
          "the movement is still admitted when it returns inside that life");

    /* Aged out: the candidate is gone, and a single reappearance can only
     * start a new one. */
    memset(&a, 0, sizeof(a));
    admission_hop(&a, applied, moved, 1);
    for (i = 0; i < AF_DELAY_CHANGE_CANDIDATE_TTL - 1; ++i)
        admission_hop(&a, applied, 0, 0);
    snprintf(label, sizeof(label),
             "the candidate survives exactly %d hops without a usable "
             "estimate (life left %d)",
             AF_DELAY_CHANGE_CANDIDATE_TTL - 1, a.ttl);
    CHECK(a.ttl > 0 && a.candidate == moved, label);
    admission_hop(&a, applied, 0, 0);
    CHECK(a.ttl == 0 && a.candidate == -1,
          "one hop later it has expired, holding nothing");
    CHECK(admission_hop(&a, applied, moved, 1) == 0 &&
          a.ttl == AF_DELAY_CHANGE_CANDIDATE_TTL,
          "a single reappearance after expiry is NOT admitted -- it is only a "
          "new first sighting");
    CHECK(admission_hop(&a, applied, moved, 1) == 1,
          "and it takes a repeat inside the new life to admit it");

    /* An estimate back at the alignment in force is absorbed without ending
     * the candidate (lib/aec's Path B has no such clear); a movement to a
     * different place replaces it with a full life. */
    memset(&a, 0, sizeof(a));
    admission_hop(&a, applied, moved, 1);
    CHECK(admission_hop(&a, applied,
                        applied + AF_DELAY_CHANGE_MIN_SAMPLES, 1) == 0 &&
          a.candidate == moved && a.ttl > 0,
          "an estimate inside the admission band is absorbed and leaves the "
          "held candidate to age");
    CHECK(admission_hop(&a, applied, 2048, 1) == 0 &&
          a.candidate == 2048 &&
          a.ttl == AF_DELAY_CHANGE_CANDIDATE_TTL,
          "a movement somewhere else entirely replaces the candidate and "
          "restarts its life");

    /* End to end: a candidate a real stream produced is cleared by reset(),
     * together with the life left on it. The path alternates in long blocks
     * and the run is long: the shared estimator's matched filter is
     * duty-cycled, so a movement is published after a number of ANALYSED
     * hops, not after a number of hops -- a scene that switches faster than
     * the estimator can answer holds no candidate to clear. */
    cfg = af_default_config(16000);
    cfg.fft_size = 512;
    p = af_test_create(&cfg);
    CHECK(p != NULL, "candidate-reset scene creates");
    if (p) {
        enum { TR_HOP = 256, TR_PAD = 16384, TR_HOPS = 1600 };
        float* far_hist = (float*)malloc(
            (size_t)(TR_HOPS * TR_HOP + TR_PAD) * sizeof(float));
        float mic[TR_HOP * AF_CHANNELS];
        float far[TR_HOP];
        int hop;
        if (far_hist) {
            realign_scene_history(far_hist, TR_HOPS * TR_HOP + TR_PAD);
            for (hop = 0; hop < TR_HOPS && held_hop < 0; ++hop) {
                AfFront pre;
                int base = hop * TR_HOP + TR_PAD;
                int dominant = ((hop / 400) % 2) ? 1024 : 512;
                realign_scene_hop(far_hist, base, TR_HOP, dominant, 0.0f, 0,
                                  mic, far);
                if (af_process_front(p, mic, far, &pre) !=
                    AF_OK) break;
                candidate_before_reset =
                    af_pending_delay_candidate(p);
                if (candidate_before_reset >= 0) {
                    held_hop = hop;
                    af_reset(p);
                    candidate_after_reset =
                        af_pending_delay_candidate(p);
                }
            }
        }
        free(far_hist);
        af_test_destroy(p);
    }
    snprintf(label, sizeof(label),
             "reset() clears a candidate a real stream was holding (hop %d, "
             "held %d, after reset %d)",
             held_hop, candidate_before_reset, candidate_after_reset);
    CHECK(held_hop >= 0 && candidate_before_reset >= 0 &&
          candidate_after_reset == -1, label);
}


/* ============================================================================
 * Runtime strength control. The lanes run a lane-level residual mode (STATE or ESTIMATE) and never reach their
 * own residual suppressor, so a level change pushed at them is a no-op: the setter must
 * move the shared post-stage suppressor (the AecRes), and the retargeted floor must survive
 * a reset (test 3 below).
 * ========================================================================== */

/* Drives `hops` hops of the shared strength/timing stimulus. When
 * last_pre_us/last_post_us are non-NULL they receive the wall-clock cost of
 * the FINAL hop's af_process_front/fuse, which is what lets the timing
 * test bracket the same workload this drives without a second copy of the
 * stimulus. */
static uint32_t elapsed_us(const struct timespec* a, const struct timespec* b) {
    return (uint32_t)((b->tv_sec - a->tv_sec) * 1000000
                      + (b->tv_nsec - a->tv_nsec) / 1000);
}

static int feed_strength_hops(Af* p, int hops,
                              uint32_t* last_pre_us,
                              uint32_t* last_post_us) {
    int hop = af_hop_size(p);
    int n = af_n_freqs(p);
    float* mics = (float*)calloc((size_t)hop * 4u, sizeof(float));
    float* far = (float*)calloc((size_t)hop, sizeof(float));
    float* out = (float*)calloc((size_t)hop, sizeof(float));
    /* Channel-major Complex[4][n_freqs] -- the weights contract in
     * aec.h. Sizing this as [n_freqs] would have af_fuse() read
     * three channels past the end. */
    Complex* w = (Complex*)calloc(
        (size_t)AF_CHANNELS * (size_t)n, sizeof(Complex));
    AfFront pre;
    int h, i, ch, k;
    for (ch = 0; ch < AF_CHANNELS; ++ch)
        for (k = 0; k < n; ++k) {
            w[(size_t)ch * (size_t)n + (size_t)k].r = 0.25f;
            w[(size_t)ch * (size_t)n + (size_t)k].i = 0.0f;
        }
    for (h = 0; h < hops; ++h) {
        struct timespec a, b;
        for (i = 0; i < hop; ++i) {
            float t = (float)(h * hop + i);
            float r = 0.6f * sinf(0.06f * t);
            far[i] = r;
            for (ch = 0; ch < 4; ++ch)
                mics[(size_t)i * 4u + (size_t)ch] = 0.5f * r + 0.02f * sinf(0.017f * t);
        }
        clock_gettime(CLOCK_MONOTONIC, &a);
        if (af_process_front(p, mics, far, &pre) != AF_OK)
            break;
        clock_gettime(CLOCK_MONOTONIC, &b);
        if (last_pre_us) *last_pre_us = elapsed_us(&a, &b);

        clock_gettime(CLOCK_MONOTONIC, &a);
        if (fuse_only(p, w) != AF_OK) break;
        clock_gettime(CLOCK_MONOTONIC, &b);
        if (last_post_us) *last_post_us = elapsed_us(&a, &b);
    }
    free(mics); free(far); free(out); free(w);
    return h;
}

/* ============================================================================
 * Matched-filter duty cycle
 * ========================================================================== */

/* Long enough that no 1-in-K schedule this file's grids produce could
 * manufacture it by accident (the coarsest K here is well under this), short
 * enough to fit inside the hold window the machine re-arms after. */
#define DUTY_FULL_RATE_RUN 8

typedef struct DutyRun {
    unsigned long long total;
    unsigned long long run;
    int hops_seen;
    /* Every hop must advance the census total exactly once: a census wired to
     * anything but the decision site would drift from what actually ran. */
    int census_total_mismatch;
    /* An unanalysed hop must publish nothing new -- that is the whole
     * contract of feeding without analysing. */
    int updates_on_skipped_hop;
    int stable_tail_hops;
    int stable_tail_analysed;
    int acquisition_hop;
    int longest_analysed_streak_after_change;
    /* Where the machine went back to full rate: the hop the first run of
     * DUTY_FULL_RATE_RUN consecutive analysed hops after the move begins on.
     * Consecutive analysed hops are impossible while the machine is engaged,
     * so this is when it disengaged -- and WHEN is the whole question, since
     * the estimate itself does not move at the point of the change. */
    int first_full_rate_hop;
    int analysed_after_change;
    int hops_after_change;
    int reacquire_hop;
    int reacquire_delay;
    /* Watchdog leak, reconstructed from the peak trajectory: hops on which
     * the peak decayed, and how many of those decayed by anything other than
     * the converted per-hop amount the core stored. A leak that is converted
     * correctly and then not USED leaves every value-level assertion green,
     * so this is the half that holds the branch. */
    int leak_hops;
    int leak_mismatches;
} DutyRun;

/* One 16 kHz/hop-256 MATCHED scene through the shared estimator, reporting
 * what the duty machine did with it. Same single-path echo generator the
 * realign scenes use; `shift_at` >= 0 moves the path to delay_b from that hop
 * on, which is the scene the disengage rule exists for. */
static void duty_run(int hops, int delay_a, int delay_b, int shift_at,
                     int stable_tail, DutyRun* out) {
    enum { DU_HOP = 256, DU_PAD = 32768 };
    AfConfig cfg = af_default_config(16000);
    Af* p;
    float* far_hist;
    float mic[DU_HOP * AF_CHANNELS];
    float far[DU_HOP];
    int hop;
    int previous_updates = 0;
    int streak = 0;
    unsigned long long previous_total = 0;
    unsigned long long previous_run = 0;

    memset(out, 0, sizeof(*out));
    out->acquisition_hop = -1;
    out->reacquire_hop = -1;
    out->reacquire_delay = -1;
    out->first_full_rate_hop = -1;
    out->leak_hops = 0;
    out->leak_mismatches = 0;

    cfg.fft_size = 512;             /* the ULCNet grid: hop 256 */
    /* The throttle is opt-in (the default is full rate) -- this scene exists
     * specifically to characterize the throttle, so it asks for the regime
     * explicitly. 0.5 s is the period this scene's numbers were measured at. */
    cfg.delay_est_period_s = 0.5f;
    p = af_test_create(&cfg);
    if (!p) return;

    far_hist = (float*)malloc((size_t)(hops * DU_HOP + DU_PAD) * sizeof(float));
    if (!far_hist) { af_test_destroy(p); return; }
    realign_scene_history(far_hist, hops * DU_HOP + DU_PAD);

    for (hop = 0; hop < hops; ++hop) {
        AfFront pre;
        int base = hop * DU_HOP + DU_PAD;
        int dominant = (shift_at >= 0 && hop >= shift_at) ? delay_b : delay_a;
        unsigned long long total_now, run_now;
        int analysed;
        float peak_before = af_duty_erle_peak(p);
        realign_scene_hop(far_hist, base, DU_HOP, dominant, 0.0f, delay_b,
                          mic, far);
        if (af_process_front(p, mic, far, &pre) !=
            AF_OK) break;
        out->hops_seen += 1;

        /* The watchdog's steady decay branch, reconstructed. A decrease is its signature (arming
         * jumps the peak to the current lane ERLE and a rise re-seeds it); the other way down,
         * a trip, zeroes the peak outright and is excluded by its value (15.50 -> 0 at hop 312
         * on the moved scene). Compared with == because it is the identical float subtraction:
         * anything else means the branch subtracted a different number than the core
         * converted. */
        {
            float peak_after = af_duty_erle_peak(p);
            if (peak_after < peak_before && peak_after != 0.0f) {
                out->leak_hops += 1;
                if (peak_after != peak_before -
                        af_duty_erle_leak_db(p))
                    out->leak_mismatches += 1;
            }
        }

        total_now = af_duty_hops_total(p);
        run_now = af_duty_hops_run(p);
        analysed = (int)(run_now - previous_run);
        if (total_now - previous_total != 1ull) out->census_total_mismatch += 1;
        if (!analysed && hop > 0 &&
            pre.delay.estimator_updates != previous_updates)
            out->updates_on_skipped_hop += 1;

        /* Tail window: the last stable_tail hops before the shift, or before
         * the end of the run when there is no shift -- so a shift-free call
         * measures its settled tail without a second, byte-identical run. */
        {
            int tail_end = (shift_at >= 0) ? shift_at : hops;
            if (stable_tail > 0 && hop >= tail_end - stable_tail &&
                hop < tail_end) {
                out->stable_tail_hops += 1;
                out->stable_tail_analysed += analysed;
            }
        }
        if (shift_at >= 0 && hop >= shift_at) {
            out->hops_after_change += 1;
            out->analysed_after_change += analysed;
            streak = analysed ? streak + 1 : 0;
            if (streak > out->longest_analysed_streak_after_change)
                out->longest_analysed_streak_after_change = streak;
            if (streak >= DUTY_FULL_RATE_RUN && out->first_full_rate_hop < 0)
                out->first_full_rate_hop = hop - streak + 1;
        }
        if (pre.delay.changed) {
            if (out->acquisition_hop < 0) out->acquisition_hop = hop;
            if (shift_at >= 0 && hop >= shift_at && out->reacquire_hop < 0) {
                out->reacquire_hop = hop;
                out->reacquire_delay = pre.delay.delay_samples;
            }
        }
        previous_total = total_now;
        previous_run = run_now;
        previous_updates = pre.delay.estimator_updates;
    }

    out->total = af_duty_hops_total(p);
    out->run = af_duty_hops_run(p);
    free(far_hist);
    af_test_destroy(p);
}

static void test_duty_cycle_machine(void) {
    DutyRun stable, moved;
    AfConfig cfg;
    Af* p;
    char label[224];

    /* STABLE: one path, held for the whole run. This is exactly the condition
     * the machine is built to decimate (estimate solid and unchanged for the
     * hold window), so "it never armed" is a failure here, not a property of
     * the stimulus. */
    duty_run(400, 2048, 2048, -1, 120, &stable);
    printf("duty cycle: static 2048-sample path -> acquired hop %d, analysed "
           "%llu of %llu hops (%.1f%%)\n", stable.acquisition_hop,
           stable.run, stable.total,
           stable.total ? 100.0 * (double)stable.run / (double)stable.total
                        : 0.0);

    snprintf(label, sizeof(label),
             "the census counts every hop through the shared delay stage "
             "(%llu counted, %d hops driven, %d hops that advanced it by "
             "anything but 1)", stable.total, stable.hops_seen,
             stable.census_total_mismatch);
    CHECK(stable.total == (unsigned long long)stable.hops_seen &&
          stable.hops_seen == 400 && stable.census_total_mismatch == 0, label);

    /* The assertion that can actually fail: a machine that never arms, or a
     * census wired to a plain hop counter, makes run == total. The band is
     * deliberately wide -- how fast the estimate goes solid is
     * stimulus-dependent, and only the decimated REGIME is being pinned. */
    snprintf(label, sizeof(label),
             "engagement drops below full rate once the estimate holds still "
             "(%llu analysed of %llu)", stable.run, stable.total);
    CHECK(stable.run > 0ull && stable.run < stable.total &&
          (double)stable.run < 0.6 * (double)stable.total, label);

    snprintf(label, sizeof(label),
             "the analysed share over the settled tail is decimated, not "
             "merely reduced (%d of the last %d hops)",
             stable.stable_tail_analysed, stable.stable_tail_hops);
    CHECK(stable.stable_tail_hops == 120 &&
          stable.stable_tail_analysed > 0 &&
          stable.stable_tail_analysed <= 30, label);

    snprintf(label, sizeof(label),
             "a hop the matched filter did not analyse publishes no new "
             "estimate (%d hops published one anyway)",
             stable.updates_on_skipped_hop);
    CHECK(stable.updates_on_skipped_hop == 0, label);

    /* MOVED: the echo path really travels, from 2048 to 4096 samples, after
     * the filters have converged on the first one. Nothing in the PUBLISHED
     * estimate changes at the move -- it keeps reporting the old delay with
     * full confidence -- so the only thing that can take the machine back to
     * full rate is the cancellation watchdog. */
    duty_run(900, 2048, 4096, 300, 120, &moved);
    printf("duty cycle: 2048 -> 4096 at hop 300 -> re-acquired hop %d applied "
           "%d, %d analysed of %d hops after the move, longest full-rate run "
           "%d hops\n", moved.reacquire_hop, moved.reacquire_delay,
           moved.analysed_after_change, moved.hops_after_change,
           moved.longest_analysed_streak_after_change);

    snprintf(label, sizeof(label),
             "control: the moved path is really re-acquired, at the new "
             "alignment (hop %d, applied %d for a true 4096)",
             moved.reacquire_hop, moved.reacquire_delay);
    CHECK(moved.reacquire_hop > 300 && moved.reacquire_delay >= 0 &&
          4096 - moved.reacquire_delay >= 0 &&
          4096 - moved.reacquire_delay <= KD_MAX_UNDERSHOOT, label);

    /* While engaged the schedule is 1-in-K, so consecutive analysed hops are
     * impossible: a run of them proves the machine DISENGAGED. Deleting the
     * cancellation watchdog leaves this at 1 -- the estimate itself never
     * moves at the point of the change, so nothing else can disengage it. */
    snprintf(label, sizeof(label),
             "the cancellation watchdog takes the machine back to full rate "
             "PROMPTLY after the move (first run of %d consecutive analysed "
             "hops starts on hop %d, %d hops after it; longest run %d)",
             DUTY_FULL_RATE_RUN, moved.first_full_rate_hop,
             moved.first_full_rate_hop >= 0 ? moved.first_full_rate_hop - 300
                                            : -1,
             moved.longest_analysed_streak_after_change);
    CHECK(moved.first_full_rate_hop >= 300 &&
          moved.first_full_rate_hop - 300 <= 60 &&
          moved.longest_analysed_streak_after_change >= DUTY_FULL_RATE_RUN,
          label);

    snprintf(label, sizeof(label),
             "and the census still accounts for every hop of the moved scene "
             "(%llu counted, %d driven, %d skipped hops that published)",
             moved.total, moved.hops_seen, moved.updates_on_skipped_hop);
    CHECK(moved.total == (unsigned long long)moved.hops_seen &&
          moved.census_total_mismatch == 0 &&
          moved.updates_on_skipped_hop == 0, label);

    /* The watchdog subtracts the CONVERTED leak, not the authored per-hop literal.
     * test_duty_watchdog_leak_is_a_rate pins the value the core stores; this pins that the
     * branch reads it (the only assertion that goes red if a use site reverts to
     * DUTY_ERLE_LEAK_DB). The decay lives on the MOVED scene; on the static one the lanes
     * converge monotonically, so the peak is re-seeded from a rising ERLE and never decays. */
    snprintf(label, sizeof(label),
             "the watchdog subtracts the converted leak on every decay hop "
             "(%d decay hops, %d subtracted something else; static scene "
             "never decays: %d)",
             moved.leak_hops, moved.leak_mismatches, stable.leak_hops);
    CHECK(moved.leak_mismatches == 0 && stable.leak_mismatches == 0, label);
    snprintf(label, sizeof(label),
             "and the decay branch was actually reached (%d hops)",
             moved.leak_hops);
    CHECK(moved.leak_hops > 0, label);

    /* Only MATCHED builds a matched filter, so only MATCHED has a cost to
     * decimate: the other two modes must leave both counters at 0 rather
     * than reporting a saving on a machine that was never built. */
    {
        int mode;
        for (mode = 0; mode < 2; ++mode) {
            float mic[128 * AF_CHANNELS];
            float far[128];
            Complex* weights;
            int hop, i, n_freqs;
            cfg = af_default_config(16000);
            if (mode == 0) {
                cfg.delay_mode = AEC_DELAY_FIXED;
                cfg.fixed_delay_samples = 256;
            } else {
                cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
                cfg.fixed_delay_samples = -1;
            }
            p = af_test_create(&cfg);
            CHECK(p != NULL, "duty census: non-MATCHED core creates");
            if (!p) continue;
            hop = af_hop_size(p);
            n_freqs = af_n_freqs(p);
            weights = (Complex*)calloc(
                (size_t)AF_CHANNELS * n_freqs, sizeof(Complex));
            for (i = 0; i < AF_CHANNELS * n_freqs; ++i)
                weights[i].r = 0.25f;
            for (i = 0; i < 20; ++i) {
                AfFront pre;
                fill_inputs(mic, far, hop, 16000, i);
                if (af_process_front(p, mic, far, &pre) !=
                    AF_OK) break;
                fuse_only(p, weights);
            }
            snprintf(label, sizeof(label),
                     "%s builds no matched filter, so its duty census stays "
                     "at 0/0 (%llu/%llu)",
                     mode == 0 ? "FIXED" : "EXTERNAL_ALIGNED",
                     af_duty_hops_run(p),
                     af_duty_hops_total(p));
            CHECK(af_duty_hops_total(p) == 0ull &&
                  af_duty_hops_run(p) == 0ull, label);
            free(weights);
            af_test_destroy(p);
        }
    }

    /* The census measures ONE run of the machine: reset() restarts it, and
     * counting resumes rather than sticking at zero. */
    {
        float mic[128 * AF_CHANNELS];
        float far[128];
        unsigned long long after_reset;
        int hop, i;
        cfg = af_default_config(16000);
        cfg.enable_post = 0;
        p = af_test_create(&cfg);
        CHECK(p != NULL, "duty census: reset-scope core creates");
        if (p) {
            hop = af_hop_size(p);
            for (i = 0; i < 12; ++i) {
                AfFront pre;
                fill_inputs(mic, far, hop, 16000, i);
                if (af_process_front(p, mic, far, &pre) !=
                    AF_OK) break;
            }
            CHECK(af_duty_hops_total(p) == 12ull,
                  "duty census counts the hops before a reset");
            af_reset(p);
            CHECK(af_duty_hops_total(p) == 0ull &&
                  af_duty_hops_run(p) == 0ull,
                  "af_reset() zeroes the duty census");
            for (i = 0; i < 5; ++i) {
                AfFront pre;
                fill_inputs(mic, far, hop, 16000, i);
                if (af_process_front(p, mic, far, &pre) !=
                    AF_OK) break;
            }
            after_reset = af_duty_hops_total(p);
            snprintf(label, sizeof(label),
                     "and counting resumes after it (%llu)", after_reset);
            CHECK(after_reset == 5ull, label);
            af_test_destroy(p);
        }
    }
}

/* ============================================================================
 * The watchdog leak is a RATE, not a per-hop amount: the duty machine's windows convert
 * from seconds at init, and so must its leak (0.001 dB/hop is 0.1 dB/s only on the 10 ms
 * hop lib/aec calibrated it on). Asserted as the rate at all three shipped grids, none of
 * which is the calibration grid, so reverting the conversion turns all rows red. The
 * calibration grid cannot be built here (hop 160 at 16 kHz), so the retime identity is
 * asserted against the pure conversion bit for bit.
 * ========================================================================== */
/* The authored per-hop amount, restated here because the core keeps it
 * file-local -- same reason LE_ERLE_COLLAPSE_DB is restated further down. A
 * divergence makes this test's premise false rather than its assertions
 * wrong, which is why the per-grid rows below also tie back to the core's own
 * conversion. */
#define DUTY_LEAK_AUTHORED_PER_HOP 0.001f

static void test_duty_watchdog_leak_is_a_rate(void) {
    static const struct { int sample_rate; int fft_size; } grids[] = {
        { 16000, 256 }, { 16000, 512 }, { 48000, 1024 }
    };
    const float target_db_per_s = 0.1f;
    char label[192];
    size_t g;

    for (g = 0; g < sizeof(grids) / sizeof(grids[0]); ++g) {
        AfConfig cfg =
            af_default_config(grids[g].sample_rate);
        Af* p;
        float leak;
        double per_s;
        int hop;

        cfg.fft_size = grids[g].fft_size;
        p = af_test_create(&cfg);
        snprintf(label, sizeof(label),
                 "watchdog leak: core creates at %d Hz / fft %d",
                 grids[g].sample_rate, grids[g].fft_size);
        CHECK(p != NULL, label);
        if (!p) continue;

        hop = af_hop_size(p);
        leak = af_duty_erle_leak_db(p);
        per_s = (double)leak * grids[g].sample_rate / hop;

        snprintf(label, sizeof(label),
                 "watchdog leak: %d Hz / fft %d leaks %.9g dB per hop = "
                 "%.6f dB/s (calibrated 0.100000)",
                 grids[g].sample_rate, grids[g].fft_size, (double)leak, per_s);
        CHECK(fabs(per_s - (double)target_db_per_s) <= 1e-6, label);

        /* No shipped grid is the calibration grid, so every row must have
         * moved off the authored literal -- this is the assertion a reverted
         * conversion fails, on all three rows. */
        snprintf(label, sizeof(label),
                 "watchdog leak: %d Hz / fft %d moved off the 10 ms literal "
                 "(%.9g != 0.001)",
                 grids[g].sample_rate, grids[g].fft_size, (double)leak);
        CHECK(leak != DUTY_LEAK_AUTHORED_PER_HOP, label);

        /* And what init stored is the conversion's own answer for this grid,
         * so the two cannot drift apart. */
        snprintf(label, sizeof(label),
                 "watchdog leak: %d Hz / fft %d stored the conversion's own "
                 "answer", grids[g].sample_rate, grids[g].fft_size);
        CHECK(leak == af_duty_leak_db_for_grid(
                          hop, grids[g].sample_rate), label);

        af_test_destroy(p);
    }

    /* The calibration point. Bit-exact, not within a tolerance: one ulp of
     * drift here would mean the 10 ms grid's behaviour had been rewritten
     * rather than preserved, which is the difference between a retime and a
     * retune. */
    snprintf(label, sizeof(label),
             "watchdog leak: the conversion is the identity at the "
             "calibration grid (hop 160 @ 16 kHz -> %.9g)",
             (double)af_duty_leak_db_for_grid(160, 16000));
    CHECK(af_duty_leak_db_for_grid(160, 16000) ==
              DUTY_LEAK_AUTHORED_PER_HOP, label);

    /* LINEAR in the hop period, which is the whole reason this is its own
     * conversion and not the power law a retention takes: doubling the hop
     * doubles the leak, exactly. A power-law conversion reproduces the
     * calibration point above and fails here. Asserted with == because the
     * two hop periods differ by a power of two, so the quotients do too and
     * neither rounds. */
    snprintf(label, sizeof(label),
             "watchdog leak: doubling the hop doubles the leak exactly "
             "(%.9g -> %.9g)",
             (double)af_duty_leak_db_for_grid(128, 16000),
             (double)af_duty_leak_db_for_grid(256, 16000));
    CHECK(af_duty_leak_db_for_grid(256, 16000) ==
              2.0f * af_duty_leak_db_for_grid(128, 16000), label);

    /* Degenerate grid: fall back to the authored per-hop amount ("not
     * converted") rather than dividing by zero. */
    CHECK(af_duty_leak_db_for_grid(0, 16000) ==
              DUTY_LEAK_AUTHORED_PER_HOP &&
          af_duty_leak_db_for_grid(160, 0) ==
              DUTY_LEAK_AUTHORED_PER_HOP,
          "watchdog leak: a non-positive grid falls back to the authored "
          "per-hop amount");
}

/* ============================================================================
 * Duty cycle vs a delay change the watchdog cannot see
 * ========================================================================== */

/* The counterpart of test_duty_cycle_machine's MOVED scene (good cancellation, watchdog
 * fires): here near-end noise of comparable power on every microphone caps lane ERLE near
 * 3 dB, below DUTY_ERLE_COLLAPSE_DB (6 dB), so the watchdog never arms and the 1-in-K
 * schedule is the whole re-lock mechanism. The scene runs with the schedule pinned at full
 * rate (af_pin_duty_full_rate()) and with the throttle opted into via
 * cfg.delay_est_period_s (K = 6 here: round(0.5/0.016)/5). The numbers are measured, so a
 * change that widens the gap fails here. The estimate-moved disengage shortens neither
 * latency (it fires after the matched filter found the new peak); the census check holds it. */

#define LE_HOP              256
#define LE_PAD              32768
#define LE_SHIFT_AT         400
#define LE_DELAY_A          2048
#define LE_DELAY_B          4096
#define LE_ECHO_GAIN        0.6f
#define LE_NEAR_NOISE_GAIN  0.15f
#define LE_SETTLED_WINDOW   60
/* Analysis period in hops at 16 kHz / hop 256, and the full-rate re-lock
 * bound (measured 319 hops, identical on both backends and both SIMD
 * settings). The scene runs exactly LE_DUTY_K * LE_FULL_RATE_BOUND hops
 * after the move, so "the duty arm re-locked before the scene ended" IS the
 * "within K times the full-rate bound" assertion -- there is no second,
 * looser number to drift. */
#define LE_DUTY_K           6
#define LE_FULL_RATE_BOUND  400
#define LE_HOPS             (LE_SHIFT_AT + LE_DUTY_K * LE_FULL_RATE_BOUND)
/* The lane-ERLE figure the watchdog arms on, restated here because the core
 * keeps it file-local; a divergence makes this test's premise false. */
#define LE_ERLE_COLLAPSE_DB 6.0f

typedef struct LowErleRun {
    int acquisition_hop;
    /* First hop at or after the move that publishes an alignment covering
     * the new path. Same acceptance window the realign scenes use. */
    int relock_hop;
    int relock_delay;
    int hops_after;
    int analysed_after;
    /* Best per-hop lane ERLE anywhere in the run, measured from the core's
     * own linear output against the microphone it was given -- audio, not a
     * counter. If this ever clears LE_ERLE_COLLAPSE_DB the scene has stopped
     * being the scene. */
    double peak_erle_db;
    /* The same figure over the settled window immediately before the move. */
    double settled_erle_db;
    unsigned long long run;
    unsigned long long total;
} LowErleRun;

static void low_erle_run(int pin_full_rate, LowErleRun* out) {
    AfConfig cfg = af_default_config(16000);
    Af* p;
    float* far_hist;
    float mic[LE_HOP * AF_CHANNELS];
    float far[LE_HOP];
    uint32_t noise_rng = 0x77aa55u;
    double settled_mic = 0.0;
    double settled_err = 0.0;
    unsigned long long previous_run = 0;
    int hop;

    memset(out, 0, sizeof(*out));
    out->acquisition_hop = -1;
    out->relock_hop = -1;
    out->relock_delay = -1;

    cfg.fft_size = 512;             /* the ULCNet grid: hop 256 */
    /* The throttle is opt-in (the default is full rate) -- this scene exists
     * to compare the throttled duty arm against a full-rate one, so it asks
     * for the throttled regime explicitly. 0.5 s is the period LE_DUTY_K was
     * measured against. */
    cfg.delay_est_period_s = 0.5f;
    p = af_test_create(&cfg);
    if (!p) return;
    if (pin_full_rate) af_pin_duty_full_rate(p);

    far_hist = (float*)malloc(
        (size_t)(LE_HOPS * LE_HOP + LE_PAD) * sizeof(float));
    if (!far_hist) { af_test_destroy(p); return; }
    realign_scene_history(far_hist, LE_HOPS * LE_HOP + LE_PAD);

    for (hop = 0; hop < LE_HOPS; ++hop) {
        AfFront pre;
        int base = hop * LE_HOP + LE_PAD;
        int dominant = (hop >= LE_SHIFT_AT) ? LE_DELAY_B : LE_DELAY_A;
        double hop_best = -1.0e30;
        unsigned long long run_now;
        int analysed;
        int i, ch;

        for (i = 0; i < LE_HOP; ++i) {
            int t = base + i;
            float echo = LE_ECHO_GAIN * far_hist[t - dominant];
            float noise;
            noise_rng ^= noise_rng << 13;
            noise_rng ^= noise_rng >> 17;
            noise_rng ^= noise_rng << 5;
            /* Independent of the far end, so no filter can cancel it: this
             * is what holds ERLE under the watchdog's arming threshold. */
            noise = LE_NEAR_NOISE_GAIN *
                (((float)(noise_rng >> 8) * (1.0f / 16777216.0f)) - 0.5f);
            far[i] = far_hist[t];
            for (ch = 0; ch < AF_CHANNELS; ++ch)
                mic[i * AF_CHANNELS + ch] = echo + noise;
        }
        if (af_process_front(p, mic, far, &pre) !=
            AF_OK) break;

        run_now = af_duty_hops_run(p);
        analysed = (int)(run_now - previous_run);
        previous_run = run_now;

        /* The watchdog reads the BEST of the four lanes, so measure the same
         * way -- one quiet lane must not be able to fake a low reading. */
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            double mic_power = 0.0;
            double err_power = 0.0;
            for (i = 0; i < LE_HOP; ++i) {
                double m = (double)mic[i * AF_CHANNELS + ch];
                double e = (double)pre.formed_hop[ch][i];
                mic_power += m * m;
                err_power += e * e;
            }
            if (mic_power > 0.0 && err_power > 0.0) {
                double db = 10.0 * log10(mic_power / err_power);
                if (db > hop_best) hop_best = db;
            }
            if (ch == 0 && hop >= LE_SHIFT_AT - LE_SETTLED_WINDOW &&
                hop < LE_SHIFT_AT) {
                settled_mic += mic_power;
                settled_err += err_power;
            }
        }
        if (hop_best > -1.0e29 && (hop == 0 || hop_best > out->peak_erle_db))
            out->peak_erle_db = hop_best;

        if (hop >= LE_SHIFT_AT) {
            out->hops_after += 1;
            out->analysed_after += analysed;
        }
        if (pre.delay.changed) {
            if (out->acquisition_hop < 0) out->acquisition_hop = hop;
            if (hop >= LE_SHIFT_AT && out->relock_hop < 0 &&
                LE_DELAY_B - pre.delay.delay_samples >= 0 &&
                LE_DELAY_B - pre.delay.delay_samples <= KD_MAX_UNDERSHOOT) {
                out->relock_hop = hop;
                out->relock_delay = pre.delay.delay_samples;
            }
        }
    }

    out->settled_erle_db = (settled_err > 0.0)
        ? 10.0 * log10(settled_mic / settled_err) : 0.0;
    out->run = af_duty_hops_run(p);
    out->total = af_duty_hops_total(p);
    free(far_hist);
    af_test_destroy(p);
}

static void test_duty_low_erle_delay_change(void) {
    LowErleRun full, duty;
    int full_latency;
    int duty_latency;
    int pure_schedule;
    char label[288];

    low_erle_run(1, &full);
    low_erle_run(0, &duty);

    full_latency = full.relock_hop >= 0 ? full.relock_hop - LE_SHIFT_AT : -1;
    duty_latency = duty.relock_hop >= 0 ? duty.relock_hop - LE_SHIFT_AT : -1;
    /* What a schedule that never went back to full rate would have analysed. */
    pure_schedule = duty.hops_after / LE_DUTY_K;

    printf("duty low-ERLE: full rate re-locks %d hops after the move "
           "(%.0f ms), duty %d hops (%.0f ms) -- %.2fx, K=%d; peak lane ERLE "
           "%.2f dB full / %.2f dB duty (settled %.2f dB, watchdog arms above "
           "%.1f dB); analysed %d of %d hops after the move against %d for a "
           "schedule that never disengaged\n",
           full_latency, (double)full_latency * 16.0,
           duty_latency, (double)duty_latency * 16.0,
           full_latency > 0 ? (double)duty_latency / (double)full_latency : 0.0,
           LE_DUTY_K, full.peak_erle_db, duty.peak_erle_db,
           duty.settled_erle_db, (double)LE_ERLE_COLLAPSE_DB,
           duty.analysed_after, duty.hops_after, pure_schedule);

    /* Premise first: if cancellation were good enough to arm the watchdog,
     * everything below would be measuring the watchdog instead of the
     * schedule, and the scene would be silently the wrong one. */
    snprintf(label, sizeof(label),
             "the scene really does hold lane ERLE under the watchdog's "
             "arming threshold (peak %.2f dB full / %.2f dB duty, threshold "
             "%.1f dB)", full.peak_erle_db, duty.peak_erle_db,
             (double)LE_ERLE_COLLAPSE_DB);
    CHECK(full.peak_erle_db < (double)LE_ERLE_COLLAPSE_DB &&
          duty.peak_erle_db < (double)LE_ERLE_COLLAPSE_DB, label);

    snprintf(label, sizeof(label),
             "both arms acquire the first alignment identically, before the "
             "schedule can differ (hop %d full, hop %d duty)",
             full.acquisition_hop, duty.acquisition_hop);
    CHECK(full.acquisition_hop >= 0 &&
          full.acquisition_hop == duty.acquisition_hop, label);

    snprintf(label, sizeof(label),
             "full rate re-locks the moved path within its measured bound "
             "(%d hops after the move, bound %d, applied %d for a true %d)",
             full_latency, LE_FULL_RATE_BOUND, full.relock_delay, LE_DELAY_B);
    CHECK(full_latency > 0 && full_latency <= LE_FULL_RATE_BOUND &&
          LE_DELAY_B - full.relock_delay >= 0 &&
          LE_DELAY_B - full.relock_delay <= KD_MAX_UNDERSHOOT, label);

    /* The scene is exactly LE_DUTY_K * LE_FULL_RATE_BOUND hops long after the
     * move, so re-locking at all inside it IS re-locking within K times the
     * full-rate bound. */
    snprintf(label, sizeof(label),
             "duty re-locks the same path within K times that bound (%d hops "
             "after the move, bound %d = %dx%d, applied %d)",
             duty_latency, LE_DUTY_K * LE_FULL_RATE_BOUND, LE_DUTY_K,
             LE_FULL_RATE_BOUND, duty.relock_delay);
    CHECK(duty_latency > 0 &&
          duty_latency <= LE_DUTY_K * LE_FULL_RATE_BOUND &&
          LE_DELAY_B - duty.relock_delay >= 0 &&
          LE_DELAY_B - duty.relock_delay <= KD_MAX_UNDERSHOOT, label);

    /* The gap is the point. A build where the two arms cost the same has
     * either stopped decimating or stopped honouring the pin, and every
     * bound above would still be green. */
    snprintf(label, sizeof(label),
             "and it really is slower than full rate -- the gap this scene "
             "exists to document (%d vs %d hops, %d hops of extra wait)",
             duty_latency, full_latency, duty_latency - full_latency);
    CHECK(duty_latency > full_latency, label);

    snprintf(label, sizeof(label),
             "the machine stayed engaged across the whole re-acquisition "
             "(%d of %d hops analysed after the move; full rate would be %d)",
             duty.analysed_after, duty.hops_after, duty.hops_after);
    CHECK(duty.analysed_after > 0 &&
          duty.analysed_after * 4 < duty.hops_after, label);

    /* ...but it did disengage, once. The estimate-moved re-arm shortens no
     * latency here (see this block's header), so the census is the only
     * evidence it ran at all: a schedule that never disengaged would analyse
     * exactly hops_after/K hops and not one more. */
    snprintf(label, sizeof(label),
             "the estimate-moved re-arm still fires once the filter finds the "
             "new peak (%d analysed, more than the %d a never-disengaging "
             "schedule would)", duty.analysed_after, pure_schedule);
    CHECK(duty.analysed_after > pure_schedule, label);

    snprintf(label, sizeof(label),
             "the pinned arm analysed every hop and the census says so "
             "(%llu of %llu)", full.run, full.total);
    CHECK(full.total == (unsigned long long)LE_HOPS &&
          full.run == full.total, label);
}

/* ============================================================================
 * Cross-lane far-end spectrum guard
 * ========================================================================== */

static void test_far_spec_provenance_guard(void) {
    enum { FS_N = 33 };
    Complex reference[FS_N];
    Complex lane[FS_N];
    AfConfig cfg;
    Af* p;
    char label[224];
    int k;

    for (k = 0; k < FS_N; ++k) {
        reference[k].r = 0.5f * (float)(k - 11);
        reference[k].i = -0.25f * (float)(k + 3);
    }
    memcpy(lane, reference, sizeof(reference));

    CHECK(af_far_spec_agrees(0, lane, reference, FS_N) == 1,
          "a lane carrying the same far spectrum is accepted on the "
          "comparison path");
    CHECK(af_far_spec_agrees(1, lane, reference, FS_N) == 1,
          "and on the provenance path");

    /* A lane whose far-end spectrum genuinely differs must still be rejected
     * -- X is one shared render reference, not a fourth observation of it.
     * The divergence is far outside complex_close()'s relative window. */
    lane[7].r = reference[7].r + 1.0f;
    CHECK(af_far_spec_agrees(0, lane, reference, FS_N) == 0,
          "a lane carrying a DIFFERENT far spectrum is rejected when nothing "
          "establishes where it came from");

    /* MUTATION, asserted rather than described: the provenance flag is what
     * licenses skipping the comparison, so a flag that says "shared" when the
     * lanes are not lets exactly this frame through. That is why nothing sets
     * it except the counter evidence in af_process_front(), and why every bail-out
     * path leaves it 0. Make af_far_spec_agrees() ignore its
     * first argument and the row above goes red. */
    CHECK(af_far_spec_agrees(1, lane, reference, FS_N) == 1,
          "and would be accepted if provenance claimed the lanes shared one "
          "spectrum (the flag is load-bearing, not decorative)");

    CHECK(af_far_spec_agrees(0, NULL, reference, FS_N) == 0 &&
          af_far_spec_agrees(1, lane, NULL, FS_N) == 0,
          "a missing spectrum is rejected on either path");

    /* End to end: production really does take the cheap path, and the
     * evidence it rests on -- one real far transform per hop across the four
     * lanes -- is the same figure the sharing test pins. */
    cfg = af_default_config(16000);
    p = af_test_create(&cfg);
    CHECK(p != NULL, "far-spec provenance: core creates");
    if (p) {
        float mic[128 * AF_CHANNELS];
        float far[128];
        Complex* weights;
        int hop = af_hop_size(p);
        int n_freqs = af_n_freqs(p);
        int established = 0;
        int hops = 0;
        int i;
        weights = (Complex*)calloc(
            (size_t)AF_CHANNELS * n_freqs, sizeof(Complex));
        for (i = 0; i < AF_CHANNELS * n_freqs; ++i)
            weights[i].r = 0.25f;
        for (i = 0; i < 40; ++i) {
            AfFront pre;
            fill_inputs(mic, far, hop, 16000, i);
            if (af_process_front(p, mic, far, &pre) !=
                AF_OK) break;
            hops += 1;
            established += af_far_spec_provenance(p) ? 1 : 0;
            if (fuse_only(p, weights) != AF_OK) break;
        }
        snprintf(label, sizeof(label),
                 "every hop establishes the shared-far provenance (%d of %d), "
                 "backed by one real far transform per hop (%ld)",
                 established, hops,
                 af_far_fft_real_compute_count(p));
        CHECK(hops == 40 && established == 40 &&
              af_far_fft_real_compute_count(p) == (long)hops,
              label);
        free(weights);
        af_test_destroy(p);
    }
}


/* A hop on which some lanes realign warm and others soft must give every lane its own far
 * spectrum: a soft realign zeroes that lane's far_buffer history, a warm one keeps it, so
 * lane 0's spectrum is wrong for the reset lanes. Borrowing it leaves no trace on the hop
 * itself (a fresh filter has zero taps) but corrupts later adaptation. The reference is four
 * stand-alone lanes through the ordinary local-far path (aec_process_linear) with the same
 * inputs and realign deltas; the core's lanes must match bit for bit, and the far-FFT counter
 * must read 4 on the mixed hop and 1 otherwise. Lanes 0-1 carry a two-path echo (warm
 * transfer on the swing), lanes 2-3 unrelated noise (filter-only reset). */
typedef struct MixedHopRun {
    int hops_run;
    int mixed_hops;              /* hops with both warm and soft lane realigns */
    int mixed_hop_first;         /* -1 when none                              */
    int mixed_fft_not_four;      /* mixed hops whose far-FFT delta was not 4   */
    int other_fft_not_one;       /* other hops whose far-FFT delta was not 1   */
    int provenance_wrong;        /* hops whose far-spectrum provenance was not 0 (mixed) / 1 (otherwise) */
    int outcome_count_mismatch;  /* realign hops where reference warm/soft
                                  * counts differ from the core's              */
    int lane_hops_mismatched;    /* lane-hops where error spectrum or formed
                                  * hop differs from the reference             */
    int first_mismatch_hop;      /* -1 when none                              */
    int far_spec_diverges;       /* mixed hops where the reference lanes' own
                                  * far spectra differ between lanes           */
    int fuse_failures;           /* hops whose fuse did not return OK          */
    int valid;
} MixedHopRun;

static int mixed_hop_far_specs_differ(const Complex* a, const Complex* b,
                                      int n) {
    return memcmp(a, b, (size_t)n * sizeof(Complex)) != 0;
}

static void mixed_hop_run(int sample_rate, int fft_size, int strong_delay,
                          int weak_delay, int swing_hop, int hops,
                          MixedHopRun* out) {
    enum { MH_PAD = 4096 };
    AfConfig cfg = af_default_config(sample_rate);
    AecTestCfg lane_cfg;
    Af* p = NULL;
    Aec* ref_lane[AF_CHANNELS] = {NULL, NULL, NULL, NULL};
    void* ref_pool[AF_CHANNELS] = {NULL, NULL, NULL, NULL};
    /* xorshift states of the synthetic near-end noise per lane (scene input, not a comfort-noise seed) */
    uint32_t noise_rng[AF_CHANNELS] = {
        0u, 0u, 0x9e3779b9u, 0x85ebca6bu};
    uint32_t rng = 0x1234567u;
    float* history = NULL;
    float* microphones = NULL;
    float* lane_mic = NULL;
    float* ref = NULL;
    Complex* weights = NULL;
    long warm_before = 0, soft_before = 0, fft_before = 0;
    int previous_solid = 0;
    int previous_delay = 0;
    int hop = 0, n_freqs = 0;
    int h, ch, i;

    memset(out, 0, sizeof(*out));
    out->mixed_hop_first = -1;
    out->first_mismatch_hop = -1;

    cfg.fft_size = fft_size;
    p = af_test_create(&cfg);
    if (!p || lane_config_for(&cfg, &lane_cfg) != 0)
        goto cleanup;
    hop = af_hop_size(p);
    n_freqs = af_n_freqs(p);
    {
        size_t bytes = aec_tcfg_mem_size(&lane_cfg);
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            if (bytes == 0 ||
                posix_memalign(&ref_pool[ch], 16, bytes) != 0) goto cleanup;
            ref_lane[ch] = aec_tcfg_init(ref_pool[ch], bytes, &lane_cfg);
            if (!ref_lane[ch]) goto cleanup;
        }
    }
    history = (float*)calloc((size_t)(hops * hop + MH_PAD), sizeof(float));
    microphones = (float*)calloc(
        (size_t)hop * AF_CHANNELS, sizeof(float));
    lane_mic = (float*)calloc((size_t)hop, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)AF_CHANNELS * n_freqs, sizeof(Complex));
    if (!history || !microphones || !lane_mic || !ref || !weights)
        goto cleanup;
    for (i = 0; i < hops * hop + MH_PAD; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        history[i] = 0.1f * (((float)(rng >> 8) *
                              (1.0f / 16777216.0f)) - 0.5f);
    }
    for (i = 0; i < AF_CHANNELS * n_freqs; ++i)
        weights[i].r = 0.25f;

    for (h = 0; h < hops; ++h) {
        AfFront pre;
        AfFrame fused;
        int base = h * hop + MH_PAD;
        float strong_gain = h >= swing_hop ? 0.25f : 0.6f;
        float weak_gain = h >= swing_hop ? 0.6f : 0.25f;
        int old_align = previous_solid ? previous_delay : 0;
        long warm_after, soft_after, fft_after;
        int warm_delta, soft_delta, fft_delta, mixed;
        int ref_warm = 0, ref_soft = 0;
        const Complex* first_far = NULL;
        const Complex* soft_far = NULL;

        for (i = 0; i < hop; ++i) {
            float echo = strong_gain * history[base + i - strong_delay] +
                         weak_gain * history[base + i - weak_delay];
            ref[i] = history[base + i];
            for (ch = 0; ch < AF_CHANNELS; ++ch) {
                float v = echo;
                if (ch >= 2) {
                    noise_rng[ch] ^= noise_rng[ch] << 13;
                    noise_rng[ch] ^= noise_rng[ch] >> 17;
                    noise_rng[ch] ^= noise_rng[ch] << 5;
                    v = 0.06f * (((float)(noise_rng[ch] >> 8) *
                                  (1.0f / 16777216.0f)) - 0.5f);
                }
                microphones[i * AF_CHANNELS + ch] = v;
            }
        }
        if (af_process_front(p, microphones, ref, &pre) !=
                AF_OK) goto cleanup;
        out->hops_run += 1;

        warm_after = af_realign_warm_lane_count(p);
        soft_after = af_realign_soft_lane_count(p);
        fft_after = af_far_fft_real_compute_count(p);
        warm_delta = (int)(warm_after - warm_before);
        soft_delta = (int)(soft_after - soft_before);
        fft_delta = (int)(fft_after - fft_before);
        mixed = pre.delay.changed && warm_delta > 0 && soft_delta > 0;
        if (mixed) {
            out->mixed_hops += 1;
            if (out->mixed_hop_first < 0) out->mixed_hop_first = h;
        }

        if (pre.delay.changed) {
            for (ch = 0; ch < AF_CHANNELS; ++ch) {
                int outcome = aec_apply_external_realign(
                    ref_lane[ch], pre.delay.delay_samples - old_align);
                if (outcome == 1) ref_warm += 1;
                if (outcome == 0) ref_soft += 1;
            }
            if (ref_warm != warm_delta || ref_soft != soft_delta)
                out->outcome_count_mismatch += 1;
        }
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            AecResidualView context;
            int lane_bad;
            for (i = 0; i < hop; ++i)
                lane_mic[i] = microphones[i * AF_CHANNELS + ch];
            aec_process_linear(ref_lane[ch], lane_mic, pre.aligned_ref);
            aec_process_residual(ref_lane[ch], NULL);
            memset(&context, 0, sizeof context); context.struct_size = sizeof context;
            aec_get_residual(ref_lane[ch], &context);
            lane_bad =
                memcmp(context.error_spec, pre.linear_spectra[ch],
                       (size_t)n_freqs * sizeof(Complex)) != 0 ||
                memcmp(context.formed_hop, pre.formed_hop[ch],
                       (size_t)hop * sizeof(float)) != 0;
            if (lane_bad) {
                out->lane_hops_mismatched += 1;
                if (out->first_mismatch_hop < 0) out->first_mismatch_hop = h;
            }
            if (ch == 0) first_far = context.far_spec;
            else if (!soft_far && first_far && mixed &&
                     mixed_hop_far_specs_differ(
                         context.far_spec, first_far, n_freqs))
                soft_far = context.far_spec;
        }
        if (mixed && soft_far) out->far_spec_diverges += 1;

        if (mixed) {
            if (fft_delta != AF_CHANNELS)
                out->mixed_fft_not_four += 1;
        } else if (fft_delta != 1) {
            out->other_fft_not_one += 1;
        }
        if (af_far_spec_provenance(p) != (mixed ? 0 : 1)) out->provenance_wrong += 1;
        if (af_fuse(p, weights, NULL, &fused) != AF_OK)
            out->fuse_failures += 1;

        previous_solid = pre.delay.solid;
        previous_delay = pre.delay.delay_samples;
        warm_before = warm_after;
        soft_before = soft_after;
        fft_before = fft_after;
    }
    out->valid = 1;

cleanup:
    free(weights);
    free(ref);
    free(lane_mic);
    free(microphones);
    free(history);
    for (ch = 0; ch < AF_CHANNELS; ++ch) free(ref_pool[ch]);
    af_test_destroy(p);
}

static void test_mixed_realign_hop_far_spectrum(
        int sample_rate, int fft_size, int strong_delay, int weak_delay,
        int swing_hop, int hops) {
    MixedHopRun run;
    char label[256];

    mixed_hop_run(sample_rate, fft_size, strong_delay, weak_delay,
                  swing_hop, hops, &run);
    snprintf(label, sizeof(label),
             "mixed realign sr=%d/fft=%d: the scene runs to completion and "
             "really produces a hop with warm AND soft lane realigns (%d "
             "mixed hop(s), first at hop %d of %d)",
             sample_rate, fft_size, run.mixed_hops, run.mixed_hop_first,
             run.hops_run);
    CHECK(run.valid && run.hops_run == hops && run.mixed_hops >= 1, label);
    snprintf(label, sizeof(label),
             "mixed realign sr=%d/fft=%d: the independent reference lanes "
             "realign with the same warm/soft split as the core's lanes (%d "
             "hops differ)",
             sample_rate, fft_size, run.outcome_count_mismatch);
    CHECK(run.valid && run.outcome_count_mismatch == 0, label);
    snprintf(label, sizeof(label),
             "mixed realign sr=%d/fft=%d: on the mixed hop the lanes' own "
             "far spectra really differ (%d of %d mixed hops)",
             sample_rate, fft_size, run.far_spec_diverges, run.mixed_hops);
    CHECK(run.valid && run.mixed_hops >= 1 &&
          run.far_spec_diverges == run.mixed_hops, label);
    snprintf(label, sizeof(label),
             "mixed realign sr=%d/fft=%d: every mixed hop runs four local "
             "far FFTs (%d mixed hops did not) and every other hop exactly "
             "one (%d hops did not)",
             sample_rate, fft_size, run.mixed_fft_not_four,
             run.other_fft_not_one);
    CHECK(run.valid && run.mixed_fft_not_four == 0 &&
          run.other_fft_not_one == 0, label);
    snprintf(label, sizeof(label),
             "mixed realign sr=%d/fft=%d: the shared-far provenance is 0 on every mixed hop and 1 on every other (%d wrong)",
             sample_rate, fft_size, run.provenance_wrong);
    CHECK(run.valid && run.provenance_wrong == 0, label);
    snprintf(label, sizeof(label),
             "mixed realign sr=%d/fft=%d: every lane's error spectrum and "
             "formed hop equals an ordinary local-far reference lane bit for "
             "bit across the mixed hop and the hops after it (%d lane-hops "
             "differ, first at hop %d)",
             sample_rate, fft_size, run.lane_hops_mismatched,
             run.first_mismatch_hop);
    CHECK(run.valid && run.lane_hops_mismatched == 0, label);
    snprintf(label, sizeof(label),
             "mixed realign sr=%d/fft=%d: the fuse accepts the mixed hop "
             "and every other (%d refusals)",
             sample_rate, fft_size, run.fuse_failures);
    CHECK(run.valid && run.fuse_failures == 0, label);
}

/* One hop of the echo stimulus the bypass tests share: a far-end tone, its
 * half-level echo on every microphone, a weak near-end tone and a
 * deterministic broadband term so the denoiser has something to shape once
 * it leaves init -- a pure tone pair can leave the gain at its ceiling and a
 * divergence row would never fire. */


/* ---- invalid configs ---- */
static void test_invalid_configs(void) {
    AfConfig cfg;
    AfMemReq req;

    cfg = af_default_config(8000);
    CHECK(af_test_create(&cfg) == NULL,
          "8 kHz is outside the 4-channel contract");
    CHECK(af_get_mem_requirements(&cfg, &req) != 0,
          "static sizing rejects 8 kHz");

    cfg = af_default_config(16000);
    cfg.fft_size = 1024;
    CHECK(af_test_create(&cfg) == NULL,
          "cross-rate FFT grid is rejected");

    cfg = af_default_config(16000);
    cfg.capture_proxy_channel = 4;
    CHECK(af_test_create(&cfg) == NULL,
          "invalid capture proxy is rejected");

    cfg = af_default_config(16000);
    CHECK(cfg.enable_post == 1, "the default config keeps the fusion on");
    cfg.enable_post = 2;
    CHECK(af_test_create(&cfg) == NULL &&
          af_get_mem_requirements(&cfg, &req) != 0,
          "an out-of-range residual mode is rejected by create and sizing");

    /* Backward-jump quarantine. The WINDOW is checked even though the enable
     * defaults to 0: a config that would misbehave the moment someone flips
     * one field must not pass validation today. The accept row at the end is
     * what stops all of this from being a comparison that cannot fail. */
    cfg = af_default_config(16000);
    CHECK(cfg.delay_backward_quarantine_enabled == 0 &&
          cfg.delay_backward_quarantine_s == 1.0f,
          "the quarantine defaults to OFF with a 1.0 s window");
    cfg.delay_backward_quarantine_enabled = 2;
    CHECK(af_test_create(&cfg) == NULL,
          "a non-boolean quarantine enable is rejected");

    cfg = af_default_config(16000);
    cfg.delay_backward_quarantine_s = -1.0f;
    CHECK(af_test_create(&cfg) == NULL,
          "a negative quarantine window is rejected");
    cfg.delay_backward_quarantine_s = 3601.0f;
    CHECK(af_test_create(&cfg) == NULL,
          "an out-of-range quarantine window is rejected");
    cfg.delay_backward_quarantine_s = (float)NAN;
    CHECK(af_test_create(&cfg) == NULL,
          "a NaN quarantine window is rejected");

    cfg = af_default_config(16000);
    cfg.delay_backward_quarantine_enabled = 1;
    cfg.delay_backward_quarantine_s = 0.25f;
    CHECK(af_get_mem_requirements(&cfg, &req) == 0,
          "a valid enabled quarantine config is ACCEPTED (so the rows above "
          "are a real check, not an identity)");

    /* Shared-estimator duty-cycle throttle: same range contract as the
     * quarantine window above, and checked for the same reason -- a config
     * that would misbehave the moment someone opts into the throttle must
     * not pass validation today. */
    cfg = af_default_config(16000);
    CHECK(cfg.delay_est_period_s == 0.0f,
          "the duty-cycle throttle defaults to OFF (full rate)");
    cfg.delay_est_period_s = -1.0f;
    CHECK(af_test_create(&cfg) == NULL,
          "a negative duty-cycle period is rejected");
    cfg.delay_est_period_s = 3601.0f;
    CHECK(af_test_create(&cfg) == NULL,
          "an out-of-range duty-cycle period is rejected");
    cfg.delay_est_period_s = (float)NAN;
    CHECK(af_test_create(&cfg) == NULL,
          "a NaN duty-cycle period is rejected");

    cfg = af_default_config(16000);
    cfg.delay_est_period_s = 0.5f;
    CHECK(af_get_mem_requirements(&cfg, &req) == 0,
          "a valid opted-in duty-cycle period is ACCEPTED (so the rows above "
          "are a real check, not an identity)");
}


/* ---- stage timing (the wrapper and the lanes share AEC_STAGE_TIMING now) ---- */
static void test_stage_timing(void) {
    AfConfig cfg = af_default_config(16000);
    Af* p = af_test_create(&cfg);
    AfLastTiming t, zero;
    uint32_t pre_wall_us = 0, post_wall_us = 0;
    uint32_t pre_sum, post_sum;

    CHECK(p != NULL, "timing: core created");
    if (!p) return;

    /* A NULL pipeline must zero the whole record rather than fail. Compared
     * as a struct, so the two always-zero fields are covered too -- they are
     * exactly where a memset-versus-field-copy slip would hide. */
    memset(&zero, 0, sizeof(zero));
    memset(&t, 0xa5, sizeof(t));
    af_get_last_timing(NULL, &t);
    CHECK(memcmp(&t, &zero, sizeof(t)) == 0,
          "timing: a NULL pipeline zeroes the record");

    /* Warm up so the delay estimator and filters are doing real work; the
     * driver hands back the final hop's wall time measured around the same
     * two calls the library stamps inside. */
    feed_strength_hops(p, 34, &pre_wall_us, &post_wall_us);
    af_get_last_timing(p, &t);

    /* Written so it can fail: four AEC lanes each run a main filter, a
     * pre-filter stage and a post/RES block over a full hop, and the shared
     * five-filter matched estimator runs once -- none of those can cost under
     * a microsecond, so a zero anywhere here means that window never ran.
     * frontend_us and lane_res_us come from aec_get_last_timing(); they read
     * zero if the AEC-side windows are ever dropped. */
    /* Each half is asserted against its OWN flag, because the two are
     * separate builds: AEC_STAGE_TIMING governs the wrapper's
     * stages, AEC_STAGE_TIMING governs the three summed off the lanes. Both
     * default off, and the zero branches are not filler -- they are what
     * proves the flag removes the measurement rather than merely hiding it,
     * and that a compiled-out build reports zeros instead of stale values. */
#if AEC_STAGE_TIMING
    CHECK(t.delay_us > 0, "timing: the delay estimator reports a real cost");
#else
    CHECK(t.delay_us == 0,
          "timing: the wrapper's stages read zero when compiled out");
#endif
#if AEC_STAGE_TIMING
    CHECK(t.frontend_us > 0,
          "timing: the lanes' pre-filter stage reports a real cost");
    CHECK(t.linear_us > 0, "timing: the lanes' main filter reports a real cost");
    CHECK(t.lane_res_us > 0,
          "timing: the lanes' post/RES block reports a real cost");
#else
    CHECK(t.frontend_us == 0 && t.linear_us == 0 && t.lane_res_us == 0,
          "timing: the lanes' stages read zero when compiled out");
#endif

    /* Every stage is bounded by the call that contains it. This is what
     * distinguishes a measurement from a number: a stray or uninitialised
     * value would almost certainly exceed its enclosing call. */
    pre_sum = t.delay_us + t.frontend_us + t.linear_us + t.lane_res_us;
    post_sum = t.fuse_us;
    CHECK(pre_sum <= pre_wall_us,
          "timing: lane stages must fit inside af_process_front's wall time");
    CHECK(post_sum <= post_wall_us,
          "timing: the fuse stage must fit inside af_fuse's wall time");

    /* A hop that is accepted and then bails out must report zeros for the
     * stages it never reached, not the previous hop's numbers. The non-finite
     * input check is the reachable bail-out: it runs after the record is
     * cleared, so without that clearing the values above would survive. */
    {
        int hop = af_hop_size(p);
        float* nan_mics = (float*)malloc((size_t)hop * 4u * sizeof(float));
        float* far = (float*)calloc((size_t)hop, sizeof(float));
        AfFront pre;
        AfLastTiming after_bail;
        int i;
        CHECK(nan_mics && far, "timing: allocation");
        if (nan_mics && far) {
            for (i = 0; i < hop * 4; ++i) nan_mics[i] = (float)NAN;
            CHECK(af_process_front(p, nan_mics, far, &pre) ==
                      AF_INVALID_ARGUMENT,
                  "timing: a non-finite hop is rejected");
            /* Read into its own record: `t` still holds the good hop the
             * report below prints. */
            af_get_last_timing(p, &after_bail);
            CHECK(after_bail.delay_us == 0 && after_bail.linear_us == 0 &&
                  after_bail.frontend_us == 0 && after_bail.lane_res_us == 0,
                  "timing: a bailed-out hop reports zeros, not the last "
                  "hop's numbers");
        }
        free(nan_mics); free(far);
    }

    af_test_destroy(p);
    printf("timing: delay=%u frontend=%u linear=%u lane_res=%u "
           "(pre wall %uus) | fuse=%u (fuse wall %uus)\n",
           t.delay_us, t.frontend_us, t.linear_us, t.lane_res_us, pre_wall_us,
           t.fuse_us, post_wall_us);
}

/* Comfort noise fills the AEC-suppressed bins from lib/aec's recipe: an LCG step per bin
 * whose top five bits index kCngTable, the imaginary part a quarter turn (+8 of 32) ahead
 * of the real one. The injected noise owes the caller bounded samples, the same expected
 * per-bin power as the unit-variance Gaussian pair and an advancing sequence; no output
 * bytes are pinned. The pipeline half differences a CNG-on run against a byte-identical
 * CNG-off one: comfort noise is added after every gain, so the difference is the noise. */

/* ---- the echo stimulus ---- */
static void synth_echo_hop(int h, int hop, float* mics, float* far,
                           uint32_t* rng) {
    int i, ch;
    for (i = 0; i < hop; ++i) {
        float t = (float)(h * hop + i);
        float r = 0.6f * sinf(0.06f * t);
        float noise;
        *rng = *rng * 1664525u + 1013904223u;
        noise = 0.02f * ((float)((*rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f);
        far[i] = r;
        for (ch = 0; ch < 4; ++ch)
            mics[(size_t)i * 4u + (size_t)ch] =
                0.5f * r + 0.02f * sinf(0.017f * t) + noise;
    }
}


/* ---- fuse frame, usable_linear fold, proxy lanes, counters ---- */
static void test_fuse_frame_contract(void) {
    AfConfig cfg = af_default_config(48000);
    AfConfig front_only;
    Af *p, *q;
    static Complex weights[4 * 513];
    static float mics[512 * 4];
    float far[512];
    AfFront front;
    AfFrame fr;
    uint32_t rng = 7u;
    int h, i, hop, n, ok = 1, finite = 1;
    int usable_linear_valid = 1, usable_linear_seen = 0;

    cfg.fft_size = 1024;
    p = af_test_create(&cfg);
    CHECK(p != NULL, "fuse frame: 48 kHz core creates");
    if (!p) return;
    hop = af_hop_size(p);
    n = af_n_freqs(p);
    for (i = 0; i < 4 * n; ++i) { weights[i].r = 0.25f; weights[i].i = 0.0f; }
    for (h = 0; h < 60; ++h) {
        synth_echo_hop(h, hop, mics, far, &rng);
        ok = ok && af_process_front(p, mics, far, &front) == AF_OK;
        ok = ok && af_fuse(p, weights, NULL, &fr) == AF_OK;
        if (!ok) break;
        if (fr.n_freqs != n || fr.hop != hop || fr.e == NULL || fr.y == NULL ||
            fr.r2 == NULL || fr.cn == NULL || fr.g_res_ctx != NULL ||
            fr.far_hop != front.aligned_ref)
            ok = 0;
        if (!(fr.max_dt >= 0.0f) || !(fr.max_sat >= 0.0f) ||
            !all_finite(fr.r2, n) || !all_finite(fr.cn, n))
            finite = 0;
        if (fr.usable_linear != 0 && fr.usable_linear != 1)
            usable_linear_valid = 0;
        if (fr.usable_linear)
            usable_linear_seen = 1;
    }
    CHECK(ok, "fuse frame: shape, pointers and no lane-level gain on every hop");
    CHECK(finite, "fuse frame: R^2, comfort noise and the worst-lane scalars are valid");
    CHECK(usable_linear_valid,
          "fuse frame: usable_linear is a 0/1 verdict on every hop");
    CHECK(usable_linear_seen,
          "fuse frame: usable_linear becomes true once four equally-weighted, "
          "equally-converging lanes clear startup (regression guard for the "
          "af914d4 lost-update: frame->usable_linear was memset to 0 and "
          "never written)");
    CHECK(af_fuse(p, weights, NULL, &fr) == AF_OK,
          "fuse frame: a second fusion on the same linear stage is allowed without a gain stage (ESTIMATE)");

    front_only = cfg;
    front_only.enable_post = 0;
    q = af_test_create(&front_only);
    CHECK(q != NULL, "fuse frame: front-only core creates");
    if (q) {
        synth_echo_hop(0, hop, mics, far, &rng);
        CHECK(af_process_front(q, mics, far, &front) == AF_OK &&
              af_fuse(q, weights, NULL, &fr) == AEC_ERR_MODE,
              "fuse frame: a lanes-only (STATE) array refuses to fuse (AEC_ERR_MODE)");
        af_test_destroy(q);
    }
    af_test_destroy(p);
}

/* Like synth_echo_hop(), but lane decorrelated_ch gets capture content with
 * zero correlation to the far-end reference (independent noise, no echo
 * term at all) instead of the shared echo. That lane's adaptive filter has
 * nothing stable to lock onto, so its own filter_converged never fires,
 * convergence_seen never latches, and usable_linear (gated on
 * convergence_seen -- see fq_update()/fq_linear_filter_usable() in
 * lib/aec/c_impl/src/filter_quality.c) stays false for the whole run. */
static void synth_echo_hop_decorrelated_lane(int h, int hop, float* mics,
                                             float* far, int decorrelated_ch,
                                             uint32_t* rng) {
    int i, ch;
    for (i = 0; i < hop; ++i) {
        float t = (float)(h * hop + i);
        float r = 0.6f * sinf(0.06f * t);
        float noise;
        *rng = *rng * 1664525u + 1013904223u;
        noise = 0.02f * ((float)((*rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f);
        far[i] = r;
        for (ch = 0; ch < 4; ++ch) {
            if (ch == decorrelated_ch) {
                uint32_t draw = (*rng = *rng * 1664525u + 1013904223u);
                mics[(size_t)i * 4u + (size_t)ch] =
                    0.3f * ((float)((draw >> 8) & 0xFFFFu) / 32768.0f - 1.0f);
            } else {
                mics[(size_t)i * 4u + (size_t)ch] =
                    0.5f * r + 0.02f * sinf(0.017f * t) + noise;
            }
        }
    }
}

/* Complement to test_fuse_frame_contract(): the usable_linear AND-fold in fuse_contexts()
 * runs over contributing lanes only (the lane_contributes gate of the all_converged fold),
 * matching Python's `all(... for c in contributors)`. Lane 3 gets capture content
 * uncorrelated with the far end, so its usable_linear stays false while lanes 0-2 become
 * usable. The same front frame is fused twice per hop: with lane 3 at zero weight
 * usable_linear must eventually read true (an AND over all four lanes would latch it
 * false); with lane 3 given real weight it must read false (an "always 1" mutation or a
 * dropped AND-fold fails this, and the equal-weight setup of test_fuse_frame_contract()
 * cannot tell it from the correct fold). */
static void test_fuse_usable_linear_and_over_contributors(void) {
    AfConfig cfg = af_default_config(48000);
    Af* p;
    static Complex noncontrib_w[4 * 513];
    static Complex contrib_w[4 * 513];
    static float mics[512 * 4];
    float far[512];
    AfFront front;
    AfFrame fr;
    uint32_t rng = 31u;
    int h, i, hop, n, k, ok = 1;
    int noncontrib_usable_seen = 0;
    int contrib_usable_wrongly_true = 0;

    cfg.fft_size = 1024;
    p = af_test_create(&cfg);
    CHECK(p != NULL, "fuse usable_linear AND-fold: core creates");
    if (!p) return;
    hop = af_hop_size(p);
    n = af_n_freqs(p);
    for (k = 0; k < n; ++k) {
        for (i = 0; i < 3; ++i) {
            noncontrib_w[(size_t)i * n + k].r = 1.0f / 3.0f;
            noncontrib_w[(size_t)i * n + k].i = 0.0f;
        }
        noncontrib_w[3 * n + k].r = 0.0f;
        noncontrib_w[3 * n + k].i = 0.0f;
        for (i = 0; i < 4; ++i) {
            contrib_w[(size_t)i * n + k].r = 0.25f;
            contrib_w[(size_t)i * n + k].i = 0.0f;
        }
    }

    for (h = 0; h < 60; ++h) {
        synth_echo_hop_decorrelated_lane(h, hop, mics, far, 3, &rng);
        ok = ok && af_process_front(p, mics, far, &front) ==
                       AF_OK;
        if (!ok) break;
        ok = ok && af_fuse(p, noncontrib_w, NULL, &fr) ==
                       AF_OK;
        if (!ok) break;
        if (fr.usable_linear) noncontrib_usable_seen = 1;

        ok = ok &&
             af_fuse(p, contrib_w, NULL, &fr) == AF_OK;
        if (!ok) break;
        if (noncontrib_usable_seen && fr.usable_linear)
            contrib_usable_wrongly_true = 1;
    }
    CHECK(ok, "fuse usable_linear AND-fold: front/fuse succeed on every hop");
    CHECK(noncontrib_usable_seen,
          "fuse usable_linear AND-fold: zero-weight non-usable lane does not "
          "block the other three (equally-weighted, equally-converging) "
          "lanes from driving usable_linear true");
    CHECK(!contrib_usable_wrongly_true,
          "fuse usable_linear AND-fold: once the same never-converging lane "
          "is given real weight, usable_linear must read false (AND over "
          "contributing lanes, not 'always true')");

    af_test_destroy(p);
}

static void test_proxy_lane_inputs(int sample_rate, int fft_size,
                                   AecDelayMode mode) {
    AfConfig cfg = af_default_config(sample_rate);
    Af* pipelines[4] = {NULL, NULL, NULL, NULL};
    AfFront views[4];
    float* microphones = NULL;
    float* ref = NULL;
    int hop = 0, bins = 0, ok = 1, distinct = 0;
    cfg.fft_size = fft_size;
    cfg.delay_mode = mode;
    cfg.fixed_delay_samples = mode == AEC_DELAY_FIXED ? 0 : -1;
    for (int proxy = 0; proxy < 4; ++proxy) {
        cfg.capture_proxy_channel = proxy;
        pipelines[proxy] = af_test_create(&cfg);
        if (!pipelines[proxy]) { ok = 0; break; }
    }
    if (ok) {
        hop = af_hop_size(pipelines[0]);
        bins = af_n_freqs(pipelines[0]);
        microphones = (float*)calloc((size_t)hop * 4, sizeof(float));
        ref = (float*)calloc((size_t)hop, sizeof(float));
        ok = microphones != NULL && ref != NULL;
    }
    /* With alignment fixed, the matcher proxy must not change any lane's
     * capture input. Distinct channels expose a reused deinterleave buffer. */
    for (int h = 0; h < 12 && ok; ++h) {
        if (h == 6)
            for (int proxy = 0; proxy < 4; ++proxy)
                af_reset(pipelines[proxy]);
        for (int i = 0; i < hop; ++i)
            for (int ch = 0; ch < 4; ++ch)
                microphones[4 * i + ch] = 0.05f * (float)(ch + 1) *
                    sinf((float)(h * hop + i) * (0.03f + 0.011f * ch));
        for (int proxy = 0; proxy < 4; ++proxy)
            if (af_process_front(pipelines[proxy], microphones, ref,
                                      &views[proxy]) != AF_OK)
                ok = 0;
        if (!ok) break;
        distinct |= memcmp(views[0].linear_spectra[0],
                           views[0].linear_spectra[1],
                           (size_t)bins * sizeof(Complex)) != 0;
        for (int proxy = 1; proxy < 4; ++proxy)
            for (int ch = 0; ch < 4; ++ch)
                if (memcmp(views[0].linear_spectra[ch],
                           views[proxy].linear_spectra[ch],
                           (size_t)bins * sizeof(Complex)) != 0 ||
                    memcmp(views[0].formed_hop[ch],
                           views[proxy].formed_hop[ch],
                           (size_t)hop * sizeof(float)) != 0)
                    ok = 0;
    }
    CHECK(ok && distinct,
          "every legal proxy preserves distinct per-lane capture across reset");
    for (int proxy = 0; proxy < 4; ++proxy)
        if (pipelines[proxy]) af_test_destroy(pipelines[proxy]);
    free(microphones);
    free(ref);
}

static void test_counter_boundaries(void) {
    CHECK(af_count_add_sat(3, 7) == 10 &&
          af_count_add_sat(LONG_MAX - 1, 1) == LONG_MAX &&
          af_count_add_sat(LONG_MAX - 1, 2) == LONG_MAX &&
          af_count_add_sat(LONG_MAX, LONG_MAX) == LONG_MAX,
          "diagnostic counts saturate without signed overflow");
    CHECK(af_far_fft_delta_matches(0, 1, 1) &&
          af_far_fft_delta_matches(7, 7, 0) &&
          !af_far_fft_delta_matches(LONG_MAX - 1, LONG_MAX, 1) &&
          !af_far_fft_delta_matches(LONG_MAX, LONG_MAX, 0) &&
          !af_far_fft_delta_matches(LONG_MAX, LONG_MAX, 1) &&
          !af_far_fft_delta_matches(10, 0, 1) &&
          !af_far_fft_delta_matches(-1, LONG_MAX, 1),
          "ambiguous counter deltas never establish borrow provenance");
}



/* ---- aligned far, flow view and grid ---- */

/* The flow half of the aligned-reference check: the
 * front's aligned reference is the delayed far, whole-hop exact, on every hop. */
static uint32_t g_scene_rng = 0x1234567u;
static float scene_frand(void) {
    uint32_t x = g_scene_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_scene_rng = x;
    return ((float)(x >> 8) * (1.0f / 16777216.0f)) - 0.5f;
}

static void scene_echo_hop(int frame, int hop, int true_delay, float* far_hist, float* far,
                           float* microphones) {
    for (int i = 0; i < hop; ++i) {
        int64_t t = (int64_t)frame * hop + i;
        float noise = 0.25f * scene_frand();
        float echo;
        far_hist[t] = noise;
        far[i] = noise;
        echo = t >= true_delay ? 0.6f * far_hist[t - true_delay] : 0.0f;
        for (int ch = 0; ch < AF_CHANNELS; ++ch)
            microphones[i * AF_CHANNELS + ch] = echo * (1.0f - 0.02f * ch);
    }
}

static void test_front_aligned_ref(void) {
    enum { FRAMES = 240, TRUE_DELAY = 400 };
    AfConfig cfg = af_default_config(16000);
    Af* p;
    AfFront pre;
    float *microphones, *far, *far_hist;
    int hop, locked = 0, aligned_checked = 0, exact = 1, frame;

    cfg.fft_size = 512;
    p = af_test_create(&cfg);
    CHECK(p != NULL && af_hop_size(p) == 256, "aligned ref: 512/256 flow creates");
    if (!p) return;
    hop = af_hop_size(p);
    microphones = (float*)malloc((size_t)hop * AF_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    for (frame = 0; microphones && far && far_hist && frame < FRAMES; ++frame) {
        scene_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        /* Zeroed first: a front that stopped assigning aligned_ref fails the NULL check
         * instead of reading stack garbage. */
        memset(&pre, 0, sizeof(pre));
        if (af_process_front(p, microphones, far, &pre) != AF_OK || !pre.aligned_ref ||
            pre.delay.delay_samples < 0) {
            exact = 0;
            break;
        }
        {
            /* The shift is a whole-hop decision: the raw far until the samples seen cover the
             * applied delay, the delayed far from then on. */
            int64_t hop_start = (int64_t)frame * hop;
            int servable = hop_start >= (int64_t)pre.delay.delay_samples;
            for (int i = 0; i < hop; ++i) {
                int64_t t = hop_start + i;
                float expected = servable ? far_hist[t - pre.delay.delay_samples] : far_hist[t];
                if (pre.aligned_ref[i] != expected) exact = 0;
                aligned_checked += 1;
            }
        }
        if (pre.delay.solid && pre.delay.delay_samples > 0) locked = 1;
    }
    CHECK(exact, "aligned ref: aligned_ref content matches the delayed far on every hop");
    CHECK(locked, "aligned ref: the delayed far eventually locks with positive delay_samples");
    CHECK(aligned_checked == FRAMES * hop, "aligned ref: every sample was checked");
    free(far_hist);
    free(far);
    free(microphones);
    af_test_destroy(p);
}

/* The flow's view and timing queries, and its grid refusals. */
static void test_flow_view_and_grid(void) {
    AfConfig cfg = af_default_config(16000);
    Af* p = af_test_create(&cfg);
    AfLastTiming t, z;
    CHECK(p != NULL && af_linear_aec_count(p) == AF_CHANNELS && af_hop_size(p) == 128 &&
              af_fft_size(p) == 256 && af_n_freqs(p) == 129 && af_sample_rate(p) == 16000,
          "flow view: lane count and grid read back");
    memset(&t, 0xff, sizeof(t));
    memset(&z, 0, sizeof(z));
    af_get_last_timing(NULL, &t);
    CHECK(memcmp(&t, &z, sizeof(t)) == 0, "flow view: a NULL handle zeroes the timing record");
    CHECK(af_linear_aec_count(NULL) == 0 && af_matched_filter_count(NULL) == 0 &&
              af_realign_warm_lane_count(NULL) == 0 && af_pending_delay_candidate(NULL) == -1,
          "flow view: a NULL handle reads 0 / -1 on the array diagnostics");
    af_test_destroy(p);

    cfg = af_default_config(48000);
    cfg.fft_size = 512;
    CHECK(af_test_create(&cfg) == NULL, "flow grid: 48 kHz rejects cross-rate FFT 512");
    cfg = af_default_config(16000);
    cfg.enable_post = 0;
    p = af_test_create(&cfg);
    CHECK(p != NULL && af_enable_post(p) == 0,
          "flow grid: the front-only profile (no fuse) is accepted");
    af_test_destroy(p);
}

/* The lane parameters are the application's: af_apply_lane_contract() stamps the contract
 * fields onto an application-composed AecConfig and nothing else, and a flow over such lanes
 * runs. */

/* ---- DSP error on overflowing weights ---- */

/* ---- refusal paths of the hop: every DSP_ERROR resets the flow, and an argument refusal
 * changes nothing ---- */

static void equal_weights(Complex* w, int n, float value) {
    int k;
    for (k = 0; k < AF_CHANNELS * n; ++k) { w[k].r = value; w[k].i = 0.0f; }
}

/* Finite weights that pass the gate but overflow the projection are a DSP refusal with the
 * flow reset, and default weights on a new front still fuse. */
static void test_fuse_overflow_is_dsp_error(void) {
    AfConfig cfg = af_default_config(16000);
    Af* p = af_test_create(&cfg);
    float mic[128 * AF_CHANNELS], ref[128];
    Complex w[AF_CHANNELS * 129];
    AfFront front;
    AfFrame view;
    int frame, warm = 1;
    CHECK(p != NULL, "fuse overflow: flow creates");
    if (!p) return;
    for (frame = 0; frame < 4; ++frame) {
        fill_inputs(mic, ref, 128, 16000, frame);
        warm &= af_process_front(p, mic, ref, &front) == AF_OK;
    }
    equal_weights(w, 129, 1e20f);
    CHECK(warm && af_weights_valid(w, (size_t)AF_CHANNELS * 129u) == 1 &&
              af_fuse(p, w, NULL, &view) == AF_DSP_ERROR,
          "fuse overflow: gate-valid weights of 1e20 are a DSP error, not inf/NaN with AF_OK");
    equal_weights(w, 129, 0.25f);
    CHECK(af_fuse(p, w, NULL, &view) == AF_SEQUENCE_ERROR && af_duty_hops_total(p) == 0,
          "fuse overflow: the refusal reset the flow (no front, census restarted)");
    fill_inputs(mic, ref, 128, 16000, frame);
    CHECK(af_process_front(p, mic, ref, &front) == AF_OK &&
              af_fuse(p, w, NULL, &view) == AF_OK && all_finite(view.r2, 129) &&
              all_finite(view.cn, 129),
          "fuse overflow: a new front fuses with normal weights");
    af_test_destroy(p);
}

/* af_create(): the array instance's own pool on the heap (the objects stay the caller's), the same flow as
 * a caller pool, freed by af_destroy(). */

/* ---- guard scene, weight shapes, double reference ---- */

/* ---- rendering base and post-chain resets (aec.h "Residual view") -------------------- */

/* Four lanes of one far end through four echo paths whose bulk delay moves at 1/3 and 2/3 of
 * the stream; lane 3's delay is a quarter of the others'. The first eighth has no echo path at
 * all (an estimate that is never usable there) and the middle delay is past the lanes' filter, so
 * the lanes' output guards fire, together and on their own. A double-talk burst follows the
 * first move. mics: [n][4] interleaved. */
typedef struct GuardScene {
    int sr, hop, nh;
    float* far;
    float* mics;
} GuardScene;

static float scene_rnd(void) {
    g_scene_rng ^= g_scene_rng << 13;
    g_scene_rng ^= g_scene_rng >> 17;
    g_scene_rng ^= g_scene_rng << 5;
    return (float)(g_scene_rng & 0xFFFFFFu) / (float)0x800000u - 1.0f;
}

static int guard_scene_make(GuardScene* s, int sr, int hop, int seconds) {
    int n, i, ch, seg = sr / 4, scale = sr / 16000;
    float env = 0.2f, lp = 0.0f, nlp[AF_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f};
    memset(s, 0, sizeof(*s));
    s->sr = sr;
    s->hop = hop;
    s->nh = seconds * sr / hop;
    n = s->nh * hop;
    s->far = (float*)calloc((size_t)n, sizeof(float));
    s->mics = (float*)calloc((size_t)n * AF_CHANNELS, sizeof(float));
    if (!s->far || !s->mics) return -1;
    g_scene_rng = 0x2545F491u;
    for (i = 0; i < n; ++i) {
        if (i % seg == 0) {
            float r = scene_rnd();
            env = r > 0.3f ? 0.35f : (r > -0.4f ? 0.12f : 0.0f);
        }
        lp = 0.6f * lp + 0.4f * scene_rnd();
        s->far[i] = env * lp * 2.0f;
    }
    for (i = 0; i < n; ++i) {
        int d = i < n / 3 ? 480 * scale : (i < 2 * n / 3 ? 1100 * scale : 240 * scale);
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            int dc = (ch == 3 ? d / 4 : d) + 5 * ch;
            float g = 0.45f - 0.06f * (float)ch;
            float e = 0.0f;
            nlp[ch] = 0.7f * nlp[ch] + 0.3f * scene_rnd();
            if (i >= n / 8 && i - dc >= 0) e += g * s->far[i - dc];
            if (i >= n / 8 && i - dc - 37 >= 0) e += 0.2f * s->far[i - dc - 37];
            s->mics[(size_t)i * AF_CHANNELS + (size_t)ch] =
                e + ((i > n / 3 && i < n / 3 + n / 6) ? 0.5f : 0.008f) * nlp[ch];
        }
    }
    return 0;
}

static void guard_scene_free(GuardScene* s) {
    free(s->far);
    free(s->mics);
    memset(s, 0, sizeof(*s));
}

/* The weight set of hop h, one of five shapes in turn: unit on lane 0, unit on lane 2, equal,
 * complex per bin and per hop, two complex lanes. */
static int render_weights(Complex* w, int n, int h) {
    int mode = (h / 7) % 5, ch, k;
    memset(w, 0, (size_t)AF_CHANNELS * (size_t)n * sizeof(Complex));
    for (ch = 0; ch < AF_CHANNELS; ++ch) {
        for (k = 0; k < n; ++k) {
            Complex* c = w + (size_t)ch * (size_t)n + (size_t)k;
            if (mode == 0) c->r = ch == 0 ? 1.0f : 0.0f;
            else if (mode == 1) c->r = ch == 2 ? 1.0f : 0.0f;
            else if (mode == 2) c->r = 0.25f;
            else if (mode == 3) {
                c->r = 0.25f + 0.2f * sinf(0.01f * (float)h + (float)ch + 0.003f * (float)k);
                c->i = 0.15f * cosf(0.02f * (float)h + 2.0f * (float)ch + 0.005f * (float)k);
            } else if (ch < 2) {
                c->r = ch == 0 ? 0.6f : 0.0f;
                c->i = ch == 1 ? 0.4f : 0.0f;
            }
        }
    }
    return mode;
}

/* Per-bin bound of the float32 four-term projection against the double reference. */
static int render_matches_reference(const Complex* b, const Complex* w, const Complex* const* in,
                                    int n) {
    int k, ch;
    for (k = 0; k < n; ++k) {
        double rr = 0.0, ri = 0.0, mag = 0.0;
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            const Complex wc = w[(size_t)ch * (size_t)n + (size_t)k];
            const Complex x = in[ch][k];
            rr += (double)wc.r * x.r - (double)wc.i * x.i;
            ri += (double)wc.r * x.i + (double)wc.i * x.r;
            mag += sqrt((double)wc.r * wc.r + (double)wc.i * wc.i) *
                   sqrt((double)x.r * x.r + (double)x.i * x.i);
        }
        if (fabs((double)b[k].r - rr) > 8.0 * FLT_EPSILON * mag + 1e-30 ||
            fabs((double)b[k].i - ri) > 8.0 * FLT_EPSILON * mag + 1e-30) return 0;
    }
    return 1;
}

/* The fuse's rendering base against an independent reference, on a scene where the lanes' guards
 * fire (all of them, some of them, none), with five weight shapes, trusted and rebuilt beams, a
 * flow reset mid-stream and, in MATCHED, shared-delay realigns:
 *   - render_spectra[ch] is linear_spectra[ch] itself when the lane's guard is off and its
 *     capture spectrum E + echo (bit for bit) when on, never on a usable hop;
 *   - guard_lanes is the set of contributing lanes that fired; b is e itself when empty, else the
 *     projection sum(w * render_spectra) within float rounding of a double reference, and a
 *     unit weight gives the lane's own rendering base exactly (the mono result);
 *   - the analysis outputs do not see the guard: e is the beam of linear_spectra;
 *   - post_reset_serial is the sum of the lanes' counters, still across realigns, +4 on reset.
 * Self-checks: the reference with E in place of the fired lanes' base must be rejected on some
 * fired hop, and every coverage count must be non-zero. */

/* ---- new forms of the tests that used to reach into caller-owned lanes ---------------------------------------------------- */

/* Stand-alone reference lanes (single-channel instances built from the lane contract of lane_config_for()), fed what the array
 * feeds its own lanes: the aligned far of the array's view and the channel's microphone. The mixed-hop test above shows they
 * equal the array's lanes bit for bit; the tests below use them to read what the array does not expose (a lane's echo spectrum,
 * usable verdict, serial). */
typedef struct RefLanes {
    Aec* lane[AF_CHANNELS];
    void* pool[AF_CHANNELS];
    AecTestCfg cfg;
    int prev_solid, prev_delay, n_freqs, hop;
    float* mic;
} RefLanes;

static int ref_lanes_init(RefLanes* r, const AfConfig* c) {
    size_t bytes;
    int ch;
    memset(r, 0, sizeof *r);
    if (lane_config_for(c, &r->cfg) != 0 || (bytes = aec_tcfg_mem_size(&r->cfg)) == 0) return -1;
    for (ch = 0; ch < AF_CHANNELS; ++ch) {
        if (posix_memalign(&r->pool[ch], 16, bytes) != 0 || !(r->lane[ch] = aec_tcfg_init(r->pool[ch], bytes, &r->cfg))) return -1;
    }
    r->hop = aec_hop_size(r->lane[0]);
    r->n_freqs = r->hop + 1;
    r->mic = (float*)calloc((size_t)r->hop, sizeof(float));
    return r->mic ? 0 : -1;
}

static void ref_lanes_free(RefLanes* r) {
    int ch;
    for (ch = 0; ch < AF_CHANNELS; ++ch) {
        if (r->lane[ch]) aec_destroy(r->lane[ch]);
        free(r->pool[ch]);
    }
    free(r->mic);
    memset(r, 0, sizeof *r);
}

static void ref_lanes_reset(RefLanes* r) {
    int ch;
    for (ch = 0; ch < AF_CHANNELS; ++ch) aec_reset(r->lane[ch]);
    r->prev_solid = 0;
    r->prev_delay = 0;
}

/* One hop of every reference lane after the array's linear stage `front` on `mics`. 0, or -1 if a lane refused. */
static int ref_lanes_hop(RefLanes* r, const AfFront* front, const float* mics) {
    int ch, i;
    if (front->delay.changed) {
        int old_align = r->prev_solid ? r->prev_delay : 0;
        for (ch = 0; ch < AF_CHANNELS; ++ch) aec_apply_external_realign(r->lane[ch], front->delay.delay_samples - old_align);
    }
    for (ch = 0; ch < AF_CHANNELS; ++ch) {
        for (i = 0; i < r->hop; ++i) r->mic[i] = mics[i * AF_CHANNELS + ch];
        if (aec_process_linear(r->lane[ch], r->mic, front->aligned_ref) != AEC_OK ||
            aec_process_residual(r->lane[ch], NULL) != AEC_OK) return -1;
    }
    r->prev_solid = front->delay.solid;
    r->prev_delay = front->delay.delay_samples;
    return 0;
}

/* The lane contract (lib/aec stamps it on each lane): on every grid and mode, with and without the fusion, the array's lanes
 * equal stand-alone EXTERNAL_ALIGNED lanes of the documented config bit for bit, and a lane's reset counter is its own. */
static void test_lane_contract_parity(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int g, v, ok = 1, tried = 0, serial_ok = 1;
    for (g = 0; g < 3; ++g) {
        for (v = 0; v < 5; ++v) {
            AfConfig cfg = af_default_config(grids[g][0]);
            RefLanes ref;
            Af* p;
            AfFront front;
            float mics[SPLIT_SIGNAL_MAX_HOP * AF_CHANNELS], far[SPLIT_SIGNAL_MAX_HOP];
            int h, hop, n, lane_bad = 0, ch;
            cfg.fft_size = grids[g][1];
            if (v == 1) cfg.enable_post = 0;
            if (v == 2) { cfg.delay_mode = AEC_DELAY_FIXED; cfg.fixed_delay_samples = 320; }
            if (v == 3) cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
            if (v == 4) cfg.delay_est_period_s = 0.5f;
            p = af_test_create(&cfg);
            if (!p || ref_lanes_init(&ref, &cfg) != 0) { ok = 0; af_test_destroy(p); continue; }
            tried += 1;
            hop = af_hop_size(p);
            n = af_n_freqs(p);
            for (h = 0; h < 60; ++h) {
                AecResidualView rv;
                fill_inputs(mics, far, hop, cfg.sample_rate, h);
                if (af_process_front(p, mics, far, &front) != AF_OK || ref_lanes_hop(&ref, &front, mics) != 0) { ok = 0; break; }
                for (ch = 0; ch < AF_CHANNELS; ++ch) {
                    memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
                    aec_get_residual(ref.lane[ch], &rv);
                    lane_bad += memcmp(rv.error_spec, front.linear_spectra[ch], (size_t)n * sizeof(Complex)) != 0;
                    lane_bad += memcmp(rv.formed_hop, front.formed_hop[ch], (size_t)hop * sizeof(float)) != 0;
                    lane_bad += memcmp(rv.base_spec, front.render_spectra[ch], (size_t)n * sizeof(Complex)) != 0;
                    lane_bad += (cfg.enable_post != 0) != (rv.r2 != NULL);
                }
            }
            if (lane_bad) ok = 0;
            {
                uint32_t sum = 0;
                for (ch = 0; ch < AF_CHANNELS; ++ch) sum += aec_post_reset_serial(ref.lane[ch]);
                serial_ok &= aec_post_reset_serial(p) == sum;
                af_reset(p);
                ref_lanes_reset(&ref);
                serial_ok &= aec_post_reset_serial(p) == 4u;
            }
            ref_lanes_free(&ref);
            af_test_destroy(p);
        }
    }
    CHECK(ok && tried == 15,
          "lane contract: the array's lanes equal stand-alone EXTERNAL_ALIGNED lanes of the documented config on every grid and mode");
    CHECK(serial_ok, "lane contract: the array's serial is the lanes' counters summed, +4 per aec_reset");
}

/* A non-finite hop after a good linear stage: AEC_ERR_ARG, the linear stage stays pending, and no state moved (a
 * twin that never saw the refused call stays bit-identical afterwards). */
static void test_nonfinite_hop_invalidates_front(void) {
    AfConfig cfg = af_default_config(16000);
    Af *p = af_test_create(&cfg), *twin = af_test_create(&cfg);
    float mic[128 * AF_CHANNELS], ref[128];
    Complex w[AF_CHANNELS * 129];
    AfFront front, tfront;
    AfFrame view, tview;
    unsigned long long calls;
    int frame, ok = 1, same = 1, ch;
    CHECK(p != NULL && twin != NULL, "non-finite hop: arrays build");
    if (!p || !twin) { af_test_destroy(p); af_test_destroy(twin); return; }
    equal_weights(w, 129, 0.25f);
    for (frame = 0; frame < 6; ++frame) {
        fill_inputs(mic, ref, 128, 16000, frame);
        ok &= af_process_front(p, mic, ref, &front) == AF_OK && af_process_front(twin, mic, ref, &tfront) == AF_OK;
    }
    calls = af_duty_hops_total(p);
    fill_inputs(mic, ref, 128, 16000, frame);
    mic[5] = (float)NAN;
    CHECK(ok && af_process_front(p, mic, ref, &front) == AF_INVALID_ARGUMENT,
          "non-finite hop: refused, the earlier linear stage stays pending");
    CHECK(af_duty_hops_total(p) == calls, "non-finite hop: the estimator census did not move");
    for (; frame < frame + 1 && frame < 40; ++frame) {
        fill_inputs(mic, ref, 128, 16000, frame);
        if (af_process_front(p, mic, ref, &front) != AF_OK || af_process_front(twin, mic, ref, &tfront) != AF_OK ||
            af_fuse(p, w, NULL, &view) != AF_OK || af_fuse(twin, w, NULL, &tview) != AF_OK) { ok = 0; break; }
        for (ch = 0; ch < AF_CHANNELS; ++ch)
            same &= memcmp(front.linear_spectra[ch], tfront.linear_spectra[ch], 129 * sizeof(Complex)) == 0;
        same &= memcmp(view.e, tview.e, 129 * sizeof(Complex)) == 0 && memcmp(view.r2, tview.r2, 129 * sizeof(float)) == 0;
    }
    CHECK(ok && same, "non-finite hop: no lane and no estimator state changed (a twin that never saw it stays bit-identical)");
    af_test_destroy(p);
    af_test_destroy(twin);
}

/* aec_create(): the heap arena holds the whole array (lanes, estimator, fusion); the same flow as a caller pool. */
static void test_create_heap_path(void) {
    AfConfig cfg = af_default_config(16000);
    AecTestCfg ac;
    AecMemReq req;
    void* pool = NULL;
    Aec *pooled = NULL, *heap;
    float mic[128 * AF_CHANNELS], ref[128];
    int frame, k, same = 1;
    af_to_aec(&cfg, &ac);
    CHECK(aec_tcfg_mem_requirements(&ac, &req) == 0 && posix_memalign(&pool, 16, (size_t)req.bytes) == 0, "create: pool and descriptor");
    heap = af_test_create(&cfg);
    if (pool) pooled = aec_init_ex(pool, (size_t)req.bytes, &ac.c, &req);
    CHECK(heap != NULL && pooled != NULL && af_hop_size(heap) == 128, "create: the heap array and the pooled array build");
    for (frame = 0; heap && pooled && frame < 20; ++frame) {
        AfFront fh, fp;
        fill_inputs(mic, ref, 128, 16000, frame);
        if (af_process_front(heap, mic, ref, &fh) != AF_OK || af_process_front(pooled, mic, ref, &fp) != AF_OK) { same = 0; break; }
        for (k = 0; k < 129; ++k)
            if (memcmp(&fh.linear_spectra[2][k], &fp.linear_spectra[2][k], sizeof(Complex)) != 0) same = 0;
    }
    CHECK(same, "create: the heap array equals a caller-pool array bit for bit");
    {
        AecTestCfg bad = ac;
        Aec *tmp = NULL;
        bad.c.capture_proxy_channel = 9;
        CHECK((tmp = aec_tcfg_create(&bad)) == NULL && (tmp = aec_create(NULL)) == NULL &&
                  aec_tcfg_init(pool, (size_t)req.bytes, &bad) == NULL,
              "create: refused configs or NULL give no handle");
    }
    af_test_destroy(heap);
    if (pooled) aec_destroy(pooled);
    free(pool);
}

/* A reset instance (aec_reset()) acquires the delay exactly like a new one. */
static void test_reinit_after_reset(void) {
    enum { FRAMES = 300, TRUE_DELAY = 608 };
    AfConfig cfg = af_default_config(16000);
    Af *p = af_test_create(&cfg), *q = af_test_create(&cfg);
    float *mic = NULL, *far = NULL, *hist = NULL;
    int frame, solid_before = 0, ok = 1, same = 1, k;
    CHECK(p != NULL && q != NULL, "re-init: arrays build");
    mic = (float*)malloc(128u * AF_CHANNELS * sizeof(float));
    far = (float*)malloc(128u * sizeof(float));
    hist = (float*)calloc((size_t)FRAMES * 128u, sizeof(float));
    for (frame = 0; p && mic && far && hist && frame < FRAMES; ++frame) {
        AfFront f;
        scene_echo_hop(frame, 128, TRUE_DELAY, hist, far, mic);
        if (af_process_front(p, mic, far, &f) != AF_OK) { ok = 0; break; }
        solid_before = f.delay.solid && f.delay.delay_samples > 0;
    }
    CHECK(ok && solid_before && af_realign_warm_lane_count(p) + af_realign_soft_lane_count(p) > 0,
          "re-init: the first run acquired the delay");
    af_reset(p);
    CHECK(af_realign_warm_lane_count(p) == 0 && af_realign_soft_lane_count(p) == 0 && af_duty_hops_total(p) == 0 &&
              af_pending_delay_candidate(p) == -1 && af_far_fft_real_compute_count(p) == 0,
          "re-init: aec_reset clears the realign counters, the census, the candidate and the far-FFT count");
    g_scene_rng = 0x1234567u;
    memset(hist, 0, (size_t)FRAMES * 128u * sizeof(float));
    for (frame = 0; p && q && frame < FRAMES; ++frame) {
        AfFront a, b;
        scene_echo_hop(frame, 128, TRUE_DELAY, hist, far, mic);
        if (af_process_front(p, mic, far, &a) != AF_OK || af_process_front(q, mic, far, &b) != AF_OK) { same = 0; break; }
        if (a.delay.solid != b.delay.solid || a.delay.changed != b.delay.changed || a.delay.delay_samples != b.delay.delay_samples)
            same = 0;
        for (k = 0; k < 129; ++k)
            if (memcmp(&a.linear_spectra[1][k], &b.linear_spectra[1][k], sizeof(Complex)) != 0) same = 0;
    }
    CHECK(same && af_realign_warm_lane_count(p) == af_realign_warm_lane_count(q) &&
              af_realign_soft_lane_count(p) == af_realign_soft_lane_count(q),
          "re-init: after aec_reset the array acquires exactly like a new one");
    free(hist); free(far); free(mic);
    af_test_destroy(p);
    af_test_destroy(q);
}

/* The fusion's rendering base against an independent reference, on a scene where the lanes' guards fire (all of them, some of
 * them, none), with five weight shapes, trusted and rebuilt beams, a reset mid-stream and, in MATCHED, shared-delay realigns:
 *   - render_spectra[ch] is linear_spectra[ch] itself when the lane's guard is off and its capture spectrum E + echo (bit for
 *     bit) when on, never on a usable hop;
 *   - guard_lanes is the set of contributing lanes that fired; b is e itself when empty, else the projection
 *     sum(w * render_spectra) within float rounding of a double reference, and a unit weight gives the lane's own rendering
 *     base exactly (the mono result);
 *   - the analysis outputs do not see the guard: e is the beam of linear_spectra;
 *   - post_reset_serial is the sum of the lanes' counters, still across realigns, +4 on reset.
 * Self-checks: the reference with E in place of the fired lanes' base must be rejected on some fired hop, and every coverage
 * count must be non-zero. The lane internals (echo spectrum, usable verdict, counters) are read from stand-alone reference lanes. */
static void test_render_base_contract(int sample_rate, int fft_size, AecDelayMode mode) {
    AfConfig cfg = af_default_config(sample_rate);
    GuardScene sc;
    Af* p = NULL;
    RefLanes rl;
    Complex *w = NULL, *beam = NULL;
    Complex* capture = NULL;
    int hop, n, h, ch, k;
    long fired_all = 0, fired_some = 0, fired_none = 0, unit_fired = 0, changed = 0;
    long bad_lane_base = 0, bad_mask = 0, bad_alias = 0, bad_ref = 0, bad_unit = 0;
    long bad_serial = 0, bad_e = 0, wrong_ref_rejected = 0, resets_seen = 0, bad_usable = 0;
    uint32_t last_serial = 0;
    char label[160];

    cfg.fft_size = fft_size;
    cfg.delay_mode = mode;
    if (mode != AEC_DELAY_MATCHED) cfg.delay_num_filters = DA_NUM_FILTERS;
    p = af_test_create(&cfg);
    if (!p || ref_lanes_init(&rl, &cfg) != 0 || guard_scene_make(&sc, sample_rate, af_hop_size(p), 12) != 0) {
        CHECK(0, "render base: flow and scene build");
        af_test_destroy(p);
        guard_scene_free(&sc);
        return;
    }
    hop = af_hop_size(p);
    n = af_n_freqs(p);
    w = (Complex*)calloc((size_t)AF_CHANNELS * (size_t)n, sizeof(Complex));
    beam = (Complex*)calloc((size_t)n, sizeof(Complex));
    capture = (Complex*)calloc((size_t)n, sizeof(Complex));
    for (h = 0; h < sc.nh && w && beam && capture; ++h) {
        AfFront front;
        AfFrame frame;
        const Complex* e_in[AF_CHANNELS];
        const float* mics = sc.mics + (size_t)h * hop * AF_CHANNELS;
        uint32_t lane_sum = 0;
        int wmode, mask = 0, contrib = 0, trusted = h & 1;
        if (h == sc.nh / 2) { af_reset(p); ref_lanes_reset(&rl); }
        if (af_process_front(p, mics, sc.far + (size_t)h * hop, &front) != AF_OK || ref_lanes_hop(&rl, &front, mics) != 0) {
            CHECK(0, "render base: every linear stage succeeds");
            break;
        }
        changed += front.delay.changed;
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            AecResidualView c;
            memset(&c, 0, sizeof c); c.struct_size = sizeof c;
            lane_sum += aec_post_reset_serial(rl.lane[ch]);
            aec_get_residual(rl.lane[ch], &c);
            e_in[ch] = front.linear_spectra[ch];
            if (!front.guard_active[ch]) {
                bad_lane_base += front.render_spectra[ch] != front.linear_spectra[ch];
            } else {
                for (k = 0; k < n; ++k) {
                    capture[k].r = c.error_spec[k].r + c.echo_spec[k].r;
                    capture[k].i = c.error_spec[k].i + c.echo_spec[k].i;
                }
                bad_lane_base += memcmp(capture, front.render_spectra[ch], (size_t)n * sizeof(Complex)) != 0;
                bad_usable += c.usable_linear != 0;
            }
        }
        if (h > 0 && front.post_reset_serial != last_serial) {
            resets_seen++;
            bad_serial += h != sc.nh / 2 || front.post_reset_serial != last_serial + 4u;
        }
        bad_serial += front.post_reset_serial != lane_sum;
        last_serial = front.post_reset_serial;

        wmode = render_weights(w, n, h);
        if (trusted) {
            memset(beam, 0, (size_t)n * sizeof(Complex));
            for (ch = 0; ch < AF_CHANNELS; ++ch)
                af_projection_cmac_scalar(beam, w + (size_t)ch * (size_t)n, e_in[ch], n);
        }
        if (af_fuse(p, w, trusted ? beam : NULL, &frame) != AF_OK) {
            CHECK(0, "render base: every fusion succeeds");
            break;
        }
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            int lane_on = 0;
            for (k = 0; k < n; ++k)
                lane_on |= w[(size_t)ch * n + k].r != 0.0f || w[(size_t)ch * n + k].i != 0.0f;
            contrib |= lane_on << ch;
            if (lane_on && front.guard_active[ch]) mask |= 1 << ch;
        }
        bad_mask += frame.guard_lanes != mask || frame.post_reset_serial != front.post_reset_serial;
        bad_e += trusted ? frame.e != beam : !render_matches_reference(frame.e, w, e_in, n);
        if (mask == 0) {
            fired_none++;
            bad_alias += frame.b != frame.e;
        } else {
            fired_all += mask == contrib && contrib == 0xF;
            fired_some += mask != contrib;
            bad_ref += !render_matches_reference(frame.b, w, front.render_spectra, n);
            wrong_ref_rejected += !render_matches_reference(frame.b, w, e_in, n);
            if (wmode <= 1) {
                const Complex* own = front.render_spectra[wmode == 0 ? 0 : 2];
                unit_fired++;
                for (k = 0; k < n; ++k)
                    bad_unit += frame.b[k].r != own[k].r || frame.b[k].i != own[k].i;
            }
        }
    }
    snprintf(label, sizeof label, "render base @ %d Hz/%d %s: ", sample_rate, fft_size,
             mode == AEC_DELAY_MATCHED ? "matched" : "external");
#define RCHECK(cond, what) do { char l_[256]; snprintf(l_, sizeof l_, "%s%s", label, what); \
                                CHECK(cond, l_); } while (0)
    printf("  %shops %d: all fired %ld, some fired %ld, none %ld, unit-weight fired %ld, delay changes %ld\n",
           label, sc.nh, fired_all, fired_some, fired_none, unit_fired, changed);
    RCHECK(h == sc.nh, "the stream runs to the end");
    RCHECK(fired_all > 0 && fired_some > 0 && fired_none > 0 && unit_fired > 0,
           "the scene covers all / some / no lanes firing and a fired unit-weight hop");
    RCHECK(mode != AEC_DELAY_MATCHED || changed > 0, "the shared delay realigns (MATCHED)");
    RCHECK(bad_lane_base == 0 && bad_usable == 0,
           "render_spectra is E, or E + echo exactly on a guard hop, never on a usable hop");
    RCHECK(bad_mask == 0, "guard_lanes is the contributing fired set; the frame carries the serial");
    RCHECK(bad_e == 0, "e is the analysis beam of linear_spectra, untouched by the guard");
    RCHECK(bad_alias == 0, "with no contributing lane fired, b is e itself");
    RCHECK(bad_ref == 0, "b is sum(w * render_spectra) within float rounding (double reference)");
    RCHECK(wrong_ref_rejected > 0, "self-check: the analysis projection fails that reference on fired hops");
    RCHECK(bad_unit == 0, "a unit weight renders the lane's own base exactly (mono result)");
    RCHECK(bad_serial == 0 && resets_seen == 1,
           "post_reset_serial = sum of lane counters, still across realigns, +4 at aec_reset");
#undef RCHECK
    free(w);
    free(beam);
    free(capture);
    guard_scene_free(&sc);
    ref_lanes_free(&rl);
    af_test_destroy(p);
}



/* ---- a non-finite beamformed spectrum ---- */
/* A beamformed spectrum handed to the fuse must be finite in every bin and
 * component: one NaN or infinity anywhere is a DSP error. */
static void test_trusted_spectrum_finite_gate(void) {
    AfConfig cfg = af_default_config(16000);
    Af* p;
    float* microphones;
    float* ref;
    Complex* weights;
    Complex* trusted_spectrum;
    int n_freqs;
    int hop;
    int ch;
    int k;
    int ok_clean = 1;
    int ok_bad = 1;
    const float bad_values[4] = {
        (float)NAN, INFINITY, -INFINITY, -(float)NAN};
    const int bin_count_offsets[9] = {0, 1, 2, 3, 4, 5, 64, 127, 128};
    int frame = 0;
    int v;
    int comp;
    int b;

    cfg.fft_size = 256;
    p = af_test_create(&cfg);
    CHECK(p != NULL, "finite gate: core creates");
    if (!p) return;
    hop = af_hop_size(p);
    n_freqs = af_n_freqs(p);
    microphones = (float*)calloc(
        (size_t)hop * AF_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)AF_CHANNELS * n_freqs, sizeof(Complex));
    trusted_spectrum = (Complex*)calloc((size_t)n_freqs, sizeof(Complex));
    if (!microphones || !ref || !weights || !trusted_spectrum) {
        CHECK(0, "finite gate: buffers allocate");
        goto cleanup;
    }
    for (ch = 0; ch < AF_CHANNELS; ++ch)
        for (k = 0; k < n_freqs; ++k) {
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
            weights[(size_t)ch * n_freqs + k].i = 0.0f;
        }
    for (k = 0; k < n_freqs; ++k) {
        trusted_spectrum[k].r = 1e-3f * (float)(k % 5);
        trusted_spectrum[k].i = -1e-3f * (float)(k % 3);
    }
    {
        AfFront front;
        AfFrame view;
        fill_inputs(microphones, ref, hop, 16000, frame++);
        ok_clean = af_process_front(p, microphones, ref, &front) ==
                       AF_OK &&
                   af_fuse(p, weights, trusted_spectrum, &view) ==
                       AF_OK;
        for (v = 0; v < 4 && ok_bad; ++v)
            for (comp = 0; comp < 2 && ok_bad; ++comp)
                for (b = 0; b < 9 && ok_bad; ++b) {
                    Complex saved;
                    int bin = bin_count_offsets[b];
                    if (bin >= n_freqs) bin = n_freqs - 1;
                    saved = trusted_spectrum[bin];
                    if (comp) trusted_spectrum[bin].i = bad_values[v];
                    else trusted_spectrum[bin].r = bad_values[v];
                    fill_inputs(microphones, ref, hop, 16000, frame++);
                    ok_bad = af_process_front(
                                 p, microphones, ref, &front) ==
                                 AF_OK &&
                             af_fuse(
                                 p, weights, trusted_spectrum, &view) ==
                                 AF_INVALID_ARGUMENT;
                    trusted_spectrum[bin] = saved;
                }
    }
    CHECK(ok_clean, "finite gate: a finite beamformed spectrum is accepted");
    CHECK(ok_bad,
          "finite gate: a NaN or infinity in any component is an argument error");
    {
        /* An invalid spectrum is refused up front: nothing is reset, the hop stays open (the next fuse of the
         * same front succeeds) and the shared estimator's census keeps counting. */
        AfFront front;
        AfFrame view;
        int warm;
        fill_inputs(microphones, ref, hop, 16000, frame++);
        warm = af_process_front(p, microphones, ref, &front) == AF_OK;
        fill_inputs(microphones, ref, hop, 16000, frame++);
        warm = warm && af_process_front(p, microphones, ref, &front) == AF_OK &&
               af_duty_hops_total(p) >= 2;
        trusted_spectrum[3].r = (float)NAN;
        CHECK(warm && af_fuse(p, weights, trusted_spectrum, &view) == AF_INVALID_ARGUMENT &&
                  af_duty_hops_total(p) >= 2 &&
                  af_fuse(p, weights, NULL, &view) == AF_OK,
              "finite gate: the argument refusal reset nothing (the open hop still fuses, census kept)");
    }

cleanup:
    free(trusted_spectrum);
    free(weights);
    free(ref);
    free(microphones);
    af_test_destroy(p);
}


/* ---- pre-frame WOLA identity ---- */
/* The WOLA identity below is measured across a realign boundary, so the scene is a delayed
 * echo that MOVES: on a zero-delay scene `changed` realigns by a delta of 0, which
 * aec_apply_external_realign() treats as a no-op. `history` holds WOLA_SCENE_PAD samples of
 * pre-roll so the echo is valid from the first sample. The run is long because the second
 * boundary is a re-lock and the shared matched filter is duty-cycled (coarsest grid:
 * 48 kHz, hop 512), so the scene has to outlast the decimated analysis. */
#define WOLA_SCENE_PAD   16384
#define WOLA_SHIFT_HOP   150
#define WOLA_SCENE_HOPS  3000

static void fill_delayed_echo(float* microphones, float* ref, int hop,
                              const float* history, int base, int delay) {
    int i;
    int ch;
    for (i = 0; i < hop; ++i) {
        float echo = 0.6f * history[base + i - delay];
        ref[i] = history[base + i];
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            microphones[i * AF_CHANNELS + ch] =
                echo * (1.0f + 0.05f * (float)ch);
        }
    }
}

static void test_pre_frame_wola_identity(int sample_rate, int fft_size) {
    AfConfig cfg = af_default_config(sample_rate);
    Af* pipeline = NULL;
    AfFront pre;
    FftHandle* fft = NULL;
    float* microphones = NULL;
    float* ref = NULL;
    float* out = NULL;
    float* previous = NULL;
    float* ola = NULL;
    float* time_frame = NULL;
    Complex* weights = NULL;
    float* window = NULL;
    float* history = NULL;
    uint32_t rng = 0x1234567u;
    float max_error = 0.0f;
    long sweeps_before = 0;
    int applied_before = -1;
    int moving_realigns = 0;
    int lanes_swept_on_moves = 0;
    int rate_scale = sample_rate / 16000;
    int base_delay = 512 * rate_scale;
    int moved_delay = 1024 * rate_scale;
    int valid = 1;
    int hop;
    int n_freqs;
    char label[224];

    cfg.fft_size = fft_size;
    pipeline = af_test_create(&cfg);
    if (pipeline) fft = fft_create(fft_size);
    CHECK(pipeline != NULL && fft != NULL,
          "4ch WOLA identity instances create");
    if (!pipeline || !fft) goto cleanup;

    hop = af_hop_size(pipeline);
    n_freqs = af_n_freqs(pipeline);
    microphones = (float*)calloc(
        (size_t)hop * AF_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    previous = (float*)calloc(
        (size_t)hop * AF_CHANNELS, sizeof(float));
    ola = (float*)calloc(
        (size_t)fft_size * AF_CHANNELS, sizeof(float));
    time_frame = (float*)calloc((size_t)fft_size, sizeof(float));
    weights = (Complex*)calloc(
        (size_t)n_freqs * AF_CHANNELS, sizeof(Complex));
    window = (float*)calloc((size_t)fft_size, sizeof(float));
    history = (float*)calloc(
        (size_t)(WOLA_SCENE_HOPS * hop + WOLA_SCENE_PAD), sizeof(float));
    if (!microphones || !ref || !out || !previous || !ola ||
        !time_frame || !weights || !window || !history) {
        valid = 0;
        goto check_result;
    }
    for (int i = 0; i < WOLA_SCENE_HOPS * hop + WOLA_SCENE_PAD; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        history[i] = 0.1f * (((float)(rng >> 8) *
                              (1.0f / 16777216.0f)) - 0.5f);
    }
    ac_window_build(window, fft_size);
    for (int ch = 0; ch < AF_CHANNELS; ++ch) {
        for (int k = 0; k < n_freqs; ++k)
            weights[(size_t)ch * n_freqs + k].r = 0.25f;
    }

    for (int frame = 0; frame < WOLA_SCENE_HOPS; ++frame) {
        long sweeps_after;
        fill_delayed_echo(
            microphones, ref, hop, history, frame * hop + WOLA_SCENE_PAD,
            frame >= WOLA_SHIFT_HOP ? moved_delay : base_delay);
        if (af_process_front(
                pipeline, microphones, ref, &pre) != AF_OK) {
            valid = 0;
            break;
        }
        /* A delay realignment does not restart any WOLA sequence: the
         * lanes realign their filters in place, so the external mirror keeps
         * its OLA running too and this identity proves the seams stay
         * continuous ACROSS the realign boundary, not merely after both
         * sides were wiped. Counted here so the claim rests on realigns this
         * scene actually performed: a generation that MOVES the alignment is
         * the only one that shifts an IR, and each must sweep four lanes. */
        sweeps_after = af_realign_warm_lane_count(pipeline) +
                       af_realign_soft_lane_count(pipeline);
        if (pre.delay.changed && applied_before >= 0 &&
            pre.delay.delay_samples != applied_before) {
            moving_realigns += 1;
            lanes_swept_on_moves += (int)(sweeps_after - sweeps_before);
        }
        applied_before = pre.delay.delay_samples;
        sweeps_before = sweeps_after;
        for (int ch = 0; ch < AF_CHANNELS; ++ch) {
            float* channel_ola = ola + (size_t)ch * fft_size;
            float* channel_previous = previous + (size_t)ch * hop;
            fft_inverse(fft, pre.linear_spectra[ch], time_frame);
            for (int i = 0; i < fft_size; ++i)
                channel_ola[i] += time_frame[i] * window[i];
            for (int i = 0; i < hop; ++i) {
                float d = fabsf(channel_ola[i] - channel_previous[i]);
                if (d > max_error) max_error = d;
                channel_previous[i] = pre.formed_hop[ch][i];
            }
            memmove(channel_ola, channel_ola + hop,
                    (size_t)(fft_size - hop) * sizeof(float));
            memset(channel_ola + (fft_size - hop), 0,
                   (size_t)hop * sizeof(float));
        }
    }

check_result:
    snprintf(label, sizeof(label),
             "4ch front time/spectrum seams share one reconstructing WOLA grid "
             "sr=%d/fft=%d (max error %.3g across %d alignment moves)",
             sample_rate, fft_size, (double)max_error, moving_realigns);
    CHECK(valid && max_error <= 1e-4f, label);
    /* Two moves at least: the acquisition off raw far, and the mid-stream
     * shift. Without them the identity above would only have been measured on
     * a stream whose alignment never moved. */
    snprintf(label, sizeof(label),
             "sr=%d/fft=%d: the scene really crosses realign boundaries (%d "
             "alignment moves, %d lane realigns)",
             sample_rate, fft_size, moving_realigns, lanes_swept_on_moves);
    CHECK(moving_realigns >= 2 &&
          lanes_swept_on_moves ==
              moving_realigns * AF_CHANNELS, label);

cleanup:
    free(history);
    free(window);
    free(weights);
    free(time_frame);
    free(ola);
    free(previous);
    free(out);
    free(ref);
    free(microphones);
    fft_destroy(fft);
    af_test_destroy(pipeline);
}

/* Group 6: lane 0 runs its own far-end FFT every hop; lanes 1-3 borrow it via
 * aec_process_linear_shared_far_from(). The total (af_far_fft_real_compute_count(),
 * summed over the four lanes) must rise by at most 1 per af_process_front() call,
 * never 4. "At most", not "exactly": the call resets all four lanes (zeroing the counter)
 * whenever the shared delay estimate changes, e.g. during initial acquisition. A drop is
 * fine; a jump by more than 1 would mean lanes 1-3 stopped sharing. */

/* ---- unit-weight rendering equals the single-channel full mode ---- */

/* The external synthesis of one fused output, as the example chains write it, with knobs that
 * break one rule each for the mutation rows. */
typedef struct OracleSynth {
    AcIstft* is;
    void* mem;
    Complex* spec;
    uint32_t rng, handled;
    int render_from_e;     /* mutation: render the analysis beam */
    int ignore_serial;     /* mutation: never follow a post-chain reset */
    uint32_t seed;         /* AEC_CNG_SEED, or a mutated seed */
} OracleSynth;

static int oracle_synth_init(OracleSynth* o, const Aec* aec, int fft, uint32_t seed) {
    AcStftConfig sc;
    size_t sz;
    int win_n = 0;
    memset(o, 0, sizeof(*o));
    ac_stft_config_default(&sc, fft);
    /* the library's own synthesis window */
    if (aec_synthesis_window(aec, &sc.window, &win_n) != 0 || win_n != fft) return -1;
    sz = ac_istft_get_mem_size(&sc);
    if (!sz || posix_memalign(&o->mem, 16, sz) != 0) return -1;
    o->is = ac_istft_init(o->mem, sz, &sc);
    o->spec = (Complex*)calloc((size_t)(fft / 2 + 1), sizeof(Complex));
    o->seed = seed;
    o->rng = seed;
    return o->is && o->spec ? 0 : -1;
}

static void oracle_synth_free(OracleSynth* o) {
    free(o->mem);
    free(o->spec);
    memset(o, 0, sizeof(*o));
}

static void oracle_synth_hop(OracleSynth* o, const AfFrame* f, const float* gain, int cng,
                             float* out) {
    if (!o->ignore_serial && f->post_reset_serial != o->handled) {
        o->handled = f->post_reset_serial;
        ac_istft_reset(o->is);
        o->rng = o->seed;
    }
    sk_capply_gain_f32(o->spec, o->render_from_e ? f->e : f->b, gain, f->n_freqs);
    if (cng) aec_cng_fill(&o->rng, o->spec, f->cn, gain, NULL, 0.0f, f->n_freqs);
    ac_istft_push(o->is, o->spec, out);
}

/* The array's rendering against the library's full mode, through public calls only: with the
 * weights [1, 0, 0, 0] the fused frame is lane 0's context, so rendering frame.b with the gain
 * of a full-mode twin of lane 0 (same lane config but residual FULL, fed the same capture and
 * the same aligned far, reset with the flow), plus its comfort noise from AEC_CNG_SEED and the
 * post_reset_serial rule, through an AcIstft with the library's synthesis window must reproduce
 * the twin's aec_process() output bit for bit on every hop, guard hops and the reset included.
 * The post-beam RES is not part of this: it is a separate stage with its own gain. Mutations
 * (rendering e, ignoring the serial, the old seed) must each break the identity. */
static void test_unit_weight_matches_full_mode(int sample_rate, int fft_size, int cng) {
    AfConfig cfg = af_default_config(sample_rate);
    GuardScene sc;
    Af* p = NULL;
    Aec* twin = NULL;
    AecTestCfg lane;
    OracleSynth good, m_e, m_serial, m_seed;
    Complex* w = NULL;
    float *out_twin = NULL, *out = NULL, *out_m = NULL, *mic0 = NULL;
    long diff = 0, diff_e = 0, diff_serial = 0, diff_seed = 0, guard_hops = 0, bad_ctx = 0;
    int hop, n, h, i, twin_ok;
    char label[200];

    cfg.fft_size = fft_size;
    cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    p = af_test_create(&cfg);
    twin_ok = p && lane_config_for(&cfg, &lane) == 0;
    if (twin_ok) {
        lane.c.residual = AEC_RESIDUAL_FULL;       /* the full-mode twin of lane 0: aec_process() applies its own gain */
        lane.c.enable_cng = cng;
        twin_ok = (twin = aec_tcfg_create(&lane)) != NULL;
    }
    if (!twin_ok || guard_scene_make(&sc, sample_rate, af_hop_size(p), 10) != 0) {
        CHECK(0, "full-mode twin: flow, twin and scene build");
        aec_destroy(twin);
        af_test_destroy(p);
        guard_scene_free(&sc);
        return;
    }
    hop = af_hop_size(p);
    n = af_n_freqs(p);
    w = (Complex*)calloc((size_t)AF_CHANNELS * (size_t)n, sizeof(Complex));
    out_twin = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    out_m = (float*)calloc((size_t)hop, sizeof(float));
    mic0 = (float*)calloc((size_t)hop, sizeof(float));
    if (oracle_synth_init(&good, twin, fft_size, AEC_CNG_SEED) ||
        oracle_synth_init(&m_e, twin, fft_size, AEC_CNG_SEED) ||
        oracle_synth_init(&m_serial, twin, fft_size, AEC_CNG_SEED) ||
        /* negative control: the retired comfort-noise seed, planted as a defect */
        oracle_synth_init(&m_seed, twin, fft_size, 0x9e3779b9u) || !w || !mic0) {
        CHECK(0, "full-mode twin: synthesizers build");
        sc.nh = 0;
    }
    m_e.render_from_e = 1;
    m_serial.ignore_serial = 1;
    for (i = 0; i < n; ++i) w[i].r = 1.0f;
    for (h = 0; h < sc.nh; ++h) {
        AfFront front;
        AfFrame frame;
        AecResidualView tc;
        const float* mics = sc.mics + (size_t)h * hop * AF_CHANNELS;
        if (h == sc.nh / 2) {
            af_reset(p);
            aec_reset(twin);
        }
        if (af_process_front(p, mics, sc.far + (size_t)h * hop, &front) != AF_OK ||
            af_fuse(p, w, NULL, &frame) != AF_OK) {
            CHECK(0, "full-mode twin: every hop of the flow succeeds");
            break;
        }
        for (i = 0; i < hop; ++i) mic0[i] = mics[(size_t)i * AF_CHANNELS];
        aec_process(twin, mic0, front.aligned_ref, out_twin);
        memset(&tc, 0, sizeof tc); tc.struct_size = sizeof tc;
        aec_get_residual(twin, &tc);
        guard_hops += front.guard_active[0];
        bad_ctx += memcmp(tc.error_spec, frame.e, (size_t)n * sizeof(Complex)) != 0;
        oracle_synth_hop(&good, &frame, tc.res_gain, cng, out);
        diff += memcmp(out, out_twin, (size_t)hop * sizeof(float)) != 0;
        oracle_synth_hop(&m_e, &frame, tc.res_gain, cng, out_m);
        diff_e += memcmp(out_m, out_twin, (size_t)hop * sizeof(float)) != 0;
        oracle_synth_hop(&m_serial, &frame, tc.res_gain, cng, out_m);
        diff_serial += memcmp(out_m, out_twin, (size_t)hop * sizeof(float)) != 0;
        oracle_synth_hop(&m_seed, &frame, tc.res_gain, cng, out_m);
        diff_seed += memcmp(out_m, out_twin, (size_t)hop * sizeof(float)) != 0;
    }
    snprintf(label, sizeof label, "full-mode twin @ %d Hz/%d cng=%d", sample_rate, fft_size, cng);
    printf("  %s: %d hops, %ld guard hops on lane 0, %ld differ (mutations: render e %ld, "
           "ignore serial %ld, old seed %ld)\n", label, sc.nh, guard_hops, diff, diff_e,
           diff_serial, diff_seed);
    CHECK(h == sc.nh && sc.nh > 0 && guard_hops > 0,
          "full-mode twin: the stream runs to the end and lane 0's guard fires");
    CHECK(bad_ctx == 0, "full-mode twin: lane 0's analysis E equals the full-mode twin's");
    CHECK(diff == 0, "full-mode twin: unit-weight rendering of frame.b == aec_process() bit for bit");
    CHECK(diff_e > 0, "full-mode twin: mutation 'render e' is detected");
    CHECK(diff_serial > 0, "full-mode twin: mutation 'ignore post_reset_serial' is detected");
    CHECK(!cng || diff_seed > 0, "full-mode twin: mutation 'old comfort-noise seed' is detected");
    oracle_synth_free(&good);
    oracle_synth_free(&m_e);
    oracle_synth_free(&m_serial);
    oracle_synth_free(&m_seed);
    aec_destroy(twin);
    free(w);
    free(out_twin);
    free(out);
    free(out_m);
    free(mic0);
    guard_scene_free(&sc);
    af_test_destroy(p);
}


/* ---- delay request rules: every row is accepted or refused alike by the pool query, aec_create and aec_init; a refused row
 * leaves the pool unwritten. ---- */
static void test_delay_request_rules(void) {
    enum { M = AEC_DELAY_MATCHED, F = AEC_DELAY_FIXED, E = AEC_DELAY_EXTERNAL_ALIGNED };
    static const struct {
        int sr, mode, filters, fixed, accept, neighbour;
    } rows[] = {
        /* 0 */ {16000, M, 1, -1, 1, -1},
        /* 1 */ {16000, M, DA_NUM_FILTERS, -1, 1, -1},
        /* 2 */ {16000, F, DA_NUM_FILTERS, 0, 1, -1},
        /* 3 */ {16000, F, DA_NUM_FILTERS, 120 * 16000, 1, -1},
        /* 4 */ {48000, F, DA_NUM_FILTERS, 120 * 16000 + 1, 1, -1},
        /* 5 */ {48000, F, DA_NUM_FILTERS, 120 * 48000, 1, -1},
        /* 6 */ {16000, E, DA_NUM_FILTERS, -1, 1, -1},
        {16000, M, 0, -1, 0, 0},
        {16000, M, -1, -1, 0, 0},
        {16000, M, DA_NUM_FILTERS + 1, -1, 0, 1},
        {16000, M, 1, 0, 0, 0},
        {16000, M, DA_NUM_FILTERS, 0, 0, 1},
        {16000, M, DA_NUM_FILTERS, -2, 0, 1},
        {16000, F, DA_NUM_FILTERS, -1, 0, 2},
        {16000, F, DA_NUM_FILTERS, -2, 0, 2},
        {16000, F, DA_NUM_FILTERS, 120 * 16000 + 1, 0, 3},
        {16000, F, DA_NUM_FILTERS, 0x7fffffff, 0, 3},
        {48000, F, DA_NUM_FILTERS, 120 * 48000 + 1, 0, 5},
        {16000, F, DA_NUM_FILTERS - 1, 0, 0, 2},
        {16000, F, DA_NUM_FILTERS + 1, 0, 0, 2},
        {16000, F, 1, 0, 0, 2},
        {16000, E, DA_NUM_FILTERS, 0, 0, 6},
        {16000, E, DA_NUM_FILTERS, -2, 0, 6},
        {16000, E, DA_NUM_FILTERS - 1, -1, 0, 6},
        {16000, E, DA_NUM_FILTERS + 1, -1, 0, 6},
        {16000, E, 2, -1, 0, 6},
        {16000, 3, DA_NUM_FILTERS, -1, 0, 6},
        {16000, -1, DA_NUM_FILTERS, -1, 0, 6},
    };
    const int count = (int)(sizeof(rows) / sizeof(rows[0]));
    int r, bad = 0, refused_untouched = 1;
    for (r = 0; r < count; ++r) {
        AfConfig cfg, host;
        AecTestCfg ac;
        AecMemReq req, host_req;
        unsigned char* pool = NULL;
        size_t i;
        int h = rows[r].accept ? r : rows[r].neighbour;
        int want = rows[r].accept ? 0 : -1;
        int got_mem, got_create, got_init = -2;
        Aec* p;
        cfg = af_default_config(rows[r].sr);
        cfg.fft_size = rows[r].sr == 48000 ? 1024 : 256;
        cfg.delay_mode = (AecDelayMode)rows[r].mode;
        cfg.delay_num_filters = rows[r].filters;
        cfg.fixed_delay_samples = rows[r].fixed;
        host = af_default_config(rows[h].sr);
        host.fft_size = cfg.fft_size;
        host.delay_mode = (AecDelayMode)rows[h].mode;
        host.delay_num_filters = rows[h].filters;
        host.fixed_delay_samples = rows[h].fixed;

        got_mem = af_get_mem_requirements(&cfg, &req) == 0 ? 0 : -1;
        { Af* c = af_test_create(&cfg); got_create = c ? 0 : -1; af_test_destroy(c); }
        af_to_aec(&cfg, &ac);
        if (af_get_mem_requirements(&host, &host_req) == 0 &&
            posix_memalign((void**)&pool, 16, (size_t)host_req.bytes) == 0) {
            memset(pool, 0xa5, (size_t)host_req.bytes);
            p = aec_tcfg_init(pool, (size_t)host_req.bytes, &ac);
            got_init = p ? 0 : -1;
            if (!rows[r].accept)
                for (i = 0; i < (size_t)host_req.bytes; ++i)
                    if (pool[i] != 0xa5) { refused_untouched = 0; break; }
            if (p) aec_destroy(p);
        }
        if (got_mem != want || got_create != want || got_init != want) {
            printf("  delay row %d (sr %d mode %d filters %d fixed %d): mem %d create %d init %d, want %d\n", r,
                   rows[r].sr, rows[r].mode, rows[r].filters, rows[r].fixed, got_mem, got_create, got_init, want);
            bad += 1;
        }
        free(pool);
    }
    CHECK(bad == 0 && count == 28,
          "delay rules: every MATCHED/FIXED/EXTERNAL_ALIGNED row is accepted or refused alike by the pool query, aec_create and aec_init");
    CHECK(refused_untouched, "delay rules: a refused delay request leaves the pool unwritten");
}

/* A beamformed spectrum the caller computed itself (the trusted path): used in place, never rebuilt, and everything else of the
 * fused frame equals the rebuilt one's. The beam here is the scalar projection of the lanes' error spectra. */
static void test_trusted_spectrum_path(void) {
    AfConfig cfg = af_default_config(16000);
    Af *a, *b;
    float mics[128 * AF_CHANNELS], ref[128];
    Complex w[AF_CHANNELS * 129], trusted[129];
    int frame, ch, k, ok = 1, aliased = 1, same_rest = 1, e_same = 1;
    cfg.fft_size = 256;
    a = af_test_create(&cfg);
    b = af_test_create(&cfg);
    CHECK(a && b, "trusted spectrum: both arrays build");
    if (!a || !b) { af_test_destroy(a); af_test_destroy(b); return; }
    for (ch = 0; ch < AF_CHANNELS; ++ch)
        for (k = 0; k < 129; ++k) {
            w[ch * 129 + k].r = 0.20f + 0.025f * (float)ch;
            w[ch * 129 + k].i = 0.0001f * (float)((k % 7) - 3) * (float)ch;
        }
    for (frame = 0; frame < 4; ++frame) {
        AfFront fa, fb;
        AfFrame ra, rb;
        fill_inputs(mics, ref, 128, 16000, frame);
        ok &= af_process_front(a, mics, ref, &fa) == AF_OK && af_process_front(b, mics, ref, &fb) == AF_OK;
        memset(trusted, 0, sizeof trusted);
        for (ch = 0; ch < AF_CHANNELS; ++ch)
            af_projection_cmac_scalar(trusted, w + ch * 129, fb.linear_spectra[ch], 129);
        ok &= af_fuse(a, w, NULL, &ra) == AF_OK && af_fuse(b, w, trusted, &rb) == AF_OK;
        aliased &= rb.e == trusted && ra.e != trusted;
        e_same &= memcmp(ra.e, rb.e, sizeof trusted) == 0;
        same_rest &= memcmp(ra.y, rb.y, sizeof trusted) == 0 && memcmp(ra.r2, rb.r2, 129 * sizeof(float)) == 0 &&
                     memcmp(ra.cn, rb.cn, 129 * sizeof(float)) == 0 && memcmp(ra.b, rb.b, sizeof trusted) == 0;
    }
    CHECK(ok, "trusted spectrum: the linear and fusion stages succeed");
    CHECK(aliased, "trusted spectrum: a supplied spectrum is used in place, a missing one rebuilt");
    CHECK(e_same && same_rest, "trusted spectrum: the scalar projection as the trusted beam gives the rebuilt frame bit for bit");
    af_test_destroy(a);
    af_test_destroy(b);
}

/* The grid contract of the array, without a post chain (the 4ch suite's run_grid minus the NR / RES chain, which stays with the
 * flow in Audio_ALG): accessors, the sequence rules around reset, retry after invalid weights, NULL arguments, frame contents. */
static int run_grid(int sample_rate, int fft_size) {
    AfConfig cfg = af_default_config(sample_rate);
    Af* p;
    AfFront pre;
    AfFrame frame_view;
    float *microphones, *ref;
    Complex* weights;
    int hop, n_freqs, frame, ch, k, rc;

    cfg.fft_size = fft_size;
    p = af_test_create(&cfg);
    CHECK(p != NULL, "create supported rate");
    if (!p) return 0;
    hop = af_hop_size(p);
    n_freqs = af_n_freqs(p);
    CHECK(hop == fft_size / 2, "rate-specific hop");
    CHECK(af_fft_size(p) == 2 * hop, "FFT is two hops");
    CHECK(af_sample_rate(p) == sample_rate, "sample-rate accessor");
    CHECK(af_matched_filter_count(p) == 1, "one shared matcher");
    CHECK(af_linear_aec_count(p) == AF_CHANNELS, "four linear AEC filters");
    microphones = (float*)calloc((size_t)hop * AF_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc((size_t)AF_CHANNELS * n_freqs, sizeof(Complex));
    CHECK(microphones && ref && weights, "test buffers allocate");
    if (!microphones || !ref || !weights) { free(microphones); free(ref); free(weights); af_test_destroy(p); return 0; }
    for (ch = 0; ch < AF_CHANNELS; ++ch)
        for (k = 0; k < n_freqs; ++k) weights[(size_t)ch * n_freqs + k].r = 0.25f;

    CHECK(af_fuse(p, weights, NULL, &frame_view) == AF_SEQUENCE_ERROR, "fuse before any front frame is a sequence error");
    for (frame = 0; frame < 3; ++frame) {
        fill_inputs(microphones, ref, hop, sample_rate, frame);
        rc = af_process_front(p, microphones, ref, &pre);
        CHECK(rc == AF_OK, "front stage succeeds");
        if (rc != AF_OK) break;
        CHECK(pre.hop_size == hop && pre.n_freqs == n_freqs && pre.n_channels == AF_CHANNELS && pre.aligned_ref != NULL,
              "front view shape contract");
        for (ch = 0; ch < AF_CHANNELS; ++ch)
            CHECK(pre.linear_spectra[ch] != NULL && all_finite(pre.formed_hop[ch], hop), "lane spectrum and finite formed hop");
        CHECK(af_fuse(p, weights, NULL, &frame_view) == AF_OK, "fusion succeeds");
    }
    fill_inputs(microphones, ref, hop, sample_rate, 4);
    CHECK(af_process_front(p, microphones, ref, &pre) == AF_OK, "front before reset succeeds");
    af_reset(p);
    CHECK(af_fuse(p, weights, NULL, &frame_view) == AF_SEQUENCE_ERROR, "reset invalidates the front frame");
    fill_inputs(microphones, ref, hop, sample_rate, 5);
    CHECK(af_process_front(p, microphones, ref, &pre) == AF_OK, "front after reset succeeds");
    memset(weights, 0, (size_t)AF_CHANNELS * n_freqs * sizeof(Complex));
    CHECK(af_fuse(p, weights, NULL, &frame_view) == AF_INVALID_ARGUMENT, "all-zero beamformer weights are rejected");
    for (ch = 0; ch < AF_CHANNELS; ++ch)
        for (k = 0; k < n_freqs; ++k) weights[(size_t)ch * n_freqs + k].r = 0.25f;
    CHECK(af_fuse(p, weights, NULL, &frame_view) == AF_OK, "the front frame may retry after invalid weights");
    CHECK(af_fuse(NULL, weights, NULL, &frame_view) == AF_INVALID_ARGUMENT &&
              af_fuse(p, NULL, NULL, &frame_view) == AF_INVALID_ARGUMENT && af_fuse(p, weights, NULL, NULL) == AF_INVALID_ARGUMENT,
          "fuse rejects NULL arguments");
    CHECK(frame_view.n_freqs == n_freqs && frame_view.hop == hop && frame_view.e != NULL && frame_view.r2 != NULL &&
              frame_view.cn != NULL && frame_view.g_res_ctx == NULL && frame_view.far_hop == pre.aligned_ref,
          "fused frame carries E, R^2, comfort noise and the aligned far");
    free(microphones); free(ref); free(weights);
    af_test_destroy(p);
    return 1;
}

/* The admission band with literal numbers (the macro-based rows above would follow a changed constant): a movement of exactly
 * 32 samples is absorbed and holds nothing, 33 starts a candidate; a repeat within 15 samples of the candidate admits it, one
 * 16 away only replaces it. */
static void test_admission_band_literals(void) {
    AfDelayAdmission a;
    memset(&a, 0, sizeof a);
    CHECK(af_admission_offer(&a, 1000, 1032) == 0 && a.ttl == 0, "admission band: a movement of exactly 32 samples is absorbed and holds nothing");
    CHECK(af_admission_offer(&a, 1000, 968) == 0 && a.ttl == 0, "admission band: -32 samples likewise");
    CHECK(af_admission_offer(&a, 1000, 1033) == 0 && a.ttl == 3 && a.candidate == 1033, "admission band: 33 samples starts a candidate with a life of 3 hops");
    CHECK(af_admission_offer(&a, 1000, 1033 + 16) == 0 && a.candidate == 1049 && a.ttl == 3, "admission band: a repeat 16 samples away only replaces the candidate");
    CHECK(af_admission_offer(&a, 1000, 1049 - 15) == 1 && a.ttl == 0, "admission band: a repeat 15 samples away admits it");
}

/* The backward-jump quarantine end to end, on the two-path scene of array_signal.h (32 ms and 48 ms paths of nearly equal gain
 * whose balance flips every 2.4 s, so the shared estimate moves between them): with the quarantine on, a backward move that
 * the capture-proxy lane still cancels through is adopted about delay_backward_quarantine_s later than with it off; forward
 * moves and the first acquisition are never held. */
static void scene_changes(int quarantine, int* hops_out, int* backward_out, int max) {
    AfConfig cfg = af_default_config(16000);
    Af* p;
    ArraySig sig;
    float mics[SPLIT_SIGNAL_MAX_HOP * AF_CHANNELS], ref[SPLIT_SIGNAL_MAX_HOP];
    int h, n_hops, count = 0, prev = 0;
    cfg.fft_size = 256;
    cfg.delay_backward_quarantine_enabled = quarantine;
    cfg.delay_backward_quarantine_s = 0.5f;
    p = af_test_create(&cfg);
    hops_out[0] = -1;
    if (!p) return;
    array_sig_init(&sig, 16000, 128, 1);
    n_hops = (int)((int64_t)ARR_SCENE_MS * 16000 / 1000 / 128);
    for (h = 0; h < n_hops; ++h) {
        AfFront f;
        array_sig_next(&sig, mics, ref);
        if (af_process_front(p, mics, ref, &f) != AF_OK) break;
        if (f.delay.changed && count < max) {
            hops_out[count] = h;
            backward_out[count] = count > 0 && f.delay.delay_samples < prev;
            count++;
        }
        if (f.delay.changed) prev = f.delay.delay_samples;
    }
    if (count < max) hops_out[count] = -1;
    af_test_destroy(p);
}

static void test_backward_quarantine_scene(void) {
    int on[8], off[8], on_back[8], off_back[8], i, n_on = 0, n_off = 0, max_delay = 0, never_early = 1, fwd_same = 1;
    scene_changes(1, on, on_back, 8);
    scene_changes(0, off, off_back, 8);
    while (n_on < 8 && on[n_on] >= 0) n_on++;
    while (n_off < 8 && off[n_off] >= 0) n_off++;
    CHECK(n_on == n_off && n_on >= 4, "quarantine scene: the same number of alignment generations with and without the quarantine");
    for (i = 0; i < n_on && i < n_off; ++i) {
        int d = on[i] - off[i];
        if (d < 0) never_early = 0;
        if (d > max_delay) max_delay = d;
        if (!off_back[i] && d != 0 && i == 0) fwd_same = 0;
    }
    CHECK(never_early && fwd_same, "quarantine scene: the quarantine never makes a move earlier and never holds the first acquisition");
    printf("  quarantine scene: %d generations, largest hold %d hops (%.2f s)\n", n_on, max_delay, (double)max_delay * 128.0 / 16000.0);
    CHECK(max_delay >= (int)(0.4 * 16000 / 128) && max_delay <= (int)(0.7 * 16000 / 128),
          "quarantine scene: a backward move is adopted about 0.5 s later with the quarantine on");
}

/* ---- far-only analysis hoist: an array with the hoist on against an identical one forced all-local ------------------- *
 * Both arrays see the same hops. Every output (lane views, fused frame, gain, delay state) must match bit for bit, and so
 * must the analysis state the hoist skips: lane 0's against the twin's lane 0, and every lane the array calls in sync
 * against the twin's own lane (the donor's state stands in for it while the lane is stale). A lane the array drops out of
 * sync must really have a different far history. */
typedef enum { HS_SPEECH, HS_MIXED, HS_TONAL, HS_GUARD } HoistScene;

typedef struct HoistRun {
    int hops_run, valid;
    int output_bad_hops, state_bad_hops, sync_unsound, stale_outside_sync, first_bad;
    int mixed_hops, sat_hops, guard_hops;
    unsigned sync_end, stale_end, twin_stale_end, sync_after_reset, sync_before_reset;
    int dropped_without_diff;     /* a lane left sync though its history never differed (mixed scene only) */
    int a_prior_ok;
} HoistRun;

typedef struct HoistSpec {
    HoistScene scene;
    int sr, fft;
    AecDelayMode mode;
    int enable_post, with_gain;
    int proxy;
    unsigned echo_mask;           /* HS_MIXED: lanes that carry the echo */
    int strong_delay, weak_delay, swing_hop, hops;
    int reset_at;                 /* >0: both arrays reset after this hop */
    int toggle_off_at, toggle_on_at;   /* >0: the hoisted array runs all-local in [off, on) */
} HoistSpec;

static int hoist_state_eq(const Aec* x, const Aec* y) {
    const int K = x->n_freqs;
    return x->rsa.n_counters == y->rsa.n_counters &&
           memcmp(x->rsa.counters, y->rsa.counters, (size_t)x->rsa.n_counters * sizeof(int64_t)) == 0 &&
           x->rsa.narrow_peak_band == y->rsa.narrow_peak_band &&
           x->rsa.narrow_peak_counter == y->rsa.narrow_peak_counter &&
           memcmp(x->a3_stat.noise.noise, y->a3_stat.noise.noise, (size_t)K * sizeof(float)) == 0 &&
           x->a3_stat.noise.block_counter == y->a3_stat.noise.block_counter;
}

/* The state both kinds of lane keep current every hop. */
static int hoist_carried_eq(const Aec* x, const Aec* y) {
    return x->non_zero_render_seen == y->non_zero_render_seen &&
           memcmp(&x->sat_ref, &y->sat_ref, sizeof x->sat_ref) == 0 &&
           x->render_activity.env_mean == y->render_activity.env_mean &&
           x->render_activity.env_var == y->render_activity.env_var &&
           x->render_activity.active_prev == y->render_activity.active_prev &&
           x->render_activity.is_stationary == y->render_activity.is_stationary &&
           x->warmup_far_active == y->warmup_far_active;
}

static int hoist_fronts_equal(const AfFront* a, const AfFront* b, int n_freqs, int hop) {
    int ch;
    if (a->delay.delay_samples != b->delay.delay_samples || a->delay.changed != b->delay.changed ||
        a->delay.solid != b->delay.solid || memcmp(&a->delay.confidence, &b->delay.confidence, sizeof(float)) != 0)
        return 0;
    if (memcmp(a->aligned_ref, b->aligned_ref, (size_t)hop * sizeof(float)) != 0) return 0;
    for (ch = 0; ch < AF_CHANNELS; ++ch) {
        if (memcmp(a->linear_spectra[ch], b->linear_spectra[ch], (size_t)n_freqs * sizeof(Complex)) != 0 ||
            memcmp(a->render_spectra[ch], b->render_spectra[ch], (size_t)n_freqs * sizeof(Complex)) != 0 ||
            memcmp(a->formed_hop[ch], b->formed_hop[ch], (size_t)hop * sizeof(float)) != 0 ||
            a->guard_active[ch] != b->guard_active[ch]) return 0;
    }
    return a->post_reset_serial == b->post_reset_serial;
}

static int hoist_frames_equal(const AfFrame* a, const AfFrame* b, int n_freqs, int full) {
    const size_t bins = (size_t)n_freqs * sizeof(Complex), fl = (size_t)n_freqs * sizeof(float);
    if (memcmp(a->e, b->e, bins) || memcmp(a->b, b->b, bins) || memcmp(a->y, b->y, bins) || memcmp(a->x, b->x, bins) ||
        memcmp(a->r2, b->r2, fl) || memcmp(a->cn, b->cn, fl)) return 0;
    if (full && memcmp(a->g_res_ctx, b->g_res_ctx, fl)) return 0;
    return a->guard_lanes == b->guard_lanes && a->post_reset_serial == b->post_reset_serial &&
           memcmp(&a->far_power, &b->far_power, sizeof(float)) == 0 &&
           memcmp(&a->max_dt, &b->max_dt, sizeof(float)) == 0 &&
           memcmp(&a->max_sat, &b->max_sat, sizeof(float)) == 0 &&
           a->all_converged == b->all_converged && a->usable_linear == b->usable_linear;
}

static void hoist_scene_hop(const HoistSpec* sp, ArraySig* sg, const GuardScene* gs, const float* history, uint32_t* rng,
                            int h, int hop, float* mics, float* ref) {
    int i, ch;
    enum { PAD = 4096 };
    if (sp->scene == HS_SPEECH) { (void)array_sig_next(sg, mics, ref); return; }
    if (sp->scene == HS_GUARD) {
        memcpy(mics, gs->mics + (size_t)h * hop * AF_CHANNELS, (size_t)hop * AF_CHANNELS * sizeof(float));
        memcpy(ref, gs->far + (size_t)h * hop, (size_t)hop * sizeof(float));
        return;
    }
    if (sp->scene == HS_MIXED) {
        const int base = h * hop + PAD;
        const float strong_gain = h >= sp->swing_hop ? 0.25f : 0.6f, weak_gain = h >= sp->swing_hop ? 0.6f : 0.25f;
        for (i = 0; i < hop; ++i) {
            const float echo = strong_gain * history[base + i - sp->strong_delay] +
                               weak_gain * history[base + i - sp->weak_delay];
            ref[i] = history[base + i];
            for (ch = 0; ch < AF_CHANNELS; ++ch) {
                float v = echo;
                if (!((sp->echo_mask >> ch) & 1u)) {
                    *rng ^= *rng << 13; *rng ^= *rng >> 17; *rng ^= *rng << 5;
                    v = 0.06f * (((float)(*rng >> 8) * (1.0f / 16777216.0f)) - 0.5f);
                }
                mics[i * AF_CHANNELS + ch] = v;
            }
        }
        return;
    }
    /* HS_TONAL: silence, two clipped tones (narrow peaks, saturation), then quiet tones. */
    for (i = 0; i < hop; ++i) {
        const int64_t t = (int64_t)h * hop + i;
        const float ph1 = 6.28318530718f * 1000.0f * (float)(t % sp->sr) / (float)sp->sr;
        const float ph2 = 6.28318530718f * 2437.5f * (float)(t % sp->sr) / (float)sp->sr;
        float amp = h < 30 ? 0.0f : (h < sp->hops * 6 / 10 ? 0.75f : 0.08f), x;
        *rng ^= *rng << 13; *rng ^= *rng >> 17; *rng ^= *rng << 5;
        x = amp * (sinf(ph1) + sinf(ph2)) +
            (h < 30 ? 0.0f : 0.004f * (((float)(*rng >> 8) * (1.0f / 16777216.0f)) - 0.5f));   /* exact silence below the peak floor first */
        if (x > 1.0f) x = 1.0f;
        if (x < -1.0f) x = -1.0f;
        ref[i] = x;
        for (ch = 0; ch < AF_CHANNELS; ++ch) {
            *rng ^= *rng << 13; *rng ^= *rng >> 17; *rng ^= *rng << 5;
            mics[i * AF_CHANNELS + ch] = (0.4f + 0.05f * (float)ch) * x * 0.5f +
                0.01f * (((float)(*rng >> 8) * (1.0f / 16777216.0f)) - 0.5f);
        }
    }
}

static void hoist_pair_run(const HoistSpec* sp, HoistRun* out) {
    AfConfig cfg = af_default_config(sp->sr);
    Af *A = NULL, *B = NULL;
    ArraySig sg;
    GuardScene gs;
    float *mics = NULL, *ref = NULL, *history = NULL;
    Complex* weights = NULL;
    uint32_t rng = 0x1234567u;
    int hop, n_freqs, h, ch, i, total_hops;
    int ever_diff[AF_CHANNELS] = {0, 0, 0, 0};
    int dropped_seen[AF_CHANNELS] = {0, 0, 0, 0};
    long warm_before = 0, soft_before = 0;
    enum { PAD = 4096 };

    memset(out, 0, sizeof(*out));
    memset(&gs, 0, sizeof gs);
    out->first_bad = -1;
    cfg.fft_size = sp->fft;
    cfg.delay_mode = sp->mode;
    cfg.enable_post = sp->enable_post;
    cfg.with_gain = sp->with_gain;
    cfg.capture_proxy_channel = sp->proxy;
    if (sp->mode == AEC_DELAY_FIXED) cfg.fixed_delay_samples = 40 * sp->sr / 1000;
    A = af_test_create(&cfg);
    B = af_test_create(&cfg);
    if (!A || !B) goto cleanup;
    hop = af_hop_size(A);
    n_freqs = af_n_freqs(A);
    total_hops = sp->hops;
    if (sp->scene == HS_SPEECH) total_hops = split_signal_n_hops(sp->sr, hop);
    if (sp->scene == HS_GUARD) {
        if (guard_scene_make(&gs, sp->sr, hop, 10) != 0) goto cleanup;
        total_hops = gs.nh;
    }
    array_sig_init(&sg, sp->sr, hop, 0);
    mics = (float*)calloc((size_t)hop * AF_CHANNELS, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    weights = (Complex*)calloc((size_t)AF_CHANNELS * n_freqs, sizeof(Complex));
    history = (float*)calloc((size_t)(sp->hops * hop + PAD), sizeof(float));
    if (!mics || !ref || !weights || !history) goto cleanup;
    for (i = 0; i < sp->hops * hop + PAD; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        history[i] = 0.1f * (((float)(rng >> 8) * (1.0f / 16777216.0f)) - 0.5f);
    }
    rng = 0x9e3779b9u;
    for (i = 0; i < AF_CHANNELS * n_freqs; ++i) weights[i].r = 0.25f;
    aec_testing_array_set_far_hoist(B, 0);

    for (h = 0; h < total_hops; ++h) {
        AfFront fa, fb;
        AfFrame ua, ub;
        unsigned sync, stale;
        int bad = 0;
        long warm_after, soft_after;
        AecLinearView lv;

        if (sp->toggle_off_at > 0 && h == sp->toggle_off_at) aec_testing_array_set_far_hoist(A, 0);
        if (sp->toggle_on_at > 0 && h == sp->toggle_on_at) aec_testing_array_set_far_hoist(A, 1);
        if (sp->reset_at > 0 && h == sp->reset_at) {
            out->sync_before_reset = aec_testing_array_far_sync_lanes(A);
            af_reset(A);
            af_reset(B);
            out->sync_after_reset = aec_testing_array_far_sync_lanes(A);
            for (ch = 0; ch < AF_CHANNELS; ++ch) ever_diff[ch] = 0;
        }
        hoist_scene_hop(sp, &sg, &gs, history, &rng, h, hop, mics, ref);
        if (af_process_front(A, mics, ref, &fa) != AF_OK || af_process_front(B, mics, ref, &fb) != AF_OK) goto cleanup;
        out->hops_run += 1;
        if (!hoist_fronts_equal(&fa, &fb, n_freqs, hop)) bad = 1;
        for (ch = 0; ch < AF_CHANNELS; ++ch) if (fa.guard_active[ch]) { out->guard_hops += 1; break; }
        if (sp->enable_post) {
            if (af_fuse(A, weights, NULL, &ua) != AF_OK || af_fuse(B, weights, NULL, &ub) != AF_OK) goto cleanup;
            if (!hoist_frames_equal(&ua, &ub, n_freqs, sp->with_gain)) bad = 1;
        }
        memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
        if (aec_get_linear(A, &lv) == AEC_OK && lv.saturation_level > 0.1f) out->sat_hops += 1;

        sync = aec_testing_array_far_sync_lanes(A);
        stale = aec_testing_array_far_stale_lanes(A);
        if (aec_testing_array_far_stale_lanes(B) != 0u) bad = 1;
        if (stale & ~sync) out->stale_outside_sync += 1;
        {
            const Aec* a0 = aec_testing_array_lane(A, 0);
            const Aec* b0 = aec_testing_array_lane(B, 0);
            if (!hoist_state_eq(a0, b0)) bad = 1;
            for (ch = 0; ch < AF_CHANNELS; ++ch) {
                const Aec* al = aec_testing_array_lane(A, ch);
                const Aec* bl = aec_testing_array_lane(B, ch);
                if (!hoist_carried_eq(al, bl)) bad = 1;
                if (ch == 0) continue;
                if (!aec_lane_far_history_equal(a0, al)) ever_diff[ch] = 1;
                if ((sync >> ch) & 1u) {
                    if (ever_diff[ch]) out->sync_unsound += 1;
                    if (!hoist_state_eq(a0, bl)) bad = 1;          /* the donor's state is the lane's */
                } else if (!dropped_seen[ch]) {
                    dropped_seen[ch] = 1;
                    if (!ever_diff[ch]) out->dropped_without_diff += 1;
                }
                if (!((stale >> ch) & 1u) && !hoist_state_eq(al, bl)) bad = 1;   /* a lane that kept its own state */
            }
        }
        warm_after = af_realign_warm_lane_count(A);
        soft_after = af_realign_soft_lane_count(A);
        if (fa.delay.changed && warm_after > warm_before && soft_after > soft_before) out->mixed_hops += 1;
        warm_before = warm_after;
        soft_before = soft_after;
        if (bad) {
            out->output_bad_hops += 1;
            if (out->first_bad < 0) out->first_bad = h;
        }
    }
    out->sync_end = aec_testing_array_far_sync_lanes(A);
    out->stale_end = aec_testing_array_far_stale_lanes(A);
    out->twin_stale_end = aec_testing_array_far_stale_lanes(B);
    out->valid = 1;

cleanup:
    guard_scene_free(&gs);
    free(history); free(weights); free(ref); free(mics);
    af_test_destroy(A);
    af_test_destroy(B);
}

static void hoist_check(const char* name, const HoistSpec* sp, HoistRun* r) {
    char label[320];
    hoist_pair_run(sp, r);
    snprintf(label, sizeof label, "far hoist %s: the run completes (%d hops)", name, r->hops_run);
    CHECK(r->valid && r->hops_run > 0, label);
    snprintf(label, sizeof label,
             "far hoist %s: outputs, delay state and the analysis state equal the all-local twin bit for bit (%d bad hops, first %d)",
             name, r->output_bad_hops, r->first_bad);
    CHECK(r->valid && r->output_bad_hops == 0, label);
    snprintf(label, sizeof label, "far hoist %s: a lane in sync never had a different far history (%d violations) and stale lanes are in sync (%d)",
             name, r->sync_unsound, r->stale_outside_sync);
    CHECK(r->valid && r->sync_unsound == 0 && r->stale_outside_sync == 0, label);
}

static void test_far_hoist_pair(int sr, int fft) {
    static const AecResidualMode modes[3] = {AEC_RESIDUAL_STATE, AEC_RESIDUAL_ESTIMATE, AEC_RESIDUAL_FULL};
    static const char* const mname[3] = {"STATE", "ESTIMATE", "FULL"};
    HoistSpec sp;
    HoistRun r;
    char name[96], label[320];
    int m, delay_mode, ch;

    for (m = 0; m < 3; ++m) {
        /* external alignment: the hoist stays engaged the whole stream; a speech-like scene, then tones + clipping + silence */
        memset(&sp, 0, sizeof sp);
        sp.sr = sr; sp.fft = fft; sp.mode = AEC_DELAY_EXTERNAL_ALIGNED;
        sp.enable_post = modes[m] != AEC_RESIDUAL_STATE; sp.with_gain = modes[m] == AEC_RESIDUAL_FULL;
        sp.scene = HS_SPEECH;
        snprintf(name, sizeof name, "%d/%d EXTERNAL %s speech", sr, fft, mname[m]);
        hoist_check(name, &sp, &r);
        snprintf(label, sizeof label, "far hoist %s: lanes 1-3 run on lane 0's analysis (sync %x, stale %x; twin stale %x)",
                 name, r.sync_end, r.stale_end, r.twin_stale_end);
        CHECK(r.valid && r.sync_end == 0xEu && r.stale_end == 0xEu && r.twin_stale_end == 0u, label);

        sp.scene = HS_TONAL; sp.hops = split_signal_n_hops(sr, fft / 2);
        snprintf(name, sizeof name, "%d/%d EXTERNAL %s tones+clip+silence", sr, fft, mname[m]);
        hoist_check(name, &sp, &r);
        snprintf(label, sizeof label, "far hoist %s: the scene really clips the far end (%d hops with saturation > 0.1)", name, r.sat_hops);
        CHECK(r.valid && r.sat_hops > 10, label);
    }

    /* the estimator's alignment moves: all lanes realign together, then mixed warm/soft */
    for (delay_mode = 0; delay_mode < 2; ++delay_mode) {
        const int strong = sr == 48000 ? 1536 : (fft == 512 ? 512 : 256), weak = 2 * strong;
        const int swing = sr == 48000 ? 150 : 150, hops = sr == 48000 ? 480 : 360;
        unsigned echo_mask = delay_mode == 0 ? 0x3u : 0xCu;
        int proxy = delay_mode == 0 ? 0 : 2;
        unsigned expect_sync = 0u;
        for (ch = 1; ch < AF_CHANNELS; ++ch)
            if (((echo_mask >> ch) & 1u) == (echo_mask & 1u)) expect_sync |= 1u << ch;
        memset(&sp, 0, sizeof sp);
        sp.sr = sr; sp.fft = fft; sp.mode = AEC_DELAY_MATCHED;
        sp.enable_post = 1; sp.with_gain = 1;
        sp.scene = HS_MIXED; sp.echo_mask = echo_mask; sp.proxy = proxy;
        sp.strong_delay = strong; sp.weak_delay = weak; sp.swing_hop = swing; sp.hops = hops;
        snprintf(name, sizeof name, "%d/%d MATCHED mixed realign, echo on lanes %x (proxy %d)", sr, fft, echo_mask, proxy);
        hoist_check(name, &sp, &r);
        snprintf(label, sizeof label, "far hoist %s: the scene has a mixed warm/soft hop (%d) and only the lanes that realigned differently from lane 0 left sync (sync %x, expected %x)",
                 name, r.mixed_hops, r.sync_end, expect_sync);
        CHECK(r.valid && r.mixed_hops >= 1 && r.sync_end == expect_sync, label);
        snprintf(label, sizeof label, "far hoist %s: no lane left sync while its far history still matched lane 0's (%d)", name, r.dropped_without_diff);
        CHECK(r.valid && r.dropped_without_diff == 0, label);
        snprintf(label, sizeof label, "far hoist %s: lanes that left sync took lane 0's state and run on their own (stale %x, sync %x)",
                 name, r.stale_end, r.sync_end);
        CHECK(r.valid && (r.stale_end & ~r.sync_end) == 0u, label);

        /* the same scene with the arrays reset after the swing, and with the hoist switched off and on again mid-stream */
        sp.reset_at = hops - 40;
        snprintf(name, sizeof name, "%d/%d MATCHED mixed realign then reset", sr, fft);
        hoist_check(name, &sp, &r);
        snprintf(label, sizeof label, "far hoist %s: lanes had left sync before the reset (%x) and every lane is back in sync right after it (%x)",
                 name, r.sync_before_reset, r.sync_after_reset);
        CHECK(r.valid && r.sync_before_reset == expect_sync && r.sync_after_reset == 0xEu, label);
        sp.reset_at = 0; sp.toggle_off_at = 40; sp.toggle_on_at = swing - 20;
        snprintf(name, sizeof name, "%d/%d MATCHED mixed realign with the hoist toggled", sr, fft);
        hoist_check(name, &sp, &r);
    }

    /* output guards firing (together and alone) while the alignment moves, with a reset in the middle */
    memset(&sp, 0, sizeof sp);
    sp.sr = sr; sp.fft = fft; sp.mode = AEC_DELAY_MATCHED; sp.enable_post = 1; sp.with_gain = 1; sp.scene = HS_GUARD;
    sp.hops = 1;
    snprintf(name, sizeof name, "%d/%d MATCHED FULL guard scene", sr, fft);
    hoist_check(name, &sp, &r);
    snprintf(label, sizeof label, "far hoist %s: the output guards fire (%d hops)", name, r.guard_hops);
    CHECK(r.valid && r.guard_hops > 0, label);

    memset(&sp, 0, sizeof sp);
    sp.sr = sr; sp.fft = fft; sp.mode = AEC_DELAY_MATCHED; sp.enable_post = 1; sp.scene = HS_SPEECH;
    snprintf(name, sizeof name, "%d/%d MATCHED ESTIMATE speech", sr, fft);
    hoist_check(name, &sp, &r);
    sp.mode = AEC_DELAY_FIXED; sp.enable_post = 1;
    snprintf(name, sizeof name, "%d/%d FIXED ESTIMATE speech", sr, fft);
    hoist_check(name, &sp, &r);
}

static void test_far_hoist(void) {
    test_far_hoist_pair(16000, 256);
    test_far_hoist_pair(16000, 512);
    test_far_hoist_pair(48000, 1024);
}

/* The standalone gain stage of aec_test_hooks.h (aec_testing_res_*), fed the array's AecResidualView, reproduces the array's own
 * G_res bit for bit on every hop; the Audio_ALG 4-channel replay relies on it to recompute G_res from a recorded frame. A stage
 * built at another level must give a different gain on some hop, so the comparison can fail. */
static void test_standalone_gain_stage_matches_array(int sr, int fft, AecLevel level, AecLevel other) {
    AecConfig cfg, std_cfg, other_cfg;
    ArraySig sig;
    float mics[SPLIT_SIGNAL_MAX_HOP * ARR_CH], ref[SPLIT_SIGNAL_MAX_HOP];
    void *mem = NULL, *std_mem = NULL, *other_mem = NULL;
    Complex* w = NULL;
    Aec* a = NULL;
    AecTestingRes *std_res = NULL, *other_res = NULL;
    size_t sz, std_sz, other_sz;
    long hops_equal = 0, hops_other_differs = 0, hops_total = 0;
    int h, hop, n, ok = 0;
    char label[200];

    aec_config_from_level(&cfg, level, sr);
    cfg.fft_size = fft;
    cfg.n_mics = AEC_ARRAY_MICS;
    cfg.residual = AEC_RESIDUAL_FULL;
    std_cfg = cfg;
    aec_config_from_level(&other_cfg, other, sr);
    other_cfg.fft_size = fft;
    sz = aec_get_mem_size(&cfg);
    std_sz = aec_testing_res_mem_size(&std_cfg);
    other_sz = aec_testing_res_mem_size(&other_cfg);
    if (sz && std_sz && other_sz && posix_memalign(&mem, 64u, sz) == 0 && posix_memalign(&std_mem, 64u, std_sz) == 0 &&
        posix_memalign(&other_mem, 64u, other_sz) == 0) {
        a = aec_init(mem, sz, &cfg);
        std_res = aec_testing_res_init(std_mem, std_sz, &std_cfg);
        other_res = aec_testing_res_init(other_mem, other_sz, &other_cfg);
        ok = a && std_res && other_res;
    }
    snprintf(label, sizeof label, "standalone gain stage %d/%d level %d: array and both stages build", sr, fft, (int)level);
    CHECK(ok, label);
    if (!ok) { free(mem); free(std_mem); free(other_mem); return; }
    hop = aec_hop_size(a);
    n = hop + 1;
    w = (Complex*)calloc((size_t)AEC_ARRAY_MICS * (size_t)n, sizeof(Complex));
    for (h = 0; h < AEC_ARRAY_MICS * n; ++h) w[h].r = 0.25f;
    array_sig_init(&sig, sr, hop, 0);
    for (h = 0; h < 400; ++h) {
        AecBeam beam;
        AecResidualView rv;
        const float *g, *go;
        beam.weights = w;
        beam.beamformed_error = NULL;
        memset(&rv, 0, sizeof rv);
        rv.struct_size = sizeof rv;
        array_sig_next(&sig, mics, ref);
        if (aec_process_linear(a, mics, ref) != AEC_OK || aec_process_residual(a, &beam) != AEC_OK ||
            aec_get_residual(a, &rv) != AEC_OK || !rv.res_gain)
            break;
        g = aec_testing_res_process(std_res, &rv);
        go = aec_testing_res_process(other_res, &rv);
        if (!g || !go) break;
        ++hops_total;
        hops_equal += memcmp(g, rv.res_gain, (size_t)n * sizeof(float)) == 0;
        hops_other_differs += memcmp(go, rv.res_gain, (size_t)n * sizeof(float)) != 0;
    }
    snprintf(label, sizeof label, "standalone gain stage %d/%d level %d: equals the array's G_res on every hop (%ld of %ld)",
             sr, fft, (int)level, hops_equal, hops_total);
    CHECK(hops_total == 400 && hops_equal == hops_total, label);
    snprintf(label, sizeof label, "standalone gain stage %d/%d: a stage at level %d differs from the array's on %ld hops, so the comparison can fail",
             sr, fft, (int)other, hops_other_differs);
    CHECK(hops_other_differs > 0, label);
    aec_destroy(a);
    free(w); free(mem); free(std_mem); free(other_mem);
}

int main(void) {
    test_projection_kernels();
    test_weights_gate();
    test_trusted_spectrum_finite_gate();
    test_trusted_spectrum_path();
    test_invalid_configs();
    run_grid(16000, 256);
    run_grid(16000, 512);
    run_grid(48000, 1024);
    test_pre_frame_wola_identity(16000, 256);
    test_pre_frame_wola_identity(16000, 512);
    test_pre_frame_wola_identity(48000, 1024);
    test_far_fft_sharing_reduces_four_to_one(16000, 256);
    test_far_fft_sharing_reduces_four_to_one(16000, 512);
    test_far_fft_sharing_reduces_four_to_one(48000, 1024);
    test_delay_modes_and_bank_sizing();
    test_known_delay_acquisition_and_coverage();
    test_known_delay_mislock_is_detectable();
    test_known_delay_memory_and_cost();
    test_shared_delay_change_admission();
    test_delay_change_candidate_ttl();
    test_admission_band_literals();
    test_backward_quarantine_scene();
    test_duty_cycle_machine();
    test_duty_watchdog_leak_is_a_rate();
    test_duty_low_erle_delay_change();
    test_far_spec_provenance_guard();
    test_mixed_realign_hop_far_spectrum(16000, 256, 256, 512, 150, 360);
    test_mixed_realign_hop_far_spectrum(16000, 512, 512, 1024, 150, 360);
    test_mixed_realign_hop_far_spectrum(48000, 1024, 1536, 3072, 150, 480);
    test_stage_timing();
    test_fuse_frame_contract();
    test_fuse_usable_linear_and_over_contributors();
    test_proxy_lane_inputs(16000, 256, AEC_DELAY_FIXED);
    test_proxy_lane_inputs(16000, 512, AEC_DELAY_FIXED);
    test_proxy_lane_inputs(48000, 1024, AEC_DELAY_FIXED);
    test_proxy_lane_inputs(16000, 256, AEC_DELAY_EXTERNAL_ALIGNED);
    test_proxy_lane_inputs(16000, 512, AEC_DELAY_EXTERNAL_ALIGNED);
    test_proxy_lane_inputs(48000, 1024, AEC_DELAY_EXTERNAL_ALIGNED);
    test_counter_boundaries();
    test_lane_contract_parity();
    test_delay_request_rules();
    test_front_aligned_ref();
    test_flow_view_and_grid();
    test_fuse_overflow_is_dsp_error();
    test_nonfinite_hop_invalidates_front();
    test_create_heap_path();
    test_reinit_after_reset();
    test_render_base_contract(16000, 256, AEC_DELAY_EXTERNAL_ALIGNED);
    test_render_base_contract(16000, 512, AEC_DELAY_EXTERNAL_ALIGNED);
    test_render_base_contract(48000, 1024, AEC_DELAY_EXTERNAL_ALIGNED);
    test_render_base_contract(16000, 256, AEC_DELAY_MATCHED);
    test_render_base_contract(16000, 512, AEC_DELAY_MATCHED);
    test_render_base_contract(48000, 1024, AEC_DELAY_MATCHED);
    test_unit_weight_matches_full_mode(16000, 256, 1);
    test_unit_weight_matches_full_mode(16000, 256, 0);
    test_unit_weight_matches_full_mode(16000, 512, 1);
    test_unit_weight_matches_full_mode(48000, 1024, 1);
    test_far_hoist();
    test_standalone_gain_stage_matches_array(16000, 256, AEC_LEVEL_BALANCED, AEC_LEVEL_AGGRESSIVE);
    test_standalone_gain_stage_matches_array(16000, 512, AEC_LEVEL_AGGRESSIVE, AEC_LEVEL_MILD);
    test_standalone_gain_stage_matches_array(48000, 1024, AEC_LEVEL_MILD, AEC_LEVEL_AGGRESSIVE);
    if (failures) {
        printf("%d test(s) failed (%d passed)\n", failures, passes);
        return 1;
    }
    printf("All array flow tests passed (%d checks)\n", passes);
    return 0;
}
