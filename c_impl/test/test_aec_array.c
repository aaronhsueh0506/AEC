/* test_aec_array.c -- the four-microphone array instance (AecConfig.n_mics == AEC_ARRAY_MICS) driven
 * hop by hop in three residual modes, plus the contract checks (status codes, sequence rules, non-finite input, DSP-error
 * reset, pool descriptor, static / heap construction, reset, pool contents, zero-weight lanes, n_mics validation, level
 * retargeting). The hashed fields form a line stream that AEC_TEST_BASELINE_WRITE / AEC_TEST_BASELINE capture and compare
 * (no baseline is shipped). The cases are run three ways, producing the same lines:
 *   STATE     the cases without a fusion: the lanes only;
 *   ESTIMATE  the cases with a fusion, the gain taken from a test-side AecRes and applied with the explicit-array helpers;
 *   FULL      the same lines with the gain, the gain application and the comfort noise taken from the instance, and the
 *             test-side AecRes / helpers kept as an independent cross-check of the instance's own.
 *
 * Run from c_impl/:  make test-aec-array */
#include "aec.h"
#include "aec_view_internal.h"
#include "aec_internal.h"
#include "aec_white_box.h"
#include "aec_res_internal.h"
#include "utility/simd_kernels.h"
#include "array_signal.h"
#include "res_gain_of.h"
#include "aec_pool_test_util.h"
#include "aec_test_tuning.h"   /* AEC_TESTING build: the admission candidate, the tuning overrides */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NB_MAX 1025

static int g_fail = 0;
static long g_checks = 0;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

/* ---- baseline ---------------------------------------------------------------------------------------------------------- */
#include "baseline_stream.h"

static void emit_line(const char* line) {
    baseline_line(line);
}

static void emit(const char* cid, const char* field, int hop, uint64_t h) {
    char hex[17], line[160];
    split_hash_hex(hex, h);
    snprintf(line, sizeof line, "%s %s %d %s", cid, field, hop, hex);
    emit_line(line);
}

/* ---- helpers --------------------------------------------------------------------------------------------------------- */
static void array_cfg(AecConfig* cfg, int sr, int fft, int mode, AecResidualMode residual) {
    aec_config_from_level(cfg, AEC_LEVEL_BALANCED, sr);
    cfg->fft_size = fft;
    cfg->n_mics = AEC_ARRAY_MICS;
    cfg->residual = residual;
    cfg->delay_mode = mode == ARR_MATCHED ? AEC_DELAY_MATCHED : mode == ARR_FIXED ? AEC_DELAY_FIXED : AEC_DELAY_EXTERNAL_ALIGNED;
    cfg->fixed_delay_samples = mode == ARR_FIXED ? ARR_ECHO_MS * sr / 1000 : -1;
}

static void case_cfg(AecTestCfg* tc, const ArrCase* c, AecResidualMode residual) {
    AecConfig* cfg = &tc->c;
    array_cfg(cfg, c->sr, c->fft, c->mode, residual);
    aec_tcfg_refresh(tc);
    if (c->scene == 1) {
        cfg->delay_backward_quarantine_enabled = 1;
        cfg->delay_backward_quarantine_s = 0.5f;
    }
    if (c->duty) {
        tc->t.delay_est_period_s = 1.0f;
        cfg->delay_backward_quarantine_enabled = 1;
        cfg->delay_backward_quarantine_s = 0.5f;
    }
}

/* ---- one case, one residual mode ------------------------------------------------------------------------------ */
typedef struct { uint32_t rng, handled; } CngState;

static int run_case(const ArrCase* c, int id, AecResidualMode residual) {
    AecTestCfg cfg;
    void *mem = NULL, *rmem = NULL;
    char cid[16];
    AecRes* res = NULL;
    CngState st = { AEC_CNG_SEED, 0u };
    ArraySig sig;
    float mics[SPLIT_SIGNAL_MAX_HOP * ARR_CH], ref[SPLIT_SIGNAL_MAX_HOP];
    Complex* W = NULL;
    const int full = (residual == AEC_RESIDUAL_FULL), estimate = (residual == AEC_RESIDUAL_ESTIMATE);
    long start_idx = g_idx;
    snprintf(cid, sizeof cid, "c%02d", id);
    case_cfg(&cfg, c, residual);
    Aec* a = aec_tcfg_new_aec(&cfg, &mem, 0xA5);
    if (!a) { printf("FAIL: aec_init %s\n", c->name); g_fail++; return -1; }
    const int hop = aec_hop_size(a);
    const int n_hops = array_case_hops(c);
    AecArrayView av;
    char line[96];
    int n = hop + 1;
    W = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
    if (c->post) {
        AecConfig rc_cfg;
        array_weights(c->wkind, n, W);
        aec_config_from_level(&rc_cfg, AEC_LEVEL_BALANCED, c->sr);
        rc_cfg.fft_size = c->fft;
        size_t rsz = aec_res_get_mem_size(&rc_cfg);
        rmem = aec_test_pool_alloc_fill(rsz, 0x5A);
        res = (rsz && rmem) ? aec_res_init(rmem, rsz, &rc_cfg) : NULL;
        CHECK(res != NULL, "test-side aec_res_init");
        if (!res) return -1;
    }
    array_sig_init(&sig, c->sr, hop, c->scene);

    snprintf(line, sizeof line, "# %s %s", cid, c->name);
    emit_line(line);
    emit(cid, "META_hop", 0, (uint64_t)hop);
    emit(cid, "META_hops", 0, (uint64_t)n_hops);
    for (int h = 0; h < n_hops; ++h) {
        uint64_t hh;
        const int light = c->scene && (h % ARR_SCENE_STRIDE) != 0;
        int changed_hop;
        if (c->reset_at && h == c->reset_at) {
            aec_reset(a);
            if (res) aec_res_reset(res);
        }
        array_sig_next(&sig, mics, ref);
        int rc = aec_process_linear(a, mics, ref);
        CHECK(rc == AEC_OK, "%s hop %d: aec_process_linear rc %d", c->name, h, rc);
        if (rc != AEC_OK) { free(W); free(rmem); free(mem); return -1; }
        memset(&av, 0, sizeof av); av.struct_size = sizeof av;
        CHECK(aec_get_array(a, &av) == AEC_OK, "aec_get_array");
        CHECK(av.n_freqs == n, "array view geometry");
        changed_hop = av.delay.changed;
        /* a scene's heavy lines are also hashed on every hop whose alignment changed */
        const int skip = light && !changed_hop;
        if (!skip) {
            hh = SPLIT_FNV_BASIS; for (int ch = 0; ch < ARR_CH; ++ch) hh = ah_cxv(hh, av.lane_error_spec[ch], n);
            emit(cid, "front_lin", h, hh);
            hh = SPLIT_FNV_BASIS; for (int ch = 0; ch < ARR_CH; ++ch) hh = ah_cxv(hh, av.lane_base_spec[ch], n);
            emit(cid, "front_ren", h, hh);
            hh = SPLIT_FNV_BASIS; for (int ch = 0; ch < ARR_CH; ++ch) hh = ah_f32v(hh, av.lane_formed_hop[ch], hop);
            emit(cid, "front_formed", h, hh);
        }
        hh = SPLIT_FNV_BASIS;
        hh = ah_f32v(hh, av.aligned_ref, hop);
        for (int ch = 0; ch < ARR_CH; ++ch) hh = ah_u32(hh, (uint32_t)av.lane_guard_active[ch]);
        hh = ah_u32(hh, (uint32_t)av.delay.delay_samples);
        hh = ah_f32(hh, av.delay.confidence);
        hh = ah_u32(hh, (uint32_t)av.delay.solid);
        hh = ah_u32(hh, (uint32_t)av.delay.changed);
        hh = ah_u32(hh, (uint32_t)(av.delay.estimator_calls & 0xFFFFFFFFu));
        hh = ah_u32(hh, (uint32_t)(av.delay.estimator_calls >> 32));
        hh = ah_u32(hh, (uint32_t)av.delay.estimator_updates);
        hh = ah_u32(hh, av.post_reset_serial);
        emit(cid, "front_meta", h, hh);
        CHECK(av.post_reset_serial == aec_post_reset_serial(a), "view serial != aec_post_reset_serial");
        if (!c->post) {
            AecBeam b = { W, NULL };
            CHECK(aec_process_residual(a, &b) == AEC_ERR_MODE, "residual on a STATE array must be AEC_ERR_MODE");
            continue;
        }

        {
            AecBeam beam = { W, NULL };
            AecResidualViewFull early; memset(&early, 0, sizeof early); early.struct_size = sizeof early;
            CHECK(aec_get_residual_full(a, &early) == AEC_ERR_SEQUENCE, "residual view before the residual stage");
            rc = aec_process_residual(a, &beam);
            CHECK(rc == AEC_OK, "%s hop %d: aec_process_residual rc %d", c->name, h, rc);
            if (rc != AEC_OK) { free(W); free(rmem); free(mem); return -1; }
        }
        AecResidualViewFull rv; AecLinearViewFull lv;
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
        CHECK(aec_get_residual_full(a, &rv) == AEC_OK, "aec_get_residual");
        CHECK(aec_get_linear_full(a, &lv) == AEC_OK, "aec_get_linear");
        if (!skip) {
            hh = SPLIT_FNV_BASIS;
            hh = ah_cxv(hh, rv.error_spec, n); hh = ah_cxv(hh, rv.base_spec, n);
            hh = ah_cxv(hh, rv.near_spec, n);  hh = ah_cxv(hh, rv.far_spec, n);
            emit(cid, "frame_spec", h, hh);
            emit(cid, "frame_r2", h, ah_f32v(SPLIT_FNV_BASIS, rv.r2, n));
            emit(cid, "frame_cn", h, ah_f32v(SPLIT_FNV_BASIS, rv.comfort_noise, n));
            emit(cid, "frame_far", h, ah_f32v(SPLIT_FNV_BASIS, rv.far_hop, hop));
            hh = SPLIT_FNV_BASIS;
            hh = ah_u32(hh, rv.guard_lanes);
            hh = ah_u32(hh, rv.post_reset_serial);
            hh = ah_f32(hh, lv.far_power);
            hh = ah_f32(hh, rv.max_dt);
            hh = ah_f32(hh, rv.max_saturation);
            hh = ah_u32(hh, (uint32_t)rv.all_converged);
            hh = ah_u32(hh, (uint32_t)rv.usable_linear);
            hh = ah_u32(hh, (uint32_t)rv.clock_drift);
            hh = ah_u32(hh, (uint32_t)av.delay.changed);
            hh = ah_u32(hh, (uint32_t)av.delay.solid);
            emit(cid, "frame_flags", h, hh);
        }
        /* view invariants */
        CHECK(rv.guard_active == (rv.guard_lanes != 0), "guard_active vs guard_lanes");
        CHECK((rv.guard_lanes == 0) == (rv.base_spec == rv.error_spec), "base_spec is error_spec exactly when no guard fired");
        CHECK(rv.res_gain == (full ? res_gain_of(a) : NULL), "gain pointer vs mode");
        CHECK(!full || rv.res_gain, "FULL exposes the gain");
        CHECK(estimate ? (rv.res_gain == NULL && res_gain_of(a) == NULL) : 1, "ESTIMATE exposes a gain");

        /* the test-side gain stage: the literal old flow (render_hop form) */
        {
            AecResInput in;
            memset(&in, 0, sizeof in);
            in.error_spec = rv.error_spec; in.near_spec = rv.near_spec; in.r2 = rv.r2; in.comfort_noise = rv.comfort_noise;
            in.render_hop = rv.far_hop; in.all_converged = rv.all_converged; in.max_dt = rv.max_dt;
            in.max_saturation = rv.max_saturation; in.clock_drift = rv.clock_drift;
            const float* g_ext = aec_res_process(res, &in);
            CHECK(g_ext != NULL, "test-side aec_res_process");
            if (!g_ext) { free(W); free(rmem); free(mem); return -1; }
            const float* g = full ? rv.res_gain : g_ext;
            if (full) CHECK(split_hash_f32(rv.res_gain, n) == split_hash_f32(g_ext, n),
                            "%s hop %d: the instance's gain differs from a standalone AecRes fed the same frame", c->name, h);
            if (!skip) emit(cid, "res_gain", h, split_hash_f32(g, n));

            /* the explicit-array reference of the apply lines */
            {
                float gx[NB_MAX]; Complex p1[NB_MAX], p2[NB_MAX], p3[NB_MAX], p4[NB_MAX];
                Complex q1[NB_MAX], q2[NB_MAX], q3[NB_MAX], q4[NB_MAX];
                if (rv.post_reset_serial != st.handled) { st.rng = AEC_CNG_SEED; st.handled = rv.post_reset_serial; }
                array_extra_gain(gx, n, h);
                aec_apply_gain_spec(q1, rv.base_spec, g_ext, gx, n);
                memcpy(q2, q1, (size_t)n * sizeof(Complex));
                aec_apply_cng_spec(&st.rng, q2, rv.comfort_noise, g_ext, gx, n);
                aec_apply_gain_spec(q3, rv.base_spec, g_ext, NULL, n);
                memcpy(q4, q3, (size_t)n * sizeof(Complex));
                aec_apply_cng_spec(&st.rng, q4, rv.comfort_noise, g_ext, NULL, n);
                if (full) {
                    CHECK(aec_apply_gain(a, gx, p1) == AEC_OK, "aec_apply_gain");
                    memcpy(p2, p1, (size_t)n * sizeof(Complex));
                    CHECK(aec_apply_cng(a, gx, p2) == AEC_OK, "aec_apply_cng");
                    CHECK(aec_apply_gain(a, NULL, p3) == AEC_OK, "aec_apply_gain NULL");
                    memcpy(p4, p3, (size_t)n * sizeof(Complex));
                    CHECK(aec_apply_cng(a, NULL, p4) == AEC_OK, "aec_apply_cng NULL");
                    CHECK(split_hash_complex(p1, n) == split_hash_complex(q1, n) && split_hash_complex(p2, n) == split_hash_complex(q2, n) &&
                          split_hash_complex(p3, n) == split_hash_complex(q3, n) && split_hash_complex(p4, n) == split_hash_complex(q4, n),
                          "%s hop %d: the instance's gain / comfort-noise helpers differ from the explicit-array forms", c->name, h);
                } else {
                    CHECK(aec_apply_gain(a, gx, p1) == (estimate ? AEC_ERR_MODE : AEC_ERR_MODE), "apply_gain below FULL");
                    memcpy(p1, q1, sizeof q1); memcpy(p2, q2, sizeof q2); memcpy(p3, q3, sizeof q3); memcpy(p4, q4, sizeof q4);
                }
                if (!skip) {
                    hh = SPLIT_FNV_BASIS; hh = ah_cxv(hh, p1, n); hh = ah_cxv(hh, p2, n);
                    emit(cid, "apply", h, hh);
                    hh = SPLIT_FNV_BASIS; hh = ah_cxv(hh, p3, n); hh = ah_cxv(hh, p4, n);
                    emit(cid, "apply_null", h, hh);
                }
            }
        }
    }
    emit(cid, "END_far_fft", 0, (uint64_t)aec_far_fft_real_compute_count(a));
    emit(cid, "END_realign_warm", 0, (uint64_t)aec_array_realign_warm_lane_count(a));
    emit(cid, "END_realign_soft", 0, (uint64_t)aec_array_realign_soft_lane_count(a));
    {
        AecDebugStatus ds; aec_debug_status(a, &ds);
        emit(cid, "END_pending", 0, (uint64_t)(uint32_t)aec_testing_array_pending_delay_candidate(a));
        emit(cid, "END_duty_total", 0, ds.duty_hops_total);
        emit(cid, "END_duty_run", 0, ds.duty_hops_run);
    }
    emit(cid, "END_matched", 0, (uint64_t)aec_array_matched_filter_count(a));
    emit(cid, "END_lanes", 0, (uint64_t)aec_array_linear_aec_count(a));
    aec_destroy(a);
    (void)start_idx;
    free(W); free(rmem); free(mem);
    return 0;
}

/* ---- contract checks that need no baseline ----------------------------------------------------------------------------- */

typedef struct { uint64_t front, frame, gain; } StreamHash;

/* A short stream (hops of the 16 kHz / 256 MATCHED unequal-weight case) hashed into one value per stage; `nan_at` >= 0 feeds a
 * non-finite sample on that hop (the call must be refused with the state untouched and the hop is then replayed clean). */
static int short_stream(Aec* a, int sr, int fft, int hops, int nan_at, int bad_weights_at, StreamHash* out) {
    ArraySig sig;
    const int hop = fft / 2, n = hop + 1;
    float mics[SPLIT_SIGNAL_MAX_HOP * ARR_CH], ref[SPLIT_SIGNAL_MAX_HOP], bad[SPLIT_SIGNAL_MAX_HOP * ARR_CH];
    Complex* W = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
    Complex* Z = (Complex*)calloc((size_t)ARR_CH * n, sizeof(Complex));
    uint64_t f = SPLIT_FNV_BASIS, r = SPLIT_FNV_BASIS, g = SPLIT_FNV_BASIS;
    array_weights(ARR_W_UNEQUAL, n, W);
    array_sig_init(&sig, sr, hop, 0);
    for (int h = 0; h < hops; ++h) {
        AecArrayView av; AecResidualViewFull rv; AecBeam beam = { W, NULL };
        array_sig_next(&sig, mics, ref);
        if (h == nan_at) {
            memcpy(bad, mics, sizeof bad);
            bad[3 * ARR_CH + 2] = NAN;
            CHECK(aec_process_linear(a, bad, ref) == AEC_ERR_ARG, "non-finite mic sample must be AEC_ERR_ARG");
            bad[3 * ARR_CH + 2] = 0.0f;
            { float rb[SPLIT_SIGNAL_MAX_HOP]; memcpy(rb, ref, sizeof rb); rb[5] = INFINITY;
              CHECK(aec_process_linear(a, mics, rb) == AEC_ERR_ARG, "non-finite far sample must be AEC_ERR_ARG"); }
            memset(&av, 0, sizeof av); av.struct_size = sizeof av;
            CHECK(aec_get_array(a, &av) == AEC_OK, "a refused non-finite hop leaves the previous hop's views readable");
            { AecResidualViewFull prv; memset(&prv, 0, sizeof prv); prv.struct_size = sizeof prv;
              CHECK(aec_get_residual_full(a, &prv) == AEC_OK, "a refused non-finite hop leaves the previous fusion readable"); }
        }
        CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "stream linear");
        memset(&av, 0, sizeof av); av.struct_size = sizeof av;
        CHECK(aec_get_array(a, &av) == AEC_OK, "stream array view");
        for (int ch = 0; ch < ARR_CH; ++ch) f = ah_cxv(f, av.lane_error_spec[ch], n);
        if (h == bad_weights_at) {
            AecBeam zero = { Z, NULL }; AecBeam nul = { NULL, NULL };
            Complex* E = (Complex*)calloc((size_t)n, sizeof(Complex));
            AecBeam nan_e = { W, E };
            E[3].r = NAN;
            CHECK(aec_process_residual(a, &nan_e) == AEC_ERR_ARG, "NaN beamformed_error must be AEC_ERR_ARG");
            E[3].r = 0.0f; E[n - 1].i = -INFINITY;
            CHECK(aec_process_residual(a, &nan_e) == AEC_ERR_ARG, "-Inf beamformed_error must be AEC_ERR_ARG");
            free(E);
            W[7].r = NAN;
            CHECK(aec_process_residual(a, &beam) == AEC_ERR_ARG, "NaN weight must be AEC_ERR_ARG");
            W[7].r = 0.25f - 0.0001f * 7.0f; W[7].i = 0.0f;       /* (lane 0 bin 7 of the unequal set is 0.4986 + 0.1i) */
            array_weights(ARR_W_UNEQUAL, n, W);
            CHECK(aec_process_residual(a, &zero) == AEC_ERR_ARG, "all-zero weights must be AEC_ERR_ARG");
            CHECK(aec_process_residual(a, &nul) == AEC_ERR_ARG, "NULL weights must be AEC_ERR_ARG");
            CHECK(aec_process_residual(a, NULL) == AEC_ERR_ARG, "NULL beam on an array must be AEC_ERR_ARG");
            memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
            CHECK(aec_get_residual_full(a, &rv) == AEC_ERR_SEQUENCE, "a refused fusion leaves the hop open");
        }
        CHECK(aec_process_residual(a, &beam) == AEC_OK, "stream residual");
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        CHECK(aec_get_residual_full(a, &rv) == AEC_OK, "stream residual view");
        r = ah_cxv(r, rv.error_spec, n); r = ah_cxv(r, rv.base_spec, n); r = ah_f32v(r, rv.r2, n);
        g = ah_f32v(g, rv.res_gain, n);
    }
    out->front = f; out->frame = r; out->gain = g;
    free(W); free(Z);
    return 0;
}

static int same_stream(const StreamHash* x, const StreamHash* y) {
    return x->front == y->front && x->frame == y->frame && x->gain == y->gain;
}

static void test_status_and_sequence(void) {
    AecConfig cfg;
    void* mem = NULL;
    float mics[SPLIT_SIGNAL_MAX_HOP * ARR_CH], ref[SPLIT_SIGNAL_MAX_HOP], out[SPLIT_SIGNAL_MAX_HOP];
    ArraySig sig;
    array_cfg(&cfg, 16000, 256, ARR_EXTERNAL, AEC_RESIDUAL_FULL);
    Aec* a = aec_test_new_aec(&cfg, &mem, 0xA5);
    CHECK(a != NULL, "status: init");
    if (!a) return;
    const int hop = aec_hop_size(a), n = hop + 1;
    Complex* W = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
    array_weights(ARR_W_EQUAL, n, W);
    array_sig_init(&sig, 16000, hop, 0);
    array_sig_next(&sig, mics, ref);
    AecBeam beam = { W, NULL };
    AecArrayView av; memset(&av, 0, sizeof av); av.struct_size = sizeof av;
    AecResidualViewFull rv; memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
    Complex spec[NB_MAX];

    CHECK(hop == 128 && aec_hop_size(a) == 128, "array hop size");
    CHECK(aec_process(a, mics, ref, out) == AEC_ERR_MODE, "aec_process on an array is AEC_ERR_MODE");
    CHECK(aec_process_linear_shared_far_from(a, mics, ref, a) == AEC_ERR_MODE, "shared-far entry on an array is AEC_ERR_MODE");
    CHECK(aec_process_linear(NULL, mics, ref) == AEC_ERR_ARG, "linear NULL instance");
    CHECK(aec_process_linear(a, NULL, ref) == AEC_ERR_ARG, "linear NULL mics");
    CHECK(aec_process_linear(a, mics, NULL) == AEC_ERR_ARG, "linear NULL ref");
    CHECK(aec_get_array(a, &av) == AEC_ERR_SEQUENCE, "array view before any hop");
    CHECK(aec_process_residual(a, &beam) == AEC_ERR_SEQUENCE, "residual before any linear");
    CHECK(aec_get_residual_full(a, &rv) == AEC_ERR_SEQUENCE, "residual view before any hop");
    CHECK(res_gain_of(a) == NULL, "gain before any hop");
    CHECK(aec_apply_gain(a, NULL, spec) == AEC_ERR_SEQUENCE, "apply_gain before any hop");
    CHECK(aec_apply_cng(a, NULL, spec) == AEC_ERR_SEQUENCE, "apply_cng before any hop");
    CHECK(aec_apply_external_realign(a, 10) == -1, "external realign on an array is refused");
    CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "linear");
    CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "a second linear stage replaces the first (no residual needed in between)");
    CHECK(aec_get_residual_full(a, &rv) == AEC_ERR_SEQUENCE, "residual view while the residual stage is pending");
    CHECK(aec_apply_gain(a, NULL, spec) == AEC_ERR_SEQUENCE, "apply_gain while the residual stage is pending");
    CHECK(aec_process_residual(a, &beam) == AEC_OK, "residual");
    CHECK(aec_process_residual(a, &beam) == AEC_ERR_SEQUENCE, "a second residual stage on the same hop is AEC_ERR_SEQUENCE");
    CHECK(aec_get_residual_full(a, &rv) == AEC_OK && rv.res_gain != NULL, "residual view after the residual stage");
    CHECK(aec_apply_gain(a, NULL, NULL) == AEC_ERR_ARG, "apply_gain NULL out");
    CHECK(aec_apply_cng(a, NULL, NULL) == AEC_ERR_ARG, "apply_cng NULL spec");
    {
        AecResidualViewFull v; memset(&v, 0, sizeof v); v.struct_size = sizeof v - 1;
        CHECK(aec_get_residual_full(a, &v) == AEC_ERR_ARG, "short struct_size (residual)");
        AecArrayView w; memset(&w, 0, sizeof w); w.struct_size = sizeof w - 1;
        CHECK(aec_get_array(a, &w) == AEC_ERR_ARG, "short struct_size (array)");
        AecLinearViewFull l; memset(&l, 0, sizeof l); l.struct_size = sizeof l - 1;
        CHECK(aec_get_linear_full(a, &l) == AEC_ERR_ARG, "short struct_size (linear)");
    }
    /* linear after a completed hop, then reset drops the pending fusion */
    CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "linear after a completed hop");
    aec_reset(a);
    CHECK(aec_process_residual(a, &beam) == AEC_ERR_SEQUENCE, "a reset drops an open hop");
    CHECK(aec_get_array(a, &av) == AEC_ERR_SEQUENCE, "a reset drops the lane views");
    CHECK(res_gain_of(a) == NULL, "a reset drops the gain");
    CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "linear after reset");
    CHECK(aec_process_residual(a, &beam) == AEC_OK, "residual after reset");
    CHECK(aec_post_reset_serial(a) == 4u, "four lane resets: the summed serial reads 4 after one aec_reset");

    /* aggregate linear view and array view */
    {
        AecLinearViewFull lv; memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
        CHECK(aec_get_linear_full(a, &lv) == AEC_OK, "aggregate linear view");
        CHECK(lv.hop_size == hop && lv.n_freqs == n && lv.delay_samples == 0 && lv.delay_state == AEC_LINEAR_DELAY_LOCKED &&
              lv.delay_confidence == 1.0f, "EXTERNAL_ALIGNED aggregate: zero offset, locked");
        memset(&av, 0, sizeof av); av.struct_size = sizeof av;
        CHECK(aec_get_array(a, &av) == AEC_OK && lv.aligned_far_hop == av.aligned_ref, "aggregate aligned far is the array's");
        CHECK(av.delay.solid == 1 && av.delay.changed == 0 && av.delay.delay_samples == 0, "EXTERNAL_ALIGNED delay state");
        CHECK(!memcmp(av.aligned_ref, ref, (size_t)hop * sizeof(float)), "EXTERNAL_ALIGNED: aligned far is the caller's far");
    }
    /* synthesis window: lane 0's */
    {
        const float* w = NULL; int len = 0;
        CHECK(aec_synthesis_window(a, &w, &len) == 0 && w != NULL && len == 256, "synthesis window of an array");
    }
    aec_destroy(a);
    aec_destroy(a);                                  /* repeatable */
    free(W); free(mem);

    /* STATE and ESTIMATE arrays */
    {
        static const AecResidualMode modes[2] = { AEC_RESIDUAL_STATE, AEC_RESIDUAL_ESTIMATE };
        for (int i = 0; i < 2; ++i) {
            array_cfg(&cfg, 16000, 256, ARR_EXTERNAL, modes[i]);
            a = aec_test_new_aec(&cfg, &mem, 0xA5);
            CHECK(a != NULL, "modes: init %d", i);
            if (!a) continue;
            W = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
            array_weights(ARR_W_EQUAL, n, W);
            beam.weights = W;
            CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "modes: linear");
            CHECK(aec_get_array(a, &av) == AEC_OK, "modes: array view");
            memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
            if (modes[i] == AEC_RESIDUAL_STATE) {
                CHECK(aec_process_residual(a, &beam) == AEC_ERR_MODE, "STATE: residual refused");
                CHECK(aec_get_residual_full(a, &rv) == AEC_ERR_MODE, "STATE: no fused frame");
                CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "STATE: the next hop needs no residual stage");
            } else {
                CHECK(aec_process_residual(a, &beam) == AEC_OK, "ESTIMATE: residual");
                CHECK(aec_get_residual_full(a, &rv) == AEC_OK && rv.r2 && rv.comfort_noise && !rv.res_gain, "ESTIMATE: r2, cn, no gain");
                CHECK(aec_process_residual(a, &beam) == AEC_OK, "ESTIMATE: a second fusion of the same hop is a stateless re-fusion");
            }
            CHECK(aec_apply_gain(a, NULL, spec) == AEC_ERR_MODE, "apply_gain below FULL (mode %d)", (int)modes[i]);
            CHECK(aec_apply_cng(a, NULL, spec) == AEC_ERR_MODE, "apply_cng below FULL (mode %d)", (int)modes[i]);
            CHECK(res_gain_of(a) == NULL, "gain below FULL");
            aec_destroy(a);
            free(W); free(mem);
        }
    }
    /* a single-channel instance: beam refused, aec_get_array answers with lane 0 */
    {
        AecConfig sc; Aec* s; void* smem = NULL; float m1[SPLIT_SIGNAL_MAX_HOP], r1[SPLIT_SIGNAL_MAX_HOP];
        aec_config_from_level(&sc, AEC_LEVEL_BALANCED, 16000);
        s = aec_test_new_aec(&sc, &smem, 0xA5);
        CHECK(s != NULL, "single-channel init");
        if (s) {
            split_signal_fill(16000, aec_hop_size(s), 0, r1, m1);
            memset(&av, 0, sizeof av); av.struct_size = sizeof av;
            CHECK(aec_process_linear(s, m1, r1) == AEC_OK, "single-channel linear");
            CHECK(aec_get_array(s, &av) == AEC_ERR_SEQUENCE, "single-channel array view before the residual stage");
            CHECK(aec_process_residual(s, NULL) == AEC_OK, "single-channel hop");
            CHECK(aec_get_array(s, &av) == AEC_OK && av.lane_error_spec[0] && av.lane_base_spec[0] &&
                  av.lane_formed_hop[0] && av.aligned_ref, "single-channel array view");
            CHECK(aec_array_linear_aec_count(s) == 1 && aec_testing_array_pending_delay_candidate(s) == -1 &&
                  aec_array_realign_warm_lane_count(s) == 0 && aec_array_realign_soft_lane_count(s) == 0,
                  "array diagnostics on a single-channel instance");
            aec_destroy(s);
        }
        free(smem);
    }
}

static void test_nonfinite_input_leaves_state_untouched(void) {
    StreamHash clean, dirty;
    AecConfig cfg;
    void *m1 = NULL, *m2 = NULL;
    array_cfg(&cfg, 16000, 256, ARR_MATCHED, AEC_RESIDUAL_FULL);
    Aec* a = aec_test_new_aec(&cfg, &m1, 0xA5);
    Aec* b = aec_test_new_aec(&cfg, &m2, 0xA5);
    CHECK(a && b, "nonfinite: init");
    if (!a || !b) return;
    short_stream(a, 16000, 256, 120, -1, -1, &clean);
    short_stream(b, 16000, 256, 120, 40, 70, &dirty);
    CHECK(same_stream(&clean, &dirty),
          "refused non-finite hops and refused weight sets (AEC_ERR_ARG) changed the stream: state was not untouched");
    CHECK(aec_post_reset_serial(a) == aec_post_reset_serial(b) && aec_post_reset_serial(a) == 0u, "no reset counted");
    aec_destroy(a); aec_destroy(b); free(m1); free(m2);
}

static void test_dsp_error_resets_the_instance(void) {
    AecConfig cfg;
    void *mem = NULL, *mem2 = NULL;
    ArraySig sig;
    float mics[SPLIT_SIGNAL_MAX_HOP * ARR_CH], ref[SPLIT_SIGNAL_MAX_HOP];
    array_cfg(&cfg, 16000, 256, ARR_EXTERNAL, AEC_RESIDUAL_FULL);
    Aec* a = aec_test_new_aec(&cfg, &mem, 0xA5);
    Aec* fresh = aec_test_new_aec(&cfg, &mem2, 0x5A);
    CHECK(a && fresh, "dsp: init");
    if (!a || !fresh) return;
    const int hop = aec_hop_size(a), n = hop + 1;
    Complex* W = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
    Complex* H = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
    array_weights(ARR_W_EQUAL, n, W);
    for (int i = 0; i < ARR_CH * n; ++i) { H[i].r = 1e30f; H[i].i = 1e30f; }          /* finite, and the projections overflow */
    array_sig_init(&sig, 16000, hop, 0);
    for (int h = 0; h < 60; ++h) {
        AecBeam beam = { W, NULL };
        array_sig_next(&sig, mics, ref);
        CHECK(aec_process_linear(a, mics, ref) == AEC_OK && aec_process_residual(a, &beam) == AEC_OK, "dsp: warm-up hop");
    }
    {
        AecBeam huge = { H, NULL };
        AecArrayView av; AecResidualViewFull rv; memset(&av, 0, sizeof av); av.struct_size = sizeof av;
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        array_sig_next(&sig, mics, ref);
        CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "dsp: linear");
        CHECK(aec_process_residual(a, &huge) == AEC_ERR_DSP, "overflowing weights are AEC_ERR_DSP");
        CHECK(aec_post_reset_serial(a) == 4u, "the DSP error reset the instance (four lane resets)");
        CHECK(aec_get_array(a, &av) == AEC_ERR_SEQUENCE && aec_get_residual_full(a, &rv) == AEC_ERR_SEQUENCE,
              "the DSP error dropped the views");
    }
    /* after the reset the instance behaves like a fresh one whose lanes were reset once: same stream as a reset-fresh instance */
    aec_reset(fresh);
    {
        ArraySig s2; array_sig_init(&s2, 16000, hop, 0);
        for (int h = 0; h < 40; ++h) {
            AecBeam beam = { W, NULL };
            AecResidualViewFull r1, r2; memset(&r1, 0, sizeof r1); r1.struct_size = sizeof r1; r2 = r1;
            array_sig_next(&s2, mics, ref);
            CHECK(aec_process_linear(a, mics, ref) == AEC_OK && aec_process_residual(a, &beam) == AEC_OK, "dsp: hop after reset (a)");
            CHECK(aec_process_linear(fresh, mics, ref) == AEC_OK && aec_process_residual(fresh, &beam) == AEC_OK, "dsp: hop (fresh)");
            CHECK(aec_get_residual_full(a, &r1) == AEC_OK && aec_get_residual_full(fresh, &r2) == AEC_OK, "dsp: views");
            CHECK(split_hash_complex(r1.error_spec, n) == split_hash_complex(r2.error_spec, n) &&
                  split_hash_f32(r1.res_gain, n) == split_hash_f32(r2.res_gain, n) &&
                  split_hash_f32(r1.r2, n) == split_hash_f32(r2.r2, n),
                  "dsp: hop %d after the DSP-error reset differs from a reset fresh instance", h);
        }
    }
    /* a non-finite beamformed spectrum is an argument error: refused up front, no reset, the hop stays open */
    {
        Complex* E = (Complex*)calloc((size_t)n, sizeof(Complex));
        AecBeam bad = { W, E };
        AecArrayView av; AecResidualViewFull rv;
        uint32_t serial;
        memset(&av, 0, sizeof av); av.struct_size = sizeof av;
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        array_sig_next(&sig, mics, ref);
        CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "dsp: linear");
        serial = aec_post_reset_serial(a);
        E[3].r = NAN;
        CHECK(aec_process_residual(a, &bad) == AEC_ERR_ARG, "a non-finite beamformed spectrum is AEC_ERR_ARG");
        CHECK(aec_post_reset_serial(a) == serial, "a refused beamformed spectrum resets nothing");
        CHECK(aec_get_array(a, &av) == AEC_OK && aec_get_residual_full(a, &rv) == AEC_ERR_SEQUENCE,
              "a refused beamformed spectrum leaves the hop open with the lane views readable");
        E[3].r = 0.0f;
        { AecBeam good = { W, E };
          CHECK(aec_process_residual(a, &good) == AEC_OK, "the open hop accepts a finite beamformed spectrum afterwards"); }
        free(E);
    }
    aec_destroy(a); aec_destroy(fresh);
    free(W); free(H); free(mem); free(mem2);
}

/* ---- use after destroy, NULL handles, in-place gain ------------------------------------------------------------------ */

/* test_handle_ready checks every entry point's refusal of a destroyed pool handle; this checks what only a flow test can read. */
static void check_destroyed_refuses(Aec* a, const char* what) {
    CHECK(res_gain_of(a) == NULL, "%s: residual gain", what);
    CHECK(aec_testing_array_pending_delay_candidate(a) == -1, "%s: pending delay candidate", what);
    CHECK(aec_hop_size(a) == 0, "%s: hop_size", what);
    aec_destroy(a);                                 /* repeatable */
}

static void test_use_after_destroy_and_null(void) {
    AecConfig cfg;
    float mics[SPLIT_SIGNAL_MAX_HOP * ARR_CH], ref[SPLIT_SIGNAL_MAX_HOP];
    ArraySig sig;
    /* heap array: aec_destroy frees the block (the pointer is dead afterwards), and a new aec_create builds a working instance */
    array_cfg(&cfg, 16000, 256, ARR_MATCHED, AEC_RESIDUAL_FULL);
    {
        Aec *heap = NULL;
        const int hop = 128, n = 129;
        Complex W[ARR_CH * NB_MAX];
        AecBeam beam = { W, NULL };
        CHECK((heap = aec_create(&cfg)) != NULL, "destroy: aec_create of an array");
        array_weights(ARR_W_EQUAL, n, W);
        array_sig_init(&sig, 16000, hop, 0);
        array_sig_next(&sig, mics, ref);
        CHECK(aec_process_linear(heap, mics, ref) == AEC_OK && aec_process_residual(heap, &beam) == AEC_OK, "destroy: one hop");
        aec_destroy(heap);
        CHECK((heap = aec_create(&cfg)) != NULL, "after a destroy, aec_create builds an instance again");
        array_sig_next(&sig, mics, ref);
        CHECK(aec_process_linear(heap, mics, ref) == AEC_OK, "re-created handle works");
        aec_destroy(heap);
    }
    /* static array: the pool outlives the destroy, and the handle stays refused */
    {
        void* mem = NULL;
        Aec* a = aec_test_new_aec(&cfg, &mem, 0xA5);
        CHECK(a != NULL, "destroy: static init");
        if (a) { aec_destroy(a); check_destroyed_refuses(a, "destroyed static array"); }
        free(mem);
    }
    /* heap and static single-channel instances */
    {
        AecConfig sc; Aec* heap = NULL; void* mem = NULL; Aec* s;
        aec_config_from_level(&sc, AEC_LEVEL_BALANCED, 16000);
        CHECK((heap = aec_create(&sc)) != NULL, "destroy: single-channel create");
        aec_destroy(heap);
        s = aec_test_new_aec(&sc, &mem, 0xA5);
        CHECK(s != NULL, "destroy: single-channel static init");
        if (s) { aec_destroy(s); check_destroyed_refuses(s, "destroyed static single-channel"); }
        free(mem);
    }
    /* NULL handles */
    {
        AecResidualViewFull rv; AecLinearViewFull lv; AecArrayView av; AecDebugStatus ds; AecStageTiming tm;
        AecBeam beam = { NULL, NULL };
        Complex spec[NB_MAX];
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
        memset(&av, 0, sizeof av); av.struct_size = sizeof av;
        memset(&ds, 0x5A, sizeof ds);
        CHECK(aec_hop_size(NULL) == 0, "NULL: hop_size");
        aec_reset(NULL); aec_destroy(NULL);
        CHECK(aec_process(NULL, mics, ref, mics) == AEC_ERR_ARG && aec_process_linear(NULL, mics, ref) == AEC_ERR_ARG &&
              aec_process_residual(NULL, &beam) == AEC_ERR_ARG, "NULL: process calls");
        CHECK(aec_get_residual_full(NULL, &rv) == AEC_ERR_ARG && aec_get_linear_full(NULL, &lv) == AEC_ERR_ARG &&
              aec_get_array(NULL, &av) == AEC_ERR_ARG, "NULL: views");
        CHECK(res_gain_of(NULL) == NULL && aec_apply_gain(NULL, NULL, spec) == AEC_ERR_ARG &&
              aec_apply_cng(NULL, NULL, spec) == AEC_ERR_ARG, "NULL: gain calls");
        CHECK(aec_set_level(NULL, AEC_LEVEL_MILD) == -1 && aec_apply_external_realign(NULL, 1) == -1, "NULL: set_level, realign");
        CHECK(aec_post_reset_serial(NULL) == 0u && aec_far_fft_real_compute_count(NULL) == 0 && aec_linear_is_cancelling(NULL) == 0,
              "NULL: counters");
        aec_debug_status(NULL, &ds);
        CHECK(ds.delay_samples == 0 && ds.duty_hops_run == 0ull, "NULL: debug_status zeroes the record");
        memset(&tm, 0x5A, sizeof tm);
        aec_get_last_timing(NULL, &tm);
        CHECK(tm.delay_us == 0 && tm.steering_us == 0, "NULL: last_timing zeroes the record");
        CHECK(aec_array_realign_warm_lane_count(NULL) == 0 && aec_testing_array_pending_delay_candidate(NULL) == -1, "NULL: array diagnostics");
    }
}

/* aec_apply_gain() in place on the caller's own beamformed_error (the array's base_spec when no output guard fired) is
 * bit-identical to the out-of-place result; afterwards the view's error_spec / base_spec, which alias that buffer, hold the
 * gained data. */
static void test_apply_gain_in_place(void) {
    AecConfig cfg;
    ArraySig sig;
    float mics[SPLIT_SIGNAL_MAX_HOP * ARR_CH], ref[SPLIT_SIGNAL_MAX_HOP];
    void* mem = NULL;
    int in_place_hops = 0;
    array_cfg(&cfg, 16000, 256, ARR_MATCHED, AEC_RESIDUAL_FULL);
    Aec* a = aec_test_new_aec(&cfg, &mem, 0xA5);
    CHECK(a != NULL, "in-place: init");
    if (!a) return;
    const int hop = aec_hop_size(a), n = hop + 1;
    Complex* W = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
    Complex* E = (Complex*)calloc((size_t)n, sizeof(Complex));
    Complex* want = (Complex*)calloc((size_t)n, sizeof(Complex));
    float* extra = (float*)malloc((size_t)n * sizeof(float));
    for (int k = 0; k < n; ++k) extra[k] = 0.15f + 0.8f * (float)((k * 37) % 101) / 100.0f;
    array_weights(ARR_W_UNEQUAL, n, W);
    array_sig_init(&sig, 16000, hop, 0);
    for (int h = 0; h < 200; ++h) {
        AecArrayView av; AecResidualViewFull rv; AecBeam beam = { W, E };
        const float* g = (h & 1) ? extra : NULL;
        array_sig_next(&sig, mics, ref);
        CHECK(aec_process_linear(a, mics, ref) == AEC_OK, "in-place: linear");
        memset(&av, 0, sizeof av); av.struct_size = sizeof av;
        CHECK(aec_get_array(a, &av) == AEC_OK, "in-place: array view");
        for (int k = 0; k < n; ++k) {                       /* the caller's beamformer: sum of weight * lane error */
            float re = 0.0f, im = 0.0f;
            for (int ch = 0; ch < ARR_CH; ++ch) {
                const Complex w = W[(size_t)ch * n + k], e = av.lane_error_spec[ch][k];
                re += w.r * e.r - w.i * e.i;
                im += w.r * e.i + w.i * e.r;
            }
            E[k].r = re; E[k].i = im;
        }
        CHECK(aec_process_residual(a, &beam) == AEC_OK, "in-place: residual");
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        CHECK(aec_get_residual_full(a, &rv) == AEC_OK, "in-place: residual view");
        CHECK(rv.error_spec == E, "the array's error_spec aliases the caller's beamformed_error");
        if (rv.base_spec != E) continue;                     /* the output guard fired: the base is an internal buffer */
        CHECK(aec_apply_gain(a, g, want) == AEC_OK, "in-place: out-of-place gain");
        CHECK(aec_apply_gain(a, g, E) == AEC_OK, "in-place: in-place gain");
        CHECK(memcmp(E, want, (size_t)n * sizeof(Complex)) == 0, "in-place gain differs from the out-of-place result (hop %d)", h);
        {
            AecResidualViewFull after; memset(&after, 0, sizeof after); after.struct_size = sizeof after;
            CHECK(aec_get_residual_full(a, &after) == AEC_OK && after.error_spec == E && after.base_spec == E &&
                  memcmp(after.base_spec, want, (size_t)n * sizeof(Complex)) == 0,
                  "after an in-place gain the view's base_spec holds the gained data");
        }
        in_place_hops++;
    }
    CHECK(in_place_hops > 150, "in-place: enough hops without a guard (%d)", in_place_hops);
    aec_destroy(a);
    free(W); free(E); free(want); free(extra); free(mem);
}

static void test_descriptor(void) {
    static const int sizes_cases[2] = { 0, 1 };
    for (int which = 0; which < 2; ++which) {
        AecConfig cfg;
        AecMemReq req, bad;
        void* mem;
        if (sizes_cases[which]) array_cfg(&cfg, 16000, 256, ARR_MATCHED, AEC_RESIDUAL_FULL);
        else aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
        CHECK(aec_get_mem_requirements(&cfg, &req) == 0, "descriptor query (%s)", which ? "array" : "single");
        CHECK(req.descriptor_version == AEC_DESCRIPTOR_VERSION && req.layout_version == AEC_LAYOUT_VERSION &&
              req.backend_id == AEC_BACKEND_NE10 && req.alignment == 16u && req.reserved == 0u &&
              req.bytes == aec_get_mem_size(&cfg) && req.build_flags_hash != 0u, "descriptor fields (%s)", which ? "array" : "single");
        mem = aec_test_pool_alloc_fill((size_t)req.bytes + 64, 0x11);
        CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, &req) != NULL, "init_ex with the current descriptor (%s)", which ? "array" : "single");
        CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, NULL) != NULL, "init_ex without a descriptor");
        CHECK(aec_init(mem, (size_t)req.bytes, &cfg) != NULL, "aec_init is init_ex(NULL)");
        { AecMemReq big = req; big.bytes += 4096;
          CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, &big) != NULL, "a descriptor from a larger pool is accepted"); }
        { void* roomy = aec_test_pool_alloc_fill((size_t)req.bytes + 4096, 0x5A);
          Aec* r = aec_init(roomy, (size_t)req.bytes + 4096, &cfg);
          CHECK(r != NULL, "a pool larger than aec_get_mem_size() is accepted (%s)", which ? "array" : "single");
          aec_destroy(r);
          free(roomy); }
        bad = req; bad.descriptor_version += 1;
        CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, "stale descriptor_version");
        bad = req; bad.layout_version += 1;
        CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, "stale layout_version");
        bad = req; bad.backend_id += 1;
        CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, "stale backend_id");
        bad = req; bad.build_flags_hash ^= 1u;
        CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, "stale build_flags_hash");
        bad = req; bad.alignment = 32u;
        CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, "stale alignment");
        bad = req; bad.reserved = 1u;
        CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, "non-zero reserved");
        bad = req; bad.bytes -= 16;
        CHECK(aec_init_ex(mem, (size_t)req.bytes, &cfg, &bad) == NULL, "descriptor bytes below the current pool");
        CHECK(aec_init_ex(mem, (size_t)req.bytes - 16, &cfg, &req) == NULL, "pool smaller than the descriptor bytes");
        CHECK(aec_init(mem, (size_t)req.bytes - 16, &cfg) == NULL, "aec_init with a short pool");
        CHECK(aec_init((uint8_t*)mem + 8, (size_t)req.bytes, &cfg) == NULL, "a pool not 16-byte aligned");
        CHECK(aec_init(NULL, (size_t)req.bytes, &cfg) == NULL && aec_init_ex(mem, (size_t)req.bytes, NULL, &req) == NULL, "NULL arguments");
        CHECK(aec_get_mem_requirements(NULL, &req) == -1 && aec_get_mem_requirements(&cfg, NULL) == -1, "descriptor NULL arguments");
        free(mem);
    }
    /* the descriptor distinguishes nothing but the pool size between the two kinds; the hash is the same carve text */
    {
        AecConfig s, a4; AecMemReq rs, ra;
        aec_config_from_level(&s, AEC_LEVEL_BALANCED, 16000);
        array_cfg(&a4, 16000, 256, ARR_MATCHED, AEC_RESIDUAL_FULL);
        CHECK(aec_get_mem_requirements(&s, &rs) == 0 && aec_get_mem_requirements(&a4, &ra) == 0 && ra.bytes > 3 * rs.bytes / 2 && ra.bytes < 5 * rs.bytes &&
              rs.build_flags_hash == ra.build_flags_hash, "array pool is four lanes plus the flow (%llu vs single %llu)",
              (unsigned long long)ra.bytes, (unsigned long long)rs.bytes);
    }
    /* invalid config: no descriptor */
    {
        AecConfig cfg; AecMemReq req;
        array_cfg(&cfg, 16000, 256, ARR_MATCHED, AEC_RESIDUAL_FULL);
        cfg.sample_rate = 44100;
        CHECK(aec_get_mem_requirements(&cfg, &req) == -1 && req.bytes == 0, "invalid config: no descriptor, out zeroed");
    }
}

static void test_static_vs_heap_and_poison(void) {
    AecConfig cfg;
    StreamHash ref_hash, h;
    array_cfg(&cfg, 16000, 256, ARR_MATCHED, AEC_RESIDUAL_FULL);
    {
        void* mem = NULL;
        Aec* a = aec_test_new_aec(&cfg, &mem, 0x00);
        CHECK(a != NULL, "poison: init 0x00");
        if (a) { short_stream(a, 16000, 256, 90, -1, -1, &ref_hash); aec_destroy(a); }
        free(mem);
    }
    static const int fills[3] = { 0xA5, 0xFF, 0x3C };
    for (int i = 0; i < 3; ++i) {
        void* mem = NULL;
        Aec* a = aec_test_new_aec(&cfg, &mem, fills[i]);
        CHECK(a != NULL, "poison: init 0x%02X", fills[i]);
        if (!a) continue;
        short_stream(a, 16000, 256, 90, -1, -1, &h);
        CHECK(same_stream(&ref_hash, &h), "the stream depends on the pool contents at init (fill 0x%02X)", fills[i]);
        aec_destroy(a);
        free(mem);
    }
    /* heap path (aec_create) */
    {
        Aec *heap = NULL;
        CHECK((heap = aec_create(&cfg)) != NULL, "aec_create of an array");
        short_stream(heap, 16000, 256, 90, -1, -1, &h);
        CHECK(same_stream(&ref_hash, &h), "heap construction differs from caller-pool construction");
        aec_destroy(heap);
    }
    {
        Aec *heap = NULL; AecConfig bad = cfg; bad.residual = AEC_RESIDUAL_OFF;
        CHECK((heap = aec_create(&bad)) == NULL, "aec_create refuses an invalid array config");
        CHECK((heap = aec_create(NULL)) == NULL, "aec_create NULL config");
    }
    /* misaligned pool offset: the pool at several 16-byte multiples gives the same stream */
    for (int off = 16; off <= 48; off += 16) {
        size_t sz = aec_get_mem_size(&cfg);
        uint8_t* raw = (uint8_t*)aec_test_pool_alloc_fill(sz + 64, 0xC3);
        Aec* a = aec_init(raw + off, sz, &cfg);
        CHECK(a != NULL, "placement: init at +%d", off);
        if (!a) { free(raw); continue; }
        short_stream(a, 16000, 256, 90, -1, -1, &h);
        CHECK(same_stream(&ref_hash, &h), "the stream depends on the pool placement (+%d)", off);
        aec_destroy(a);
        free(raw);
    }
}

static void test_reset_equals_fresh(void) {
    static const int grids[3][2] = { {16000, 256}, {16000, 512}, {48000, 1024} };
    for (int g = 0; g < 3; ++g) {
        for (int mode = 0; mode < 3; ++mode) {
            AecConfig cfg;
            StreamHash first, again, fresh;
            void *m1 = NULL, *m2 = NULL;
            array_cfg(&cfg, grids[g][0], grids[g][1], mode, AEC_RESIDUAL_FULL);
            Aec* a = aec_test_new_aec(&cfg, &m1, 0xA5);
            Aec* b = aec_test_new_aec(&cfg, &m2, 0x5A);
            CHECK(a && b, "reset: init");
            if (!a || !b) continue;
            short_stream(a, grids[g][0], grids[g][1], 70, -1, -1, &first);
            aec_reset(a);
            short_stream(a, grids[g][0], grids[g][1], 70, -1, -1, &again);
            short_stream(b, grids[g][0], grids[g][1], 70, -1, -1, &fresh);
            CHECK(same_stream(&first, &fresh), "reset: the first run differs from a fresh instance");
            CHECK(same_stream(&again, &fresh), "reset: %d/%d mode %d: a reset instance differs from a fresh one", grids[g][0], grids[g][1], mode);
            aec_destroy(a); aec_destroy(b); free(m1); free(m2);
        }
    }
}

static void test_cng_reseeds_on_serial_change(void) {
    /* The instance's generator restarts at the seed when the lanes' summed serial moved (aec_reset), before any draw. */
    AecConfig cfg;
    void* mem = NULL;
    ArraySig sig;
    float mics[SPLIT_SIGNAL_MAX_HOP * ARR_CH], ref[SPLIT_SIGNAL_MAX_HOP];
    array_cfg(&cfg, 16000, 256, ARR_EXTERNAL, AEC_RESIDUAL_FULL);
    Aec* a = aec_test_new_aec(&cfg, &mem, 0xA5);
    CHECK(a != NULL, "cng: init");
    if (!a) return;
    const int hop = aec_hop_size(a), n = hop + 1;
    Complex* W = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
    array_weights(ARR_W_EQUAL, n, W);
    array_sig_init(&sig, 16000, hop, 0);
    uint32_t rng = AEC_CNG_SEED;
    for (int h = 0; h < 80; ++h) {
        AecBeam beam = { W, NULL };
        AecResidualViewFull rv; Complex s1[NB_MAX], s2[NB_MAX];
        if (h == 40) { aec_reset(a); }
        array_sig_next(&sig, mics, ref);
        if (aec_process_linear(a, mics, ref) != AEC_OK || aec_process_residual(a, &beam) != AEC_OK) { CHECK(0, "cng: hop"); break; }
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        aec_get_residual_full(a, &rv);
        if (h == 40) rng = AEC_CNG_SEED;           /* the serial moved: the generator restarts */
        memset(s1, 0, sizeof s1); memset(s2, 0, sizeof s2);
        aec_apply_cng_spec(&rng, s1, rv.comfort_noise, rv.res_gain, NULL, n);
        CHECK(aec_apply_cng(a, NULL, s2) == AEC_OK, "cng apply");
        CHECK(split_hash_complex(s1, n) == split_hash_complex(s2, n), "cng: hop %d differs from the generator reseeded on the serial change", h);
    }
    aec_destroy(a);
    free(W); free(mem);
}

static void test_set_level(void) {
    AecConfig cfg, rc_cfg;
    void *mem = NULL, *rmem = NULL;
    ArraySig sig;
    float mics[SPLIT_SIGNAL_MAX_HOP * ARR_CH], ref[SPLIT_SIGNAL_MAX_HOP];
    array_cfg(&cfg, 16000, 256, ARR_EXTERNAL, AEC_RESIDUAL_FULL);
    Aec* a = aec_test_new_aec(&cfg, &mem, 0xA5);
    CHECK(a != NULL, "level: init");
    if (!a) return;
    const int hop = aec_hop_size(a), n = hop + 1;
    aec_config_from_level(&rc_cfg, AEC_LEVEL_BALANCED, 16000);
    size_t rsz = aec_res_get_mem_size(&rc_cfg);
    rmem = aec_test_pool_alloc_fill(rsz, 0x5A);
    AecRes* res = aec_res_init(rmem, rsz, &rc_cfg);
    Complex* W = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
    array_weights(ARR_W_EQUAL, n, W);
    array_sig_init(&sig, 16000, hop, 0);
    CHECK(aec_set_level(a, (AecLevel)9) == -1 && aec_res_set_level(res, (AecLevel)9) == -1, "set_level refusals");
    CHECK(a->cfg.level == AEC_LEVEL_BALANCED && a->tun.min_gain_floor_far_active_db == -28.0f, "a refused level leaves the stored level");
    for (int h = 0; h < 250; ++h) {
        AecBeam beam = { W, NULL };
        AecResidualViewFull rv; AecResInput in;
        if (h == 100) {
            CHECK(aec_set_level(a, AEC_LEVEL_AGGRESSIVE) == 0, "set_level aggressive");
            CHECK(a->cfg.level == AEC_LEVEL_AGGRESSIVE && a->tun.min_gain_floor_far_active_db == -38.0f &&
                  a->tun.min_gain_floor_dt_db == -30.0f, "set_level stores the level and its floors on an array");
            CHECK(aec_res_set_level(res, AEC_LEVEL_AGGRESSIVE) == 0, "res set_level");
        }
        if (h == 170) {
            CHECK(aec_set_level(a, AEC_LEVEL_MILD) == 0, "set_level mild with a ramp");
            CHECK(aec_res_set_level(res, AEC_LEVEL_MILD) == 0, "res set_level ramp");
        }
        array_sig_next(&sig, mics, ref);
        CHECK(aec_process_linear(a, mics, ref) == AEC_OK && aec_process_residual(a, &beam) == AEC_OK, "level: hop");
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        aec_get_residual_full(a, &rv);
        memset(&in, 0, sizeof in);
        in.error_spec = rv.error_spec; in.near_spec = rv.near_spec; in.r2 = rv.r2; in.comfort_noise = rv.comfort_noise;
        in.render_hop = rv.far_hop; in.all_converged = rv.all_converged; in.max_dt = rv.max_dt;
        in.max_saturation = rv.max_saturation; in.clock_drift = rv.clock_drift;
        const float* g = aec_res_process(res, &in);
        CHECK(g && split_hash_f32(g, n) == split_hash_f32(rv.res_gain, n), "level: hop %d gain differs from a standalone AecRes at the same level", h);
    }
    aec_destroy(a);
    free(W); free(mem); free(rmem);
    /* ESTIMATE / STATE arrays accept the call (no gain stage to move) */
    {
        AecConfig c2; void* m2 = NULL; Aec* b;
        array_cfg(&c2, 16000, 256, ARR_EXTERNAL, AEC_RESIDUAL_ESTIMATE);
        b = aec_test_new_aec(&c2, &m2, 0xA5);
        CHECK(b && aec_set_level(b, AEC_LEVEL_AGGRESSIVE) == 0 && aec_set_level(b, (AecLevel)9) == -1,
              "set_level on an ESTIMATE array");
        CHECK(b && b->cfg.level == AEC_LEVEL_AGGRESSIVE, "an ESTIMATE array stores the level too");
        if (b) aec_destroy(b);
        free(m2);
    }
}

static void test_config_validation(void) {
    AecConfig cfg, c;
    array_cfg(&cfg, 16000, 256, ARR_MATCHED, AEC_RESIDUAL_FULL);
    CHECK(aec_get_mem_size(&cfg) > 0, "the array default config is valid");
    c = cfg; c.n_mics = 2;  CHECK(aec_get_mem_size(&c) == 0, "n_mics 2 refused");
    c = cfg; c.n_mics = 3;  CHECK(aec_get_mem_size(&c) == 0, "n_mics 3 refused");
    c = cfg; c.n_mics = 5;  CHECK(aec_get_mem_size(&c) == 0, "n_mics 5 refused");
    c = cfg; c.n_mics = -1; CHECK(aec_get_mem_size(&c) == 0, "n_mics -1 refused");
    c = cfg; c.n_mics = 0;  CHECK(aec_get_mem_size(&c) == 0, "n_mics 0 (a config not built by aec_config_from_level) refused");
    { AecConfig s; aec_config_from_level(&s, AEC_LEVEL_BALANCED, 16000); AecConfig z = s;
      CHECK(aec_get_mem_size(&s) > 0, "the single-channel default config is valid");
      CHECK(s.n_mics == 1 && s.delay_backward_quarantine_s == 1.0f && s.delay_backward_quarantine_enabled == 0 &&
            s.capture_proxy_channel == 0, "the config defaults of the array fields");
      z = s; z.capture_proxy_channel = 1; CHECK(aec_get_mem_size(&z) == 0, "a single channel with a proxy channel is refused");
      z = s; z.delay_backward_quarantine_enabled = 1; CHECK(aec_get_mem_size(&z) == 0, "a single channel with the quarantine on is refused"); }
    c = cfg; c.residual = AEC_RESIDUAL_OFF; CHECK(aec_get_mem_size(&c) == 0, "array: residual OFF refused");
    c = cfg; c.residual = AEC_RESIDUAL_STATE;    CHECK(aec_get_mem_size(&c) > 0, "array: STATE accepted");
    c = cfg; c.residual = AEC_RESIDUAL_ESTIMATE; CHECK(aec_get_mem_size(&c) > 0, "array: ESTIMATE accepted");
    c = cfg; c.capture_proxy_channel = 3;  CHECK(aec_get_mem_size(&c) > 0, "proxy channel 3 accepted");
    c = cfg; c.capture_proxy_channel = 4;  CHECK(aec_get_mem_size(&c) == 0, "proxy channel 4 refused");
    c = cfg; c.capture_proxy_channel = -1; CHECK(aec_get_mem_size(&c) == 0, "proxy channel -1 refused");
    c = cfg; c.delay_backward_quarantine_enabled = 2; CHECK(aec_get_mem_size(&c) == 0, "a non-boolean quarantine enable refused");
    c = cfg; c.delay_backward_quarantine_s = -1.0f;   CHECK(aec_get_mem_size(&c) == 0, "a negative quarantine window refused");
    c = cfg; c.delay_backward_quarantine_s = 3601.0f; CHECK(aec_get_mem_size(&c) == 0, "a too long quarantine window refused");
    c = cfg; c.delay_backward_quarantine_s = (float)NAN; CHECK(aec_get_mem_size(&c) == 0, "a NaN quarantine window refused");
    c = cfg; c.delay_backward_quarantine_enabled = 1; c.delay_backward_quarantine_s = 0.25f;
    CHECK(aec_get_mem_size(&c) > 0, "a valid enabled quarantine accepted");
    c = cfg; c.delay_backward_quarantine_enabled = 0; c.delay_backward_quarantine_s = 5000.0f;
    CHECK(aec_get_mem_size(&c) == 0, "the window is checked while the enable is 0");
    {   /* internal tuning values the array's own rules bound (through the hooks) */
        AecTestCfg t; t.c = cfg; aec_tcfg_refresh(&t);
        t.t.delay_est_period_s = -1.0f;   CHECK(aec_tcfg_mem_size(&t) == 0, "negative duty period refused");
        t.t.delay_est_period_s = 3601.0f; CHECK(aec_tcfg_mem_size(&t) == 0, "too long duty period refused");
        t.t.delay_est_period_s = 1.0f;    CHECK(aec_tcfg_mem_size(&t) > 0, "duty period 1 s accepted");
    }
    /* grids */
    c = cfg; c.sample_rate = 8000;  c.fft_size = 256;  CHECK(aec_get_mem_size(&c) == 0, "array: 8 kHz refused");
    c = cfg; c.fft_size = 128;  CHECK(aec_get_mem_size(&c) == 0, "array: 16k/128 refused");
    c = cfg; c.fft_size = 512;  CHECK(aec_get_mem_size(&c) > 0, "array: 16k/512 accepted");
    c = cfg; c.fft_size = 1024; CHECK(aec_get_mem_size(&c) == 0, "array: 16k/1024 refused");
    c = cfg; c.sample_rate = 48000; c.fft_size = 1024; CHECK(aec_get_mem_size(&c) > 0, "array: 48k/1024 accepted");
    c = cfg; c.sample_rate = 48000; c.fft_size = 512;  CHECK(aec_get_mem_size(&c) == 0, "array: 48k/512 refused");
    c = cfg; c.fft_size = 0;    CHECK(aec_get_mem_size(&c) == 0, "array: fft_size 0 refused");
    /* delay request rules (shared with the lanes' contract) */
    c = cfg; c.delay_num_filters = 0;  CHECK(aec_get_mem_size(&c) == 0, "bank size 0 refused");
    c = cfg; c.delay_num_filters = DA_NUM_FILTERS + 1; CHECK(aec_get_mem_size(&c) == 0, "bank size above the maximum refused");
    c = cfg; c.delay_num_filters = 1;  CHECK(aec_get_mem_size(&c) > 0, "bank size 1 accepted");
    c = cfg; c.delay_mode = AEC_DELAY_FIXED; c.fixed_delay_samples = 640;
    CHECK(aec_get_mem_size(&c) > 0, "FIXED with a delay accepted");
    c.fixed_delay_samples = -1; CHECK(aec_get_mem_size(&c) == 0, "FIXED without a delay refused");
    c = cfg; c.delay_mode = AEC_DELAY_FIXED; c.fixed_delay_samples = 120 * 16000 + 1; CHECK(aec_get_mem_size(&c) == 0, "FIXED above two minutes refused");
    c = cfg; c.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED; CHECK(aec_get_mem_size(&c) > 0, "EXTERNAL_ALIGNED accepted");
    c.delay_num_filters = 3; CHECK(aec_get_mem_size(&c) == 0, "EXTERNAL_ALIGNED with a bank size refused");
    c = cfg; c.delay_mode = AEC_DELAY_MATCHED; c.fixed_delay_samples = 10; CHECK(aec_get_mem_size(&c) == 0, "MATCHED with a fixed delay refused");
    /* the memory breakdown */
    {
        AecMemBreakdown b; AecMemBreakdown sb; AecConfig s;
        CHECK(aec_get_mem_breakdown(&cfg, &b) == AEC_OK && b.total_bytes == aec_get_mem_size(&cfg), "array breakdown total");
        CHECK(b.estimator_bytes > 0 && b.ring_bytes > 0 && b.lane_bytes > 0 && b.fusion_bytes > 0, "array breakdown parts");
        CHECK(b.estimator_bytes + b.ring_bytes + b.lane_bytes + b.fusion_bytes + ((sizeof(Aec) + 15u) & ~(size_t)15u) <= b.total_bytes,
              "array breakdown parts fit in the total");
        c = cfg; c.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
        CHECK(aec_get_mem_breakdown(&c, &b) == AEC_OK && b.estimator_bytes == 0 && b.ring_bytes == 0, "EXTERNAL_ALIGNED: no estimator, no ring");
        c = cfg; c.delay_mode = AEC_DELAY_FIXED; c.fixed_delay_samples = 640;
        CHECK(aec_get_mem_breakdown(&c, &b) == AEC_OK && b.estimator_bytes == 0 && b.ring_bytes > 0, "FIXED: a ring, no estimator");
        aec_config_from_level(&s, AEC_LEVEL_BALANCED, 16000);
        CHECK(aec_get_mem_breakdown(&s, &sb) == AEC_OK && sb.lane_bytes == 0 && sb.fusion_bytes == 0 && sb.total_bytes == aec_get_mem_size(&s),
              "single-channel breakdown meaning unchanged");
        c = cfg; c.n_mics = 2;
        CHECK(aec_get_mem_breakdown(&c, &b) == AEC_ERR_ARG && b.total_bytes == 0, "invalid config: breakdown zeroed");
    }
    /* sizes: ESTIMATE / STATE arrays are smaller than FULL, FIXED / EXTERNAL smaller than MATCHED */
    {
        AecConfig f = cfg, e = cfg, s = cfg, x = cfg;
        e.residual = AEC_RESIDUAL_ESTIMATE; s.residual = AEC_RESIDUAL_STATE; x.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
        CHECK(aec_get_mem_size(&s) < aec_get_mem_size(&e) && aec_get_mem_size(&e) < aec_get_mem_size(&f), "pool size by residual mode");
        CHECK(aec_get_mem_size(&x) < aec_get_mem_size(&f), "pool size by delay mode");
    }
}

int main(void) {
    ArrCase cases[ARR_MAX_CASES];
    int nc = array_cases(cases);
    g_have_baseline = baseline_load("aec_array");
    test_config_validation();
    test_status_and_sequence();
    test_nonfinite_input_leaves_state_untouched();
    test_dsp_error_resets_the_instance();
    test_use_after_destroy_and_null();
    test_apply_gain_in_place();
    test_descriptor();
    test_static_vs_heap_and_poison();
    test_reset_equals_fresh();
    test_cng_reseeds_on_serial_change();
    test_set_level();
    for (int i = 0; i < nc; ++i) {
        const ArrCase* c = &cases[i];
        long start = g_idx;
        if (!c->post) {
            run_case(c, i, AEC_RESIDUAL_STATE);
        } else {
            run_case(c, i, AEC_RESIDUAL_ESTIMATE);
            g_idx = start;                      /* the same lines again with the instance's own gain stage */
            g_replay = 1;
            run_case(c, i, AEC_RESIDUAL_FULL);
            g_replay = 0;
        }
    }
    g_fail += baseline_finish();
    printf("%d checks, %d failed\n", (int)g_checks, g_fail);
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
