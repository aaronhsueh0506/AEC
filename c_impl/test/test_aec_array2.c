/* test_aec_array2.c -- the second scene set of the array instance (AecConfig.n_mics == AEC_ARRAY_MICS): a clipped and a silent
 * microphone, double talk, a clipping far end, capture_proxy_channel != 0, a caller-supplied beamformed_error, clock drift,
 * aec_set_level in flight, lanes without the shadow filter, an admission scene with the 32/36-sample band, the backward and
 * forward moves under the quarantine and the duty cycle, and a reset inside double talk. Run through the array instance; the
 * chained hashes form a line stream that AEC_TEST_BASELINE_WRITE / AEC_TEST_BASELINE capture and compare (no baseline is shipped).
 *
 * Like test_aec_array.c the cases run in the residual modes they can: a case without a fusion in STATE (the lanes only), a
 * case with a fusion in ESTIMATE (the gain from the stand-alone gain stage of baseline2_array.h) and in FULL (the same lines
 * with the instance's own gain, compared with it hop by hop). The cases, stimuli, hashing and chaining are baseline2_array.h.
 *
 * Also checked per hop, with no baseline: the view invariants (res_floor_protect is the max_dt decision, guard_active vs
 * guard_lanes, base_spec is error_spec exactly when no guard fired), the caller's beamformed_error flowing into the fused
 * error, and the instance's gain / gain application against the explicit forms.
 *
 * Run from c_impl/:  make test-aec-array-2   (G2_VERBOSE=<file> writes every hop's unchained group hashes) */
#include "aec.h"
#include "../src/aec_res_internal.h"
#include "res_gain_of.h"
#include "aec_test_tuning.h"   /* AEC_TESTING build: the admission candidate, the tuning overrides */
#include "utility/simd_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static long g_checks = 0;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_fail++; if (g_fail < 40) { printf("FAIL: " __VA_ARGS__); printf("\n"); } } } while (0)

/* ---- baseline ---------------------------------------------------------------------------------------------------------- */
#include "baseline_stream.h"

static void g2_sink(const char* line) {
    baseline_line(line);
}

/* ---- the array instance behind baseline2_array.h --------------------------------------------------------------------------- */
typedef struct NewH {
    Aec* a;
    void* mem;
    int mode;
    AecResidualView rv;
    AecLinearView lv;
    const Complex* beam_err;
    int hop, n;
} NewH;

struct G2ACase;
static void* g2a_create(const struct G2ACase* c, int mode);
static void  g2a_destroy(void* h);
struct G2AFront; struct G2AFrame;
static int   g2a_front(void* h, const float* mics, const float* ref, struct G2AFront* f);
static int   g2a_fuse(void* h, const Complex* weights, const Complex* beamformed_error, struct G2AFrame* fr);
static void  g2a_reset(void* h);
static void  g2a_set_level(void* h, int level);
static void  g2a_counters(void* h, uint64_t out[8]);

static void frame_hook(void* h, const struct G2ACase* c, int hop, const struct G2AFront* f, const struct G2AFrame* fr, const float* gain);
#define G2A_HOOK_FRAME(h, c, hop, f, fr, g) frame_hook((h), (c), (hop), (f), (fr), (g))

#include "baseline2_array.h"

static void* g2a_create(const G2ACase* c, int mode) {
    AecTestCfg cfg;
    NewH* h = (NewH*)calloc(1, sizeof *h);
    aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, c->sr);
    cfg.c.fft_size = c->fft;
    cfg.c.n_mics = AEC_ARRAY_MICS;
    cfg.c.residual = mode == 0 ? AEC_RESIDUAL_STATE : mode == 1 ? AEC_RESIDUAL_ESTIMATE : AEC_RESIDUAL_FULL;
    cfg.c.delay_mode = c->delay_mode == 0 ? AEC_DELAY_MATCHED : c->delay_mode == 1 ? AEC_DELAY_FIXED : AEC_DELAY_EXTERNAL_ALIGNED;
    cfg.c.fixed_delay_samples = c->delay_mode == 1 ? (int)((int64_t)c->stim.dly[0].d16 * c->sr / 16000) : -1;
    cfg.c.capture_proxy_channel = c->proxy;
    cfg.c.delay_backward_quarantine_enabled = c->quarantine;
    cfg.c.delay_backward_quarantine_s = c->quarantine_s;
    cfg.t.delay_est_period_s = c->duty_s;
    if (c->noshadow) cfg.t.enable_shadow = 0;
    size_t sz = aec_tcfg_mem_size(&cfg);
    h->mem = g2_pool(sz);
    h->a = h->mem ? aec_tcfg_init(h->mem, sz, &cfg) : NULL;
    h->mode = mode; h->hop = c->fft / 2; h->n = h->hop + 1;
    if (!h->a) { free(h->mem); free(h); return NULL; }
    return h;
}

static void g2a_destroy(void* p) {
    NewH* h = (NewH*)p;
    aec_destroy(h->a);
    free(h->mem); free(h);
}

static int g2a_front(void* p, const float* mics, const float* ref, G2AFront* o) {
    NewH* h = (NewH*)p;
    AecArrayView av;
    if (aec_process_linear(h->a, mics, ref) != AEC_OK) return -1;
    memset(&av, 0, sizeof av); av.struct_size = sizeof av;
    if (aec_get_array(h->a, &av) != AEC_OK) return -1;
    CHECK(av.n_channels == ARR_CH && av.n_freqs == h->n && av.hop_size == h->hop, "array view geometry");
    CHECK(av.post_reset_serial == aec_post_reset_serial(h->a), "view serial != aec_post_reset_serial");
    memset(o, 0, sizeof *o);
    o->n_freqs = av.n_freqs; o->hop_size = av.hop_size;
    for (int ch = 0; ch < ARR_CH; ++ch) {
        o->lane_error[ch] = av.lane_error_spec[ch]; o->lane_base[ch] = av.lane_base_spec[ch];
        o->lane_formed[ch] = av.lane_formed_hop[ch]; o->lane_guard[ch] = av.lane_guard_active[ch];
    }
    o->aligned_ref = av.aligned_ref;
    o->delay_samples = av.delay.delay_samples; o->delay_conf = av.delay.confidence; o->delay_solid = av.delay.solid;
    o->delay_changed = av.delay.changed; o->est_calls = av.delay.estimator_calls; o->est_updates = av.delay.estimator_updates;
    o->serial = av.post_reset_serial;
    if (h->mode == 0) {
        AecBeam b = { NULL, NULL };
        CHECK(aec_process_residual(h->a, &b) == AEC_ERR_MODE || aec_process_residual(h->a, NULL) == AEC_ERR_MODE,
              "residual on a STATE array must be AEC_ERR_MODE");
    }
    return 0;
}

static int g2a_fuse(void* p, const Complex* weights, const Complex* beamformed_error, G2AFrame* o) {
    NewH* h = (NewH*)p;
    AecBeam beam;
    beam.weights = weights; beam.beamformed_error = beamformed_error;
    h->beam_err = beamformed_error;
    {
        AecResidualView early; memset(&early, 0, sizeof early); early.struct_size = sizeof early;
        CHECK(aec_get_residual(h->a, &early) == AEC_ERR_SEQUENCE, "residual view before the residual stage");
    }
    if (aec_process_residual(h->a, &beam) != AEC_OK) return -1;
    memset(&h->rv, 0, sizeof h->rv); h->rv.struct_size = sizeof h->rv;
    memset(&h->lv, 0, sizeof h->lv); h->lv.struct_size = sizeof h->lv;
    if (aec_get_residual(h->a, &h->rv) != AEC_OK || aec_get_linear(h->a, &h->lv) != AEC_OK) return -1;
    memset(o, 0, sizeof *o);
    o->e = h->rv.error_spec; o->b = h->rv.base_spec; o->y = h->rv.near_spec; o->x = h->rv.far_spec;
    o->r2 = h->rv.r2; o->cn = h->rv.comfort_noise; o->far_hop = h->rv.far_hop;
    o->guard_lanes = h->rv.guard_lanes; o->serial = h->rv.post_reset_serial; o->far_power = h->lv.far_power;
    o->max_dt = h->rv.max_dt; o->max_sat = h->rv.max_saturation; o->all_converged = h->rv.all_converged;
    o->usable_linear = h->rv.usable_linear; o->clock_drift = h->rv.clock_drift; o->delay_changed = h->rv.delay_changed;
    o->delay_locked = h->rv.delay_locked;
    o->instance_gain = h->rv.res_gain;
    return 0;
}

static void g2a_reset(void* p) { aec_reset(((NewH*)p)->a); }
static void g2a_set_level(void* p, int level) {
    CHECK(aec_set_level(((NewH*)p)->a, (AecLevel)level) == 0, "aec_set_level(%d)", level);
}

static void g2a_counters(void* p, uint64_t k[8]) {
    const Aec* a = ((NewH*)p)->a;
    AecDebugStatus ds;
    aec_debug_status(a, &ds);
    k[0] = (uint64_t)aec_far_fft_real_compute_count(a);
    k[1] = (uint64_t)aec_array_realign_warm_lane_count(a);
    k[2] = (uint64_t)aec_array_realign_soft_lane_count(a);
    k[3] = (uint64_t)(uint32_t)aec_testing_array_pending_delay_candidate(a);
    k[4] = ds.duty_hops_total;
    k[5] = ds.duty_hops_run;
    k[6] = (uint64_t)aec_array_matched_filter_count(a);
    k[7] = (uint64_t)aec_array_linear_aec_count(a);
}

static long g_hook_hops;
static void frame_hook(void* p, const G2ACase* c, int hop, const G2AFront* f, const G2AFrame* fr, const float* gain) {
    NewH* h = (NewH*)p;
    const AecResidualView* rv = &h->rv;
    const int n = h->n;
    float gx[G2_NB_MAX];
    Complex t[G2_NB_MAX], q[G2_NB_MAX];
    (void)f;
    g_hook_hops++;
    CHECK(rv->res_floor_protect == (rv->max_dt > AEC_RES_DT_PROTECT_THRESHOLD), "%s hop %d: res_floor_protect != (max_dt > threshold)", c->name, hop);
    CHECK(rv->guard_active == (rv->guard_lanes != 0), "%s hop %d: guard_active vs guard_lanes", c->name, hop);
    CHECK((rv->guard_lanes == 0) == (rv->base_spec == rv->error_spec), "%s hop %d: base_spec is error_spec exactly when no guard fired", c->name, hop);
    CHECK(rv->formed_hop == NULL && rv->echo_spec == NULL, "array view: formed_hop / echo_spec are NULL");
    if (c->ext_beam)
        CHECK(g2_hc(G2_FNV_BASIS, rv->error_spec, n) == g2_hc(G2_FNV_BASIS, h->beam_err, n),
              "%s hop %d: the fused error is not the caller's beamformed_error", c->name, hop);
    else
        CHECK(h->beam_err == NULL, "no caller error expected");
    if (h->mode == 2) {
        CHECK(rv->res_gain && res_gain_of(h->a) == rv->res_gain, "FULL exposes the gain");
        CHECK(g2_fnv_f32v(G2_FNV_BASIS, rv->res_gain, n) == g2_fnv_f32v(G2_FNV_BASIS, gain, n),
              "%s hop %d: the instance's gain differs from a stand-alone AecRes fed the same frame", c->name, hop);
        array_extra_gain(gx, n, hop);
        CHECK(aec_apply_gain(h->a, gx, t) == AEC_OK, "aec_apply_gain");
        g2a_apply_gain_spec(q, fr->b, gain, gx, n);
        CHECK(g2_hc(G2_FNV_BASIS, t, n) == g2_hc(G2_FNV_BASIS, q, n), "%s hop %d: aec_apply_gain != explicit form", c->name, hop);
    } else {
        CHECK(rv->res_gain == NULL && res_gain_of(h->a) == NULL, "ESTIMATE exposes a gain");
        CHECK(aec_apply_gain(h->a, NULL, t) == AEC_ERR_MODE, "apply_gain below FULL");
    }
}

int main(void) {
    static G2ACase cases[G2A_MAX_CASES];
    int nc = g2a_cases(cases);
    g_have_baseline = baseline_load("aec_array2");
    for (int i = 0; i < nc; ++i) {
        const G2ACase* c = &cases[i];
        long start = g_idx;
        if (!g2a_want(c)) continue;
        if (!c->post) {
            if (g2a_run_case(c, 0) != 0) { g_fail++; break; }
        } else {
            if (g2a_run_case(c, 1) != 0) { g_fail++; break; }
            g_idx = start;                          /* the same lines again, with the instance's own gain */
            g_replay = 1;
            if (g2a_run_case(c, 2) != 0) { g_fail++; break; }
            g_replay = 0;
        }
    }
    g_fail += baseline_finish();
    CHECK(g_hook_hops > 0, "no view invariant was checked (G2_ONLY / the frame hook)");
    printf("view invariants checked on %ld hops\n", g_hook_hops);
    printf("%ld checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
