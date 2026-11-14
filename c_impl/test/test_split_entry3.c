/* test_split_entry3.c -- the third scene set of the single-channel API: bad-relock judgments that the stimulus produces by itself
 * (no field of the instance is written). Two echo paths whose gains swap during a bursty far end move the delay estimate; the
 * windowed ERLE after the shift is judged against the value before it. The scenes make that comparison come out in every way:
 * ERLE improving after the shift (no revert), clearly worse (revert), a few tenths of a dB either side of the 1 dB margin, and
 * worse with no estimate yet (no revert), on the three product grids; the soft path is reached with kept judgments only. Every
 * judgment is checked against the expectations of the case table below and printed; the chained hashes form a line
 * stream that AEC_TEST_BASELINE_WRITE / AEC_TEST_BASELINE capture and compare (no baseline is shipped). The runner, views and
 * hashing are baseline2_mono.h. No field of the instance is written.
 *
 * Scene: a far end of eight bursts (on_ms / off_ms from t0_ms) and two echo paths at d1 and d2 (16 kHz samples), gains {a0, a1} for the
 * first flip_ms and {b0, b1} for the next, swapping thereafter. d2 = 1600 lies beyond the tap span of the filter, d2 = 768 inside it.
 * A near-end burst (1.2 s, 0.6 s ramp) at near_t_ms keeps the near-end-recent latch up (soft path). Cases:
 *   rj_up      ERLE after the shift above the stored one (kept)
 *   rj_down    ERLE 8 to 10 dB below a converged stored one (natural revert)
 *   rj_keep    0.5 dB above the revert threshold (kept)
 *   rj_rev     0.6 dB below the revert threshold (natural revert)
 *   rj_noest   ERLE below the threshold but error power not below near power (kept)
 *   rj_soft    soft path (kept)
 *
 * Run from c_impl/:  make test-split-entry-3   (G2_VERBOSE=<file> writes every hop's unchained group hashes, G2_ONLY=<substring>)
 * The expectation checks run on the NEON and the scalar build only. */
#include "aec.h"
#include "aec_res_internal.h"
#include "aec_test_tuning.h"
#include "utility/simd_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define G2_JOUT stdout

static int g_fail = 0;
static long g_checks = 0;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

/* ---- baseline ---------------------------------------------------------------------------------------------------------- */
#include "baseline_stream.h"

static void g2_sink(const char* line) {
    baseline_line(line);
}

/* The scene expectations below were scanned on the NEON and the scalar build only; elsewhere those checks are skipped. */
#if defined(SIMD_KERNELS_FORCE_SCALAR) || (defined(SK_HAVE_NEON) && SK_HAVE_NEON)
#define HOST_VALIDATED 1
#else
#define HOST_VALIDATED 0
#endif

/* ---- the new entry points behind baseline2_mono.h ------------------------------------------------------------------------- */
static void g2_mode_cfg(AecConfig* c, int ctx) { (void)ctx; c->residual = AEC_RESIDUAL_FULL; }

struct G2View;
static int g2_step_ctx(Aec* a, const float* mic, const float* ref, struct G2View* v);
static int g2_step_out(Aec* a, const float* mic, const float* ref, float* out) {
    return aec_process(a, mic, ref, out) == AEC_OK ? 0 : -1;
}

#include "baseline2_mono.h"

typedef struct {
    const char* name;
    int sr, fft, dur_ms;
    int on_ms, off_ms, t0_ms;       /* eight far-end bursts */
    int flip_ms, d1_16, d2_16;      /* path gains swap every flip_ms; path delays at 16 kHz */
    float g[4];                     /* a0, a1, b0, b1 */
    int near_t_ms; float near_amp;  /* near-end burst of 1.2 s with a 0.6 s ramp (0 = none) */
    int n_exp; G2JudgeExp exp[1];
} G3Scene;

static const G3Scene g3_scenes[] = {
    { "rj_up", 16000, 256, 6600, 1000, 400, 500, 3600, 512, 1600, { 0.5f, 0.5f, 0.05f, 0.55f }, 0, 0.0f, 1, { { 0, 1, 0.30f, 100.0f } } },
    { "rj_down", 16000, 256, 6900, 1000, 300, 1000, 3600, 512, 768, { 0.5f, 0.5f, 0.2f, 0.55f }, 0, 0.0f, 1, { { 1, 1, -100.0f, -2.00f } } },
    { "rj_keep", 16000, 256, 6900, 2200, 700, 1000, 4400, 512, 1600, { 0.5f, 0.5f, 0.15f, 0.55f }, 0, 0.0f, 1, { { 0, 1, -0.85f, -0.15f } } },
    { "rj_rev", 16000, 256, 6800, 2200, 700, 500, 3600, 512, 1600, { 0.5f, 0.3f, 0.05f, 0.55f }, 0, 0.0f, 1, { { 1, 1, -1.85f, -1.15f } } },
    { "rj_noest", 16000, 256, 6600, 1800, 300, 1000, 4400, 512, 1600, { 0.5f, 0.5f, 0.15f, 0.55f }, 0, 0.0f, 1, { { 0, 0, -100.0f, -1.20f } } },
    { "rj_up", 16000, 512, 6700, 1000, 300, 100, 4400, 512, 1600, { 0.5f, 0.5f, 0.1f, 0.55f }, 0, 0.0f, 1, { { 0, 1, 0.30f, 100.0f } } },
    { "rj_down", 16000, 512, 7200, 2200, 700, 500, 3600, 512, 768, { 0.5f, 0.5f, 0.2f, 0.55f }, 0, 0.0f, 1, { { 1, 1, -100.0f, -2.00f } } },
    { "rj_keep", 16000, 512, 8400, 1800, 400, 500, 4400, 512, 1600, { 0.5f, 0.5f, 0.1f, 0.55f }, 0, 0.0f, 1, { { 0, 1, -0.85f, -0.15f } } },
    { "rj_rev", 16000, 512, 7700, 2200, 500, 100, 3600, 512, 1600, { 0.5f, 0.3f, 0.2f, 0.55f }, 0, 0.0f, 1, { { 1, 1, -1.85f, -1.15f } } },
    { "rj_noest", 16000, 512, 7400, 1800, 300, 1000, 4400, 512, 1600, { 0.5f, 0.5f, 0.15f, 0.55f }, 0, 0.0f, 1, { { 0, 0, -100.0f, -1.20f } } },
    { "rj_up", 48000, 1024, 6900, 1400, 400, 1000, 3600, 512, 1600, { 0.5f, 0.5f, 0.05f, 0.55f }, 0, 0.0f, 1, { { 0, 1, 0.30f, 100.0f } } },
    { "rj_down", 48000, 1024, 6900, 2200, 700, 500, 3600, 512, 768, { 0.5f, 0.5f, 0.2f, 0.55f }, 0, 0.0f, 1, { { 1, 1, -100.0f, -2.00f } } },
    { "rj_keep", 48000, 1024, 8900, 1000, 400, 1000, 5000, 512, 1600, { 0.5f, 0.2f, 0.1f, 0.55f }, 0, 0.0f, 1, { { 0, 1, -0.85f, -0.15f } } },
    { "rj_rev", 48000, 1024, 7200, 1800, 500, 500, 3600, 512, 1600, { 0.5f, 0.2f, 0.2f, 0.55f }, 0, 0.0f, 1, { { 1, 1, -1.85f, -1.15f } } },
    { "rj_noest", 48000, 1024, 6700, 2200, 300, 1000, 6000, 512, 1600, { 0.5f, 0.5f, 0.3f, 0.55f }, 0, 0.0f, 1, { { 0, 0, -100.0f, -1.20f } } },
    { "rj_soft", 16000, 256, 9800, 1000, 300, 100, 4400, 512, 1600, { 0.5f, 0.2f, 0.2f, 0.55f }, 5500, 0.5f, 1, { { 0, 1, -0.05f, 0.15f } } },
    { "rj_soft", 16000, 512, 12300, 1400, 500, 1000, 6000, 512, 768, { 0.5f, 0.4f, 0.2f, 0.55f }, 7100, 0.5f, 1, { { 0, 1, -0.05f, 0.15f } } },
    { "rj_soft", 48000, 1024, 9800, 1000, 400, 500, 5000, 512, 1600, { 0.5f, 0.2f, 0.2f, 0.55f }, 5500, 0.5f, 1, { { 0, 1, -0.05f, 0.15f } } },
};

static int g3m_cases(G2MonoCase* out) {
    int n = 0;
    (void)g2m_cases;                       /* the second set's table is not used here */
    for (size_t i = 0; i < sizeof g3_scenes / sizeof g3_scenes[0]; ++i) {
        const G3Scene* s = &g3_scenes[i];
        G2MonoCase* c = g2m_new(out, &n, s->name, s->sr, s->fft, s->dur_ms);
        int t = s->t0_ms;
        memset(&c->stim, 0, sizeof c->stim);
        for (int k = 0; k < 8; ++k) { g2_seg(c->stim.far, &c->stim.n_far, t, t + s->on_ms); t += s->on_ms + s->off_ms; }
        c->stim.far_amp = 0.12f;
        c->stim.flip_ms = s->flip_ms; c->stim.fd1_16 = s->d1_16; c->stim.fd2_16 = s->d2_16;
        for (int k = 0; k < 4; ++k) c->stim.fg[k] = s->g[k];
        if (s->near_t_ms) {
            g2_seg(c->stim.near, &c->stim.n_near, s->near_t_ms, s->near_t_ms + 1200);
            c->stim.near_amp = s->near_amp; c->stim.near_ramp_ms = 600;
        }
        c->judge = 1;
        c->n_exp = s->n_exp;
        for (int k = 0; k < s->n_exp; ++k) c->exp[k] = s->exp[k];
    }
    return n;
}

/* Number of expectations violated by the judgments of the case just run (g2_jrec): their count, and per judgment the outcome,
 * the estimate flag and the range of cur - pre. A case also fails when it was judged on the soft path against its name. */
static int g3m_judge_check(const G2MonoCase* c) {
    int bad = 0;
    const int soft_case = strcmp(c->name, "rj_soft") == 0;
    if (g2_jn != c->n_exp) {
        fprintf(G2_JOUT, "FAIL: %d/%d %s: %d judgments, expected %d\n", c->sr, c->fft, c->name, g2_jn, c->n_exp);
        bad++;
    }
    for (int i = 0; i < g2_jn && i < c->n_exp; ++i) {
        const G2JudgeRec* r = &g2_jrec[i];
        const G2JudgeExp* e = &c->exp[i];
        const float d = r->cur - r->pre;
        if (r->reverted != e->revert || r->has_est != e->has_est || d < e->d_lo || d > e->d_hi || r->soft != soft_case) {
            fprintf(G2_JOUT, "FAIL: %d/%d %s judgment #%d: %s, estimate %d, delta %+.2f, %s path; expected %s, estimate %d, delta in [%.2f, %.2f], %s path\n",
                    c->sr, c->fft, c->name, i, r->reverted ? "reverted" : "kept", r->has_est, d, r->soft ? "soft" : "hard",
                    e->revert ? "reverted" : "kept", e->has_est, e->d_lo, e->d_hi, soft_case ? "soft" : "hard");
            bad++;
        }
    }
    return bad;
}

static int g2_step_ctx(Aec* a, const float* mic, const float* ref, G2View* v) {
    AecLinearView lv; AecResidualView rv;
    if (aec_process_linear(a, mic, ref) != AEC_OK) return -1;
    if (aec_process_residual(a, NULL) != AEC_OK) return -1;
    memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
    memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
    if (aec_get_linear(a, &lv) != AEC_OK || aec_get_residual(a, &rv) != AEC_OK) return -1;
    memset(v, 0, sizeof *v);
    v->n_freqs = rv.n_freqs; v->hop_size = rv.hop_size;
    v->linear_hop = lv.linear_hop; v->far_spec = lv.far_spec; v->lin_formed = rv.formed_hop;
    v->aligned_far = lv.aligned_far_hop; v->delay_samples = lv.delay_samples; v->delay_conf = lv.delay_confidence;
    v->delay_state = (int)lv.delay_state; v->generation = lv.generation;
    v->error_spec = rv.error_spec; v->echo_spec = rv.echo_spec; v->near_spec = rv.near_spec; v->base_spec = rv.base_spec;
    v->formed_hop = rv.formed_hop; v->r2 = rv.r2; v->comfort_noise = rv.comfort_noise; v->res_gain = rv.res_gain;
    v->usable_linear = rv.usable_linear; v->filter_converged = lv.filter_converged; v->epc_active = lv.epc_active;
    v->is_stationary_dt = lv.is_stationary_dt; v->res_floor_protect = rv.res_floor_protect; v->guard_active = rv.guard_active;
    v->filter_once_converged = lv.filter_once_converged; v->clock_drift = rv.clock_drift;
    v->dt_indicator = lv.dt_indicator; v->far_power = lv.far_power; v->erle_factor = lv.erle_factor;
    v->saturation_level = lv.saturation_level; v->erl_estimate = lv.erl_estimate; v->divergence = lv.divergence;
    v->shadow_dt = lv.shadow_dt;
    return 0;
}

int main(void) {
    static G2MonoCase cases[G2M_MAX_CASES];
    int nc = g3m_cases(cases), n_run = 0, n_judge = 0, n_rev = 0, n_keep = 0, n_noest = 0, n_soft = 0;
    g_have_baseline = baseline_load("split_entry3");
    for (int i = 0; i < nc; ++i) {
        if (!g2m_want(&cases[i])) continue;
        if (g2m_run_case(&cases[i]) != 0) { g_fail++; break; }
        n_run++;
        n_judge += g2_jn;
        for (int j = 0; j < g2_jn; ++j) {
            n_rev += g2_jrec[j].reverted;
            n_keep += !g2_jrec[j].reverted && g2_jrec[j].has_est;
            n_noest += !g2_jrec[j].has_est;
            n_soft += g2_jrec[j].soft;
        }
        if (HOST_VALIDATED) {
            int v = g3m_judge_check(&cases[i]);
            CHECK(v == 0, "%d judgment expectation(s) of case %d/%d %s violated", v, cases[i].sr, cases[i].fft, cases[i].name);
        }
    }
    printf("%d cases, %d judgments: %d natural reverts, %d kept with an estimate, %d kept without one (%d on the soft path)\n",
           n_run, n_judge, n_rev, n_keep, n_noest, n_soft);
    if (HOST_VALIDATED && !getenv("G2_ONLY"))
        CHECK(n_rev > 0 && n_keep > 0 && n_noest > 0 && n_soft > 0, "the scenes must reach reverts, kept judgments with and without an estimate, and the soft path");
    g_fail += baseline_finish();
    printf("%ld checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
