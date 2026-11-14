/* test_split_entry3.c -- the third scene set of the single-channel API: bad-relock judgments that the stimulus produces by itself
 * (no field of the instance is written). Two echo paths whose gains swap during a bursty far end move the delay estimate; the
 * windowed ERLE after the shift is judged against the value before it. The scenes make that comparison come out in every way:
 * ERLE improving after the shift (no revert), clearly worse (revert), a few tenths of a dB either side of the 1 dB margin, and
 * worse with no estimate yet (no revert), on the three product grids; the soft path is reached with kept judgments only. Every
 * judgment is checked against the expectations of the case table (baseline3_mono.h) and printed; the chained hashes form a line
 * stream that AEC_TEST_BASELINE_WRITE / AEC_TEST_BASELINE capture and compare (no baseline is shipped).
 *
 * Run from c_impl/:  make test-split-entry-3   (G2_VERBOSE=<file> writes every hop's unchained group hashes, G2_ONLY=<substring>)
 * The expectation checks run on the NEON and the scalar build only. */
#include "aec.h"
#include "../src/aec_res_internal.h"
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

#include "baseline3_mono.h"

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
