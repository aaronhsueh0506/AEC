/* test_res_nearend_gate.c -- product RES floor policy. The near-recent latch
 * is the only source of the DT floor, and AecResContext exports that exact
 * decision for fused consumers. Standalone runner (make
 * test-res-nearend-gate), exit != 0 on any failure. */
#include "aec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

static unsigned g_seed = 0x2545F491u;
static float urand(void) {
    g_seed = g_seed * 1664525u + 1013904223u;
    return ((float)(g_seed >> 8) / (float)(1u << 24)) - 0.5f;
}

static void fill_noise(float *mic, float *ref, int hop) {
    int i;
    for (i = 0; i < hop; i++) { mic[i] = 0.2f * urand(); ref[i] = 0.2f * urand(); }
}

/* Far-end noise through a one-tap echo path (40 samples, 0.5) plus a small
 * near-end noise floor, so the linear filter converges and the usable-linear
 * verdict becomes true on its own. */
#define ECHO_DELAY 40
static float g_echo_ring[ECHO_DELAY];
static int g_echo_pos = 0;
static void fill_echo(float *mic, float *ref, int hop) {
    int i;
    for (i = 0; i < hop; i++) {
        float x = 0.3f * urand();
        float delayed = g_echo_ring[g_echo_pos];
        g_echo_ring[g_echo_pos] = x;
        g_echo_pos = (g_echo_pos + 1) % ECHO_DELAY;
        ref[i] = x;
        mic[i] = 0.5f * delayed + 0.002f * urand();
    }
}

/* Force the detector state, the latch and the usable-linear verdict, then
 * run one hop of the echo-path signal. dne_hold_counter keeps
 * dominant_nearend_update() from clearing the state in the same hop;
 * ne_recent_frames is re-armed by the hop's own gate only from a sustained
 * near-end indicator, which this signal never produces, so the forced value
 * is what the post reads. The converged filter keeps the analyzer's own
 * verdict at usable; the unusable case switches the linear filter off in
 * the quality analyzer, which the hop does not re-derive. */
static void run_hop(Aec *a, int dne, int latch, int usable,
                    float *mic, float *ref, float *out, int hop) {
    fill_echo(mic, ref, hop);
    a->a3_sg.dne_nearend_state = dne;
    a->a3_sg.dne_hold_counter = dne ? 1000000 : 0;
    a->ne_recent_frames = latch ? 1000000 : 0;
    a->ne_above = 0;
    a->a3_state.filter_quality.use_linear_filter = usable ? 1 : 0;
    aec_process(a, mic, ref, out);
    CHECK(aec_state_usable_linear_estimate(&a->a3_state) == usable,
          "precondition: usable-linear verdict is %d (got %d)", usable,
          aec_state_usable_linear_estimate(&a->a3_state));
    a->a3_state.filter_quality.use_linear_filter = 1;
}

static int nonzero_r2(Aec *a) {
    AecResContext ctx; int k, n = 0;
    memset(&ctx, 0, sizeof ctx);
    aec_get_res_context(a, &ctx);
    if (ctx.r2) for (k = 0; k < ctx.n_freqs; k++) if (ctx.r2[k] > 0.0f) n++;
    return ctx.r2 ? n : -1;
}

static void flag_every_band(Aec *a) {
    int k;
    for (k = 0; k < a->a3_stat.n_freqs; k++) { a->a3_stat.stationarity_flags[k] = 1; a->a3_stat.hangovers[k] = 0; }
}

int main(void) {
    AecConfig cfg;
    Aec a;
    int hop, i, nonzero;
    float *mic, *ref, *out;

    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
    cfg.enable_res = 0;
    cfg.return_res_context = 1;         /* r2 is readable through the seam */
    CHECK(aec_create(&a, &cfg) == 0, "aec_create succeeds");
    hop = aec_hop_size(&a);
    mic = (float*)calloc((size_t)hop, sizeof(float));
    ref = (float*)calloc((size_t)hop, sizeof(float));
    out = (float*)calloc((size_t)hop, sizeof(float));
    CHECK(mic && ref && out, "hop buffers allocated");

    /* Warm up on the echo path: render active, post state initialised,
     * stationarity estimator past its convergence count, linear filter
     * converged (the usable-linear verdict is a precondition below). */
    for (i = 0; i < 600; i++) { fill_echo(mic, ref, hop); aec_process(&a, mic, ref, out); }
    CHECK(a.cfg.dt_aware_res_floor_enabled == 1, "balanced enables the DT floor lift");
    CHECK(aec_state_usable_linear_estimate(&a.a3_state) == 1,
          "the converged echo path yields a usable linear estimate");
    (void)fill_noise;

    /* Product policy -- the near-recent latch alone. */
    CHECK(a.cfg.min_gain_floor_dt_db == -20.0f,
          "the selected DT-floor Pareto point is -20 dB");
    run_hop(&a, 0, 1, 1, mic, ref, out, hop);
    CHECK(a.a3_sg.dt_protect_active == 1, "default: latch armed, usable linear: lifted");
    {
        AecResContext ctx;
        memset(&ctx, 0, sizeof ctx);
        aec_get_res_context(&a, &ctx);
        CHECK(ctx.res_floor_protect == 1,
              "context exports the lane's actual floor decision");
    }
    run_hop(&a, 0, 1, 0, mic, ref, out, hop);
    CHECK(a.a3_sg.dt_protect_active == 1, "default: latch armed, unusable linear: lifted");
    run_hop(&a, 1, 0, 1, mic, ref, out, hop);
    CHECK(a.a3_sg.dt_protect_active == 0, "default: detector on, latch off: NOT lifted");
    flag_every_band(&a); run_hop(&a, 0, 0, 1, mic, ref, out, hop);
    nonzero = nonzero_r2(&a);
    CHECK(nonzero == 0, "default: usable linear does not spare the stationary bands (%d nonzero bins)", nonzero);

    /* The existing enable switch still turns the lift off. */
    a.cfg.dt_aware_res_floor_enabled = 0;
    run_hop(&a, 1, 1, 0, mic, ref, out, hop);
    CHECK(a.a3_sg.dt_protect_active == 0, "dt_aware_res_floor_enabled=0: never lifted");

    free(mic); free(ref); free(out);
    aec_destroy(&a);
    printf(g_fail ? ">>> FAIL\n" : ">>> PASS\n");
    return g_fail;
}
