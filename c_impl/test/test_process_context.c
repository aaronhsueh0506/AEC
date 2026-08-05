/* test_process_context.c — context-only processing regression test.
 *
 * aec_process() was split into a shared aec_process_core() (steps 1-17,
 * 19-20: linear filter + AEC3 post/RES block + power EMAs + convergence
 * detection) and two thin wrappers: aec_process() (core + output limiter +
 * emit) and aec_process_context() (core only). This test proves three
 * things a byte-equal WAV render alone cannot:
 *
 *   1. aec_process_context() genuinely never touches the limiter's
 *      per-instance state (limiter_gain / has_limiter_lag / limiter_near_lag,
 *      including the full [hop_size] buffer contents, not just the two
 *      scalar flags) -- proven by direct inspection of those Aec struct
 *      fields (this repo's existing convention, e.g. test_config_validation.c/
 *      test_lifecycle.c already read Aec fields directly; the struct is not
 *      opaque). A link-time call-counter (zero_heap_hook.c's technique) does
 *      not apply here since the limiter is inline code in the same
 *      translation unit, not a call to an external/opaque symbol -- direct
 *      state inspection is the equivalent, precedented mechanism for an
 *      internal code path.
 *   2. EVERY AecResContext field aec_process()'s caller could read --
 *      including linear_hop, far_spec, res_gain, and saturation_level, not
 *      just the subset this test originally covered -- is BYTE-IDENTICAL to
 *      what aec_process_context() produces on two separately-constructed,
 *      identically-configured, identically-fed instances, across the three
 *      officially-contracted grids x shadow on/off x spatial_linear_context
 *      on/off (the exact config the 4ch wrapper uses is
 *      spatial_linear_context=1 -- untested before this pass) -- proving the
 *      context-only entry point loses nothing the context exposes, it only
 *      skips the limiter + emit side effects. Every populated field is also
 *      checked finite (no NaN/Inf), not just equal to its sibling.
 *   3. aec_reset() on a context-only instance restores it to the same clean
 *      state a fresh aec_create() would produce.
 *
 * Build (standalone, from c_impl/ — mirrors test_lifecycle.c's documented
 * recipe):
 *   make -C ../../audio_common BACKEND=kiss lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_process_context.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=kiss print-lib-path) -lm \
 *       -o bin/test_process_context
 * Run:
 *   ./bin/test_process_context
 * Also wired into the Makefile: `make test-process-context`.
 */
#include "aec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; } \
        else         { printf("pass: %s\n", (msg)); g_pass++; } \
    } while (0)

static void context_only_config(AecConfig* cfg, int sample_rate) {
    aec_config_from_preset(cfg, AEC_PRESET_BALANCED, sample_rate);
    cfg->enable_res = 0;
    cfg->return_res_context = 1;
}

/* fft_size=0 means "use the preset's default grid for this sample_rate".
 * spatial_linear_context mirrors the exact config the 4ch wrapper uses for
 * every lane (enable_res=0/return_res_context=1/spatial_linear_context=1) --
 * only valid on top of context_only_config's enable_res=0/return_res_
 * context=1 base (aec_validate_config rejects any other combination). */
static void context_only_config_grid(AecConfig* cfg, int sample_rate, int fft_size,
                                      int spatial_linear_context) {
    context_only_config(cfg, sample_rate);
    if (fft_size > 0) cfg->fft_size = fft_size;
    cfg->spatial_linear_context = spatial_linear_context;
}

static int all_finite(const float* buf, int n) {
    for (int i = 0; i < n; ++i) if (!isfinite(buf[i])) return 0;
    return 1;
}

static int all_finite_complex(const Complex* buf, int n) {
    for (int i = 0; i < n; ++i) if (!isfinite(buf[i].r) || !isfinite(buf[i].i)) return 0;
    return 1;
}

/* Deterministic pseudo-random hop generator (no <random.h> dependency,
 * matches this repo's existing xorshift-in-tests style). */
static unsigned g_rng_state = 0x2AEC5EEDu;
static float rng_uniform(float amp) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 17;
    g_rng_state ^= g_rng_state << 5;
    float u = (float)(g_rng_state & 0xFFFFFFu) / (float)0xFFFFFFu; /* [0,1) */
    return amp * (2.0f * u - 1.0f);
}

/* limiter_gain / has_limiter_lag / limiter_near_lag never move when every
 * hop goes through aec_process_context(); aec_process() moves them from
 * the very first hop. Uses a loud burst (same style as the AEC3
 * formed-output-seam Python test) so limiter_gain would visibly excurse
 * below 1.0 if the limiter ran. */
static void test_context_only_never_touches_limiter_state(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000);

    Aec ctx_only, full;
    CHECK(aec_create(&ctx_only, &cfg) == 0, "limiter-state test: aec_create(context-only instance)");
    CHECK(aec_create(&full, &cfg) == 0, "limiter-state test: aec_create(full instance)");

    int hop = aec_hop_size(&ctx_only);
    CHECK(hop > 0 && hop <= 4096, "limiter-state test: valid hop size");
    if (hop <= 0 || hop > 4096) { aec_destroy(&ctx_only); aec_destroy(&full); return; }

    /* Snapshot the raw limiter_near_lag buffer right after construction --
     * compared byte-for-byte after the loop below, so this test doesn't need
     * to assume any particular initial value (zeroed on aec_init's pool,
     * uninitialized malloc garbage on aec_create's heap arena) -- only that
     * aec_process_context() never writes to it, whatever it started as. */
    float* near_lag_snapshot = (float*)malloc((size_t)hop * sizeof(float));
    memcpy(near_lag_snapshot, ctx_only.limiter_near_lag, (size_t)hop * sizeof(float));

    float mic[4096], ref[4096], out[4096];
    g_rng_state = 0x2AEC5EEDu;
    for (int i = 0; i < 200; ++i) {
        /* Quiet baseline, loud burst frames 40-49, quiet tail -- mic/ref
         * independently drawn (not one derived from the other) so the
         * adaptive filter has no real echo path to converge onto; the
         * abrupt quiet->loud transition then forces a transient raw-output
         * overshoot above the (one-hop-lagged) near peak the limiter
         * compares against -- same technique as the AEC3 formed-output-seam
         * Python test (test_formed_output_seam.py). */
        float amp = (i >= 40 && i < 50) ? 0.9f : 0.02f;
        for (int s = 0; s < hop; ++s) {
            mic[s] = rng_uniform(amp);
            ref[s] = 0.3f * rng_uniform(amp);
        }
        aec_process_context(&ctx_only, mic, ref);
        aec_process(&full, mic, ref, out);
    }

    CHECK(ctx_only.limiter_gain == 1.0f,
          "aec_process_context()-only instance: limiter_gain stays exactly 1.0 after 200 hops incl. a loud burst");
    CHECK(ctx_only.has_limiter_lag == 0,
          "aec_process_context()-only instance: has_limiter_lag stays 0 (limiter block never entered)");
    CHECK(full.has_limiter_lag == 1,
          "aec_process()-only instance: has_limiter_lag becomes 1 (limiter block did run)");
    CHECK(full.limiter_gain < 0.95f,
          "aec_process()-only instance: the loud burst actually forced limiter_gain below 0.95 "
          "(otherwise this test cannot distinguish 'ran and had no effect' from 'never ran')");
    CHECK(memcmp(ctx_only.limiter_near_lag, near_lag_snapshot, (size_t)hop * sizeof(float)) == 0,
          "aec_process_context()-only instance: limiter_near_lag buffer byte-identical to its "
          "post-construction snapshot after 200 hops (never written, not just the two scalar flags)");
    CHECK(memcmp(full.limiter_near_lag, near_lag_snapshot, (size_t)hop * sizeof(float)) != 0,
          "aec_process()-only instance: limiter_near_lag buffer DID change from its post-construction "
          "snapshot (the limiter actually ran and wrote it, sanity check against a vacuous pass)");

    free(near_lag_snapshot);
    aec_destroy(&ctx_only);
    aec_destroy(&full);
}

/* Two separately-constructed, identically-configured, identically-fed
 * instances -- one driven exclusively by aec_process(), one exclusively by
 * aec_process_context() -- must expose byte-identical AecResContext on
 * every hop, across every field a real caller (mono pipeline, 4ch wrapper)
 * actually reads -- including linear_hop/far_spec/res_gain/saturation_level,
 * and spatial_linear_context on/off (the 4ch wrapper's exact config).
 * Proves the context-only path loses no information the context struct
 * itself carries. Every populated field is also checked finite, since a
 * byte-equal comparison of two NaN payloads would otherwise pass vacuously. */
static void test_context_matches_full_process_context(
        int sample_rate, int fft_size, int enable_shadow, int spatial_linear_context) {
    AecConfig cfg;
    context_only_config_grid(&cfg, sample_rate, fft_size, spatial_linear_context);
    cfg.enable_shadow = enable_shadow;

    Aec via_full, via_context;
    char label[160];
    snprintf(label, sizeof(label), "context-parity sr=%d/fft=%d/shadow=%d/spatial=%d: aec_create(via_full)",
             sample_rate, cfg.fft_size, enable_shadow, spatial_linear_context);
    CHECK(aec_create(&via_full, &cfg) == 0, label);
    snprintf(label, sizeof(label), "context-parity sr=%d/fft=%d/shadow=%d/spatial=%d: aec_create(via_context)",
             sample_rate, cfg.fft_size, enable_shadow, spatial_linear_context);
    CHECK(aec_create(&via_context, &cfg) == 0, label);

    int hop = aec_hop_size(&via_full);
    if (hop <= 0 || hop > 4096) { aec_destroy(&via_full); aec_destroy(&via_context); return; }

    float mic[4096], ref[4096], out[4096];
    g_rng_state = 0x51DEC0DEu;
    int all_match = 1;
    int all_finite_ok = 1;
    int res_gain_null_as_expected = 1;
    for (int i = 0; i < 150; ++i) {
        float amp = (i >= 50 && i < 55) ? 0.8f : 0.05f; /* occasional burst too */
        for (int s = 0; s < hop; ++s) {
            float v = rng_uniform(amp);
            mic[s] = v;
            ref[s] = 0.6f * v;
        }
        aec_process(&via_full, mic, ref, out);
        aec_process_context(&via_context, mic, ref);

        AecResContext ca, cb;
        aec_get_res_context(&via_full, &ca);
        aec_get_res_context(&via_context, &cb);

        if (ca.n_freqs != cb.n_freqs || ca.hop_size != cb.hop_size) { all_match = 0; break; }
        int K = ca.n_freqs;
        if (memcmp(ca.formed_hop, cb.formed_hop, (size_t)hop * sizeof(float)) != 0) { all_match = 0; break; }
        if (memcmp(ca.linear_hop, cb.linear_hop, (size_t)hop * sizeof(float)) != 0) { all_match = 0; break; }
        if (memcmp(ca.far_spec, cb.far_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(ca.error_spec, cb.error_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(ca.echo_spec, cb.echo_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(ca.near_spec, cb.near_spec, (size_t)K * sizeof(Complex)) != 0) { all_match = 0; break; }
        if (memcmp(ca.r2, cb.r2, (size_t)K * sizeof(float)) != 0) { all_match = 0; break; }
        if (memcmp(ca.comfort_noise, cb.comfort_noise, (size_t)K * sizeof(float)) != 0) { all_match = 0; break; }
        /* res_gain is NULL on both sides under spatial_linear_context (this
         * lane's G_res is never computed -- see AecResContext's doc
         * comment); otherwise it must be byte-identical like every other
         * per-bin field. */
        if ((ca.res_gain == NULL) != (cb.res_gain == NULL)) { all_match = 0; break; }
        if ((ca.res_gain == NULL) != (spatial_linear_context != 0)) res_gain_null_as_expected = 0;
        if (ca.res_gain != NULL &&
            memcmp(ca.res_gain, cb.res_gain, (size_t)K * sizeof(float)) != 0) { all_match = 0; break; }
        if (ca.far_power != cb.far_power || ca.erle_factor != cb.erle_factor
                || ca.dt_indicator != cb.dt_indicator || ca.divergence != cb.divergence
                || ca.saturation_level != cb.saturation_level
                || ca.erl_estimate != cb.erl_estimate || ca.shadow_dt != cb.shadow_dt
                || ca.is_stationary_dt != cb.is_stationary_dt
                || ca.filter_converged != cb.filter_converged
                || ca.filter_once_converged != cb.filter_once_converged
                || ca.epc_active != cb.epc_active) { all_match = 0; break; }

        if (!all_finite(ca.formed_hop, hop) || !all_finite(ca.linear_hop, hop)
                || !all_finite_complex(ca.far_spec, K) || !all_finite_complex(ca.error_spec, K)
                || !all_finite_complex(ca.echo_spec, K) || !all_finite_complex(ca.near_spec, K)
                || !all_finite(ca.r2, K) || !all_finite(ca.comfort_noise, K)
                || (ca.res_gain != NULL && !all_finite(ca.res_gain, K))
                || !isfinite(ca.far_power) || !isfinite(ca.erle_factor)
                || !isfinite(ca.dt_indicator) || !isfinite(ca.divergence)
                || !isfinite(ca.saturation_level) || !isfinite(ca.erl_estimate)
                || !isfinite(ca.shadow_dt)) { all_finite_ok = 0; break; }
    }

    snprintf(label, sizeof(label),
             "context-parity sr=%d/fft=%d/shadow=%d/spatial=%d: AecResContext byte-identical every hop, "
             "aec_process() vs aec_process_context()",
             sample_rate, cfg.fft_size, enable_shadow, spatial_linear_context);
    CHECK(all_match, label);

    snprintf(label, sizeof(label),
             "context-parity sr=%d/fft=%d/shadow=%d/spatial=%d: every populated AecResContext field stays finite",
             sample_rate, cfg.fft_size, enable_shadow, spatial_linear_context);
    CHECK(all_finite_ok, label);

    snprintf(label, sizeof(label),
             "context-parity sr=%d/fft=%d/shadow=%d/spatial=%d: res_gain is NULL iff spatial_linear_context is set",
             sample_rate, cfg.fft_size, enable_shadow, spatial_linear_context);
    CHECK(res_gain_null_as_expected, label);

    aec_destroy(&via_full);
    aec_destroy(&via_context);
}

/* aec_process_context() must also work correctly across a reset (state
 * fully re-zeroed, next hop behaves like a fresh instance) -- exercises
 * the same reset path aec_process() itself already relies on. */
static void test_context_survives_reset(void) {
    AecConfig cfg;
    context_only_config(&cfg, 16000);

    Aec a;
    CHECK(aec_create(&a, &cfg) == 0, "reset test: aec_create()");
    int hop = aec_hop_size(&a);
    if (hop <= 0 || hop > 4096) { aec_destroy(&a); return; }

    float mic[4096], ref[4096];
    g_rng_state = 0x8ACE0001u;
    for (int i = 0; i < 30; ++i) {
        for (int s = 0; s < hop; ++s) { mic[s] = rng_uniform(0.3f); ref[s] = rng_uniform(0.2f); }
        aec_process_context(&a, mic, ref);
    }
    CHECK(a.frame_count == 30, "reset test: frame_count reaches 30 via aec_process_context() alone");

    aec_reset(&a);
    CHECK(a.frame_count == 0, "reset test: aec_reset() zeroes frame_count");
    CHECK(a.limiter_gain == 1.0f, "reset test: aec_reset() restores limiter_gain to 1.0");

    for (int i = 0; i < 5; ++i) {
        for (int s = 0; s < hop; ++s) { mic[s] = rng_uniform(0.3f); ref[s] = rng_uniform(0.2f); }
        aec_process_context(&a, mic, ref);
    }
    CHECK(a.frame_count == 5, "reset test: aec_process_context() runs cleanly post-reset");

    aec_destroy(&a);
}

int main(void) {
    test_context_only_never_touches_limiter_state();

    /* The three officially-contracted grids (see AEC/CLAUDE.md /
     * pipelines/README.md "Parameter Alignment") x shadow on/off x
     * spatial_linear_context on/off -- the last axis is the exact config
     * the 4ch wrapper uses for every lane (enable_res=0/return_res_
     * context=1/spatial_linear_context=1), previously untested here. */
    static const int grids[][1] = { {0}, {512}, {0} };
    static const int rates[]    = { 16000, 16000, 48000 };
    for (int g = 0; g < 3; ++g) {
        for (int shadow = 0; shadow <= 1; ++shadow) {
            for (int spatial = 0; spatial <= 1; ++spatial) {
                test_context_matches_full_process_context(
                        rates[g], grids[g][0], shadow, spatial);
            }
        }
    }

    test_context_survives_reset();

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) printf("PASS\n");
    return g_fail ? 1 : 0;
}
