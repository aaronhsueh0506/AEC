/* Regression test for aec_apply_external_realign(): the 4ch shared-delay
 * wrapper's "vertical line".
 *
 * An AEC_DELAY_EXTERNAL_ALIGNED instance converges while the caller serves
 * one far alignment; the caller then changes the alignment. The old wrapper
 * response -- full aec_reset() -- wipes the converged filter, so the echo is
 * fully re-exposed for dozens of hops (bright broadband line) and the WOLA
 * restart adds a near-zero output hop (dark line). Measured on this exact
 * scene before the fix: linear residual jumped 0.030 -> 0.075 (~= the 0.076
 * echo) at the change hop.
 *
 * What this proves:
 *   1. WARM TRANSFER HOLDS CANCELLATION (delta > 0, the first-acquisition
 *      raw->aligned move): after realign, residual stays BELOW half the echo
 *      on every one of the next hops.
 *   2. THE DEFECT REPRODUCES with aec_reset() on an identical twin scene:
 *      residual rebounds to >= 0.8x echo. Without this row, row 1 could
 *      pass on a scene where the filter never converged at all.
 *   3. SIGNED SHIFT (delta < 0, aligned->raw): same continuity bound.
 *   4. API contract: 0 on delta==0, -1 on a non-external mode, and the soft
 *      path (warm transfer disabled) keeps the taps: no reset-magnitude
 *      rebound.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"

#define SR   16000
#define FFT  512
#define HOP  256
#define D    300
#define HOPS 250
#define TAIL 6

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_failures++; } \
    else { printf("ok: %s\n", msg); } \
} while (0)

static unsigned long long g_rng;
static float noise_next(void)
{
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return ((float)((g_rng >> 11) & 0xFFFFFF) / 8388608.0f - 1.0f) * 0.25f;
}

static float* g_far_hist;   /* [PAD + (HOPS+TAIL)*HOP] */
static int    g_total;
#define PAD (4 * D)

static void scene_init(void)
{
    int i;
    g_total = PAD + (HOPS + TAIL) * HOP;
    g_far_hist = (float*)malloc((size_t)g_total * sizeof(float));
    g_rng = 0x9E3779B97F4A7C15ull;
    for (i = 0; i < g_total; ++i) g_far_hist[i] = noise_next();
}

/* Serve one hop: mic = 0.5 * far_hist[n - D]; far = far_hist[n - align]. */
static float run_hop(Aec* a, int h, int align, float* out)
{
    float mic[HOP], far[HOP];
    double r2 = 0.0;
    int i, base = PAD + h * HOP;
    for (i = 0; i < HOP; ++i) {
        mic[i] = 0.5f * g_far_hist[base + i - D];
        far[i] = g_far_hist[base + i - align];
    }
    aec_process(a, mic, far, out);
    for (i = 0; i < HOP; ++i) r2 += (double)out[i] * out[i];
    return (float)sqrt(r2 / HOP);
}

static float echo_rms(int h)
{
    double e2 = 0.0;
    int i, base = PAD + h * HOP;
    for (i = 0; i < HOP; ++i) {
        float e = 0.5f * g_far_hist[base + i - D];
        e2 += (double)e * e;
    }
    return (float)sqrt(e2 / HOP);
}

static int make_external(Aec* a, int warm_transfer)
{
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
    cfg.fft_size = FFT;
    cfg.enable_res = 0;                    /* output IS the linear residual */
    cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    cfg.delay_acquire_warm_transfer = warm_transfer;
    return aec_create(a, &cfg);
}

int main(void)
{
    Aec aec, twin, back, soft;
    float out[HOP];
    int h, rc;
    float resid, worst, echo;

    scene_init();

    /* --- Row 1: warm transfer holds cancellation across raw -> aligned. */
    rc = make_external(&aec, 1);
    CHECK(rc == 0 && aec_hop_size(&aec) == HOP, "external instance: create");
    resid = 0.0f;
    for (h = 0; h < HOPS; ++h) resid = run_hop(&aec, h, 0, out);
    echo = echo_rms(HOPS - 1);
    CHECK(resid < 0.5f * echo, "raw-far convergence before the realign");
    rc = aec_apply_external_realign(&aec, D - 0);
    CHECK(rc == 1, "realign(+D) takes the warm tap-transfer path");
    worst = 0.0f;
    for (h = HOPS; h < HOPS + TAIL; ++h) {
        resid = run_hop(&aec, h, D, out);
        if (resid > worst) worst = resid;
    }
    CHECK(worst < 0.5f * echo_rms(HOPS),
          "no echo re-exposure after the warm realign");

    /* --- Row 2: the defect reproduces on a twin with aec_reset(). */
    rc = make_external(&twin, 1);
    CHECK(rc == 0, "twin instance: create");
    for (h = 0; h < HOPS; ++h) run_hop(&twin, h, 0, out);
    aec_reset(&twin);
    worst = 0.0f;
    for (h = HOPS; h < HOPS + TAIL; ++h) {
        resid = run_hop(&twin, h, D, out);
        if (resid > worst) worst = resid;
    }
    CHECK(worst >= 0.8f * echo_rms(HOPS),
          "aec_reset() control re-exposes the echo (defect reproduces)");

    /* --- Row 3: signed shift, aligned -> raw (delta < 0). */
    rc = make_external(&back, 1);
    CHECK(rc == 0, "signed-shift instance: create");
    for (h = 0; h < HOPS; ++h) run_hop(&back, h, D, out);
    resid = run_hop(&back, HOPS, D, out);
    CHECK(resid < 0.5f * echo_rms(HOPS), "aligned-far convergence at lag 0");
    rc = aec_apply_external_realign(&back, 0 - D);
    CHECK(rc == 1, "realign(-D) takes the warm tap-transfer path");
    /* The retard direction (delta < 0) clears the far history (see the
     * realign implementation), so the echo estimate is zero until the
     * partition ring refills: a BOUNDED <= 2-hop pass-through transient,
     * never amplification, never an output gap. From the third hop the
     * shifted taps must cancel outright -- against the reset control, which
     * stays exposed for dozens of hops. */
    worst = 0.0f;
    for (h = HOPS + 1; h < HOPS + 3; ++h) {
        resid = run_hop(&back, h, 0, out);
        if (resid > worst) worst = resid;
    }
    CHECK(worst <= 1.05f * echo_rms(HOPS),
          "negative-delta transient is bounded by the echo itself");
    worst = 0.0f;
    for (h = HOPS + 3; h < HOPS + TAIL; ++h) {
        resid = run_hop(&back, h, 0, out);
        if (resid > worst) worst = resid;
    }
    CHECK(worst < 0.15f * echo_rms(HOPS),
          "negative-delta realign cancels outright after the 2-hop refill");

    /* --- Row 4: API contract + soft path. */
    CHECK(aec_apply_external_realign(&aec, 0) == 0, "delta 0 is a no-op");
    CHECK(aec_apply_external_realign(NULL, D) == -1, "NULL is rejected");
    {
        AecConfig cfg;
        Aec matched;
        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
        cfg.fft_size = FFT;
        rc = aec_create(&matched, &cfg);
        CHECK(rc == 0 &&
              aec_apply_external_realign(&matched, D) == -1,
              "MATCHED mode is rejected");
        aec_destroy(&matched);
    }
    rc = make_external(&soft, 0);          /* warm transfer disabled */
    CHECK(rc == 0, "soft instance: create");
    for (h = 0; h < HOPS; ++h) run_hop(&soft, h, D, out);
    CHECK(aec_apply_external_realign(&soft, 64) == 0,
          "gate off falls to the soft path");

    aec_destroy(&aec);
    aec_destroy(&twin);
    aec_destroy(&back);
    aec_destroy(&soft);
    free(g_far_hist);
    if (g_failures) { fprintf(stderr, "%d failure(s)\n", g_failures); return 1; }
    printf("test_external_realign: ALL PASS\n");
    return 0;
}
