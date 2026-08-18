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
 *   3. SIGNED SHIFT (delta < 0, aligned->raw): the refill transient is
 *      bounded and cancellation resumes outright afterwards.
 *   4. API contract: 0 on delta==0, -1 on a non-external mode / NULL, and the
 *      generation token advances on BOTH outcomes.
 *   5. THE GATE SURVIVES FAR SILENCE. The inst-ERLE ring holds ~15 entries and
 *      every far-silent hop pushes a 0.0 into it, so a realign that arrives
 *      >= 16 hops after the last far burst -- a wrapper that re-measures its
 *      alignment between bursts, i.e. the common case -- used to read "not
 *      cancelling" on a filter cancelling at 13 dB and fall to the reset
 *      branch for nothing. aec_linear_is_cancelling()'s windowed reading is
 *      what still knows. (Needs the AEC3 post chain to run -- context-only is
 *      enough -- since that is what maintains the windowed ERLE.)
 *   6. THE RESET BRANCH ACTUALLY RECONVERGES. When the gate legitimately
 *      rejects, the taps sit at the alignment the caller abandoned. Keeping
 *      them (the counters-only echo-path-change this branch used to be) leaves
 *      the filter SUBTRACTING a decorrelated echo copy: measured 1.3-1.5x the
 *      echo, above pass-through, for 300+ hops. Pinned against an aec_reset()
 *      twin, whose trajectory is the target the filter-only reset has to match
 *      without the twin's synthesis restart.
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
/* Rows 5/6 run past HOPS+TAIL: a far-silent GAP then a long observation
 * window, and a 140-hop reconvergence trajectory. The scene buffer is sized
 * for the longest of them. */
#define GAP   24                      /* > the 15-entry inst-ERLE ring */
#define OBS   12                      /* far-active hops watched after the gap */
#define TRAJ 140                       /* reconvergence hops compared to the twin */
#define EXTRA (GAP + OBS + TRAJ + 4)

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
    g_total = PAD + (HOPS + TAIL + EXTRA) * HOP;
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

/* Far silence: both far and mic go to zero (no far => no echo). */
static void run_silent_hop(Aec* a, float* out)
{
    float zero[HOP];
    memset(zero, 0, sizeof zero);
    aec_process(a, zero, zero, out);
}

/* res_context=1 keeps the AEC3 post chain running in context-only mode: the
 * emitted samples are still exactly the linear residual (aec3_post_run's
 * context_only path forwards raw_output untouched), but the windowed-ERLE
 * bookkeeping the realign gate reads is maintained. That is the seam
 * configuration a 4-lane wrapper -- the caller this API exists for -- runs. */
static int make_external_ctx(Aec* a, int warm_transfer, int res_context)
{
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
    cfg.fft_size = FFT;
    cfg.enable_res = 0;                    /* output IS the linear residual */
    cfg.return_res_context = res_context;
    cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    cfg.delay_acquire_warm_transfer = warm_transfer;
    return aec_create(a, &cfg);
}

static int make_external(Aec* a, int warm_transfer)
{
    return make_external_ctx(a, warm_transfer, 0);
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
    /* The retard direction (delta < 0) clears the far spectra (see the realign
     * implementation), so the echo is exposed until the partition ring refills
     * past the shifted response. The general bound is
     * ceil((|delta| + tap offset)/hop) + 1 hops and the residual inside the
     * window is NOT bounded by the echo (see the aec.h contract, which carries
     * the measurements); THIS scene is the mildest corner of it -- |delta|=300
     * with the response at tap 0 -- so two hops and 1.05x is what it costs
     * here, and that is what is pinned. What matters generally is the second
     * check: cancellation resumes OUTRIGHT once the ring has refilled, against
     * the reset control that stays exposed for dozens of hops. */
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
    {
        AecLinearContext c0, c1;
        aec_get_linear_context(&soft, &c0);
        CHECK(aec_apply_external_realign(&soft, 64) == 0,
              "gate off falls to the reset path");
        aec_get_linear_context(&soft, &c1);
        /* EXTERNAL_ALIGNED owns no ring offset, so this call is the only thing
         * that can ever move the token a consumer polls to flush far caches.
         * Both outcomes must move it -- the warm instance from row 1 realigned
         * too. */
        CHECK(c1.generation != c0.generation,
              "reset outcome advances the alignment generation");
    }

    /* --- Row 5: the gate survives a far-silent gap (>= 16 hops).
     * Same scene as row 1, with the caller's realign arriving after a stretch
     * of silence -- every silent hop pushes 0.0 into the 15-entry inst-ERLE
     * ring, so a ring-only gate reads "not cancelling" and throws away a
     * filter that is cancelling at ~13 dB windowed. */
    {
        Aec gap;
        AecLinearContext g0, g1;
        rc = make_external_ctx(&gap, 1, 1);
        CHECK(rc == 0, "silent-gap instance: create");
        for (h = 0; h < HOPS; ++h) resid = run_hop(&gap, h, 0, out);
        CHECK(resid < 0.5f * echo_rms(HOPS - 1),
              "silent-gap: raw-far convergence before the silence");
        for (h = 0; h < GAP; ++h) run_silent_hop(&gap, out);
        aec_get_linear_context(&gap, &g0);
        rc = aec_apply_external_realign(&gap, D - 0);
        CHECK(rc == 1,
              "silent-gap realign still takes the warm tap-transfer path");
        aec_get_linear_context(&gap, &g1);
        CHECK(g1.generation != g0.generation,
              "warm outcome advances the alignment generation");
        worst = 0.0f;
        for (h = HOPS + GAP; h < HOPS + GAP + OBS; ++h) {
            resid = run_hop(&gap, h, D, out);
            if (resid / echo_rms(h) > worst) worst = resid / echo_rms(h);
        }
        CHECK(worst < 0.5f,
              "silent-gap warm realign keeps the residual under half the echo");
        aec_destroy(&gap);
    }

    /* --- Row 6: the reset branch reconverges like an aec_reset() twin.
     * Warm transfer off forces the branch on a genuinely converged filter, so
     * the taps really are parked at the abandoned alignment. The twin gets the
     * identical scene and an aec_reset() at the identical hop: it is the
     * trajectory a full restart achieves, and the filter-only reset has to
     * match it while leaving synthesis alone. */
    {
        Aec sres, sref;
        static const int CK[] = { 5, 10, 20, 40, 80, 120 };
        float rs[TRAJ + 1], rt[TRAJ + 1];
        size_t ci;
        char msg[96];
        CHECK(make_external_ctx(&sres, 0, 1) == 0 &&
              make_external_ctx(&sref, 0, 1) == 0, "reset-trajectory pair: create");
        for (h = 0; h < HOPS; ++h) { run_hop(&sres, h, 0, out); run_hop(&sref, h, 0, out); }
        resid = run_hop(&sres, HOPS, 0, out);
        (void)run_hop(&sref, HOPS, 0, out);
        CHECK(resid < 0.1f * echo_rms(HOPS),
              "reset-trajectory: converged before the realign");
        CHECK(aec_apply_external_realign(&sres, D - 0) == 0,
              "reset-trajectory: warm transfer off takes the reset path");
        aec_reset(&sref);
        for (h = 1; h <= TRAJ; ++h) {
            int hh = HOPS + h;
            float e = echo_rms(hh);
            rs[h] = run_hop(&sres, hh, D, out) / e;
            rt[h] = run_hop(&sref, hh, D, out) / e;
        }
        for (ci = 0; ci < sizeof(CK) / sizeof(CK[0]); ++ci) {
            int k = CK[ci];
            snprintf(msg, sizeof msg,
                     "reset-trajectory +%d: %.4f <= 1.25 x twin %.4f", k, rs[k], rt[k]);
            CHECK(rs[k] <= 1.25f * rt[k], msg);
        }
        worst = 0.0f;
        for (h = TRAJ - 19; h <= TRAJ; ++h) if (rs[h] > worst) worst = rs[h];
        snprintf(msg, sizeof msg,
                 "reset-trajectory settles at %.4f x echo (< 0.1)", worst);
        CHECK(worst < 0.1f, msg);
        aec_destroy(&sres);
        aec_destroy(&sref);
    }

    aec_destroy(&aec);
    aec_destroy(&twin);
    aec_destroy(&back);
    aec_destroy(&soft);
    free(g_far_hist);
    if (g_failures) { fprintf(stderr, "%d failure(s)\n", g_failures); return 1; }
    printf("test_external_realign: ALL PASS\n");
    return 0;
}
