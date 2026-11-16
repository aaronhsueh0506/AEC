/* Regression test for aec_apply_external_realign(): the output discontinuity when an EXTERNAL_ALIGNED caller (e.g. a
 * multi-lane shared-delay wrapper) changes its far alignment after the filter has converged. A full aec_reset() response
 * wipes the converged filter, so the echo is re-exposed for dozens of hops and the WOLA restart adds a near-zero output
 * hop; on this scene the linear residual jumps 0.030 -> 0.075 (about the 0.076 echo) at the change hop.
 *
 * What this proves:
 *   1. WARM TRANSFER HOLDS CANCELLATION (delta > 0, the first-acquisition raw->aligned move): after realign, the residual
 *      stays BELOW half the echo on every one of the next hops.
 *   2. THE FULL-RESET FAILURE REPRODUCES with aec_reset() on an identical twin scene: the residual rebounds to >= 0.8x
 *      echo. Without this row, row 1 could pass on a scene where the filter never converged at all.
 *   3. SIGNED SHIFT (delta < 0, aligned->raw): the refill transient is bounded and cancellation resumes outright afterwards.
 *   4. API contract: 0 on delta==0, -1 on a non-external mode / NULL, and the generation token advances on BOTH outcomes.
 *   5. THE GATE SURVIVES FAR SILENCE. The inst-ERLE ring holds ~15 entries and every far-silent hop pushes a 0.0 into it,
 *      so a realign arriving >= 16 hops after the last far burst (a wrapper re-measuring its alignment between bursts, the
 *      common case) would read "not cancelling" on a filter cancelling at 13 dB and take the reset branch for nothing.
 *      aec_linear_is_cancelling()'s windowed reading still knows. The silent-gap rows run the lane the way a multi-lane
 *      wrapper does, with the residual stage present (residual FULL driven by aec_process_linear() +
 *      aec_process_residual()); the windowed ERLE they rely on is kept by the linear stage in every residual mode.
 *   6. THE RESET BRANCH RECONVERGES. When the gate legitimately rejects, the taps sit at the alignment the caller
 *      abandoned; keeping them (a counters-only echo-path-change) leaves the filter subtracting a decorrelated echo copy,
 *      measured at 1.3-1.5x the echo for 300+ hops. Pinned against an aec_reset() twin, whose trajectory the filter-only
 *      reset has to match without the twin's synthesis restart.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aec_view_internal.h"
#include "aec_white_box.h"
#include "aec_test_tuning.h"
#include "utility/simd_aec.h"

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

/* Count bins where the filter's |X_buf|² mirror disagrees with a fresh
 * magnitude pass over the history it mirrors. The two X² partition sums read
 * the mirror, so any row the realign path clears in one array and not the
 * other feeds the gain magnitudes of a history that is now zero -- a defect
 * whose signature is a slightly wrong step size, far too small for the
 * residual bounds below to see, but exact here. */
static long x2_mirror_mismatches(const PBFDAF *p) {
    long bad = 0;
    int part, k;
    float *ref = (float *)malloc((size_t)p->n_freqs * sizeof(float));
    if (!ref) return -1;
    for (part = 0; part < p->n_partitions; ++part) {
        const Complex *X = p->X_buf + (size_t)part * p->n_freqs;
        const float *cache = p->x2_cache + (size_t)part * p->n_freqs;
        sk_cmag2_np_f32(X, ref, p->n_freqs);
        for (k = 0; k < p->n_freqs; ++k)
            if (memcmp(&ref[k], &cache[k], sizeof(float)) != 0) ++bad;
    }
    free(ref);
    return bad;
}
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

/* One hop of the linear residual. A residual-OFF instance is driven by aec_process(), whose output is the linear residual;
 * a FULL instance by the split pair (the library applies no gain there), whose linear residual is raw_output. */
static void process_hop(Aec* a, const float* mic, const float* far, float* out)
{
    int st;
    if (a->cfg.residual == AEC_RESIDUAL_FULL) {
        st = aec_process_linear(a, mic, far);
        if (st == AEC_OK) st = aec_process_residual(a, NULL);
        if (st == AEC_OK) memcpy(out, a->raw_output, (size_t)HOP * sizeof(float));
    } else {
        st = aec_process(a, mic, far, out);
    }
    if (st != AEC_OK) { fprintf(stderr, "FAIL: process status %d\n", st); g_failures++; }
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
    process_hop(a, mic, far, out);
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
    process_hop(a, zero, zero, out);
}

/* res_context=1 runs the post-filter chain through the split entry points (residual FULL, no gain applied): the emitted
 * samples are still exactly the linear residual (raw_output). That is the seam configuration a multi-lane wrapper -- the
 * caller this API exists for -- runs. */
static Aec* make_external_ctx(int warm_transfer, int res_context)
{
    AecTestCfg cfg;
    aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, SR);
    cfg.c.fft_size = FFT;
    cfg.c.residual = res_context ? AEC_RESIDUAL_FULL : AEC_RESIDUAL_OFF;   /* output IS the linear residual */
    cfg.c.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
    cfg.t.delay_acquire_warm_transfer = warm_transfer;
    return aec_tcfg_create(&cfg);
}

static Aec* make_external(int warm_transfer)
{
    return make_external_ctx(warm_transfer, 0);
}

int main(void)
{
    Aec *aec = NULL, *twin = NULL, *back = NULL, *soft = NULL;
    float out[HOP];
    int h, rc;
    float resid, worst, echo;

    scene_init();

    /* --- Row 1: warm transfer holds cancellation across raw -> aligned. */
    rc = ((aec = make_external(1)) ? 0 : -1);
    CHECK(rc == 0 && aec_hop_size(aec) == HOP, "external instance: create");
    resid = 0.0f;
    for (h = 0; h < HOPS; ++h) resid = run_hop(aec, h, 0, out);
    echo = echo_rms(HOPS - 1);
    CHECK(resid < 0.5f * echo, "raw-far convergence before the realign");
    rc = aec_apply_external_realign(aec, D - 0);
    CHECK(rc == 1, "realign(+D) takes the warm tap-transfer path");
    worst = 0.0f;
    for (h = HOPS; h < HOPS + TAIL; ++h) {
        resid = run_hop(aec, h, D, out);
        if (resid > worst) worst = resid;
    }
    CHECK(worst < 0.5f * echo_rms(HOPS),
          "no echo re-exposure after the warm realign");

    /* --- Row 2: the defect reproduces on a twin with aec_reset(). */
    rc = ((twin = make_external(1)) ? 0 : -1);
    CHECK(rc == 0, "twin instance: create");
    for (h = 0; h < HOPS; ++h) run_hop(twin, h, 0, out);
    aec_reset(twin);
    worst = 0.0f;
    for (h = HOPS; h < HOPS + TAIL; ++h) {
        resid = run_hop(twin, h, D, out);
        if (resid > worst) worst = resid;
    }
    CHECK(worst >= 0.8f * echo_rms(HOPS),
          "aec_reset() control re-exposes the echo (defect reproduces)");

    /* --- Row 3: signed shift, aligned -> raw (delta < 0). */
    rc = ((back = make_external(1)) ? 0 : -1);
    CHECK(rc == 0, "signed-shift instance: create");
    for (h = 0; h < HOPS; ++h) run_hop(back, h, D, out);
    resid = run_hop(back, HOPS, D, out);
    CHECK(resid < 0.5f * echo_rms(HOPS), "aligned-far convergence at lag 0");
    rc = aec_apply_external_realign(back, 0 - D);
    CHECK(rc == 1, "realign(-D) takes the warm tap-transfer path");
    /* Asserted HERE, before the refill hops: the retard branch is the only
     * place the far history is wiped mid-stream, and the frontend rewrites one
     * mirror row per hop, so a few hops of processing would repair the very
     * inconsistency this is looking for. */
    CHECK(x2_mirror_mismatches(&back->main_filter.base) == 0,
          "retard realign leaves the main filter's |X|² mirror consistent with X_buf");
    CHECK(!back->has_shadow ||
          x2_mirror_mismatches(&back->shadow_filter) == 0,
          "retard realign leaves the shadow filter's |X|² mirror consistent with X_buf");
    /* The retard direction (delta < 0) clears the far spectra, so the echo is exposed until the
     * partition ring refills past the shifted response. The general bound is ceil((|delta| + tap
     * offset)/hop) + 1 hops and the residual inside the window is NOT bounded by the echo (see
     * aec_apply_external_realign in aec.h). This scene is the mildest corner (|delta|=300, response
     * at tap 0): two hops and 1.05x is what it costs and is what is pinned. The second check
     * matters in general: cancellation resumes OUTRIGHT once the ring has refilled, against the
     * reset control that stays exposed for dozens of hops. */
    worst = 0.0f;
    for (h = HOPS + 1; h < HOPS + 3; ++h) {
        resid = run_hop(back, h, 0, out);
        if (resid > worst) worst = resid;
    }
    CHECK(worst <= 1.05f * echo_rms(HOPS),
          "negative-delta transient is bounded by the echo itself");
    worst = 0.0f;
    for (h = HOPS + 3; h < HOPS + TAIL; ++h) {
        resid = run_hop(back, h, 0, out);
        if (resid > worst) worst = resid;
    }
    CHECK(worst < 0.15f * echo_rms(HOPS),
          "negative-delta realign cancels outright after the 2-hop refill");

    /* --- Row 4: API contract + reset path. */
    CHECK(aec_apply_external_realign(aec, 0) == 0, "delta 0 is a no-op");
    CHECK(aec_apply_external_realign(NULL, D) == -1, "NULL is rejected");
    {
        AecConfig cfg;
        Aec *matched = NULL;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, SR);
        cfg.fft_size = FFT;
        rc = ((matched = aec_create(&cfg)) ? 0 : -1);
        CHECK(rc == 0 &&
              aec_apply_external_realign(matched, D) == -1,
              "MATCHED mode is rejected");
        aec_destroy(matched);
    }
    rc = ((soft = make_external(0)) ? 0 : -1);          /* warm transfer disabled */
    CHECK(rc == 0, "soft instance: create");
    for (h = 0; h < HOPS; ++h) run_hop(soft, h, D, out);
    {
        AecLinearViewFull c0, c1;
        c0.struct_size = c1.struct_size = sizeof c0;
        CHECK(aec_get_linear_full(soft, &c0) == AEC_OK, "linear view before the realign");
        CHECK(aec_apply_external_realign(soft, 64) == 0,
              "gate off falls to the reset path");
        CHECK(aec_get_linear_full(soft, &c1) == AEC_OK, "linear view after the realign");
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
        Aec *gap = NULL;
        AecLinearViewFull g0, g1;
        g0.struct_size = g1.struct_size = sizeof g0;
        rc = ((gap = make_external_ctx(1, 1)) ? 0 : -1);
        CHECK(rc == 0, "silent-gap instance: create");
        for (h = 0; h < HOPS; ++h) resid = run_hop(gap, h, 0, out);
        CHECK(resid < 0.5f * echo_rms(HOPS - 1),
              "silent-gap: raw-far convergence before the silence");
        for (h = 0; h < GAP; ++h) run_silent_hop(gap, out);
        CHECK(aec_get_linear_full(gap, &g0) == AEC_OK, "silent-gap: linear view before the realign");
        rc = aec_apply_external_realign(gap, D - 0);
        CHECK(rc == 1,
              "silent-gap realign still takes the warm tap-transfer path");
        CHECK(aec_get_linear_full(gap, &g1) == AEC_OK, "silent-gap: linear view after the realign");
        CHECK(g1.generation != g0.generation,
              "warm outcome advances the alignment generation");
        worst = 0.0f;
        for (h = HOPS + GAP; h < HOPS + GAP + OBS; ++h) {
            resid = run_hop(gap, h, D, out);
            if (resid / echo_rms(h) > worst) worst = resid / echo_rms(h);
        }
        CHECK(worst < 0.5f,
              "silent-gap warm realign keeps the residual under half the echo");
        aec_destroy(gap);
    }

    /* --- Row 6: the reset branch reconverges like an aec_reset() twin.
     * Warm transfer off forces the branch on a genuinely converged filter, so
     * the taps really are parked at the abandoned alignment. The twin gets the
     * identical scene and an aec_reset() at the identical hop: it is the
     * trajectory a full restart achieves, and the filter-only reset has to
     * match it while leaving synthesis alone. */
    {
        Aec *sres = NULL, *sref = NULL;
        static const int CK[] = { 5, 10, 20, 40, 80, 120 };
        float rs[TRAJ + 1], rt[TRAJ + 1];
        size_t ci;
        char msg[96];
        CHECK((sres = make_external_ctx(0, 1)) != NULL &&
              (sref = make_external_ctx(0, 1)) != NULL, "reset-trajectory pair: create");
        for (h = 0; h < HOPS; ++h) { run_hop(sres, h, 0, out); run_hop(sref, h, 0, out); }
        resid = run_hop(sres, HOPS, 0, out);
        (void)run_hop(sref, HOPS, 0, out);
        CHECK(resid < 0.1f * echo_rms(HOPS),
              "reset-trajectory: converged before the realign");
        CHECK(aec_apply_external_realign(sres, D - 0) == 0,
              "reset-trajectory: warm transfer off takes the reset path");
        aec_reset(sref);
        for (h = 1; h <= TRAJ; ++h) {
            int hh = HOPS + h;
            float e = echo_rms(hh);
            rs[h] = run_hop(sres, hh, D, out) / e;
            rt[h] = run_hop(sref, hh, D, out) / e;
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
        aec_destroy(sres);
        aec_destroy(sref);
    }

    /* --- Row 7: the reset path leaves no coarse/leakage sustain evidence behind. Those counters accumulate evidence about the
     * taps (how the refined error compares to the coarse one, how many bins sit on the leakage-diverged branch), so carrying
     * them across a tap wipe lets the abandoned alignment's history drive the new filter's detectors.
     * This is a STATE row on purpose: the contamination is invisible in the audio. After the wipe erl_per_bin is all zeros,
     * so both arms of the H_error leakage refresh add the same zero and the output is bit-identical whether or not the
     * counters were cleared. The evidence is the counter itself, plus the invariant that a counter starting at zero and
     * stepping by at most one per hop cannot exceed the hop index, which separates a cleared counter from one still draining
     * its pre-realign value. Only leakage_div_sustained_counter is dirty here (the pre-call check ensures that):
     * poor_coarse_counter cannot be observed non-zero on this grid, since its threshold is one hop and it self-clears within
     * the hop that sets it, and coarse_reset_hangover only arms behind that fire; those two carry no can-fail evidence. */
    {
        Aec *ctr = NULL;
        int k;
        CHECK((ctr = make_external_ctx(0, 1)) != NULL, "counter-hygiene instance: create");
        for (h = 0; h <= HOPS; ++h) run_hop(ctr, h, 0, out);
        CHECK(ctr->leakage_div_sustained_counter > 0,
              "counter-hygiene: the scene really does dirty the leakage counter");
        CHECK(aec_apply_external_realign(ctr, D - 0) == 0,
              "counter-hygiene: takes the reset path");
        CHECK(ctr->poor_coarse_counter == 0 &&
              ctr->coarse_reset_hangover == 0 &&
              ctr->leakage_div_sustained_counter == 0,
              "counter-hygiene: coarse/leakage counters cleared by the realign");
        char msg[112];
        for (k = 1; k <= 3; ++k) {
            run_hop(ctr, HOPS + k, D, out);
            snprintf(msg, sizeof msg,
                     "counter-hygiene +%d: leakage counter %d <= %d (no pre-realign drain)",
                     k, ctr->leakage_div_sustained_counter, k);
            CHECK(ctr->leakage_div_sustained_counter <= k, msg);
        }
        aec_destroy(ctr);
    }

    aec_destroy(aec);
    aec_destroy(twin);
    aec_destroy(back);
    aec_destroy(soft);
    free(g_far_hist);
    if (g_failures) { fprintf(stderr, "%d failure(s)\n", g_failures); return 1; }
    printf("test_external_realign: ALL PASS\n");
    return 0;
}
