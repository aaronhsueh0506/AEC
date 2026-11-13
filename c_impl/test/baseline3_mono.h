/* baseline3_mono.h -- third single-channel scene set: bad-relock judgments (aec_judge_delay_shift: AEC_DELAY_SHIFT_JUDGMENT_HOPS hops
 * after a confirmed shift, error power below near power and windowed ERLE more than 1 dB below the value read at the shift reverts
 * it) that free-running scenes produce by themselves. Case table and expectations of test/test_split_entry3.c; the runner, views
 * and hashing are baseline2_mono.h, so the includer provides what that header lists.
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
 * No field of the instance is written. */
#ifndef AEC_TEST_GOLDEN3_MONO_H
#define AEC_TEST_GOLDEN3_MONO_H

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

#endif /* AEC_TEST_GOLDEN3_MONO_H */
