/* baseline2_signal.h -- deterministic stimulus generator for the second scene sets (test_split_entry2/3.c, test_aec_array2.c).
 * Header-only, plain C, no libm, no AEC API (the LCG step, the syllabic modulation and the FNV-1a hash are split_signal.h's):
 * every includer regenerates bit-identical inputs.
 *
 * A stimulus (G2Stim) is a far end made of coloured LCG-noise bursts (5 ms ramps, 5 Hz syllabic modulation), optionally
 * driven into hard clipping at +-1 inside "loud" segments; a microphone made of the far end through a 5-tap echo path whose
 * bulk delay follows a schedule (steps in samples at 16 kHz, scaled to the rate), near-end bursts and low noise; and
 * optional hard clipping of the microphone. Everything is float add/multiply in a fixed order (build with
 * -ffp-contract=off) over integer LCG streams.
 *
 * g2_mono_signal() fills whole-scenario arrays (n samples) in one pass; hop h is samples [h * hop, (h + 1) * hop). */
#ifndef AEC_TEST_GOLDEN2_SIGNAL_H
#define AEC_TEST_GOLDEN2_SIGNAL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "split_signal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define G2_MAXSEG 8
#define G2_MAXDLY 12

typedef struct { int t0, t1; } G2Seg;            /* [t0, t1) ms */
typedef struct { int t_ms; int d16; } G2Dly;     /* from t_ms the echo bulk delay is d16 samples at 16 kHz (scaled to the rate) */

typedef struct G2Stim {
    int   n_far;  G2Seg far[G2_MAXSEG];  float far_amp;
    int   n_loud; G2Seg loud[G2_MAXSEG]; float loud_amp;      /* far amplitude inside: hard-clipped at +-1 */
    float loud_echo_gain;                                      /* echo multiplier inside the loud segments (0 = 1) */
    int   n_near; G2Seg near[G2_MAXSEG]; float near_amp;
    int   n_dly;  G2Dly dly[G2_MAXDLY];                        /* schedule, ascending t_ms, first entry t_ms = 0 */
    float pre_gain; int pre_adv16;                             /* pre-echo: an extra path pre_adv16 samples (at 16 kHz) earlier than the bulk delay, with this gain (0 = none) */
    int   flip_ms; int fd1_16, fd2_16; float fg[4];            /* two-path scene (replaces the bulk delay): paths fd1 / fd2 (16 kHz samples), gains fg[0..1] then fg[2..3], swapped every flip_ms */
    int   clip_mic;                                            /* 1: the microphone is hard-clipped at +-1 */
    float mic_gain;                                            /* microphone gain before the clip (0 = 1) */
    int   near_ramp_ms;                                        /* near-end burst ramp (0 = 5 ms) */
} G2Stim;

#define G2_FIR_MAX 8

static inline uint32_t g2_lcg(uint32_t* s) { *s = ss_lcg_step(*s); return *s; }
static inline float g2_white(uint32_t* s) { return (float)(g2_lcg(s) >> 8) * (1.0f / 8388608.0f) - 1.0f; }

static inline float g2_burst_env(const G2Seg* seg, int nseg, int sr, int64_t n, int ramp_ms) {
    float best = 0.0f;
    int64_t ramp = (int64_t)sr * (ramp_ms > 0 ? ramp_ms : 5) / 1000;
    if (ramp < 1) ramp = 1;
    for (int k = 0; k < nseg; ++k) {
        int64_t s = (int64_t)seg[k].t0 * sr / 1000, e = (int64_t)seg[k].t1 * sr / 1000;
        if (n < s || n >= e) continue;
        float up = (float)(n - s + 1) / (float)ramp, dn = (float)(e - n) / (float)ramp;
        float v = up < dn ? up : dn;
        if (v > 1.0f) v = 1.0f;
        if (v > best) best = v;
    }
    return best;
}

static inline int g2_in_segs(const G2Seg* seg, int nseg, int sr, int64_t n) {
    for (int k = 0; k < nseg; ++k)
        if (n >= (int64_t)seg[k].t0 * sr / 1000 && n < (int64_t)seg[k].t1 * sr / 1000) return 1;
    return 0;
}

#define g2_syllable ss_syllable

static inline float g2_clip1(float x) { return x > 1.0f ? 1.0f : (x < -1.0f ? -1.0f : x); }

/* Echo bulk delay in samples at time sample n. */
static inline int g2_delay_at(const G2Stim* st, int sr, int64_t n) {
    int d16 = st->n_dly ? st->dly[0].d16 : 0;
    for (int k = 0; k < st->n_dly; ++k)
        if (n >= (int64_t)st->dly[k].t_ms * sr / 1000) d16 = st->dly[k].d16;
    return (int)((int64_t)d16 * sr / 16000);
}

/* ref[0..n): the far end as played (clipped); mic[0..n): the microphone. */
static inline void g2_mono_signal(const G2Stim* st, int sr, int n, float* ref, float* mic) {
    static const float h_far[G2_FIR_MAX] = { 1.0f, 0.8f, 0.55f, 0.3f, 0.12f, -0.05f, -0.1f, -0.06f };
    static const float h_near[6] = { 1.0f, -0.6f, 0.3f, -0.15f, 0.08f, -0.04f };
    static const int   off[5] = { 0, 5, 11, 19, 30 };
    static const float eg[5] = { 0.35f, 0.18f, -0.10f, 0.06f, -0.03f };
    uint32_t sf = 0x13579BDFu, sn = 0x2468ACE0u, sz = 0x0F1E2D3Cu;
    float wf[G2_FIR_MAX] = {0}, wn[6] = {0};
    const float lg = st->loud_echo_gain > 0.0f ? st->loud_echo_gain : 1.0f;
    const float mg = st->mic_gain > 0.0f ? st->mic_gain : 1.0f;
    for (int i = 0; i < n; ++i) {
        float cf = 0.0f;
        for (int t = G2_FIR_MAX - 1; t > 0; --t) wf[t] = wf[t - 1];
        wf[0] = g2_white(&sf);
        for (int t = 0; t < G2_FIR_MAX; ++t) cf += h_far[t] * wf[t];
        float env = g2_burst_env(st->far, st->n_far, sr, i, 5);
        float amp = g2_in_segs(st->loud, st->n_loud, sr, i) ? st->loud_amp : st->far_amp;
        float x = (env > 0.0f) ? amp * env * g2_syllable(sr, i, 0) * cf : 0.0f;
        ref[i] = g2_clip1(x);
    }
    for (int i = 0; i < n; ++i) {
        float cn = 0.0f, e = 0.0f;
        int d = g2_delay_at(st, sr, i);
        for (int t = 5; t > 0; --t) wn[t] = wn[t - 1];
        wn[0] = g2_white(&sn);
        for (int t = 0; t < 6; ++t) cn += h_near[t] * wn[t];
        if (st->flip_ms > 0) {
            const int ph = (int)(((int64_t)i * 1000 / sr / st->flip_ms) % 2);
            const int dd[2] = { (int)((int64_t)st->fd1_16 * sr / 16000), (int)((int64_t)st->fd2_16 * sr / 16000) };
            for (int q = 0; q < 2; ++q)
                for (int t = 0; t < 5; ++t) {
                    int j = i - dd[q] - off[t];
                    e += st->fg[ph * 2 + q] * eg[t] * (j >= 0 ? ref[j] : 0.0f);
                }
        } else {
            for (int t = 0; t < 5; ++t) {
                int j = i - d - off[t];
                e += eg[t] * (j >= 0 ? ref[j] : 0.0f);
            }
        }
        if (st->pre_gain != 0.0f) {
            int dp = d - (int)((int64_t)st->pre_adv16 * sr / 16000);
            for (int t = 0; t < 5; ++t) {
                int j = i - dp - off[t];
                e += st->pre_gain * eg[t] * (j >= 0 && dp >= 0 ? ref[j] : 0.0f);
            }
        }
        if (g2_in_segs(st->loud, st->n_loud, sr, i)) e *= lg;
        float envn = g2_burst_env(st->near, st->n_near, sr, i, st->near_ramp_ms);
        float nr = (envn > 0.0f) ? st->near_amp * envn * g2_syllable(sr, i, sr / 15) * cn : 0.0f;
        float m = mg * (e + nr) + 0.002f * g2_white(&sz);
        mic[i] = st->clip_mic ? g2_clip1(m) : m;
    }
}

/* --- stimulus building blocks shared by the single-channel and the array cases ------------------------------------------- */
static inline void g2_seg(G2Seg* s, int* n, int t0, int t1) { s[*n].t0 = t0; s[*n].t1 = t1; (*n)++; }
static inline void g2_dly(G2Stim* st, int t_ms, int d16) { st->dly[st->n_dly].t_ms = t_ms; st->dly[st->n_dly].d16 = d16; st->n_dly++; }

/* Two slow-onset near-end bursts over a converged filter (the first one absorbs the echo-path-change detector, the second one is
 * a double talk that the detector scores), then a loud clipped far end and clipped microphone, then near-end only. */
static inline void g2_stim_satdt(G2Stim* st) {
    memset(st, 0, sizeof *st);
    g2_seg(st->far, &st->n_far, 100, 7000); st->far_amp = 0.12f;
    g2_seg(st->loud, &st->n_loud, 5200, 6000); st->loud_amp = 3.0f; st->loud_echo_gain = 3.0f;
    g2_seg(st->near, &st->n_near, 2000, 3200); g2_seg(st->near, &st->n_near, 3600, 4800);
    g2_seg(st->near, &st->n_near, 7100, 8100);
    st->near_amp = 0.5f; st->near_ramp_ms = 600;
    g2_dly(st, 0, 640);
    st->clip_mic = 1;
}
/* Continuous far end, the echo path jumps 40 -> 90 ms and back; near end optional around the jumps. */
/* Continuous far end, two echo paths (32 and 48 ms) whose gains swap every 2.4 s: the shared estimate moves between them, forward
 * and backward. */
static inline void g2_stim_flip(G2Stim* st) {
    memset(st, 0, sizeof *st);
    g2_seg(st->far, &st->n_far, 100, 11900); st->far_amp = 0.12f;
    st->flip_ms = 2400; st->fd1_16 = 512; st->fd2_16 = 768;
    st->fg[0] = 0.50f; st->fg[1] = 0.55f; st->fg[2] = 0.55f; st->fg[3] = 0.50f;
}

static inline void g2_stim_jump(G2Stim* st, int near_end) {
    memset(st, 0, sizeof *st);
    g2_seg(st->far, &st->n_far, 100, 4000); g2_seg(st->far, &st->n_far, 4100, 8800);
    st->far_amp = 0.12f;
    if (near_end) {
        g2_seg(st->near, &st->n_near, 2000, 3400); g2_seg(st->near, &st->n_near, 4600, 5800);
        st->near_amp = 0.25f;
    }
    g2_dly(st, 0, 640); g2_dly(st, 3000, 1440); g2_dly(st, 5500, 640);
}

/* The echo path jumps 40 -> 90 ms and back; with `near_end` a slow-onset near-end burst keeps the near-end-recent latch up
 * when the relock is decided (soft path), without one the relock is the hard path. */
static inline void g2_stim_bad(G2Stim* st, int near_end) {
    memset(st, 0, sizeof *st);
    g2_seg(st->far, &st->n_far, 100, 10800); st->far_amp = 0.12f;
    if (near_end) { g2_seg(st->near, &st->n_near, 4700, 5900); st->near_amp = 0.5f; st->near_ramp_ms = 600; }
    g2_dly(st, 0, 640); g2_dly(st, 4000, 1440); g2_dly(st, 8200, 640);
}

/* --- FNV-1a 64 (split_signal.h's) ------------------------------------------------------------------------------------ */
#define G2_FNV_BASIS SPLIT_FNV_BASIS
#define g2_fnv_u32   split_fnv_u32
#define g2_f32_bits  split_f32_bits
#define g2_fnv_f32   split_fnv_f32
/* Running hash over n floats; a NULL pointer folds a fixed marker so that "absent" differs from any value. */
static inline uint64_t g2_fnv_f32v(uint64_t h, const float* p, int n) {
    if (!p) return g2_fnv_u32(h, 0xFFFFFFFFu);
    for (int i = 0; i < n; ++i) h = g2_fnv_f32(h, p[i]);
    return h;
}
static inline uint64_t g2_fnv_cxv(uint64_t h, const float* p_ri, int n) {      /* interleaved re,im pairs */
    return g2_fnv_f32v(h, p_ri, 2 * n);
}

#ifdef __cplusplus
}
#endif
#endif /* AEC_TEST_GOLDEN2_SIGNAL_H */
