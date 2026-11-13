/* array_signal.h -- deterministic four-microphone test signal, weight sets, case table and hash helpers for the array
 * instance (header-only, plain C, no libm, no AEC API).
 *
 * Every includer must regenerate bit-identical inputs (a baseline captured from one build is compared with another), so
 * everything is a pure function of (sample_rate, hop_size, hop_index) and the case table below.
 *
 * Signal: split_signal_fill() gives the far-end hop `ref` and a one-channel microphone hop m0 (echo of the far end with a
 * 40 ms bulk delay and a short tail + near-end bursts + low noise). Channel c is then
 *     mic_c[n] = g_c * m0[n - d_c] + g2_c * m0[n - d2_c] + 0.001 * white_c(n)
 * with the lags/gains of ARR_TAPS (different per channel; the lags are in samples at every rate), so the four channels carry
 * the same echo and near-end bursts through different small delays and gains, plus independent noise.
 */
#ifndef AEC_TEST_ARRAY_SIGNAL_H
#define AEC_TEST_ARRAY_SIGNAL_H

#include <stdio.h>
#include "split_signal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ARR_CH       4
#define ARR_HIST      64      /* >= largest lag + 1 */
#define ARR_ECHO_MS   40      /* split_signal's bulk delay */

typedef struct { int d, d2; float g, g2; uint32_t seed; } ArrTap;
static const ArrTap ARR_TAPS[ARR_CH] = {
    {  0,  5,  1.00f,  0.15f, 0x11111111u },
    {  3,  9,  0.80f, -0.10f, 0x22222222u },
    {  7, 14, -0.60f,  0.20f, 0x33333333u },
    { 12, 20,  0.45f,  0.10f, 0x44444444u },
};

#define ARR_SCENE_BLOCK_MS 2400   /* the scene's path balance flips every 2.4 s */
#define ARR_SCENE_MS       12000
#define ARR_SCENE_HIST     (4096 + 1024 + 64)

typedef struct {
    int      sr, hop, next, scene;
    float    hist[ARR_HIST + SPLIT_SIGNAL_MAX_HOP];   /* m0 history: hist[ARR_HIST + j] is the current hop, j >= 0 */
    /* scene: continuous noise far end and a two-path echo, 32 ms and 48 ms, with nearly equal gains (0.50/0.55) whose
     * balance flips every ARR_SCENE_BLOCK_MS. The shared estimator moves between the two paths: forward moves, and backward
     * moves that the quarantine (scene 1) holds while the capture-proxy lane still cancels the other path. */
    uint32_t xs;
    float    fhist[ARR_SCENE_HIST + SPLIT_SIGNAL_MAX_HOP];
} ArraySig;

static inline void array_sig_init(ArraySig* s, int sr, int hop, int scene) {
    memset(s, 0, sizeof *s);
    s->sr = sr; s->hop = hop; s->next = 0; s->scene = scene; s->xs = 0x1234567u;
}

/* Fills ref (far end) and m0 (the one-channel echo + near mix before the per-channel taps) for the next hop. */
static inline int array_sig_base(ArraySig* s, float* ref, float* m0) {
    const int hop = s->hop;
    if (!s->scene) return split_signal_fill(s->sr, hop, s->next, ref, m0);
    {
        const int64_t block = (int64_t)ARR_SCENE_BLOCK_MS * s->sr / 1000 / hop;
        const int d1 = s->sr * 32 / 1000, d2 = s->sr * 48 / 1000;
        const int ph = (int)(((int64_t)s->next / block) % 2);
        const float g1 = ph ? 0.55f : 0.50f, g2 = ph ? 0.50f : 0.55f;
        memmove(s->fhist, s->fhist + hop, (size_t)ARR_SCENE_HIST * sizeof(float));
        for (int j = 0; j < hop; ++j) {
            s->xs ^= s->xs << 13; s->xs ^= s->xs >> 17; s->xs ^= s->xs << 5;
            s->fhist[ARR_SCENE_HIST + j] = 0.25f * (((float)(s->xs >> 8) * (1.0f / 16777216.0f)) - 0.5f);
        }
        for (int j = 0; j < hop; ++j) {
            ref[j] = s->fhist[ARR_SCENE_HIST + j];
            m0[j] = g1 * s->fhist[ARR_SCENE_HIST + j - d1] + g2 * s->fhist[ARR_SCENE_HIST + j - d2];
        }
    }
    return 0;
}

/* Fills mics (interleaved [hop][ARR_CH]) and ref (far end) for the next hop. Returns 0, or -1. */
static inline int array_sig_next(ArraySig* s, float* mics, float* ref) {
    float m0[SPLIT_SIGNAL_MAX_HOP], w[SPLIT_SIGNAL_MAX_HOP];
    const int hop = s->hop;
    if (array_sig_base(s, ref, m0) != 0) return -1;
    /* slide the history: keep the last ARR_HIST samples, append this hop */
    memmove(s->hist, s->hist + hop, (size_t)ARR_HIST * sizeof(float));
    memcpy(s->hist + ARR_HIST, m0, (size_t)hop * sizeof(float));
    for (int c = 0; c < ARR_CH; ++c) {
        ss_white(ARR_TAPS[c].seed, (int64_t)s->next * hop, hop, w);
        for (int j = 0; j < hop; ++j) {
            const float a = s->hist[ARR_HIST + j - ARR_TAPS[c].d];
            const float b = s->hist[ARR_HIST + j - ARR_TAPS[c].d2];
            mics[j * ARR_CH + c] = ARR_TAPS[c].g * a + ARR_TAPS[c].g2 * b + 0.001f * w[j];
        }
    }
    s->next++;
    return 0;
}

/* --- weights: channel-major Complex[ARR_CH][n] ---------------------------------------------------------------------- */
enum { ARR_W_NONE = -1, ARR_W_EQUAL = 0, ARR_W_UNEQUAL = 1 };

static inline void array_weights(int kind, int n, Complex* w) {
    for (int c = 0; c < ARR_CH; ++c) {
        for (int k = 0; k < n; ++k) {
            Complex v = { 0.0f, 0.0f };
            if (kind == ARR_W_EQUAL) {
                v.r = 0.25f - 0.0001f * (float)k;
                v.i = 0.02f + 0.00005f * (float)k;
            } else if (kind == ARR_W_UNEQUAL) {
                if (c == 0)      { v.r = 0.50f - 0.0002f * (float)k; v.i = 0.10f; }
                else if (c == 1) { v.r = 0.30f; v.i = -0.20f + 0.0001f * (float)k; }
                else if (c == 2) { v.r = 0.0f; v.i = 0.0f; }                          /* a zero-weight lane */
                else             { v.r = 0.10f + 0.0003f * (float)(k % 7); v.i = 0.30f - 0.0002f * (float)k; }
            }
            w[c * n + k] = v;
        }
    }
}

/* --- case table ------------------------------------------------------------------------------------------------------ */
enum { ARR_MATCHED = 0, ARR_FIXED = 1, ARR_EXTERNAL = 2 };

typedef struct {
    int         sr, fft;
    int         mode;       /* ARR_MATCHED / ARR_FIXED / ARR_EXTERNAL */
    int         post;       /* 1: fusion and gain (residual FULL); 0: lanes only (STATE) */
    int         wkind;      /* ARR_W_* (post only) */
    int         reset_at;   /* >0: the instance is reset after this many hops */
    int         duty;       /* 1: delay_est_period_s = 1.0 and the backward quarantine on (0.5 s) */
    int         scene;      /* 1: admission scene (quarantine on, 0.5 s), 2: the same with the quarantine off */
    char        name[40];
} ArrCase;

#define ARR_MAX_CASES 64

static inline int array_cases(ArrCase* out) {
    static const int sr_fft[3][2] = { {16000, 256}, {16000, 512}, {48000, 1024} };
    static const char* const mname[3] = { "MATCHED", "FIXED", "EXTERNAL" };
    int n = 0;
    for (int g = 0; g < 3; ++g) {
        const int sr = sr_fft[g][0], fft = sr_fft[g][1];
        const int hops = split_signal_n_hops(sr, fft / 2);
        for (int m = 0; m < 3; ++m) {
            for (int v = 0; v < 3; ++v) {         /* v: 0 = lanes only, 1 = equal weights, 2 = unequal weights */
                ArrCase c; memset(&c, 0, sizeof c);
                c.sr = sr; c.fft = fft; c.mode = m;
                c.post = (v != 0); c.wkind = (v == 0) ? ARR_W_NONE : (v == 1 ? ARR_W_EQUAL : ARR_W_UNEQUAL);
                snprintf(c.name, sizeof c.name, "%d/%d_%s_%s", sr, fft, mname[m],
                         v == 0 ? "front" : (v == 1 ? "equalW" : "unequalW"));
                out[n++] = c;
            }
        }
        {
            ArrCase c; memset(&c, 0, sizeof c);
            c.sr = sr; c.fft = fft; c.mode = ARR_MATCHED; c.post = 1; c.wkind = ARR_W_EQUAL; c.reset_at = hops / 2;
            snprintf(c.name, sizeof c.name, "%d/%d_MATCHED_equalW_reset", sr, fft);
            out[n++] = c;
        }
        {
            ArrCase c; memset(&c, 0, sizeof c);
            c.sr = sr; c.fft = fft; c.mode = ARR_MATCHED; c.post = 1; c.wkind = ARR_W_UNEQUAL; c.duty = 1;
            snprintf(c.name, sizeof c.name, "%d/%d_MATCHED_unequalW_duty", sr, fft);
            out[n++] = c;
        }
        for (int sc = 1; sc <= 2; ++sc) {
            ArrCase c; memset(&c, 0, sizeof c);
            c.sr = sr; c.fft = fft; c.mode = ARR_MATCHED; c.post = 1; c.wkind = ARR_W_UNEQUAL; c.scene = sc;
            snprintf(c.name, sizeof c.name, "%d/%d_MATCHED_unequalW_scene%s", sr, fft, sc == 1 ? "_q" : "");
            out[n++] = c;
        }
    }
    return n;
}

static inline int array_case_hops(const ArrCase* c) {
    const int hop = c->fft / 2;
    if (c->scene) return (int)((int64_t)ARR_SCENE_MS * c->sr / 1000 / hop);
    return split_signal_n_hops(c->sr, hop);
}

/* Heavy lines (spectra, gains) of an admission scene are hashed every ARR_SCENE_STRIDE-th hop (and on every hop whose
 * alignment changed); the small front_meta line is hashed every hop. */
#define ARR_SCENE_STRIDE 8

/* --- incremental FNV-1a 64 over several buffers ------------------------------------------------------------------------ */
static inline uint64_t ah_u32(uint64_t h, uint32_t v) { return split_fnv_u32(h, v); }
static inline uint64_t ah_f32(uint64_t h, float v) { return split_fnv_f32(h, v); }
static inline uint64_t ah_f32v(uint64_t h, const float* p, int n) {
    if (!p) return split_fnv_u32(h, 0xFFFFFFFFu);
    for (int i = 0; i < n; ++i) h = split_fnv_f32(h, p[i]);
    return h;
}
static inline uint64_t ah_cxv(uint64_t h, const Complex* p, int n) {
    if (!p) return split_fnv_u32(h, 0xFFFFFFFFu);
    for (int i = 0; i < n; ++i) { h = split_fnv_f32(h, p[i].r); h = split_fnv_f32(h, p[i].i); }
    return h;
}

/* The comfort-noise-free synthetic denoiser gain of the apply lines. */
static inline void array_extra_gain(float* gx, int n, int hop_index) {
    /* no libm: a triangle wave in [0.35, 0.95] that moves with k and the hop */
    for (int k = 0; k < n; ++k) {
        int ph = (k * 3 + hop_index * 5) % 40;
        int tri = ph < 20 ? ph : 40 - ph;
        gx[k] = 0.35f + 0.6f * ((float)tri / 20.0f);
    }
}

#ifdef __cplusplus
}
#endif
#endif /* AEC_TEST_ARRAY_SIGNAL_H */
