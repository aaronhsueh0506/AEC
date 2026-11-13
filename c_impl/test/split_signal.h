/* split_signal.h -- deterministic far-end / microphone test signal and hash helpers (header-only, plain C).
 *
 * Every includer must regenerate bit-identical inputs (a baseline captured from one build is compared with another), so
 * everything here is a pure function of (sample_rate, hop_size, hop_index): no static state, no libm, no <random.h>.
 *
 * Signal (4.0 s per grid; ms times below are absolute, the sample index of a time is ms * sample_rate / 1000):
 *   far  (ref): coloured LCG noise (8-tap FIR), speech-like bursts with 5 ms ramps and a 5 Hz syllabic modulation:
 *               [50,600) [700,1250) [1350,1900) [2000,2600) [2700,3050) ms; silent afterwards.
 *   mic       : far convolved with a fixed echo path (40 ms bulk delay + taps at +0/5/11/19/30 samples, gains
 *               0.35 0.18 -0.10 0.06 -0.03) + near-end bursts + low noise (uniform, +-0.002):
 *               near bursts [2100,2500) ms (double talk) and [2800,3800) ms (double talk until 3050, then near-end only).
 *   Stretches: far-end bursts with a silent near end [50,2100) and [2500,2800) ms (gaps between bursts included); double talk [2100,2500) and [2800,3050) ms;
 *              near-end only [3050,3800) ms; noise floor only [0,50) and [3800,4000) ms.
 *   Amplitude: audio scale, |x| < 0.5 by construction (bound 0.477).
 * Hops are hop_size samples; hop h covers samples [h * hop_size, (h + 1) * hop_size). split_signal_n_hops() is the number
 * of whole hops in 4.0 s (16000/hop 128: 500; 16000/hop 256: 250; 48000/hop 512: 375).
 *
 * The three noise streams are 32-bit LCGs (x' = 1664525 x + 1013904223) with fixed seeds, evaluated at an arbitrary sample
 * index by O(log n) jump-ahead, so any hop can be generated without generating the ones before it.
 *
 * Hashes: FNV-1a 64 over the little-endian bytes of the 32-bit patterns (float bits / integers), so the value does not
 * depend on host byte order. A NULL pointer hashes to 0 (a real hash is never expected to be 0). */
#ifndef AEC_TEST_SPLIT_SIGNAL_H
#define AEC_TEST_SPLIT_SIGNAL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ac_complex.h"   /* Complex { float r, i; } (also reached through fft_wrapper.h) */

#ifdef __cplusplus
extern "C" {
#endif

#define SPLIT_SIGNAL_MS         4000
#define SPLIT_SIGNAL_MAX_HOP    1024
#define SPLIT_SIGNAL_MAX_RATE   48000

#define SS_FAR_SEED   0x13579BDFu
#define SS_NEAR_SEED  0x2468ACE0u
#define SS_NOISE_SEED 0x0F1E2D3Cu
#define SS_ECHO_DELAY_MS 40
#define SS_ECHO_TAPS     5
#define SS_FIR_MAX       8
/* Largest generated span: hop + bulk delay + last tap + FIR memory. */
#define SS_SPAN_MAX (SPLIT_SIGNAL_MAX_HOP + (SPLIT_SIGNAL_MAX_RATE * SS_ECHO_DELAY_MS / 1000) + 32 + SS_FIR_MAX)

static inline int split_signal_n_hops(int sample_rate, int hop_size) {
    if (sample_rate <= 0 || hop_size <= 0) return 0;
    return (int)(((int64_t)SPLIT_SIGNAL_MS * sample_rate / 1000) / hop_size);
}

/* --- LCG with jump-ahead ------------------------------------------------------------------------------------------ */

static inline uint32_t ss_lcg_step(uint32_t x) { return x * 1664525u + 1013904223u; }

/* x advanced by n steps. */
static inline uint32_t ss_lcg_jump(uint32_t x, uint64_t n) {
    uint32_t a = 1664525u, c = 1013904223u, acc_a = 1u, acc_c = 0u;
    while (n) {
        if (n & 1u) { acc_c = acc_c * a + c; acc_a = acc_a * a; }
        c = (a + 1u) * c;
        a = a * a;
        n >>= 1;
    }
    return acc_a * x + acc_c;
}

/* White noise in [-1, 1): out[m] = w(first + m), w(i) = (state after i + 1 steps from seed >> 8) / 2^23 - 1; w(i) = 0 for
 * i < 0. */
static inline void ss_white(uint32_t seed, int64_t first, int count, float* out) {
    uint32_t st = 0u;
    int have = 0;
    for (int m = 0; m < count; ++m) {
        int64_t idx = first + m;
        if (idx < 0) { out[m] = 0.0f; continue; }
        if (!have) { st = ss_lcg_jump(seed, (uint64_t)idx + 1u); have = 1; }
        else       { st = ss_lcg_step(st); }
        out[m] = (float)(st >> 8) * (1.0f / 8388608.0f) - 1.0f;
    }
}

/* out[j] = sum_t h[t] * w(first + j - t), t ascending. */
static inline void ss_coloured(uint32_t seed, const float* h, int taps, int64_t first, int count, float* out) {
    float w[SS_SPAN_MAX];
    ss_white(seed, first - (taps - 1), count + taps - 1, w);
    for (int j = 0; j < count; ++j) {
        float acc = 0.0f;
        for (int t = 0; t < taps; ++t) acc += h[t] * w[j - t + taps - 1];
        out[j] = acc;
    }
}

/* --- envelopes ----------------------------------------------------------------------------------------------------- */

/* Burst envelope in [0, 1]: max over the segments of min(ramp up, ramp down, 1), 5 ms linear ramps. */
static inline float ss_burst_env(const int (*seg)[2], int nseg, int sample_rate, int64_t n) {
    float best = 0.0f;
    int64_t ramp = sample_rate / 200;
    if (ramp < 1) ramp = 1;
    for (int k = 0; k < nseg; ++k) {
        int64_t s = (int64_t)seg[k][0] * sample_rate / 1000;
        int64_t e = (int64_t)seg[k][1] * sample_rate / 1000;
        if (n < s || n >= e) continue;
        float up = (float)(n - s + 1) / (float)ramp;
        float dn = (float)(e - n) / (float)ramp;
        float v = up < dn ? up : dn;
        if (v > 1.0f) v = 1.0f;
        if (v > best) best = v;
    }
    return best;
}

/* 5 Hz syllabic modulation in [0.55, 1.0] (triangle); shift moves the phase. */
static inline float ss_syllable(int sample_rate, int64_t n, int64_t shift) {
    int64_t period = sample_rate / 5;
    int64_t ph = (n + shift) % period;
    int64_t tri = ph < period / 2 ? ph : period - ph;
    return 0.55f + 0.45f * ((float)(2 * tri) / (float)period);
}

/* --- generator ----------------------------------------------------------------------------------------------------- */

/* Fills ref[0 .. hop_size) with the far-end hop and mic[0 .. hop_size) with the microphone hop of hop index hop_index
 * (>= 0; hops past the 4 s signal are valid and are silence plus noise). Returns 0, or -1 (outputs zeroed) for an
 * unsupported hop_size / sample_rate (hop_size in [1, 1024], sample_rate in [8000, 48000]). */
static inline int split_signal_fill(int sample_rate, int hop_size, int hop_index, float* ref, float* mic) {
    static const int ss_far_segments_ms[5][2] = {
        {50, 600}, {700, 1250}, {1350, 1900}, {2000, 2600}, {2700, 3050}
    };
    static const int ss_near_segments_ms[2][2] = { {2100, 2500}, {2800, 3800} };
    static const float h_far[SS_FIR_MAX]  = { 1.0f, 0.8f, 0.55f, 0.3f, 0.12f, -0.05f, -0.1f, -0.06f };
    static const float h_near[6]          = { 1.0f, -0.6f, 0.3f, -0.15f, 0.08f, -0.04f };
    static const int   echo_off[SS_ECHO_TAPS]    = { 0, 5, 11, 19, 30 };
    static const float echo_gain[SS_ECHO_TAPS]   = { 0.35f, 0.18f, -0.10f, 0.06f, -0.03f };
    const float a_far = 0.10f, a_near = 0.12f, a_noise = 0.002f;

    if (hop_size < 1 || hop_size > SPLIT_SIGNAL_MAX_HOP || sample_rate < 8000 || sample_rate > SPLIT_SIGNAL_MAX_RATE ||
        hop_index < 0 || !ref || !mic) {
        if (ref && hop_size > 0) memset(ref, 0, (size_t)hop_size * sizeof(float));
        if (mic && hop_size > 0) memset(mic, 0, (size_t)hop_size * sizeof(float));
        return -1;
    }

    const int64_t start = (int64_t)hop_index * hop_size;
    const int delay = (int)((int64_t)sample_rate * SS_ECHO_DELAY_MS / 1000);
    const int lead = delay + echo_off[SS_ECHO_TAPS - 1];       /* samples of far history the echo reads */
    const int far_count = hop_size + lead;
    const int64_t far_first = start - lead;

    float farbuf[SS_SPAN_MAX];
    ss_coloured(SS_FAR_SEED, h_far, SS_FIR_MAX, far_first, far_count, farbuf);
    for (int m = 0; m < far_count; ++m) {
        int64_t n = far_first + m;
        float env = ss_burst_env(ss_far_segments_ms, 5, sample_rate, n);
        farbuf[m] = (env > 0.0f) ? a_far * env * ss_syllable(sample_rate, n, 0) * farbuf[m] : 0.0f;
    }

    float nearbuf[SPLIT_SIGNAL_MAX_HOP];
    float noisebuf[SPLIT_SIGNAL_MAX_HOP];
    ss_coloured(SS_NEAR_SEED, h_near, 6, start, hop_size, nearbuf);
    ss_white(SS_NOISE_SEED, start, hop_size, noisebuf);

    for (int j = 0; j < hop_size; ++j) {
        int64_t n = start + j;
        ref[j] = farbuf[j + lead];
        float echo = 0.0f;
        for (int t = 0; t < SS_ECHO_TAPS; ++t)
            echo += echo_gain[t] * farbuf[j + lead - delay - echo_off[t]];
        float envn = ss_burst_env(ss_near_segments_ms, 2, sample_rate, n);
        float nr = (envn > 0.0f) ? a_near * envn * ss_syllable(sample_rate, n, sample_rate / 15) * nearbuf[j] : 0.0f;
        mic[j] = echo + nr + a_noise * noisebuf[j];
    }
    return 0;
}

/* --- FNV-1a 64 hashes ----------------------------------------------------------------------------------------------- */

#define SPLIT_FNV_BASIS 0xCBF29CE484222325ull
#define SPLIT_FNV_PRIME 0x00000100000001B3ull

static inline uint64_t split_fnv_u32(uint64_t h, uint32_t v) {
    for (int b = 0; b < 4; ++b) {
        h ^= (uint64_t)((v >> (8 * b)) & 0xFFu);
        h *= SPLIT_FNV_PRIME;
    }
    return h;
}

static inline uint32_t split_f32_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static inline uint64_t split_fnv_f32(uint64_t h, float f) { return split_fnv_u32(h, split_f32_bits(f)); }

/* Hash of n floats (bit patterns); 0 for NULL. */
static inline uint64_t split_hash_f32(const float* p, int n) {
    uint64_t h = SPLIT_FNV_BASIS;
    if (!p) return 0u;
    for (int i = 0; i < n; ++i) h = split_fnv_f32(h, p[i]);
    return h;
}

/* Hash of n Complex values, real part then imaginary part of each; 0 for NULL. */
static inline uint64_t split_hash_complex(const Complex* p, int n) {
    uint64_t h = SPLIT_FNV_BASIS;
    if (!p) return 0u;
    for (int i = 0; i < n; ++i) { h = split_fnv_f32(h, p[i].r); h = split_fnv_f32(h, p[i].i); }
    return h;
}

/* Hash of n 32-bit integers (as uint32 patterns); 0 for NULL. */
static inline uint64_t split_hash_u32(const uint32_t* p, int n) {
    uint64_t h = SPLIT_FNV_BASIS;
    if (!p) return 0u;
    for (int i = 0; i < n; ++i) h = split_fnv_u32(h, p[i]);
    return h;
}

/* 16 lower-case hex digits plus NUL into buf[17]. */
static inline void split_hash_hex(char* buf, uint64_t h) {
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) buf[i] = digits[(h >> (60 - 4 * i)) & 0xFu];
    buf[16] = '\0';
}

#ifdef __cplusplus
}
#endif

#endif /* AEC_TEST_SPLIT_SIGNAL_H */
