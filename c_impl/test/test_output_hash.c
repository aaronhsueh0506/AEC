/* test_output_hash.c -- whole-output regression gate.
 *
 * Drives the library over a seeded synthetic scene and hashes, per hop, the
 * emitted hop (aec_process only) and every non-NULL field of AecResContext and
 * AecLinearContext. One row = one configuration: the 3 supported grids x
 * shadow filter on/off x entry point {aec_process, aec_process_context at
 * levels 0, 1, 2}, 6000 hops each.
 *
 * usage: test_output_hash <golden-file> <variant>
 *        test_output_hash --print <variant>        (golden lines to stdout)
 *
 * <variant> names the FFT/kernel build the golden belongs to: kiss, ne10_neon
 * (NE10 with NEON kernels) or ne10_scalar (NE10 built without NEON). The
 * output stream does not depend on the SIMD switch for a given FFT code, so
 * such builds share one key.
 *
 * Each golden row carries two digests of the output (full stream, and the
 * subset that does not depend on the gain path) plus one of the input
 * stimulus, so a generator difference (libm, compiler) is told apart from a
 * library change: a row whose stimulus digest differs FAILS with its own
 * message. The golden is valid for the host class named in its header;
 * regenerate it with --print on another toolchain.
 *
 * The Makefile compiles this file with contraction off whatever the library's
 * policy is, so the stimulus is the same for every build and a change of the
 * library's own policy shows up in the output digests alone.
 *
 * Also checked without any golden: the gain-path-free digest is identical
 * across context levels 0, 1 and 2. */
#include "aec.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- sha256 ---- */
typedef struct { uint32_t s[8]; uint8_t b[64]; uint64_t n; size_t k; } Sha;
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha_block(Sha* h, const uint8_t* p) {
    uint32_t w[64], a, b, c, d, e, f, g, hh;
    int i;
    for (i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 |
               (uint32_t)p[4*i+2] << 8 | p[4*i+3];
    for (i = 16; i < 64; i++) {
        uint32_t s0 = ROR32(w[i-15], 7) ^ ROR32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ROR32(w[i-2], 17) ^ ROR32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    a = h->s[0]; b = h->s[1]; c = h->s[2]; d = h->s[3];
    e = h->s[4]; f = h->s[5]; g = h->s[6]; hh = h->s[7];
    for (i = 0; i < 64; i++) {
        uint32_t S1 = ROR32(e, 6) ^ ROR32(e, 11) ^ ROR32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + K256[i] + w[i];
        uint32_t S0 = ROR32(a, 2) ^ ROR32(a, 13) ^ ROR32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h->s[0] += a; h->s[1] += b; h->s[2] += c; h->s[3] += d;
    h->s[4] += e; h->s[5] += f; h->s[6] += g; h->s[7] += hh;
}

static void sha_init(Sha* h) {
    static const uint32_t i0[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(h->s, i0, sizeof i0);
    h->n = 0;
    h->k = 0;
}

static void sha_update(Sha* h, const void* d, size_t n) {
    const uint8_t* p = (const uint8_t*)d;
    h->n += n;
    while (n) {
        size_t c = 64 - h->k;
        if (c > n) c = n;
        memcpy(h->b + h->k, p, c);
        h->k += c; p += c; n -= c;
        if (h->k == 64) { sha_block(h, h->b); h->k = 0; }
    }
}

static void sha_final(Sha* h, char* hex) {
    uint64_t bits = h->n * 8;
    uint8_t pad = 0x80, z = 0, l[8];
    int i;
    sha_update(h, &pad, 1);
    while (h->k != 56) sha_update(h, &z, 1);
    for (i = 0; i < 8; i++) l[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha_update(h, l, 8);
    for (i = 0; i < 8; i++) sprintf(hex + 8 * i, "%08x", h->s[i]);
}

/* ---- the two output streams: full, and the gain-path-free subset ---- */
static Sha g_full, g_lin;

static void put_both(const void* p, size_t n) {
    if (p && n) { sha_update(&g_full, p, n); sha_update(&g_lin, p, n); }
}
static void put_full_only(const void* p, size_t n) { if (p && n) sha_update(&g_full, p, n); }
static void put_int(int v) { put_both(&v, sizeof v); }
static void put_flt(float v) { put_both(&v, sizeof v); }

/* ---- stimulus: xorshift-driven far-end, 40-tap echo path delayed 24 ms,
 * near-end bursts; the scene repeats every 12 s (far-only, near-only, double
 * talk, loud segment, level step, silence). ---- */
static uint64_t g_rs;
static uint64_t rnd64(void) {
    g_rs ^= g_rs >> 12; g_rs ^= g_rs << 25; g_rs ^= g_rs >> 27;
    return g_rs * 2685821657736338717ull;
}
static double urand(void) { return (double)(rnd64() >> 11) / 9007199254740992.0; }

typedef struct {
    int sr, fft, shadow;
    const char* entry;        /* "proc", "ctx0", "ctx1", "ctx2" */
} Row;

typedef struct { char full[65], lin[65], stim[65]; } RowHash;

#define ROW_HOPS 6000
#define ROW_SEED 5
#define ECHO_TAPS 40

static int run_row(const Row* r, RowHash* out) {
    int is_proc = !strcmp(r->entry, "proc");
    int level = is_proc ? 0 : r->entry[3] - '0';
    AecConfig cfg;
    Aec a;
    int hop, delay, ring, hp = 0, hh, i, k;
    long t_samp = 0;
    double h[ECHO_TAPS], lp1 = 0, lp2 = 0, ph = 0;
    float *mic, *ref, *emit, *far_hist;
    Sha stim;

    g_rs = (uint64_t)ROW_SEED * 0x9E3779B97F4A7C15ull + 3;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, r->sr);
    cfg.fft_size = r->fft;
    cfg.enable_shadow = r->shadow;
    if (!is_proc) {
        cfg.enable_res = 0;
        cfg.return_res_context = 1;
        cfg.spatial_linear_context = level;
    }
    if (aec_create(&a, &cfg) != 0) return 0;
    hop = aec_hop_size(&a);
    mic = (float*)malloc((size_t)hop * sizeof(float));
    ref = (float*)malloc((size_t)hop * sizeof(float));
    emit = (float*)malloc((size_t)hop * sizeof(float));
    delay = (int)(0.024 * r->sr);
    ring = delay + ECHO_TAPS + 8;
    far_hist = (float*)calloc((size_t)ring, sizeof(float));
    if (!mic || !ref || !emit || !far_hist) return 0;
    for (i = 0; i < ECHO_TAPS; i++) h[i] = (urand() * 2 - 1) * exp(-i / 9.0) * 0.5;

    sha_init(&g_full); sha_init(&g_lin); sha_init(&stim);
    for (hh = 0; hh < ROW_HOPS; hh++) {
        AecResContext rc;
        AecLinearContext lc;
        int nf, floor_protect;
        for (i = 0; i < hop; i++) {
            double t = (double)(t_samp++) / r->sr, tc = fmod(t, 12.0);
            int far_on = (tc < 4.0) || (tc >= 6.0 && tc < 10.0) || (tc >= 11.0);
            int near_on = (tc >= 4.0 && tc < 9.0);
            double farg = (tc >= 9.0 && tc < 10.0) ? 0.75 : 0.25;
            double nz = urand() * 2 - 1, x, echo = 0, sp = 0, env, nearv, m;
            int q;
            lp1 = 0.9 * lp1 + 0.1 * nz;
            lp2 = 0.6 * lp2 + 0.4 * nz;
            x = far_on ? farg * (0.8 * lp1 * 3.0 + 0.2 * lp2) : 0.0;
            if (x > 0.95) x = 0.95;
            if (x < -0.95) x = -0.95;
            far_hist[hp] = (float)x;
            for (k = 0; k < ECHO_TAPS; k++)
                echo += h[k] * far_hist[((hp - delay - k) % ring + ring) % ring];
            hp = (hp + 1) % ring;
            ph += 2 * M_PI * (120 + 40 * sin(2 * M_PI * 0.3 * t)) / r->sr;
            for (q = 1; q <= 12; q++) sp += sin(q * ph) / q;
            env = 0.5 + 0.5 * sin(2 * M_PI * 3.1 * t);
            nearv = near_on ? 0.12 * sp * env : 0.0;
            m = echo + nearv + 0.002 * (urand() * 2 - 1);
            if (m > 0.99) m = 0.99;
            if (m < -0.99) m = -0.99;
            mic[i] = (float)m;
            ref[i] = (float)x;
        }
        sha_update(&stim, mic, (size_t)hop * sizeof(float));
        sha_update(&stim, ref, (size_t)hop * sizeof(float));
        if (is_proc) {
            aec_process(&a, mic, ref, emit);
            put_full_only(emit, (size_t)hop * sizeof(float));
        } else {
            aec_process_context(&a, mic, ref);
        }
        aec_get_res_context(&a, &rc);
        aec_get_linear_context(&a, &lc);
        nf = rc.n_freqs;
        put_int(rc.n_freqs); put_int(rc.hop_size);
        put_both(rc.linear_hop, (size_t)rc.hop_size * 4);
        put_both(rc.formed_hop, (size_t)rc.hop_size * 4);
        put_both(rc.echo_spec, (size_t)nf * 8);
        put_both(rc.far_spec, (size_t)nf * 8);
        put_both(rc.near_spec, (size_t)nf * 8);
        put_both(rc.error_spec, (size_t)nf * 8);
        put_flt(rc.far_power); put_flt(rc.erle_factor); put_flt(rc.dt_indicator);
        put_flt(rc.divergence); put_flt(rc.saturation_level); put_flt(rc.erl_estimate);
        put_flt(rc.shadow_dt);
        put_int(rc.is_stationary_dt); put_int(rc.filter_converged);
        put_int(rc.filter_once_converged); put_int(rc.epc_active); put_int(rc.usable_linear);
        put_full_only(rc.res_gain, (size_t)nf * 4);
        put_full_only(rc.r2, (size_t)nf * 4);
        put_full_only(rc.comfort_noise, (size_t)nf * 4);
        floor_protect = rc.res_floor_protect;
        put_full_only(&floor_protect, sizeof floor_protect);
        put_int(lc.hop_size);
        put_both(lc.formed_linear_hop, (size_t)lc.hop_size * 4);
        put_both(lc.aligned_far_hop, (size_t)lc.hop_size * 4);
        put_int(lc.delay_samples); put_flt(lc.delay_confidence);
        put_int((int)lc.delay_state); put_int((int)lc.generation);
    }
    sha_final(&g_full, out->full);
    sha_final(&g_lin, out->lin);
    sha_final(&stim, out->stim);
    aec_destroy(&a);
    free(mic); free(ref); free(emit); free(far_hist);
    return 1;
}

/* ---- row table: index = grid * 8 + shadow * 4 + entry ---- */
static const int GRIDS[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
static const char* ENTRIES[4] = {"proc", "ctx0", "ctx1", "ctx2"};
#define N_ROWS (3 * 2 * 4)

static Row row_at(int idx) {
    Row r;
    r.entry = ENTRIES[idx % 4];
    r.shadow = (idx / 4) % 2;
    r.sr = GRIDS[idx / 8][0];
    r.fft = GRIDS[idx / 8][1];
    return r;
}

static void row_id(const Row* r, char* buf, size_t n) {
    snprintf(buf, n, "%d/%d/sh%d/%s", r->sr, r->fft, r->shadow, r->entry);
}

typedef struct { char id[40], stim[80], full[80], lin[80]; } Golden;

static int load_golden(const char* path, const char* variant, Golden* g, int cap) {
    FILE* f = fopen(path, "r");
    char line[512];
    int n = 0;
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) {
        char var[32], id[40], stim[80], full[80], lin[80];
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "%31s %39s %79s %79s %79s", var, id, stim, full, lin) != 5) continue;
        if (strcmp(var, variant) != 0 || n >= cap) continue;
        strcpy(g[n].id, id); strcpy(g[n].stim, stim);
        strcpy(g[n].full, full); strcpy(g[n].lin, lin);
        n++;
    }
    fclose(f);
    return n;
}

int main(int argc, char** argv) {
    int print_mode, idx, fails = 0, ok = 0, stim_fail = 0, n_gold = 0;
    const char* variant;
    Golden gold[N_ROWS];
    RowHash res[N_ROWS];

    if (argc != 3) {
        fprintf(stderr, "usage: %s <golden-file> <variant> | --print <variant>\n", argv[0]);
        return 2;
    }
    print_mode = !strcmp(argv[1], "--print");
    variant = argv[2];
    if (!print_mode) {
        n_gold = load_golden(argv[1], variant, gold, N_ROWS);
        if (n_gold < 0) { printf("FAIL: cannot read golden %s\n", argv[1]); return 1; }
        if (n_gold == 0) { printf("FAIL: golden has no rows for variant %s\n", variant); return 1; }
    }

    for (idx = 0; idx < N_ROWS; idx++) {
        Row r = row_at(idx);
        char id[40];
        row_id(&r, id, sizeof id);
        if (!run_row(&r, &res[idx])) { printf("FAIL: %s: setup\n", id); return 1; }
        if (print_mode) {
            printf("%s %s %s %s %s\n", variant, id, res[idx].stim, res[idx].full, res[idx].lin);
        } else {
            int j, found = -1;
            for (j = 0; j < n_gold; j++) if (!strcmp(gold[j].id, id)) found = j;
            if (found < 0) {
                printf("FAIL: %s: no golden row\n", id);
                fails++;
            } else if (strcmp(gold[found].stim, res[idx].stim) != 0) {
                printf("FAIL: %s: stimulus digest differs from the golden host (generator, not library)\n", id);
                stim_fail++;
                fails++;
            } else if (strcmp(gold[found].full, res[idx].full) != 0 ||
                       strcmp(gold[found].lin, res[idx].lin) != 0) {
                printf("FAIL: %s: output digest differs\n", id);
                fails++;
            } else {
                ok++;
            }
        }
    }
    if (print_mode) return 0;

    /* The gain-path-free digest does not depend on the context level. */
    for (idx = 0; idx < N_ROWS; idx += 4) {
        const char* base = res[idx + 1].lin;
        if (strcmp(base, res[idx + 2].lin) != 0 || strcmp(base, res[idx + 3].lin) != 0) {
            Row r = row_at(idx);
            printf("FAIL: %d/%d/sh%d: linear digest differs across context levels\n",
                   r.sr, r.fft, r.shadow);
            fails++;
        } else {
            ok++;
        }
    }
    printf("test_output_hash [%s]: %d ok, %d failed (%d on the stimulus digest)\n",
           variant, ok, fails, stim_fail);
    return fails ? 1 : 0;
}
