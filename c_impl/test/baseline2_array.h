/* baseline2_array.h -- the second four-microphone scene set: case table and runner of the test of the array instance
 * (test/test_aec_array2.c).
 *
 * The includer provides, before including this header: <aec.h>, the header of the stand-alone residual gain stage (AecRes),
 * <utility/simd_kernels.h>, `static void g2_sink(const char* line)` and the instance operations declared below (g2a_*), and
 * may define G2A_HOOK_FRAME(h, c, hop, front, frame, gain) for extra per-hop checks.
 *
 * Per hop and group (front / frame / gain / apply) the hashes are chained and a line is printed every G2_STRIDE hops, on the
 * last hop and on and after every event (see baseline2_common.h). The gain is always the stand-alone AecRes fed the fused frame
 * (the literal old flow); the new tree additionally compares the instance's own gain to it in FULL. */
#ifndef AEC_TEST_GOLDEN2_ARRAY_H
#define AEC_TEST_GOLDEN2_ARRAY_H

#include "baseline2_common.h"
#include "array_signal.h"

enum { G2A_W_EQUAL = 0, G2A_W_UNEQUAL = 1, G2A_W_NONTRIV = 2 };

typedef struct G2ACase {
    char  name[28];
    int   sr, fft, dur_ms;
    int   post;              /* 1: fusion and gain (ESTIMATE / FULL); 0: the lanes only (STATE) */
    int   wkind;             /* G2A_W_* */
    int   delay_mode;        /* 0 MATCHED, 1 FIXED, 2 EXTERNAL_ALIGNED */
    int   proxy;             /* capture_proxy_channel */
    int   quarantine; float quarantine_s;
    float duty_s;
    int   noshadow;          /* lanes without the shadow filter */
    int   ext_beam;          /* the caller supplies beamformed_error (the fusion of the lane errors, then scaled bin by bin) */
    float chan_gain[ARR_CH]; /* microphone gain before the clip (0 = 1) */
    int   clip_mask;         /* bit ch: microphone ch hard-clipped at +-1 */
    int   silent_mask;       /* bit ch: microphone ch carries only its own noise */
    G2Stim stim;
    int   n_ev; G2Ev ev[6];
} G2ACase;

#define G2A_MAX_CASES 48

/* ---- neutral per-hop records filled by the includer ------------------------------------------------------------------- */
typedef struct G2AFront {
    int   n_freqs, hop_size;
    const Complex* lane_error[ARR_CH];
    const Complex* lane_base[ARR_CH];
    const float*   lane_formed[ARR_CH];
    int            lane_guard[ARR_CH];
    const float*   aligned_ref;
    int            delay_samples; float delay_conf; int delay_solid, delay_changed; uint64_t est_calls; int est_updates;
    uint32_t       serial;
} G2AFront;

typedef struct G2AFrame {
    const Complex *e, *b, *y, *x;
    const float   *r2, *cn, *far_hop;
    unsigned       guard_lanes;
    uint32_t       serial;
    float          far_power, max_dt, max_sat;
    int            all_converged, usable_linear, clock_drift;
    const float*   instance_gain;      /* the instance's own gain (new tree, FULL), else NULL */
} G2AFrame;

static void* g2a_create(const G2ACase* c, int mode);                       /* mode: 0 STATE, 1 ESTIMATE, 2 FULL (the old flow ignores it) */
static void  g2a_destroy(void* h);
static int   g2a_front(void* h, const float* mics, const float* ref, G2AFront* f);
static int   g2a_fuse(void* h, const Complex* weights, const Complex* beamformed_error, G2AFrame* fr);
static void  g2a_reset(void* h);
static void  g2a_set_level(void* h, int level);
static void  g2a_counters(void* h, uint64_t out[8]);

/* ---- cases ----------------------------------------------------------------------------------------------------------- */
static G2ACase* g2a_new(G2ACase* out, int* n, const char* name, int sr, int fft, int dur_ms) {
    G2ACase* c = &out[(*n)++];
    memset(c, 0, sizeof *c);
    snprintf(c->name, sizeof c->name, "%s", name);
    c->sr = sr; c->fft = fft; c->dur_ms = dur_ms; c->post = 1; c->wkind = G2A_W_NONTRIV; c->quarantine_s = 0.5f;
    return c;
}

/* satdt with the microphone channels doing the clipping (gain, then a hard clip at +-1 on the masked channels). */
static void g2a_stim_sat(G2ACase* c) {
    g2_stim_satdt(&c->stim);
    c->stim.clip_mic = 0;
    c->chan_gain[0] = 1.0f; c->chan_gain[1] = 1.0f; c->chan_gain[2] = 1.0f; c->chan_gain[3] = 4.0f;
    c->clip_mask = 1 << 3;
}

/* The shared estimate moves: forward (40 -> 64 ms), backward (-> 40 ms), forward again. */
static void g2a_stim_moves(G2Stim* st) {
    memset(st, 0, sizeof *st);
    g2_seg(st->far, &st->n_far, 100, 11900); st->far_amp = 0.12f;
    g2_dly(st, 0, 640); g2_dly(st, 3000, 1024); g2_dly(st, 6000, 640); g2_dly(st, 9000, 1024);
}

/* Moves of 32 samples (not admitted), 36 (admitted) and back (-36), at 16 kHz sample counts. */
static void g2a_stim_band(G2Stim* st) {
    memset(st, 0, sizeof *st);
    g2_seg(st->far, &st->n_far, 100, 11900); st->far_amp = 0.12f;
    g2_dly(st, 0, 640); g2_dly(st, 2500, 672); g2_dly(st, 5000, 676); g2_dly(st, 7500, 640); g2_dly(st, 9500, 676);
}

static void g2a_stim_drift(G2Stim* st) {
    memset(st, 0, sizeof *st);
    g2_seg(st->far, &st->n_far, 100, 11900); st->far_amp = 0.12f;
    for (int k = 0; k < 8; ++k) g2_dly(st, k ? 2000 + 1200 * (k - 1) : 0, 640 + 4 * k);
}

static int g2a_cases(G2ACase* out) {
    static const int grid[3][2] = { {16000, 256}, {16000, 512}, {48000, 1024} };
    int n = 0;
    for (int g = 0; g < 3; ++g) {
        G2ACase* c = g2a_new(out, &n, "satdt", grid[g][0], grid[g][1], 8200);
        g2a_stim_sat(c);
    }
    {
        G2ACase* c = g2a_new(out, &n, "satdt_front", 16000, 256, 8200);
        c->post = 0; g2a_stim_sat(c);
        c = g2a_new(out, &n, "silent_mic", 16000, 256, 8200);
        g2a_stim_sat(c);
        c->silent_mask = 1 << 1; c->clip_mask = 0; c->chan_gain[3] = 1.0f;
        c = g2a_new(out, &n, "noshadow", 16000, 256, 8200);
        g2a_stim_sat(c);
        c->noshadow = 1; c->wkind = G2A_W_EQUAL;
        c = g2a_new(out, &n, "fixed_sat", 16000, 256, 8200);
        g2a_stim_sat(c);
        c->delay_mode = 1; c->wkind = G2A_W_UNEQUAL;
        c = g2a_new(out, &n, "reset_dt", 16000, 256, 8200);
        g2a_stim_sat(c);
        c->n_ev = 2; c->ev[0].t_ms = 3000; c->ev[0].type = G2E_RESET; c->ev[1].t_ms = 6500; c->ev[1].type = G2E_RESET;
        c = g2a_new(out, &n, "levels", 16000, 256, 8200);
        g2a_stim_sat(c);
        c->wkind = G2A_W_UNEQUAL;
        c->n_ev = 5;
        c->ev[0].t_ms = 1500; c->ev[0].type = G2E_LEVEL; c->ev[0].a = 2;
        c->ev[1].t_ms = 3800; c->ev[1].type = G2E_LEVEL; c->ev[1].a = 0;
        c->ev[2].t_ms = 4600; c->ev[2].type = G2E_LEVEL; c->ev[2].a = 3;
        c->ev[3].t_ms = 5800; c->ev[3].type = G2E_LEVEL; c->ev[3].a = 1;
        c->ev[4].t_ms = 7000; c->ev[4].type = G2E_LEVEL; c->ev[4].a = 2;
    }
    for (int g = 0; g < 3; g += 2) {
        G2ACase* c = g2a_new(out, &n, "extbeam", grid[g][0], grid[g][1], 8200);
        g2a_stim_sat(c);
        c->ext_beam = 1;
        c = g2a_new(out, &n, "drift", grid[g][0], grid[g][1], 12000);
        g2a_stim_drift(&c->stim); c->wkind = G2A_W_EQUAL;
        c = g2a_new(out, &n, "proxy2_q", grid[g][0], grid[g][1], 12000);
        g2a_stim_moves(&c->stim);
        c->proxy = 2; c->quarantine = 1; c->wkind = G2A_W_UNEQUAL;
    }
    for (int g = 0; g < 3; ++g) {                       /* the shared estimate flips between two paths: quarantine on / off */
        G2ACase* c = g2a_new(out, &n, "flip_q", grid[g][0], grid[g][1], 12000);
        g2_stim_flip(&c->stim);
        c->proxy = 2; c->quarantine = 1; c->wkind = G2A_W_UNEQUAL;
        if (g == 0) {
            c = g2a_new(out, &n, "flip_noq", grid[g][0], grid[g][1], 12000);
            g2_stim_flip(&c->stim);
            c->proxy = 2; c->wkind = G2A_W_UNEQUAL;
        }
    }
    for (int g = 1; g < 3; ++g) {
        G2ACase* c = g2a_new(out, &n, "levels", grid[g][0], grid[g][1], 8200);
        g2a_stim_sat(c);
        c->wkind = G2A_W_UNEQUAL;
        c->n_ev = 3;
        c->ev[0].t_ms = 1500; c->ev[0].type = G2E_LEVEL; c->ev[0].a = 2;
        c->ev[1].t_ms = 3800; c->ev[1].type = G2E_LEVEL; c->ev[1].a = 3;
        c->ev[2].t_ms = 5800; c->ev[2].type = G2E_LEVEL; c->ev[2].a = 0;
        c = g2a_new(out, &n, "noshadow", grid[g][0], grid[g][1], 8200);
        g2a_stim_sat(c);
        c->noshadow = 1; c->wkind = G2A_W_EQUAL;
    }
    {
        G2ACase* c = g2a_new(out, &n, "proxy3_noq", 16000, 512, 12000);
        g2a_stim_moves(&c->stim);
        c->proxy = 3; c->wkind = G2A_W_UNEQUAL;
        c = g2a_new(out, &n, "band_q", 16000, 256, 12000);
        g2a_stim_band(&c->stim);
        c->quarantine = 1; c->proxy = 1;
        c = g2a_new(out, &n, "duty_q", 16000, 256, 12000);
        g2a_stim_moves(&c->stim);
        c->duty_s = 1.0f; c->quarantine = 1; c->proxy = 2; c->wkind = G2A_W_UNEQUAL;
    }
    return n;
}

/* ---- signal and weights ------------------------------------------------------------------------------------------------ */
static void g2a_weights(int kind, int n, Complex* w) {
    if (kind != G2A_W_NONTRIV) { array_weights(kind == G2A_W_EQUAL ? ARR_W_EQUAL : ARR_W_UNEQUAL, n, w); return; }
    for (int c = 0; c < ARR_CH; ++c)
        for (int k = 0; k < n; ++k) {
            w[c * n + k].r = 0.45f / (float)(c + 1) - 0.0003f * (float)((k * (7 + c)) % 13);
            w[c * n + k].i = 0.06f * (float)(((k + 3 * c) % 5) - 2);
        }
}

/* ref[ns] and the interleaved microphones mics[ns][ARR_CH]. */
static void g2a_signal(const G2ACase* c, int ns, float* ref, float* mics) {
    float* m0 = (float*)malloc((size_t)ns * sizeof(float));
    float* w = (float*)malloc((size_t)ns * sizeof(float));
    g2_mono_signal(&c->stim, c->sr, ns, ref, m0);
    for (int ch = 0; ch < ARR_CH; ++ch) {
        const float cg = c->chan_gain[ch] > 0.0f ? c->chan_gain[ch] : 1.0f;
        ss_white(ARR_TAPS[ch].seed, 0, ns, w);
        for (int i = 0; i < ns; ++i) {
            const int ja = i - ARR_TAPS[ch].d, jb = i - ARR_TAPS[ch].d2;
            float v = (ARR_TAPS[ch].g * (ja >= 0 ? m0[ja] : 0.0f) + ARR_TAPS[ch].g2 * (jb >= 0 ? m0[jb] : 0.0f)) * cg;
            if (c->silent_mask & (1 << ch)) v = 0.0f;
            v += 0.001f * w[i];
            if (c->clip_mask & (1 << ch)) v = g2_clip1(v);
            mics[(size_t)i * ARR_CH + ch] = v;
        }
    }
    free(m0); free(w);
}

/* The caller's beamformed spectrum: the fusion of the lane errors, then a bin-dependent scale, so that it differs from the
 * rebuild the library would otherwise do from the weights. */
static void g2a_beamformed(const G2AFront* f, const Complex* W, Complex* out) {
    const int n = f->n_freqs;
    for (int k = 0; k < n; ++k) {
        float re = 0.0f, im = 0.0f;
        for (int ch = 0; ch < ARR_CH; ++ch) {
            const Complex w = W[ch * n + k], e = f->lane_error[ch][k];
            re += w.r * e.r - w.i * e.i;
            im += w.r * e.i + w.i * e.r;
        }
        const float s = 0.97f + 0.004f * (float)(k % 5);
        out[k].r = re * s; out[k].i = im * s;
    }
}

/* ---- the AecRes (stand-alone), apply ------------------------------------------------------------------------------------- */
static void g2a_apply_gain_spec(Complex* out, const Complex* base, const float* g_res, const float* g_extra, int n) {
    if (!g_extra) { sk_capply_gain_f32(out, base, g_res, (size_t)n); return; }
    {
        float total[64];
        int k = 0;
        while (k < n) {
            int m = (n - k < 64) ? (n - k) : 64;
            sk_min_f32(total, g_extra + k, g_res + k, (size_t)m);
            sk_capply_gain_f32(out + k, base + k, total, (size_t)m);
            k += m;
        }
    }
}

enum { G2AG_FRONT, G2AG_FRAME, G2AG_GAIN, G2AG_APPLY, G2AG_COUNT };
static const char* const G2AG_NAME[G2AG_COUNT] = { "front", "frame", "gain", "apply" };

static void g2a_apply_cfg(AecConfig* cfg, const G2ACase* c) {
    aec_config_from_level(cfg, AEC_LEVEL_BALANCED, c->sr);
    cfg->fft_size = c->fft;
}

static const char* g2a_grid(const G2ACase* c, char* buf, size_t cap) { snprintf(buf, cap, "%d/%d", c->sr, c->fft); return buf; }

static int g2a_want(const G2ACase* c) {
    const char* f = getenv("G2_ONLY");
    char id[64];
    if (!f || !*f) return 1;
    snprintf(id, sizeof id, "%d/%d %s", c->sr, c->fft, c->name);
    return strstr(id, f) != NULL;
}

typedef struct { int hops, dt02, sat05, drift, guard, usable, conv, changed, solid; } G2ACensus;

/* One case. `mode`: 0 STATE (post == 0 cases), 1 ESTIMATE, 2 FULL (post == 1 cases). */
static int g2a_run_case(const G2ACase* c, int mode) {
    char grid[24];
    AecConfig rcfg;
    G2ACensus cen; memset(&cen, 0, sizeof cen);
    g2a_grid(c, grid, sizeof grid);
    void* h = g2a_create(c, mode);
    if (!h) { fprintf(stderr, "create failed %s %s\n", grid, c->name); return -1; }
    const int hop = c->fft / 2, n = hop + 1;
    const int n_hops = (int)((int64_t)c->dur_ms * c->sr / 1000 / hop), ns = n_hops * hop;
    float* ref = (float*)malloc((size_t)ns * sizeof(float));
    float* mics = (float*)malloc((size_t)ns * ARR_CH * sizeof(float));
    Complex* W = (Complex*)malloc((size_t)ARR_CH * n * sizeof(Complex));
    Complex* Ebf = (Complex*)malloc((size_t)n * sizeof(Complex));
    AecRes* res = NULL; void* rmem = NULL;
    uint64_t chain[G2AG_COUNT], hh[G2AG_COUNT];
    uint32_t handled = 0u, rng = AEC_CNG_SEED;
    int ev_hop[6], emit_next = 0, ev_now;
    g2a_signal(c, ns, ref, mics);
    g2a_weights(c->wkind, n, W);
    for (int g = 0; g < G2AG_COUNT; ++g) chain[g] = G2_FNV_BASIS;
    for (int e = 0; e < c->n_ev; ++e) ev_hop[e] = (int)((int64_t)c->ev[e].t_ms * c->sr / 1000 / hop);
    if (c->post) {
        g2a_apply_cfg(&rcfg, c);
        size_t rsz = aec_res_get_mem_size(&rcfg);
        rmem = g2_pool(rsz);
        res = (rsz && rmem) ? aec_res_init(rmem, rsz, &rcfg) : NULL;
        if (!res) { fprintf(stderr, "aec_res_init failed\n"); return -1; }
    }
    {
        uint64_t hi = G2_FNV_BASIS;
        hi = g2_fnv_f32v(hi, ref, ns); hi = g2_fnv_f32v(hi, mics, ns * ARR_CH);
        g2_line(grid, c->name, "INPUT", 0, hi);
        g2_line(grid, c->name, "META_hops", 0, (uint64_t)n_hops);
        g2_line(grid, c->name, "META_hop", 0, (uint64_t)hop);
    }
    for (int hp = 0; hp < n_hops; ++hp) {
        G2AFront f; G2AFrame fr;
        uint64_t x;
        ev_now = 0;
        for (int e = 0; e < c->n_ev; ++e) {
            if (ev_hop[e] != hp) continue;
            ev_now = 1;
            if (c->ev[e].type == G2E_RESET) { g2a_reset(h); if (res) aec_res_reset(res); }
            else if (c->ev[e].type == G2E_LEVEL) {
                g2a_set_level(h, c->ev[e].a);
                if (res) aec_res_set_level(res, (AecLevel)c->ev[e].a);
            }
        }
        if (g2a_front(h, mics + (size_t)hp * hop * ARR_CH, ref + (size_t)hp * hop, &f) != 0) {
            fprintf(stderr, "front failed %s %s hop %d\n", grid, c->name, hp); return -1;
        }
        x = G2_FNV_BASIS;
        for (int ch = 0; ch < ARR_CH; ++ch) x = g2_hc(x, f.lane_error[ch], n);
        for (int ch = 0; ch < ARR_CH; ++ch) x = g2_hc(x, f.lane_base[ch], n);
        for (int ch = 0; ch < ARR_CH; ++ch) x = g2_fnv_f32v(x, f.lane_formed[ch], hop);
        x = g2_fnv_f32v(x, f.aligned_ref, hop);
        for (int ch = 0; ch < ARR_CH; ++ch) x = g2_fnv_u32(x, (uint32_t)f.lane_guard[ch]);
        x = g2_fnv_u32(x, (uint32_t)f.delay_samples); x = g2_fnv_f32(x, f.delay_conf);
        x = g2_fnv_u32(x, (uint32_t)f.delay_solid); x = g2_fnv_u32(x, (uint32_t)f.delay_changed);
        x = g2_fnv_u32(x, (uint32_t)(f.est_calls & 0xFFFFFFFFu)); x = g2_fnv_u32(x, (uint32_t)(f.est_calls >> 32));
        x = g2_fnv_u32(x, (uint32_t)f.est_updates); x = g2_fnv_u32(x, f.serial);
        hh[G2AG_FRONT] = x;
        cen.hops++; cen.changed += f.delay_changed; cen.solid += f.delay_solid;
        for (int ch = 0; ch < ARR_CH; ++ch) cen.guard += f.lane_guard[ch];
        hh[G2AG_FRAME] = hh[G2AG_GAIN] = hh[G2AG_APPLY] = 0;
        if (c->post) {
            const float* g;
            const Complex* eb = NULL;
            AecResInput in;
            if (c->ext_beam) { g2a_beamformed(&f, W, Ebf); eb = Ebf; }
            if (g2a_fuse(h, W, eb, &fr) != 0) { fprintf(stderr, "fuse failed %s %s hop %d\n", grid, c->name, hp); return -1; }
            x = G2_FNV_BASIS;
            x = g2_hc(x, fr.e, n); x = g2_hc(x, fr.b, n); x = g2_hc(x, fr.y, n); x = g2_hc(x, fr.x, n);
            x = g2_fnv_f32v(x, fr.r2, n); x = g2_fnv_f32v(x, fr.cn, n); x = g2_fnv_f32v(x, fr.far_hop, hop);
            x = g2_fnv_u32(x, fr.guard_lanes); x = g2_fnv_u32(x, fr.serial); x = g2_fnv_f32(x, fr.far_power);
            x = g2_fnv_f32(x, fr.max_dt); x = g2_fnv_f32(x, fr.max_sat); x = g2_fnv_u32(x, (uint32_t)fr.all_converged);
            x = g2_fnv_u32(x, (uint32_t)fr.usable_linear); x = g2_fnv_u32(x, (uint32_t)fr.clock_drift);
            x = g2_fnv_u32(x, (uint32_t)f.delay_changed); x = g2_fnv_u32(x, (uint32_t)f.delay_solid);
            hh[G2AG_FRAME] = x;
            memset(&in, 0, sizeof in);
            in.error_spec = fr.e; in.near_spec = fr.y; in.r2 = fr.r2; in.comfort_noise = fr.cn;
            in.render_hop = fr.far_hop; in.all_converged = fr.all_converged; in.max_dt = fr.max_dt;
            in.max_saturation = fr.max_sat; in.clock_drift = fr.clock_drift;
            g = aec_res_process(res, &in);
            if (!g) { fprintf(stderr, "aec_res_process NULL\n"); return -1; }
            hh[G2AG_GAIN] = g2_fnv_f32v(G2_FNV_BASIS, g, n);
            if (fr.serial != handled) { rng = AEC_CNG_SEED; handled = fr.serial; }
            {
                float gx[G2_NB_MAX]; Complex p1[G2_NB_MAX], p2[G2_NB_MAX], p3[G2_NB_MAX], p4[G2_NB_MAX];
                array_extra_gain(gx, n, hp);
                g2a_apply_gain_spec(p1, fr.b, g, gx, n);
                memcpy(p2, p1, (size_t)n * sizeof(Complex));
                aec_cng_fill(&rng, p2, fr.cn, g, gx, 0.31622777f, n);
                g2a_apply_gain_spec(p3, fr.b, g, NULL, n);
                memcpy(p4, p3, (size_t)n * sizeof(Complex));
                aec_cng_fill(&rng, p4, fr.cn, g, NULL, 0.0f, n);
                x = G2_FNV_BASIS;
                x = g2_hc(x, p1, n); x = g2_hc(x, p2, n); x = g2_hc(x, p3, n); x = g2_hc(x, p4, n);
                hh[G2AG_APPLY] = x;
            }
            cen.dt02 += fr.max_dt > 0.2f; cen.sat05 += fr.max_sat > 0.5f; cen.drift += fr.clock_drift;
            cen.usable += fr.usable_linear; cen.conv += fr.all_converged;
#ifdef G2A_HOOK_FRAME
            G2A_HOOK_FRAME(h, c, hp, &f, &fr, g);
#endif
        }
        if (getenv("G2_TRACE") && c->post)
            fprintf(stderr, "T %s %s h%d t%d dt %.3f sat %.2f drift %d guard %u conv %d usable %d chg %d solid %d dly %d\n", grid, c->name,
                    hp, (int)((int64_t)hp * hop * 1000 / c->sr), fr.max_dt, fr.max_sat, fr.clock_drift, fr.guard_lanes,
                    fr.all_converged, fr.usable_linear, f.delay_changed, f.delay_solid, f.delay_samples);
        for (int g = 0; g < G2AG_COUNT; ++g) {
            if (g != G2AG_FRONT && !c->post) continue;
            g2_vline(grid, c->name, G2AG_NAME[g], hp, hh[g]);
            chain[g] = g2_chain(chain[g], hh[g]);
        }
        if ((hp % G2_STRIDE) == G2_STRIDE - 1 || hp == n_hops - 1 || emit_next || ev_now) {
            for (int g = 0; g < G2AG_COUNT; ++g) {
                if (g != G2AG_FRONT && !c->post) continue;
                g2_line(grid, c->name, G2AG_NAME[g], hp, chain[g]);
            }
        }
        emit_next = ev_now;
    }
    {
        uint64_t k[8];
        g2a_counters(h, k);
        g2_line(grid, c->name, "END_far_fft", 0, k[0]);
        g2_line(grid, c->name, "END_realign_warm", 0, k[1]);
        g2_line(grid, c->name, "END_realign_soft", 0, k[2]);
        g2_line(grid, c->name, "END_pending", 0, k[3]);
        g2_line(grid, c->name, "END_duty_total", 0, k[4]);
        g2_line(grid, c->name, "END_duty_run", 0, k[5]);
        g2_line(grid, c->name, "END_matched", 0, k[6]);
        g2_line(grid, c->name, "END_lanes", 0, k[7]);
        fprintf(stderr, "  census %-10s %-12s hops %5d dt>0.2 %4d sat>0.5 %4d drift %4d laneguards %4d usable %4d conv %4d "
                "changed %2d solid %4d warm %llu soft %llu duty %llu/%llu\n", grid, c->name, cen.hops, cen.dt02, cen.sat05,
                cen.drift, cen.guard, cen.usable, cen.conv, cen.changed, cen.solid, (unsigned long long)k[1],
                (unsigned long long)k[2], (unsigned long long)k[5], (unsigned long long)k[4]);
    }
    g2a_destroy(h);
    free(ref); free(mics); free(W); free(Ebf); free(rmem);
    return 0;
}

#endif /* AEC_TEST_GOLDEN2_ARRAY_H */
