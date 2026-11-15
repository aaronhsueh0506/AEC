/* test_dsp_refusal.c -- a hop whose results are not finite (finite input samples whose products overflow) is refused with
 * AEC_ERR_DSP and the instance is reset (aec.h, AecStatus).
 *
 * Single channel, three grids, residual OFF / STATE / ESTIMATE / FULL, driven by aec_process(), by the split pair and by
 * aec_process_linear_shared_far_from(); one poisoned sample on hop POISON_HOP. Per row: the refused call and hop, `out` zeroed,
 * the reset serial moved, the views after the refusal, then every later hop bit-identical to a fresh instance started on the
 * hop after the refusal. A sweep of large finite samples checks that every accepted hop leaves finite outputs and power
 * trackers; silence and full-scale signals are never refused. Non-finite input samples stay AEC_ERR_ARG with the instance untouched. One array row covers the shared reset
 * path.
 *
 * Run from c_impl/:  make test-dsp-refusal */
#include "aec.h"
#include "aec_internal.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NB_MAX     513
#define HOP_MAX    512
#define POISON_HOP 50
#define HOPS       140
#define POISON_AT  3

static int g_fail = 0;
static long g_rows = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

typedef struct { int sr, fft; const char* name; } Grid;
static const Grid GRIDS[3] = { {16000, 256, "16000/256"}, {16000, 512, "16000/512"}, {48000, 1024, "48000/1024"} };
static const char* const MODE_NAME[4] = { "OFF", "STATE", "ESTIMATE", "FULL" };

typedef enum { E_PROCESS, E_SPLIT, E_SHARED } Entry;
static const char* const ENTRY_NAME[3] = { "aec_process", "linear+residual", "shared_far_from" };

/* Where a poison is expected to be refused: in the linear stage (linear view not finite) or by the call that closes the hop
 * (residual view or power trackers not finite). */
typedef enum { AT_LINEAR, AT_CLOSE } Where;

typedef struct {
    const char* name;
    int on_ref;          /* 0: the sample is on the microphone, 1: on the reference */
    float value;
    Where where;         /* with AEC_RESIDUAL_OFF the linear call is the closing one */
} Poison;

/* All refused on POISON_HOP itself. mic / ref 3e38: the linear stage's spectra overflow. ref 1e20 and mic 1e25: the linear view
 * stays finite; the hop is refused when it closes. */
static const Poison POISONS[] = {
    { "mic 3e38", 0, 3.0e38f, AT_LINEAR },
    { "ref 3e38", 1, 3.0e38f, AT_LINEAR },
    { "ref 1e20", 1, 1.0e20f, AT_CLOSE },
    { "mic 1e25", 0, 1.0e25f, AT_CLOSE },
};
#define N_POISONS ((int)(sizeof(POISONS) / sizeof(POISONS[0])))

/* An echo scene: far = noise, mic = delayed far plus a small near-end noise. */
static float g_far[HOPS][HOP_MAX], g_mic[HOPS][HOP_MAX];
static void make_scene(int hop) {
    uint32_t s = 12345u;
    static float far_lin[HOPS * HOP_MAX];
    int h, i, n = HOPS * hop;
    for (i = 0; i < n; ++i) { s = s * 1664525u + 1013904223u; far_lin[i] = ((float)(s >> 9) / 4194304.0f - 1.0f) * 0.3f; }
    for (h = 0; h < HOPS; ++h)
        for (i = 0; i < hop; ++i) {
            int j = h * hop + i - 40;
            s = s * 1664525u + 1013904223u;
            g_far[h][i] = far_lin[h * hop + i];
            g_mic[h][i] = 0.5f * (j >= 0 ? far_lin[j] : 0.0f) + 0.01f * (((float)(s >> 9) / 4194304.0f - 1.0f));
        }
}

static int all_finite(const float* x, int n) {
    int i;
    if (!x) return 1;
    for (i = 0; i < n; ++i) if (!isfinite(x[i])) return 0;
    return 1;
}
static int all_zero(const float* x, int n) {
    int i;
    for (i = 0; i < n; ++i) if (x[i] != 0.0f) return 0;
    return 1;
}

static Aec* make(const Grid* g, int mode) {
    AecConfig c;
    aec_config_from_level(&c, AEC_LEVEL_BALANCED, g->sr);
    c.fft_size = g->fft;
    c.residual = (AecResidualMode)mode;
    return aec_create(&c);
}

/* Everything a caller can read after an accepted hop is finite (views as far as the mode reaches, out, the power trackers). */
static int hop_results_finite(const Aec* a, int mode, const float* out, int hop) {
    AecLinearView lv;
    AecResidualView rv;
    AecDebugStatus st;
    int n, ok = 1;
    memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
    if (aec_get_linear(a, &lv) != AEC_OK) return 0;
    n = lv.n_freqs;
    ok = all_finite(lv.linear_hop, hop) && all_finite((const float*)lv.echo_spec, 2 * n) &&
         all_finite((const float*)lv.near_spec, 2 * n) && all_finite((const float*)lv.far_spec, 2 * n);
    if (mode != AEC_RESIDUAL_OFF) {
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        if (aec_get_residual(a, &rv) != AEC_OK) return 0;
        ok = ok && all_finite(rv.formed_hop, hop) && all_finite((const float*)rv.error_spec, 2 * n) &&
             all_finite((const float*)rv.echo_spec, 2 * n) && all_finite((const float*)rv.near_spec, 2 * n) &&
             all_finite((const float*)rv.base_spec, 2 * n) && all_finite(rv.r2, n) && all_finite(rv.comfort_noise, n) &&
             all_finite(rv.res_gain, n);
    }
    aec_debug_status(a, &st);
    return ok && all_finite(out, out ? hop : 0) && isfinite(st.near_power) && isfinite(st.out_power);
}

/* What a hop leaves for comparison: out (aec_process) or the views (the split pair), packed into one float record. */
static int hop_record(const Aec* a, int mode, const float* out, int hop, float* rec) {
    AecLinearView lv;
    AecResidualView rv;
    int n, k = 0;
    memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
    if (aec_get_linear(a, &lv) != AEC_OK) return -1;
    n = lv.n_freqs;
    if (out) { memcpy(rec + k, out, (size_t)hop * sizeof(float)); k += hop; }
    memcpy(rec + k, lv.linear_hop, (size_t)hop * sizeof(float)); k += hop;
    if (mode != AEC_RESIDUAL_OFF) {
        memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
        if (aec_get_residual(a, &rv) != AEC_OK) return -1;
        memcpy(rec + k, rv.base_spec, (size_t)n * sizeof(Complex)); k += 2 * n;
        memcpy(rec + k, rv.error_spec, (size_t)n * sizeof(Complex)); k += 2 * n;
        if (rv.r2) { memcpy(rec + k, rv.r2, (size_t)n * sizeof(float)); k += n; }
        if (rv.res_gain) { memcpy(rec + k, rv.res_gain, (size_t)n * sizeof(float)); k += n; }
    }
    return k;
}

/* One hop through the chosen entry. *stage is set to AT_LINEAR or AT_CLOSE when a call refuses, naming the call. */
static int run_hop(Aec* a, Aec* donor, const float* donor_mic, Entry e, int mode, const float* mic, const float* ref, float* out,
                   Where* stage) {
    int rc;
    if (e == E_PROCESS) {
        rc = aec_process(a, mic, ref, out);
        *stage = AT_CLOSE;   /* refined by the caller from the linear-view state */
        return rc;
    }
    if (e == E_SHARED) {
        if (aec_process_linear(donor, donor_mic, ref) != AEC_OK) return -100;
        rc = aec_process_linear_shared_far_from(a, mic, ref, donor);
    } else {
        rc = aec_process_linear(a, mic, ref);
    }
    *stage = mode == AEC_RESIDUAL_OFF ? AT_CLOSE : AT_LINEAR;
    if (rc != AEC_OK || mode == AEC_RESIDUAL_OFF) {
        if (e == E_SHARED && mode != AEC_RESIDUAL_OFF) (void)aec_process_residual(donor, NULL);
        return rc;
    }
    if (e == E_SHARED) (void)aec_process_residual(donor, NULL);
    *stage = AT_CLOSE;
    return aec_process_residual(a, NULL);
}

/* One poisoned row: grid x mode x entry x poison. */
static void run_row(const Grid* g, int mode, Entry e, const Poison* p) {
    Aec* a = make(g, mode);
    Aec* donor = e == E_SHARED ? make(g, mode) : NULL;
    Aec* fresh = NULL;
    Aec* fresh_donor = NULL;
    static float out[HOP_MAX], out_f[HOP_MAX], mic[HOP_MAX], ref[HOP_MAX];
    static float rec[4 * HOP_MAX + 8 * NB_MAX], rec_f[4 * HOP_MAX + 8 * NB_MAX];
    int hop, h, refused = -1, compared = 0, differ = 0, accepted_bad = 0;
    char tag[160];
    snprintf(tag, sizeof tag, "%s %s %s %s", g->name, MODE_NAME[mode], ENTRY_NAME[e], p->name);
    g_rows++;
    if (!a || (e == E_SHARED && !donor)) { CHECK(0, "%s: create", tag); aec_destroy(a); aec_destroy(donor); return; }
    hop = aec_hop_size(a);
    for (h = 0; h < HOPS; ++h) {
        int rc;
        Where st;
        uint32_t serial_before = aec_post_reset_serial(a);
        memcpy(mic, g_mic[h], (size_t)hop * sizeof(float));
        memcpy(ref, g_far[h], (size_t)hop * sizeof(float));
        if (h == POISON_HOP) { if (p->on_ref) ref[POISON_AT] = p->value; else mic[POISON_AT] = p->value; }
        for (int i = 0; i < hop; ++i) out[i] = 7.0f;
        rc = run_hop(a, donor, g_mic[h], e, mode, mic, ref, e == E_PROCESS ? out : NULL, &st);
        if (fresh) {
            int rf = run_hop(fresh, fresh_donor, g_mic[h], e, mode, mic, ref, e == E_PROCESS ? out_f : NULL, &st);
            int k = hop_record(a, mode, e == E_PROCESS ? out : NULL, hop, rec);
            int kf = hop_record(fresh, mode, e == E_PROCESS ? out_f : NULL, hop, rec_f);
            compared++;
            if (rc != AEC_OK || rf != AEC_OK || k < 0 || k != kf || memcmp(rec, rec_f, (size_t)k * sizeof(float)) != 0) differ++;
            continue;
        }
        if (rc == AEC_OK) {
            if (!hop_results_finite(a, mode, e == E_PROCESS ? out : NULL, hop)) accepted_bad++;
            continue;
        }
        /* the refused hop */
        refused = h;
        CHECK(rc == AEC_ERR_DSP, "%s: hop %d refused with %d, want AEC_ERR_DSP", tag, h, rc);
        if (e == E_PROCESS) {
            CHECK(all_zero(out, hop), "%s: aec_process() out is zeroed on the refused hop", tag);
            st = mode == AEC_RESIDUAL_OFF ? AT_CLOSE : p->where;   /* one call: the stage is not observable */
        }
        {
            Where want = mode == AEC_RESIDUAL_OFF ? AT_CLOSE : p->where;
            CHECK(st == want, "%s: refused by the %s call, want the %s one", tag, st == AT_LINEAR ? "linear" : "closing",
                  want == AT_LINEAR ? "linear" : "closing");
        }
        CHECK(h == POISON_HOP, "%s: refused on hop %d, want %d", tag, h, POISON_HOP);
        CHECK(aec_post_reset_serial(a) != serial_before, "%s: the reset serial moves on the refusal", tag);
        {
            AecLinearView lv;
            AecResidualView rv;
            memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
            memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
            CHECK(aec_get_linear(a, &lv) == AEC_OK && all_zero(lv.linear_hop, hop) &&
                  all_zero((const float*)lv.echo_spec, 2 * lv.n_freqs) && all_zero((const float*)lv.near_spec, 2 * lv.n_freqs) &&
                  all_zero((const float*)lv.far_spec, 2 * lv.n_freqs),
                  "%s: the linear view reads zeros after the refusal", tag);
            CHECK(aec_get_residual(a, &rv) == (mode == AEC_RESIDUAL_OFF ? AEC_ERR_MODE : AEC_ERR_SEQUENCE),
                  "%s: no residual view after the refusal", tag);
            if (mode == AEC_RESIDUAL_FULL) {
                static Complex spec[NB_MAX];
                CHECK(aec_apply_gain(a, NULL, spec) == AEC_ERR_SEQUENCE, "%s: no gain after the refusal", tag);
            }
            if (st == AT_LINEAR && mode != AEC_RESIDUAL_OFF && e != E_PROCESS)
                CHECK(aec_process_residual(a, NULL) == AEC_ERR_SEQUENCE, "%s: a refused linear stage leaves no hop open", tag);
        }
        /* recovery: from the next hop, a fresh instance (and, sharing, a fresh donor) fed the same hops */
        fresh = make(g, mode);
        if (e == E_SHARED) {
            /* the donor was not refused: restart it too, so both pairs see the same far history */
            aec_reset(donor);
            fresh_donor = make(g, mode);
        }
        if (!fresh || (e == E_SHARED && !fresh_donor)) { CHECK(0, "%s: fresh create", tag); break; }
    }
    CHECK(refused >= 0, "%s: never refused", tag);
    CHECK(accepted_bad == 0, "%s: %d accepted hops left a non-finite result", tag, accepted_bad);
    CHECK(refused < 0 || (compared == HOPS - 1 - refused && differ == 0),
          "%s: %d of %d hops after the refusal differ from a fresh instance", tag, differ, compared);
    aec_destroy(a); aec_destroy(donor); aec_destroy(fresh); aec_destroy(fresh_donor);
}

/* Large finite samples, mic and ref, every mode, aec_process(): an accepted hop never leaves a non-finite result. */
static void run_sweep(const Grid* g) {
    static const float values[] = { 1e10f, 1e15f, 1e19f, 2e19f, 1e20f, 1e25f, 1e30f, 1e35f, 3e38f, FLT_MAX, -FLT_MAX };
    static float out[HOP_MAX], mic[HOP_MAX], ref[HOP_MAX];
    int v, side, mode, h, hop;
    for (mode = 0; mode < 4; ++mode)
        for (side = 0; side < 2; ++side)
            for (v = 0; v < (int)(sizeof(values) / sizeof(values[0])); ++v) {
                Aec* a = make(g, mode);
                int bad = 0, refused = 0;
                if (!a) { CHECK(0, "sweep create"); return; }
                hop = aec_hop_size(a);
                g_rows++;
                for (h = 0; h < 90; ++h) {
                    int rc;
                    memcpy(mic, g_mic[h], (size_t)hop * sizeof(float));
                    memcpy(ref, g_far[h], (size_t)hop * sizeof(float));
                    if (h >= POISON_HOP && h < POISON_HOP + 3) { if (side) ref[POISON_AT + h] = values[v]; else mic[POISON_AT + h] = values[v]; }
                    rc = aec_process(a, mic, ref, out);
                    if (rc == AEC_OK) bad += !hop_results_finite(a, mode, out, hop);
                    else if (rc == AEC_ERR_DSP) { refused++; bad += !all_zero(out, hop); }
                    else bad++;
                }
                CHECK(bad == 0, "sweep %s %s %s %g: %d hops with a non-finite result or a bad refusal", g->name, MODE_NAME[mode],
                      side ? "ref" : "mic", (double)values[v], bad);
                CHECK(values[v] < 1e20f || refused > 0, "sweep %s %s %s %g: an overflowing sample is refused", g->name,
                      MODE_NAME[mode], side ? "ref" : "mic", (double)values[v]);
                aec_destroy(a);
            }
}

/* Legal extremes are never refused: digital silence, a full-scale square wave on both inputs, full-scale DC of opposite signs.
 * Every hop of every mode, through aec_process() and the split pair, is AEC_OK with finite results. */
static void run_full_scale_controls(const Grid* g) {
    static float out[HOP_MAX], mic[HOP_MAX], ref[HOP_MAX];
    int sig, mode, e, h, i, hop;
    for (sig = 0; sig < 3; ++sig)
        for (mode = 0; mode < 4; ++mode)
            for (e = 0; e < 2; ++e) {
                Aec* a = make(g, mode);
                int bad = 0;
                if (!a) { CHECK(0, "control create"); return; }
                hop = aec_hop_size(a);
                g_rows++;
                for (h = 0; h < 300; ++h) {
                    Where st;
                    for (i = 0; i < hop; ++i) {
                        int n = h * hop + i;
                        float sq = ((n / 16) & 1) ? 1.0f : -1.0f;
                        ref[i] = sig == 0 ? 0.0f : sig == 1 ? sq : -1.0f;
                        mic[i] = sig == 0 ? 0.0f : sig == 1 ? sq : 1.0f;
                    }
                    if (run_hop(a, NULL, NULL, (Entry)e, mode, mic, ref, e == E_PROCESS ? out : NULL, &st) != AEC_OK ||
                        !hop_results_finite(a, mode, e == E_PROCESS ? out : NULL, hop)) bad++;
                }
                CHECK(bad == 0, "control %s %s %s %s: %d hops refused or not finite", g->name, MODE_NAME[mode], ENTRY_NAME[e],
                      sig == 0 ? "silence" : sig == 1 ? "full-scale square" : "full-scale DC", bad);
                aec_destroy(a);
            }
}

/* A non-finite input sample is AEC_ERR_ARG and changes nothing: the instance continues as if the hop never happened. */
static void run_nonfinite_inputs(const Grid* g) {
    static float out[HOP_MAX], out_t[HOP_MAX], mic[HOP_MAX], ref[HOP_MAX];
    const float bad_values[3] = { NAN, INFINITY, -INFINITY };
    int b, side, h, hop, differ;
    for (b = 0; b < 3; ++b)
        for (side = 0; side < 2; ++side) {
            Aec* a = make(g, AEC_RESIDUAL_FULL);
            Aec* t = make(g, AEC_RESIDUAL_FULL);
            if (!a || !t) { CHECK(0, "nonfinite create"); aec_destroy(a); aec_destroy(t); return; }
            hop = aec_hop_size(a);
            g_rows++;
            differ = 0;
            for (h = 0; h < 80; ++h) {
                if (h == POISON_HOP) {
                    uint32_t serial = aec_post_reset_serial(a);
                    memcpy(mic, g_mic[h], (size_t)hop * sizeof(float));
                    memcpy(ref, g_far[h], (size_t)hop * sizeof(float));
                    if (side) ref[POISON_AT] = bad_values[b]; else mic[POISON_AT] = bad_values[b];
                    for (int i = 0; i < hop; ++i) out[i] = 7.0f;
                    CHECK(aec_process(a, mic, ref, out) == AEC_ERR_ARG, "%s nonfinite %d/%d: AEC_ERR_ARG", g->name, b, side);
                    CHECK(out[0] == 7.0f && out[hop - 1] == 7.0f, "%s nonfinite %d/%d: out untouched", g->name, b, side);
                    CHECK(aec_post_reset_serial(a) == serial, "%s nonfinite %d/%d: no reset", g->name, b, side);
                    CHECK(aec_process_linear(a, side ? g_mic[h] : mic, side ? ref : g_far[h]) == AEC_ERR_ARG,
                          "%s nonfinite %d/%d: aec_process_linear AEC_ERR_ARG", g->name, b, side);
                    continue;
                }
                if (aec_process(a, g_mic[h], g_far[h], out) != AEC_OK || aec_process(t, g_mic[h], g_far[h], out_t) != AEC_OK ||
                    memcmp(out, out_t, (size_t)hop * sizeof(float)) != 0) differ++;
            }
            CHECK(differ == 0, "%s nonfinite %d/%d: %d hops differ from an instance that never saw the hop", g->name, b, side,
                  differ);
            aec_destroy(a); aec_destroy(t);
        }
}

#ifndef AEC_NO_ARRAY
/* The array answers the same way through the same reset: ESTIMATE, mic 3e38 on lane 0, refused by the fusion. */
static void run_array_row(void) {
    AecConfig c;
    Aec *a, *fresh = NULL;
    static Complex w[4 * 129];
    static float mics[4 * 128];
    int h, i, hop, n, refused = -1, differ = 0;
    aec_config_from_level(&c, AEC_LEVEL_BALANCED, 16000);
    c.n_mics = AEC_ARRAY_MICS;
    c.residual = AEC_RESIDUAL_ESTIMATE;
    a = aec_create(&c);
    if (!a) { CHECK(0, "array create"); return; }
    hop = aec_hop_size(a);
    n = hop + 1;
    g_rows++;
    for (i = 0; i < 4 * n; ++i) { w[i].r = i < n ? 1.0f : 0.0f; w[i].i = 0.0f; }
    for (h = 0; h < 100; ++h) {
        AecBeam beam = { w, NULL };
        int rc;
        for (i = 0; i < hop; ++i) for (int ch = 0; ch < 4; ++ch) mics[i * 4 + ch] = g_mic[h][i];
        if (h == POISON_HOP) mics[POISON_AT * 4] = 3.0e38f;
        rc = aec_process_linear(a, mics, g_far[h]);
        if (rc == AEC_OK) rc = aec_process_residual(a, &beam);
        if (fresh) {
            AecResidualView r1, r2;
            int rf = aec_process_linear(fresh, mics, g_far[h]);
            if (rf == AEC_OK) rf = aec_process_residual(fresh, &beam);
            memset(&r1, 0, sizeof r1); r1.struct_size = sizeof r1;
            memset(&r2, 0, sizeof r2); r2.struct_size = sizeof r2;
            if (rc != AEC_OK || rf != AEC_OK || aec_get_residual(a, &r1) != AEC_OK || aec_get_residual(fresh, &r2) != AEC_OK ||
                memcmp(r1.error_spec, r2.error_spec, (size_t)n * sizeof(Complex)) != 0 ||
                memcmp(r1.r2, r2.r2, (size_t)n * sizeof(float)) != 0) differ++;
            continue;
        }
        if (rc == AEC_OK) continue;
        refused = h;
        CHECK(rc == AEC_ERR_DSP && h == POISON_HOP, "array: hop %d refused with %d", h, rc);
        fresh = aec_create(&c);
        if (!fresh) break;
    }
    CHECK(refused == POISON_HOP && differ == 0, "array: refused on hop %d, %d later hops differ from a fresh instance", refused,
          differ);
    aec_destroy(a); aec_destroy(fresh);
}
#endif

int main(void) {
    int g, mode, e, p;
    for (g = 0; g < 3; ++g) {
            make_scene(GRIDS[g].fft / 2);
        for (mode = 0; mode < 4; ++mode)
            for (e = 0; e < 3; ++e) {
                if (e == E_SHARED && g != 0) continue;
                for (p = 0; p < N_POISONS; ++p) {
                    if (e == E_SHARED && POISONS[p].on_ref) continue;   /* the donor shares the reference: mic poisons only */
                    run_row(&GRIDS[g], mode, (Entry)e, &POISONS[p]);
                }
            }
        run_sweep(&GRIDS[g]);
        run_full_scale_controls(&GRIDS[g]);
        run_nonfinite_inputs(&GRIDS[g]);
    }
#ifndef AEC_NO_ARRAY
    make_scene(128);
    run_array_row();
#endif
    if (g_fail) { printf("test_dsp_refusal: %d failures over %ld rows\n", g_fail, g_rows); return 1; }
    printf("test_dsp_refusal: %ld rows passed\n", g_rows);
    return 0;
}
