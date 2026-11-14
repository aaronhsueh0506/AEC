/* test_split_entry.c -- the three AEC entry points (aec_process / aec_process_linear / aec_process_residual), their
 * views and the gain / comfort-noise helpers, on three grids. Modes: OFF, STATE, ESTIMATE and FULL are the AecConfig.residual
 * levels driven by aec_process_linear + aec_process_residual; APPLY is FULL driven by aec_process. Every field of every hop
 * is hashed into a line stream (AEC_TEST_BASELINE_WRITE / AEC_TEST_BASELINE: capture it from one version, compare against it
 * from the next; no baseline is shipped).
 *
 * Checked here with no baseline: aec_process (apply) and the split pair leave the same residual view; the instance helpers
 * aec_apply_gain / aec_apply_cng equal the explicit-array forms; calls out of order are refused without touching the
 * instance.
 *
 * Run from c_impl/:  make test-split-entry */
#include "aec.h"
#include "aec_white_box.h"
#include "../src/aec_res_internal.h"
#include "utility/simd_kernels.h"
#include "split_signal.h"
#include "res_gain_of.h"
#include "aec_pool_test_util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { M_OFF, M_STATE, M_ESTIMATE, M_FULL, M_APPLY, M_ARES, M_COUNT } Mode;
static const char* const MODE_NAME[M_COUNT] = { "OFF", "STATE", "ESTIMATE", "FULL", "APPLY", "ARES" };

typedef struct { int sr, fft; const char* name; } Grid;
static const Grid GRIDS[3] = { {16000, 256, "16000/256"}, {16000, 512, "16000/512"}, {48000, 1024, "48000/1024"} };

#define NB_MAX 1025

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

/* ---- baseline ---- */
#include "baseline_stream.h"

static void emit(const Grid* g, const char* mode, const char* field, int hop, uint64_t h) {
    char hex[17], line[160];
    split_hash_hex(hex, h);
    snprintf(line, sizeof line, "%s %s %s %d %s", g->name, mode, field, hop, hex);
    baseline_line(line);
}

static void make_cfg(AecConfig* c, const Grid* g, Mode m) {
    aec_config_from_level(c, AEC_LEVEL_BALANCED, g->sr);
    c->fft_size = g->fft;
    switch (m) {
    case M_OFF:      c->residual = AEC_RESIDUAL_OFF; break;
    case M_STATE:    c->residual = AEC_RESIDUAL_STATE; break;
    case M_ESTIMATE:
    case M_ARES:     c->residual = AEC_RESIDUAL_ESTIMATE; break;
    case M_FULL:
    case M_APPLY:    c->residual = AEC_RESIDUAL_FULL; break;
    default: break;
    }
}

static void* pool_alloc(size_t bytes) { return aec_test_pool_alloc_fill(bytes, 0xA5); }

static Aec* new_aec(const AecConfig* cfg, void** mem_out) { return aec_test_new_aec(cfg, mem_out, 0xA5); }

static void views(const Aec* a, AecLinearView* lv, AecResidualView* rv, int* have_rv) {
    memset(lv, 0, sizeof *lv); lv->struct_size = sizeof *lv;
    memset(rv, 0, sizeof *rv); rv->struct_size = sizeof *rv;
    CHECK(aec_get_linear(a, lv) == AEC_OK, "aec_get_linear");
    *have_rv = (aec_get_residual(a, rv) == AEC_OK);
}

/* ---- fields common to every non-APPLY mode ---- */
static void dump_common(const Grid* g, const char* mode, int hop, const Aec* a, const AecLinearView* lv,
                        const AecResidualView* rv, int have_rv) {
    uint32_t ls[5];
    uint64_t h;
    int usable = have_rv ? rv->usable_linear : 0;
    int floor_protect = have_rv ? rv->res_floor_protect : 0;
    int guard = have_rv ? rv->guard_active : 0;
    emit(g, mode, "linear_hop", hop, split_hash_f32(lv->linear_hop, lv->hop_size));
    emit(g, mode, "far_spec", hop, split_hash_complex(lv->far_spec, lv->n_freqs));
    emit(g, mode, "lin_formed", hop,
         split_hash_f32(have_rv ? rv->formed_hop : lv->linear_hop, lv->hop_size));
    emit(g, mode, "lin_far", hop, split_hash_f32(lv->aligned_far_hop, lv->hop_size));
    ls[0] = (uint32_t)lv->hop_size; ls[1] = (uint32_t)lv->delay_samples; ls[2] = split_f32_bits(lv->delay_confidence);
    ls[3] = (uint32_t)lv->delay_state; ls[4] = lv->generation;
    emit(g, mode, "lin_state", hop, split_hash_u32(ls, 5));
    emit(g, mode, "post_serial", hop, (uint64_t)aec_post_reset_serial(a));

    h = SPLIT_FNV_BASIS;
    h = split_fnv_u32(h, (uint32_t)usable);
    h = split_fnv_u32(h, (uint32_t)lv->filter_converged);
    h = split_fnv_u32(h, (uint32_t)lv->epc_active);
    h = split_fnv_u32(h, (uint32_t)lv->is_stationary_dt);
    h = split_fnv_u32(h, (uint32_t)floor_protect);
    h = split_fnv_u32(h, (uint32_t)guard);
    h = split_fnv_f32(h, lv->dt_indicator);
    h = split_fnv_f32(h, lv->far_power);
    h = split_fnv_f32(h, lv->erle_factor);
    h = split_fnv_f32(h, lv->saturation_level);
    h = split_fnv_f32(h, lv->erl_estimate);
    emit(g, mode, "flags", hop, h);
    {
        int v[7] = { usable, lv->filter_converged, lv->epc_active, lv->is_stationary_dt, floor_protect,
                     guard, lv->filter_once_converged };
        uint64_t packed = 0;
        for (int i = 0; i < 7; ++i) packed |= (uint64_t)(v[i] & 0xF) << (60 - 4 * i);
        emit(g, mode, "flag_ints", hop, packed);
    }
    h = SPLIT_FNV_BASIS;
    h = split_fnv_u32(h, (uint32_t)lv->filter_once_converged);
    h = split_fnv_f32(h, lv->divergence);
    h = split_fnv_f32(h, lv->shadow_dt);
    emit(g, mode, "flags_extra", hop, h);
}

/* The apply computations with an explicit-array reference (what a caller holding its own state does). */
typedef struct { uint32_t rng, rng_null, serial; } SpecState;

static void apply_hop(const Grid* g, const char* mode, const char* pfx, int hop, Aec* a, Aec* a_null,
                      SpecState* st, const float* gain, const Complex* base, const float* cn, int n, int use_lib) {
    char f[48];
    float gx[NB_MAX];
    Complex p1[NB_MAX], p2[NB_MAX], q1[NB_MAX], q2[NB_MAX];
    uint32_t serial = aec_post_reset_serial(a);
    if (serial != st->serial) { st->rng = st->rng_null = AEC_CNG_SEED; st->serial = serial; }
    for (int k = 0; k < n; ++k) gx[k] = 0.35f + 0.6f * (0.5f + 0.5f * sinf(0.07f * k + 0.11f * hop));

    /* explicit-array reference */
    aec_apply_gain_spec(q1, base, gain, gx, n);
    memcpy(q2, q1, (size_t)n * sizeof(Complex));
    aec_apply_cng_spec(&st->rng, q2, cn, gain, gx, n);
    if (use_lib) {
        /* the instance forms must agree with it, and are what the baseline line is taken from */
        CHECK(aec_apply_gain(a, gx, p1) == AEC_OK, "aec_apply_gain hop %d", hop);
        memcpy(p2, p1, (size_t)n * sizeof(Complex));
        CHECK(aec_apply_cng(a, gx, p2) == AEC_OK, "aec_apply_cng hop %d", hop);
        CHECK(split_hash_complex(p1, n) == split_hash_complex(q1, n), "apply_gain instance != array form hop %d", hop);
        CHECK(split_hash_complex(p2, n) == split_hash_complex(q2, n), "apply_cng instance != array form hop %d", hop);
    } else {
        memcpy(p1, q1, (size_t)n * sizeof(Complex));
        memcpy(p2, q2, (size_t)n * sizeof(Complex));
    }
    snprintf(f, sizeof f, "%sapply_gain", pfx); emit(g, mode, f, hop, split_hash_complex(p1, n));
    snprintf(f, sizeof f, "%sapply_cng", pfx); emit(g, mode, f, hop, split_hash_complex(p2, n));
    if (a_null) {
        /* a second instance fed the same signal produces the NULL-extra lines from its own generator */
        aec_apply_gain_spec(q1, base, gain, NULL, n);
        memcpy(q2, q1, (size_t)n * sizeof(Complex));
        aec_apply_cng_spec(&st->rng_null, q2, cn, gain, NULL, n);
        CHECK(aec_apply_gain(a_null, NULL, p1) == AEC_OK, "aec_apply_gain NULL hop %d", hop);
        memcpy(p2, p1, (size_t)n * sizeof(Complex));
        CHECK(aec_apply_cng(a_null, NULL, p2) == AEC_OK, "aec_apply_cng NULL hop %d", hop);
        CHECK(split_hash_complex(p1, n) == split_hash_complex(q1, n), "apply_gain_null instance != array form hop %d", hop);
        CHECK(split_hash_complex(p2, n) == split_hash_complex(q2, n), "apply_cng_null instance != array form hop %d", hop);
        snprintf(f, sizeof f, "%sapply_gain_null", pfx); emit(g, mode, f, hop, split_hash_complex(p1, n));
        snprintf(f, sizeof f, "%sapply_cng_null", pfx); emit(g, mode, f, hop, split_hash_complex(p2, n));
    }
}

static int run_mode(const Grid* g, Mode m, int n_hops) {
    AecConfig cfg;
    void *mem = NULL, *mem2 = NULL, *mem3 = NULL;
    make_cfg(&cfg, g, m);
    Aec* a = new_aec(&cfg, &mem);
    if (!a) { printf("FAIL: aec_init %s %s\n", g->name, MODE_NAME[m]); g_fail++; return -1; }
    int hop_size = aec_hop_size(a);
    float ref[SPLIT_SIGNAL_MAX_HOP], mic[SPLIT_SIGNAL_MAX_HOP], out[SPLIT_SIGNAL_MAX_HOP], out2[SPLIT_SIGNAL_MAX_HOP];
    SpecState ast; ast.rng = ast.rng_null = AEC_CNG_SEED; ast.serial = 0u;
    Aec* a_null = NULL;       /* FULL: second instance for the NULL-extra lines */
    Aec* a_proc = NULL;       /* FULL: aec_process twin, must leave the same residual view */
    AecRes* res = NULL; void* rmem = NULL;
    if (m == M_FULL) {
        a_null = new_aec(&cfg, &mem2);
        a_proc = new_aec(&cfg, &mem3);
        if (!a_null || !a_proc) { printf("FAIL: second aec_init\n"); g_fail++; return -1; }
    }
    if (m == M_ARES) {
        size_t rsz = aec_res_get_mem_size(&cfg);
        rmem = pool_alloc(rsz);
        res = (rsz && rmem) ? aec_res_init(rmem, rsz, &cfg) : NULL;
        if (!res) { printf("FAIL: aec_res_init\n"); g_fail++; return -1; }
    }
    const char* mn = MODE_NAME[m];
    for (int hop = 0; hop < n_hops; ++hop) {
        AecLinearView lv; AecResidualView rv; int have_rv;
        if (split_signal_fill(g->sr, hop_size, hop, ref, mic) != 0) { printf("FAIL: fill\n"); g_fail++; return -1; }
        if (m == M_APPLY) {
            CHECK(aec_process(a, mic, ref, out) == AEC_OK, "aec_process hop %d", hop);
            emit(g, mn, "out", hop, split_hash_f32(out, hop_size));
            continue;
        }
        CHECK(aec_process_linear(a, mic, ref) == AEC_OK, "aec_process_linear %s hop %d", mn, hop);
        if (m == M_OFF) {
            CHECK(aec_process_residual(a, NULL) == AEC_ERR_MODE, "residual on OFF must be AEC_ERR_MODE");
            views(a, &lv, &rv, &have_rv);
            CHECK(!have_rv, "OFF must have no residual view");
            dump_common(g, mn, hop, a, &lv, &rv, 0);
            emit(g, mn, "echo_spec", hop, split_hash_complex(lv.echo_spec, lv.n_freqs));
            emit(g, mn, "near_spec", hop, split_hash_complex(lv.near_spec, lv.n_freqs));
            continue;
        }
        /* a view requested while the residual stage is pending is refused */
        {
            AecResidualView early; memset(&early, 0, sizeof early); early.struct_size = sizeof early;
            CHECK(aec_get_residual(a, &early) == AEC_ERR_SEQUENCE, "residual view before the residual stage");
        }
        CHECK(aec_process_residual(a, NULL) == AEC_OK, "aec_process_residual %s hop %d", mn, hop);
        views(a, &lv, &rv, &have_rv);
        CHECK(have_rv, "residual view %s hop %d", mn, hop);
        if (!have_rv) return -1;
        if (m != M_ARES) {
            dump_common(g, mn, hop, a, &lv, &rv, 1);
            emit(g, mn, "error_spec", hop, split_hash_complex(rv.error_spec, rv.n_freqs));
            emit(g, mn, "echo_spec", hop, split_hash_complex(rv.echo_spec, rv.n_freqs));
            emit(g, mn, "near_spec", hop, split_hash_complex(rv.near_spec, rv.n_freqs));
            emit(g, mn, "formed_hop", hop, split_hash_f32(rv.formed_hop, rv.hop_size));
            emit(g, mn, "base_spec", hop, split_hash_complex(rv.base_spec, rv.n_freqs));
            if (m == M_STATE) {
                CHECK(!rv.r2 && !rv.comfort_noise && !rv.res_gain, "STATE exposes r2/cn/gain");
                CHECK(res_gain_of(a) == NULL, "STATE gain");
            }
            if (m == M_ESTIMATE || m == M_FULL) {
                emit(g, mn, "r2", hop, split_hash_f32(rv.r2, rv.n_freqs));
                emit(g, mn, "comfort_noise", hop, split_hash_f32(rv.comfort_noise, rv.n_freqs));
            }
            if (m == M_ESTIMATE) CHECK(!rv.res_gain && res_gain_of(a) == NULL, "ESTIMATE exposes gain");
            if (m == M_FULL) {
                emit(g, mn, "res_gain", hop, split_hash_f32(rv.res_gain, rv.n_freqs));
                /* second instances, same signal */
                CHECK(aec_process_linear(a_null, mic, ref) == AEC_OK && aec_process_residual(a_null, NULL) == AEC_OK,
                      "a_null split hop %d", hop);
                CHECK(aec_process(a_proc, mic, ref, out2) == AEC_OK, "a_proc aec_process hop %d", hop);
                {
                    AecResidualView pv; memset(&pv, 0, sizeof pv); pv.struct_size = sizeof pv;
                    CHECK(aec_get_residual(a_proc, &pv) == AEC_OK, "a_proc view");
                    CHECK(split_hash_complex(pv.error_spec, pv.n_freqs) == split_hash_complex(rv.error_spec, rv.n_freqs),
                          "aec_process vs split: error_spec hop %d", hop);
                    CHECK(split_hash_complex(pv.base_spec, pv.n_freqs) == split_hash_complex(rv.base_spec, rv.n_freqs),
                          "aec_process vs split: base_spec hop %d", hop);
                    CHECK(split_hash_f32(pv.r2, pv.n_freqs) == split_hash_f32(rv.r2, rv.n_freqs),
                          "aec_process vs split: r2 hop %d", hop);
                    CHECK(split_hash_f32(pv.res_gain, pv.n_freqs) == split_hash_f32(rv.res_gain, rv.n_freqs),
                          "aec_process vs split: res_gain hop %d", hop);
                }
                apply_hop(g, mn, "", hop, a, a_null, &ast, rv.res_gain, rv.base_spec, rv.comfort_noise, rv.n_freqs, 1);
            }
        } else {
            AecResInput in;
            memset(&in, 0, sizeof in);
            in.error_spec = rv.error_spec; in.near_spec = rv.near_spec; in.r2 = rv.r2; in.comfort_noise = rv.comfort_noise;
            in.render_hop = ref; in.all_converged = lv.filter_converged; in.max_dt = lv.dt_indicator;
            in.max_saturation = lv.saturation_level; in.clock_drift = 0;
            const float* gr = aec_res_process(res, &in);
            if (!gr) { printf("FAIL: aec_res_process NULL hop %d\n", hop); g_fail++; return -1; }
            emit(g, mn, "res_gain_ares", hop, split_hash_f32(gr, rv.n_freqs));
            apply_hop(g, mn, "ares_", hop, a, NULL, &ast, gr, rv.base_spec, rv.comfort_noise, rv.n_freqs, 0);
        }
    }
    free(rmem); free(mem); free(mem2); free(mem3);
    return 0;
}

/* ---- order / argument checks that need no baseline ---- */
static void test_sequence(void) {
    AecConfig cfg;
    void* mem = NULL;
    float ref[SPLIT_SIGNAL_MAX_HOP], mic[SPLIT_SIGNAL_MAX_HOP], out[SPLIT_SIGNAL_MAX_HOP];
    Aec* a;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    a = new_aec(&cfg, &mem);
    CHECK(a != NULL, "sequence: aec_init");
    if (!a) return;
    split_signal_fill(16000, aec_hop_size(a), 0, ref, mic);
    CHECK(aec_process_residual(a, NULL) == AEC_ERR_SEQUENCE, "residual before any linear");
    CHECK(aec_far_fft_real_compute_count(NULL) == 0, "far FFT count of NULL");
    aec_apply_gain_spec(NULL, NULL, NULL, NULL, -1);           /* refused, no access */
    {
        Complex o[NB_MAX]; AecResidualView v; memset(&v, 0, sizeof v); v.struct_size = sizeof v;
        CHECK(res_gain_of(a) == NULL && aec_apply_gain(a, NULL, o) == AEC_ERR_SEQUENCE &&
              aec_apply_cng(a, NULL, o) == AEC_ERR_SEQUENCE && aec_get_residual(a, &v) == AEC_ERR_SEQUENCE,
              "residual getters before the first completed hop");
    }
    {
        float bad[SPLIT_SIGNAL_MAX_HOP]; memcpy(bad, mic, sizeof bad); bad[3] = NAN;
        CHECK(aec_process_linear(a, bad, ref) == AEC_ERR_ARG && aec_process(a, mic, bad, out) == AEC_ERR_ARG,
              "non-finite input is AEC_ERR_ARG");
        CHECK(aec_process_residual(a, NULL) == AEC_ERR_SEQUENCE, "a refused hop opens nothing");
    }
    CHECK(aec_process_linear(NULL, mic, ref) == AEC_ERR_ARG, "linear NULL instance");
    CHECK(aec_process_linear(a, NULL, ref) == AEC_ERR_ARG, "linear NULL mic");
    CHECK(aec_process(a, mic, ref, NULL) == AEC_ERR_ARG, "process NULL out");
    CHECK(aec_process_linear(a, mic, ref) == AEC_OK, "linear");
    CHECK(aec_process_linear(a, mic, ref) == AEC_ERR_SEQUENCE, "linear twice");
    CHECK(aec_process(a, mic, ref, out) == AEC_ERR_SEQUENCE, "process while a hop is open");
    {
        Complex o[NB_MAX];
        CHECK(res_gain_of(a) == NULL, "gain while pending");
        CHECK(aec_apply_gain(a, NULL, o) == AEC_ERR_SEQUENCE, "apply_gain while pending");
    }
    CHECK(aec_process_residual(a, (const AecBeam*)mic) == AEC_ERR_ARG, "residual with a beam on a single-channel instance");
    CHECK(aec_process_residual(a, NULL) == AEC_OK, "residual");
    CHECK(aec_process_residual(a, NULL) == AEC_ERR_SEQUENCE, "residual twice");
    {
        Complex o[NB_MAX];
        CHECK(res_gain_of(a) != NULL && aec_apply_gain(a, NULL, o) == AEC_OK, "residual getters after a completed hop");
    }
    CHECK(aec_process(a, mic, ref, out) == AEC_OK, "process after a completed hop");
    aec_reset(a);
    {
        Complex o[NB_MAX];
        CHECK(res_gain_of(a) == NULL && aec_apply_gain(a, NULL, o) == AEC_ERR_SEQUENCE, "a reset drops the last residual");
    }
    CHECK(aec_process_linear(a, mic, ref) == AEC_OK, "linear after reset");
    aec_reset(a);
    CHECK(aec_process_residual(a, NULL) == AEC_ERR_SEQUENCE, "a reset drops an open hop");
    {
        AecResidualView v; memset(&v, 0, sizeof v);
        v.struct_size = sizeof v - 1;
        CHECK(aec_get_residual(a, &v) == AEC_ERR_ARG, "short struct_size");
    }
    free(mem);
}

/* A finite but overflowing sample poisons the filters; aec_reset() must bring the instance back (stale per-bin state included). */
static void test_reset_recovers_from_overflow(void) {
    AecConfig cfg;
    void* mem = NULL;
    float ref[SPLIT_SIGNAL_MAX_HOP], mic[SPLIT_SIGNAL_MAX_HOP], out[SPLIT_SIGNAL_MAX_HOP];
    int hop_size, bad_after = 0, h, i;
    Aec* a;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    a = new_aec(&cfg, &mem);
    CHECK(a != NULL, "overflow: aec_init");
    if (!a) return;
    hop_size = aec_hop_size(a);
    for (h = 0; h < 90; ++h) {
        split_signal_fill(16000, hop_size, h, ref, mic);
        if (h == 10) mic[7] = 3.0e38f;
        if (h == 30) aec_reset(a);
        (void)aec_process(a, mic, ref, out);
        if (h >= 30) for (i = 0; i < hop_size; ++i) if (!isfinite(out[i])) { bad_after++; break; }
    }
    CHECK(bad_after == 0, "output non-finite on %d hops after aec_reset()", bad_after);
    free(mem);
}

/* A residual stage below FULL refuses the gain helpers. */
static void test_modes(void) {
    static const AecResidualMode modes[3] = { AEC_RESIDUAL_OFF, AEC_RESIDUAL_STATE, AEC_RESIDUAL_ESTIMATE };
    for (int i = 0; i < 3; ++i) {
        AecConfig cfg; void* mem = NULL; Aec* a; Complex o[NB_MAX]; float mic[SPLIT_SIGNAL_MAX_HOP], ref[SPLIT_SIGNAL_MAX_HOP];
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
        cfg.residual = modes[i];
        a = new_aec(&cfg, &mem);
        CHECK(a != NULL, "modes: init %d", i);
        if (!a) continue;
        split_signal_fill(16000, aec_hop_size(a), 0, ref, mic);
        CHECK(aec_process_linear(a, mic, ref) == AEC_OK, "modes: linear");
        if (modes[i] != AEC_RESIDUAL_OFF) CHECK(aec_process_residual(a, NULL) == AEC_OK, "modes: residual");
        CHECK(aec_apply_gain(a, NULL, o) == AEC_ERR_MODE, "apply_gain below FULL (mode %d)", (int)modes[i]);
        CHECK(aec_apply_cng(a, NULL, o) == AEC_ERR_MODE, "apply_cng below FULL (mode %d)", (int)modes[i]);
        free(mem);
    }
    {
        AecConfig cfg; aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
        cfg.residual = (AecResidualMode)7;
        CHECK(aec_get_mem_size(&cfg) == 0, "an out-of-range residual mode is refused");
        cfg.residual = (AecResidualMode)-1;
        CHECK(aec_get_mem_size(&cfg) == 0, "a negative residual mode is refused");
    }
}

int main(void) {
    g_have_baseline = baseline_load("split_entry");
    test_sequence();
    test_modes();
    test_reset_recovers_from_overflow();
    for (int gi = 0; gi < 3; ++gi) {
        const Grid* g = &GRIDS[gi];
        AecConfig cfg;
        make_cfg(&cfg, g, M_OFF);
        int hop_size = g->fft / 2;
        int n_hops = split_signal_n_hops(g->sr, hop_size);
        emit(g, "META", "hops", 0, (uint64_t)n_hops);
        emit(g, "META", "hop_size", 0, (uint64_t)hop_size);
        for (int m = 0; m < M_COUNT; ++m) {
            if (m == M_ARES) {
                AecConfig c2; make_cfg(&c2, g, M_ARES);
                emit(g, "META", "aec_res_mem_size", 0, (uint64_t)aec_res_get_mem_size(&c2));
                continue;
            }
            AecConfig c2; make_cfg(&c2, g, (Mode)m);
            char f[48]; snprintf(f, sizeof f, "mem_size_%s", MODE_NAME[m]);
            emit(g, "META", f, 0, (uint64_t)aec_get_mem_size(&c2));
        }
        emit(g, "META", "sizeof_Aec", 0, (uint64_t)sizeof(Aec));
        {
            float ref[SPLIT_SIGNAL_MAX_HOP], mic[SPLIT_SIGNAL_MAX_HOP];
            for (int hop = 0; hop < n_hops; ++hop) {
                split_signal_fill(g->sr, hop_size, hop, ref, mic);
                emit(g, "INPUT", "ref", hop, split_hash_f32(ref, hop_size));
                emit(g, "INPUT", "mic", hop, split_hash_f32(mic, hop_size));
            }
        }
        for (int m = 0; m < M_COUNT; ++m)
            if (run_mode(g, (Mode)m, n_hops) != 0) break;
    }
    g_fail += baseline_finish();
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
