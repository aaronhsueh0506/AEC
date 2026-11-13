/* test_split_entry2.c -- the second scene set of the single-channel API: saturation, double talk, delay shifts with the
 * bad-relock revert and its veto, clock drift, reset, FIXED and EXTERNAL_ALIGNED with aec_apply_external_realign, levels in
 * flight, switched-off blocks, live observer, windowed noise tracker, duty cycle and the 8 kHz grid, replayed through the new
 * entry points. The cases, stimuli, hashing and chaining are baseline2_mono.h; the chained hashes form a line stream that
 * AEC_TEST_BASELINE_WRITE / AEC_TEST_BASELINE capture and compare (no baseline is shipped). Also here, with no baseline:
 * the instance's apply helpers against the explicit-array computation on every hop, aec_process alternating with the split
 * calls hop by hop, and the carve of every instance kind ending exactly at aec_get_mem_size().
 *
 * Run from c_impl/:  make test-split-entry-2   (G2_VERBOSE=<file> writes every hop's unchained group hashes) */
#include "aec.h"
#include "../src/aec_res_internal.h"
#include "utility/simd_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static long g_checks = 0;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

/* ---- baseline ---------------------------------------------------------------------------------------------------------- */
#include "baseline_stream.h"

static void g2_sink(const char* line) {
    baseline_line(line);
}

/* ---- the new entry points behind baseline2_mono.h ------------------------------------------------------------------------- */
static void g2_mode_cfg(AecConfig* c, int ctx) { (void)ctx; c->residual = AEC_RESIDUAL_FULL; }

struct G2View;
static int g2_step_ctx(Aec* a, const float* mic, const float* ref, struct G2View* v);
static int g2_step_out(Aec* a, const float* mic, const float* ref, float* out) {
    return aec_process(a, mic, ref, out) == AEC_OK ? 0 : -1;
}

/* The instance's own gain / comfort-noise helpers must agree with the explicit-array computation. */
static void apply_hook(const Aec* a, const void* view, int hop, const float* gx, const Complex* p1, const Complex* p2,
                       const Complex* p3, const Complex* p4);
#define G2_HOOK_APPLY(a, v, hop, gx, p1, p2, p3, p4) apply_hook((a), (v), (hop), (gx), (p1), (p2), (p3), (p4))

#include "baseline2_mono.h"

static int g2_step_ctx(Aec* a, const float* mic, const float* ref, G2View* v) {
    AecLinearView lv; AecResidualView rv;
    if (aec_process_linear(a, mic, ref) != AEC_OK) return -1;
    if (aec_process_residual(a, NULL) != AEC_OK) return -1;
    memset(&lv, 0, sizeof lv); lv.struct_size = sizeof lv;
    memset(&rv, 0, sizeof rv); rv.struct_size = sizeof rv;
    if (aec_get_linear(a, &lv) != AEC_OK || aec_get_residual(a, &rv) != AEC_OK) return -1;
    memset(v, 0, sizeof *v);
    v->n_freqs = rv.n_freqs; v->hop_size = rv.hop_size;
    v->linear_hop = lv.linear_hop; v->far_spec = lv.far_spec; v->lin_formed = rv.formed_hop;
    v->aligned_far = lv.aligned_far_hop; v->delay_samples = lv.delay_samples; v->delay_conf = lv.delay_confidence;
    v->delay_state = (int)lv.delay_state; v->generation = lv.generation;
    v->error_spec = rv.error_spec; v->echo_spec = rv.echo_spec; v->near_spec = rv.near_spec; v->base_spec = rv.base_spec;
    v->formed_hop = rv.formed_hop; v->r2 = rv.r2; v->comfort_noise = rv.comfort_noise; v->res_gain = rv.res_gain;
    v->usable_linear = rv.usable_linear; v->filter_converged = lv.filter_converged; v->epc_active = lv.epc_active;
    v->is_stationary_dt = lv.is_stationary_dt; v->res_floor_protect = rv.res_floor_protect; v->guard_active = rv.guard_active;
    v->filter_once_converged = lv.filter_once_converged; v->clock_drift = rv.clock_drift;
    v->dt_indicator = lv.dt_indicator; v->far_power = lv.far_power; v->erle_factor = lv.erle_factor;
    v->saturation_level = lv.saturation_level; v->erl_estimate = lv.erl_estimate; v->divergence = lv.divergence;
    v->shadow_dt = lv.shadow_dt;
    /* the view's own consistency with the instance */
    CHECK(rv.max_dt == lv.dt_indicator && rv.max_saturation == lv.saturation_level && rv.all_converged == lv.filter_converged,
          "residual view inputs != linear view");
    CHECK(rv.guard_lanes == (rv.guard_active ? 1u : 0u), "guard_lanes");
    CHECK(rv.delay_locked == (lv.delay_state != AEC_LINEAR_DELAY_UNLOCKED) && rv.delay_changed == (lv.delay_state == AEC_LINEAR_DELAY_CHANGED),
          "delay flags");
    return 0;
}

static long g_apply_checks;
static void apply_hook(const Aec* a, const void* view, int hop, const float* gx, const Complex* p1, const Complex* p2,
                       const Complex* p3, const Complex* p4) {
    Complex t1[G2_NB_MAX], t2[G2_NB_MAX];
    const G2View* v = (const G2View*)view;
    (void)p3; (void)p4;
    if (!v) return;                                            /* the stand-alone gain stage's apply */
    CHECK(aec_apply_gain(a, gx, t1) == AEC_OK, "aec_apply_gain hop %d", hop);
    CHECK(g2_hc(G2_FNV_BASIS, t1, v->n_freqs) == g2_hc(G2_FNV_BASIS, p1, v->n_freqs), "aec_apply_gain != explicit form hop %d", hop);
    memcpy(t2, t1, (size_t)v->n_freqs * sizeof(Complex));
    CHECK(aec_apply_cng((Aec*)a, gx, t2) == AEC_OK, "aec_apply_cng hop %d", hop);
    CHECK(g2_hc(G2_FNV_BASIS, t2, v->n_freqs) == g2_hc(G2_FNV_BASIS, p2, v->n_freqs), "aec_apply_cng != explicit form hop %d", hop);
    g_apply_checks++;
}

/* ---- aec_process alternating with the split calls ----------------------------------------------------------------------- */
/* aec.h: "Within a stream the two forms may alternate hop by hop; the library's own overlap-add tail and comfort-noise
 * generator advance only in aec_process() and aec_apply_cng()". So the analysis side (linear view, residual view, gain) of an
 * alternating stream equals that of a stream driven by aec_process alone on every hop; the finished hop of aec_process
 * differs after a split hop (its tail and its noise stream did not advance) and is compared only where nothing but the
 * preceding aec_process hops fed it: with the comfort noise off, on a hop whose predecessor was an aec_process hop. */
static uint64_t view_hash(const Aec* a, const AecLinearView* lv, const AecResidualView* rv) {
    uint64_t h = G2_FNV_BASIS;
    const int n = rv->n_freqs;
    h = g2_fnv_f32v(h, lv->linear_hop, lv->hop_size); h = g2_hc(h, lv->far_spec, n);
    h = g2_fnv_f32v(h, lv->aligned_far_hop, lv->hop_size);
    h = g2_fnv_u32(h, (uint32_t)lv->delay_samples); h = g2_fnv_u32(h, lv->generation);
    h = g2_fnv_f32(h, lv->dt_indicator); h = g2_fnv_f32(h, lv->erle_factor); h = g2_fnv_f32(h, lv->saturation_level);
    h = g2_fnv_u32(h, (uint32_t)lv->filter_converged);
    h = g2_hc(h, rv->error_spec, n); h = g2_hc(h, rv->echo_spec, n); h = g2_hc(h, rv->near_spec, n);
    h = g2_hc(h, rv->base_spec, n); h = g2_fnv_f32v(h, rv->formed_hop, rv->hop_size);
    h = g2_fnv_f32v(h, rv->r2, n); h = g2_fnv_f32v(h, rv->comfort_noise, n); h = g2_fnv_f32v(h, rv->res_gain, n);
    h = g2_fnv_u32(h, (uint32_t)rv->res_floor_protect); h = g2_fnv_u32(h, (uint32_t)rv->guard_active);
    h = g2_fnv_u32(h, aec_post_reset_serial(a));
    return h;
}

static void test_alternation(const G2MonoCase* c, int enable_cng, const char* label) {
    AecConfig cfg;
    void *m0 = NULL, *m1 = NULL;
    float out0[1024], out1[1024];
    int hop = c->fft / 2, n_hops, ns, same_out = 0, cmp_out = 0;
    float *ref, *mic;
    G2MonoCase cc = *c;
    cc.tog = enable_cng ? 0 : G2T_NOCNG;
    cc.n_ev = 0;
    g2m_apply_cfg(&cfg, &cc, 0);
    size_t sz = aec_get_mem_size(&cfg);
    Aec* p = (m0 = g2_pool(sz)) ? aec_init(m0, sz, &cfg) : NULL;
    Aec* x = (m1 = g2_pool(sz)) ? aec_init(m1, sz, &cfg) : NULL;
    CHECK(p && x, "alternation: init %s", label);
    if (!p || !x) return;
    n_hops = (int)((int64_t)cc.dur_ms * cc.sr / 1000 / hop); ns = n_hops * hop;
    ref = (float*)malloc((size_t)ns * sizeof(float)); mic = (float*)malloc((size_t)ns * sizeof(float));
    g2_mono_signal(&cc.stim, cc.sr, ns, ref, mic);
    /* pattern per hop: 1 = aec_process, 0 = split; blocks of 3 so that hops with an aec_process predecessor occur */
    for (int h = 0; h < n_hops; ++h) {
        const float* r = ref + (size_t)h * hop; const float* m = mic + (size_t)h * hop;
        const int use_process = ((h / 3) % 2) == 0;
        const int prev_process = h > 0 && (((h - 1) / 3) % 2) == 0;
        AecLinearView lp, lx; AecResidualView rp, rx;
        CHECK(aec_process(p, m, r, out0) == AEC_OK, "alternation %s: pure process hop %d", label, h);
        if (use_process) CHECK(aec_process(x, m, r, out1) == AEC_OK, "alternation %s: process hop %d", label, h);
        else { CHECK(aec_process_linear(x, m, r) == AEC_OK && aec_process_residual(x, NULL) == AEC_OK, "alternation %s: split hop %d", label, h); }
        memset(&lp, 0, sizeof lp); lp.struct_size = sizeof lp; lx = lp;
        memset(&rp, 0, sizeof rp); rp.struct_size = sizeof rp; rx = rp;
        CHECK(aec_get_linear(p, &lp) == AEC_OK && aec_get_residual(p, &rp) == AEC_OK, "alternation views (pure) hop %d", h);
        CHECK(aec_get_linear(x, &lx) == AEC_OK && aec_get_residual(x, &rx) == AEC_OK, "alternation views hop %d", h);
        CHECK(view_hash(p, &lp, &rp) == view_hash(x, &lx, &rx), "alternation %s: views differ at hop %d (%s)", label, h,
              use_process ? "aec_process" : "split");
        if (use_process && prev_process && !enable_cng) {
            cmp_out++;
            if (memcmp(out0, out1, (size_t)hop * sizeof(float)) == 0) same_out++;
        }
    }
    if (!enable_cng)
        CHECK(cmp_out > 0 && same_out == cmp_out, "alternation %s: finished hops after an aec_process hop: %d of %d equal", label, same_out, cmp_out);
    free(ref); free(mic); free(m0); free(m1);
}

/* ---- the carve of every instance kind ends exactly at aec_get_mem_size() ---------------------------------------------------- */
#define GUARD 256
static void check_pool_end(const AecConfig* cfg, const char* label) {
    size_t sz = aec_get_mem_size(cfg);
    uint8_t* mem = NULL;
    int ok = 1;
    CHECK(sz > 0, "carve %s: size", label);
    if (!sz) return;
    if (posix_memalign((void**)&mem, 64u, sz + GUARD) != 0) { CHECK(0, "carve %s: alloc", label); return; }
    memset(mem, 0xC5, sz + GUARD);
    {
        Aec* a = aec_init(mem, sz, cfg);
        CHECK(a != NULL, "carve %s: aec_init with exactly aec_get_mem_size() bytes", label);
        if (a) {
            float mic[4096], ref[2048], out[2048];
            const int hop = aec_hop_size(a), mics = cfg->n_mics > 0 ? cfg->n_mics : 1;
            for (int h = 0; h < 6; ++h) {
                for (int i = 0; i < hop; ++i) {
                    ref[i] = 0.1f * (float)(((i * 7 + h * 13) % 17) - 8) / 8.0f;
                    for (int ch = 0; ch < mics; ++ch) mic[i * mics + ch] = 0.05f * (float)(((i * 5 + h * 3 + ch) % 11) - 5) / 5.0f;
                }
                if (mics == 1) (void)aec_process(a, mic, ref, out);
                else (void)aec_process_linear(a, mic, ref);
            }
            aec_reset(a);
            aec_destroy(a);
        }
    }
    for (size_t i = 0; i < GUARD; ++i) if (mem[sz + i] != 0xC5) { ok = 0; break; }
    CHECK(ok, "carve %s: the pool was written past aec_get_mem_size()", label);
    /* one byte short: refused */
    CHECK(aec_init(mem, sz - 16, cfg) == NULL, "carve %s: a pool 16 bytes short must be refused", label);
    free(mem);
}

static void test_carve_end(void) {
    static const struct { int sr, fft; } grid[4] = { {16000, 256}, {16000, 512}, {48000, 1024}, {8000, 256} };
    static const AecResidualMode rm[4] = { AEC_RESIDUAL_OFF, AEC_RESIDUAL_STATE, AEC_RESIDUAL_ESTIMATE, AEC_RESIDUAL_FULL };
    for (int g = 0; g < 4; ++g) {
        for (int m = 0; m < 3; ++m) {                                   /* delay modes */
            for (int r = 0; r < 4; ++r) {
                AecConfig cfg; char label[64];
                aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, grid[g].sr);
                cfg.fft_size = grid[g].fft; cfg.residual = rm[r];
                cfg.delay_mode = m == 0 ? AEC_DELAY_MATCHED : m == 1 ? AEC_DELAY_FIXED : AEC_DELAY_EXTERNAL_ALIGNED;
                cfg.fixed_delay_samples = m == 1 ? 40 * grid[g].sr / 1000 : -1;
                snprintf(label, sizeof label, "%d/%d mode%d res%d", grid[g].sr, grid[g].fft, m, r);
                check_pool_end(&cfg, label);
            }
        }
    }
    {
        static const struct { const char* name; int tog; } tg[7] = { {"noshadow", G2T_NOSHADOW}, {"nohpf", G2T_NOHPF}, {"nocng", G2T_NOCNG},
            {"nosat", G2T_NOSAT}, {"observer", G2T_OBSERVER}, {"n2win", G2T_N2WIN}, {"all-off", G2T_NOSHADOW | G2T_NOHPF | G2T_NOCNG | G2T_NOSAT} };
        for (int k = 0; k < 7; ++k) {
            AecConfig cfg; G2MonoCase c; char label[64];
            memset(&c, 0, sizeof c); c.sr = 16000; c.fft = 256; c.level = 1; c.tog = tg[k].tog;
            g2m_apply_cfg(&cfg, &c, 1);
            snprintf(label, sizeof label, "16000/256 %s", tg[k].name);
            check_pool_end(&cfg, label);
        }
    }
#ifdef AEC_NO_ARRAY
    {                                                                    /* arrays: refused by every construction path */
        AecConfig cfg;
        void* mem = NULL;
        Aec heap;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
        cfg.n_mics = AEC_ARRAY_MICS;
        cfg.residual = AEC_RESIDUAL_FULL;
        CHECK(aec_get_mem_size(&cfg) == 0, "no-array: n_mics 4 has no pool size");
        CHECK(aec_init(&mem, sizeof mem, &cfg) == NULL, "no-array: aec_init refuses n_mics 4");
        CHECK(aec_create(&heap, &cfg) == -1, "no-array: aec_create refuses n_mics 4");
    }
#else
    for (int g = 0; g < 3; ++g) {                                        /* arrays */
        for (int r = 1; r < 4; ++r) {
            for (int m = 0; m < 3; ++m) {
                AecConfig cfg; char label[64];
                aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, grid[g].sr);
                cfg.fft_size = grid[g].fft; cfg.n_mics = AEC_ARRAY_MICS; cfg.residual = rm[r];
                cfg.delay_mode = m == 0 ? AEC_DELAY_MATCHED : m == 1 ? AEC_DELAY_FIXED : AEC_DELAY_EXTERNAL_ALIGNED;
                cfg.fixed_delay_samples = m == 1 ? 40 * grid[g].sr / 1000 : -1;
                snprintf(label, sizeof label, "array %d/%d mode%d res%d", grid[g].sr, grid[g].fft, m, r);
                check_pool_end(&cfg, label);
            }
        }
    }
#endif
}

static const G2MonoCase* find_case(const G2MonoCase* cs, int n, int sr, const char* name) {
    for (int i = 0; i < n; ++i) if (cs[i].sr == sr && strcmp(cs[i].name, name) == 0) return &cs[i];
    return NULL;
}

int main(void) {
    static G2MonoCase cases[G2M_MAX_CASES];
    int nc = g2m_cases(cases);
    g_have_baseline = baseline_load("split_entry2");
    for (int i = 0; i < nc; ++i) {
        if (!g2m_want(&cases[i])) continue;
        if (g2m_run_case(&cases[i]) != 0) { g_fail++; break; }
    }
    printf("instance apply helpers cross-checked on %ld hops\n", g_apply_checks);
    CHECK(g_apply_checks > 0, "no apply cross-check ran");
    {
        const G2MonoCase* s16 = find_case(cases, nc, 16000, "satdt");
        const G2MonoCase* s48 = find_case(cases, nc, 48000, "satdt");
        CHECK(s16 && s48, "alternation cases");
        if (s16 && s48) {
            test_alternation(s16, 1, "satdt 16000/256 cng on");
            test_alternation(s16, 0, "satdt 16000/256 cng off");
            test_alternation(s48, 1, "satdt 48000/1024 cng on");
            test_alternation(s48, 0, "satdt 48000/1024 cng off");
        }
    }
    test_carve_end();
    g_fail += baseline_finish();
    printf("%ld checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
