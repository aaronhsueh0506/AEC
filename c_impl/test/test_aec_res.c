/* test_aec_res.c -- include/internal/aec_res_internal.h: the residual-echo suppression gain stage (internal; the array instance's gain).
 *
 *  A  configuration: the suppressor an AecRes builds is bit-for-bit the one an
 *     Aec lane builds (heap and caller-pool lanes), on every supported grid
 *     and level.
 *  D  contract: argument checks, sizing, alignment, no write outside the
 *     block, reset and level semantics.
 *  E  stage details: the exact 32768 scale of the render block (a block
 *     straddling the split-floor latch power), the initial-state exit (with
 *     the two flags that make it inert in the shipped configuration cleared),
 *     and the floor_first_increase limit on the first recovery hop after a
 *     saturated echo.
 *  F  far-end detectors: 400 hops of a far-end signal that crosses both
 *     branches of the low-render detector and the far-active latch, on every
 *     grid and level.
 */
#include "src/aec_res.c"
#include "aec_white_box.h"
#include "aec_ref_tables.h"

#include "utility/simd_aec.h"

#include <stdio.h>
#include <stdlib.h>

static int g_pass, g_fail;
#define CHECK(cond, ...) \
    do { \
        if (cond) { g_pass++; } \
        else { g_fail++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
    } while (0)

static uint64_t g_rs = 0x9E3779B97F4A7C15ull;
static uint64_t rnd64(void) {
    g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17;
    return g_rs;
}
static float urand(void) { return (float)(rnd64() >> 40) / 16777216.0f; }
static float srand_f(void) { return 2.0f * urand() - 1.0f; }

static const int GRIDS[4][2] = {{8000, 256}, {16000, 256}, {16000, 512}, {48000, 1024}};
#define N_GRIDS 4
#define N_LEVELS 3

static AecConfig make_cfg(int g, int level) {
    AecConfig cfg;
    aec_config_from_level(&cfg, (AecLevel)level, GRIDS[g][0]);
    cfg.fft_size = GRIDS[g][1];
    return cfg;
}

/* The ENR/SU tables are decoded into each instance's pool, so two tunings are equal by content, not by address. */
static int tun_equal(const SuppressionGainTuning* a, const SuppressionGainTuning* b) {
    size_t bytes = (size_t)a->table_len * sizeof(float);
    return a->table_len == b->table_len &&
           memcmp(a->nearend_enr_tr, b->nearend_enr_tr, bytes) == 0 && memcmp(a->nearend_enr_su, b->nearend_enr_su, bytes) == 0 &&
           memcmp(a->nearend_emr_tr, b->nearend_emr_tr, bytes) == 0 && memcmp(a->normal_enr_tr, b->normal_enr_tr, bytes) == 0 &&
           memcmp(a->normal_enr_su, b->normal_enr_su, bytes) == 0 && memcmp(a->normal_emr_tr, b->normal_emr_tr, bytes) == 0;
}

/* The tuning's tables against the reference tables of aec_ref_tables.h (the twin's values, outside the decoder); the EMR
 * thresholds are the constant 0.3f. */
static int tun_matches_ref(const SuppressionGainTuning* t, int sr, int fft) {
    int i, k;
    for (i = 0; i < AECREF_N_GRIDS; ++i) {
        const AecRefTables* r = &AECREF_GRIDS[i];
        if (r->sample_rate != sr || r->fft_size != fft) continue;
        if (t->table_len != r->n_bins) return 0;
        for (k = 0; k < r->n_bins; ++k)
            if (t->nearend_emr_tr[k] != 0.3f || t->normal_emr_tr[k] != 0.3f) return 0;
        return t->nearend_enr_tr && memcmp(t->nearend_enr_tr, r->nearend_enr_tr, (size_t)r->n_bins * sizeof(float)) == 0 &&
               memcmp(t->nearend_enr_su, r->nearend_enr_su, (size_t)r->n_bins * sizeof(float)) == 0 &&
               memcmp(t->normal_enr_tr, r->normal_enr_tr, (size_t)r->n_bins * sizeof(float)) == 0 &&
               memcmp(t->normal_enr_su, r->normal_enr_su, (size_t)r->n_bins * sizeof(float)) == 0;
    }
    return 0;
}

static void* aligned_block(size_t bytes) {
    void* p = NULL;
    if (posix_memalign(&p, 16, bytes) != 0) return NULL;
    return p;
}

/* ---- A ---- */
static void test_config(void) {
    int g, p;
    for (g = 0; g < N_GRIDS; ++g) {
        for (p = 0; p < N_LEVELS; ++p) {
            AecConfig cfg = make_cfg(g, p);
            Aec *heap_lane = NULL;
            Aec* pool_lane;
            void *pool, *res_mem;
            AecRes* r;
            size_t need = aec_res_get_mem_size(&cfg);
            char id[48];
            snprintf(id, sizeof id, "%d/%d/level%d", GRIDS[g][0], GRIDS[g][1], p);

            CHECK(need > 0, "%s: aec_res_get_mem_size", id);
            res_mem = aligned_block(need);
            r = aec_res_init(res_mem, need, &cfg);
            CHECK(r != NULL, "%s: aec_res_init", id);
            CHECK((heap_lane = aec_create(&cfg)) != NULL, "%s: aec_create", id);
            pool = aligned_block(aec_get_mem_size(&cfg));
            pool_lane = aec_init(pool, aec_get_mem_size(&cfg), &cfg);
            CHECK(pool_lane != NULL, "%s: aec_init", id);
            if (!r || !pool_lane) continue;

            CHECK(memcmp(&heap_lane->a3_sg.cfg, &r->sg.cfg, sizeof r->sg.cfg) == 0,
                  "%s: config bytes equal the heap lane's", id);
            CHECK(memcmp(&pool_lane->a3_sg.cfg, &r->sg.cfg, sizeof r->sg.cfg) == 0,
                  "%s: config bytes equal the pool lane's", id);
            CHECK(tun_equal(&heap_lane->a3_sg.tun, &r->sg.tun), "%s: tuning tables equal the lane's", id);
            CHECK(tun_matches_ref(&r->sg.tun, GRIDS[g][0], GRIDS[g][1]) &&
                  tun_matches_ref(&heap_lane->a3_sg.tun, GRIDS[g][0], GRIDS[g][1]) &&
                  tun_matches_ref(&pool_lane->a3_sg.tun, GRIDS[g][0], GRIDS[g][1]),
                  "%s: decoded ENR/SU tables equal the reference tables (AecRes, heap and pool Aec)", id);
            CHECK(r->n_bins == heap_lane->n_freqs && r->hop_size == heap_lane->hop_size,
                  "%s: dimensions equal the lane's", id);

            aec_res_reset(r);
            aec_res_set_level(r, AEC_LEVEL_AGGRESSIVE);
            CHECK(tun_matches_ref(&r->sg.tun, GRIDS[g][0], GRIDS[g][1]),
                  "%s: tables survive aec_res_reset and a level change", id);

            aec_destroy(heap_lane);
            aec_destroy(pool_lane);
            free(pool);
            free(res_mem);
        }
    }
}

/* ---- D ---- */
/* The far-end-active floor as a power ratio: `live` is the value in force now, `target` the one it is heading to. */
static void split_floor(const AecRes* r, float* live, float* target) {
    *live = r->sg.split_floor_far_active_live;
    *target = r->sg.cfg.split_floor_far_active;
}

static void test_contract(void) {
    AecConfig cfg = make_cfg(1, 1), bad = cfg;
    size_t need = aec_res_get_mem_size(&cfg);
    unsigned char* blk = (unsigned char*)aligned_block(need + 64);
    AecRes* r;
    AecResInput in;
    int n, hop, k, i;
    Complex* sp;
    float *r2, *cn, *rend, live, target, floor_mild, floor_aggr;
    const float *g1 = NULL, *g2;
    unsigned char canary[64];

    memset(canary, 0xC3, sizeof canary);
    memset(blk, 0xA5, need + 64);
    memset(blk + need, 0xC3, 64);

    CHECK(aec_res_get_mem_size(NULL) == 0, "mem size of NULL config is 0");
    bad.fft_size = 300;
    CHECK(aec_res_get_mem_size(&bad) == 0, "mem size of an unsupported grid is 0");
    bad = cfg; bad.sample_rate = 44100;
    CHECK(aec_res_get_mem_size(&bad) == 0, "mem size of an unsupported rate is 0");
    CHECK(aec_res_init(NULL, need, &cfg) == NULL, "init rejects NULL memory");
    CHECK(aec_res_init(blk + 8, need, &cfg) == NULL, "init rejects misaligned memory");
    CHECK(aec_res_init(blk, need - 1, &cfg) == NULL, "init rejects a block one byte short");
    CHECK(aec_res_init(blk, need, NULL) == NULL, "init rejects a NULL config");
    CHECK(aec_res_init(blk, need, &bad) == NULL, "init rejects an invalid config");
    r = aec_res_init(blk, need, &cfg);
    CHECK(r != NULL, "init on an exact-size block");
    if (!r) { free(blk); return; }
    n = aec_res_n_bins(r); hop = aec_res_hop_size(r);
    CHECK(n == 129 && hop == 128, "16 kHz / 256 dimensions are 129 bins, 128-sample hop");
    CHECK(aec_res_n_bins(NULL) == 0 && aec_res_hop_size(NULL) == 0, "accessors accept NULL");

    sp = (Complex*)calloc((size_t)n, sizeof(Complex));
    r2 = (float*)calloc((size_t)n, sizeof(float));
    cn = (float*)calloc((size_t)n, sizeof(float));
    rend = (float*)calloc((size_t)hop, sizeof(float));
    for (k = 0; k < n; ++k) { sp[k].r = 0.01f * srand_f(); sp[k].i = 0.01f * srand_f(); r2[k] = 1e4f; cn[k] = 1e2f; }
    in.error_spec = sp; in.near_spec = sp; in.r2 = r2; in.comfort_noise = cn; in.render_hop = rend;
    in.all_converged = 0; in.max_dt = 0.0f; in.max_saturation = 0.0f; in.clock_drift = 0;

    CHECK(aec_res_process(NULL, &in) == NULL, "process rejects NULL instance");
    CHECK(aec_res_process(r, NULL) == NULL, "process rejects NULL input");
    { AecResInput t = in; t.error_spec = NULL; CHECK(aec_res_process(r, &t) == NULL, "process rejects NULL error_spec"); }
    { AecResInput t = in; t.near_spec = NULL; CHECK(aec_res_process(r, &t) == NULL, "process rejects NULL near_spec"); }
    { AecResInput t = in; t.r2 = NULL; CHECK(aec_res_process(r, &t) == NULL, "process rejects NULL r2"); }
    { AecResInput t = in; t.comfort_noise = NULL; CHECK(aec_res_process(r, &t) == NULL, "process rejects NULL comfort_noise"); }
    { AecResInput t = in; t.render_hop = NULL; CHECK(aec_res_process(r, &t) == NULL, "process rejects NULL render_hop"); }

    for (i = 0; i < 300; ++i) {
        g1 = aec_res_process(r, &in);
        if (!g1) break;
        for (k = 0; k < n; ++k) if (!(g1[k] >= 0.0f && g1[k] <= 1.0f)) { g1 = NULL; break; }
        if (!g1) break;
    }
    CHECK(g1 != NULL, "300 hops return finite gains in [0, 1]");
    CHECK(memcmp(blk + need, canary, 64) == 0, "no write past the end of the block");

    /* levels */
    split_floor(r, &live, &target);
    floor_mild = 0.0f; floor_aggr = 0.0f;
    CHECK(live == target && target > 0.0f, "fresh instance: live floor equals target");
    CHECK(aec_res_set_level(NULL, AEC_LEVEL_MILD) == -1, "set_level rejects NULL");
    CHECK(aec_res_set_level(r, (AecLevel)9) == -1, "set_level rejects an unknown level");
    { float l2, t2; split_floor(r, &l2, &t2);
      CHECK(l2 == live && t2 == target, "a rejected level change writes nothing"); }
    CHECK(aec_res_set_level(r, AEC_LEVEL_AGGRESSIVE) == 0, "set_level aggressive");
    split_floor(r, &live, &floor_aggr);
    CHECK(aec_res_process(r, &in) != NULL, "process after level change");
    split_floor(r, &live, &target);
    CHECK(target == floor_aggr && live > floor_aggr, "the level change walks: the live floor trails the target");
    for (i = 0; i < 64 && live != target; ++i) { aec_res_process(r, &in); split_floor(r, &live, &target); }
    CHECK(live == floor_aggr && target == floor_aggr, "the level ramp lands on the target");
    CHECK(aec_res_set_level(r, AEC_LEVEL_MILD) == 0, "set_level mild");
    split_floor(r, &live, &floor_mild);
    CHECK(floor_mild > floor_aggr, "mild floor is above aggressive floor");
    aec_res_reset(r);
    split_floor(r, &live, &target);
    CHECK(target == floor_mild && live == floor_mild, "reset keeps the retargeted floor");
    aec_res_reset(NULL);

    /* NO_FLOOR: the gate steps immediately (no ramp) and survives aec_res_reset(), which rebuilds
     * the suppressor from r->sg.cfg. */
    CHECK(r->sg.cfg.split_floor_enabled == 1,
          "no_floor: mild left the gate enabled");
    CHECK(aec_res_set_level(r, AEC_LEVEL_NO_FLOOR) == 0,
          "no_floor: retarget to NO_FLOOR accepted");
    CHECK(r->sg.cfg.split_floor_enabled == 0,
          "no_floor: gate off immediately, before any reset");
    aec_res_reset(r);
    CHECK(r->sg.cfg.split_floor_enabled == 0,
          "no_floor: aec_res_reset() preserves the gate OFF across reset");
    CHECK(aec_res_set_level(r, AEC_LEVEL_BALANCED) == 0,
          "no_floor: retarget back to BALANCED accepted");
    CHECK(r->sg.cfg.split_floor_enabled == 1,
          "no_floor: gate back on immediately");
    split_floor(r, &live, &target);
    CHECK(live == target && target > 0.0f,
          "no_floor: the far-active floor is a real positive value again, "
          "not left at NO_FLOOR's inert reading");
    aec_res_reset(r);
    CHECK(r->sg.cfg.split_floor_enabled == 1,
          "no_floor: aec_res_reset() preserves the gate ON across reset too");

    /* reset: a reset instance reproduces a fresh one */
    {
        unsigned char* blk2 = (unsigned char*)aligned_block(need);
        AecRes* f;
        int same = 1;
        memset(blk2, 0x5A, need);
        f = aec_res_init(blk2, need, &cfg);
        aec_res_set_level(f, AEC_LEVEL_MILD);
        aec_res_reset(f);   /* a reset lands the floor on its target at once, as r has */
        aec_res_reset(r);
        for (i = 0; i < 50; ++i) {
            g1 = aec_res_process(r, &in);
            g2 = aec_res_process(f, &in);
            if (!g1 || !g2 || memcmp(g1, g2, (size_t)n * sizeof(float)) != 0) same = 0;
        }
        CHECK(same, "reset instance equals a fresh instance with the same level");
        free(blk2);
    }
    /* the result does not depend on what the block held before init */
    {
        unsigned char* blk3 = (unsigned char*)aligned_block(need);
        AecRes* f;
        int same = 1;
        memset(blk3, 0x00, need);
        f = aec_res_init(blk3, need, &cfg);
        aec_res_set_level(f, AEC_LEVEL_MILD);
        aec_res_reset(f);   /* a reset lands the floor on its target at once, as r has */
        aec_res_reset(r);
        for (i = 0; i < 50; ++i) {
            g1 = aec_res_process(r, &in);
            g2 = aec_res_process(f, &in);
            if (!g1 || !g2 || memcmp(g1, g2, (size_t)n * sizeof(float)) != 0) same = 0;
        }
        CHECK(same, "result is independent of the block's previous contents");
        free(blk3);
    }

    free(sp); free(r2); free(cn); free(rend); free(blk);
}

/* ---- E: stage details ---- */

/* Reference model of the stage written out with the suppressor primitives: the
 * test_initial_phase run compares the stage to it hop by hop. */
typedef struct {
    int n_freqs, hop_size;
    SuppressionGain post_sg;
    SuppressionGainConfig post_sg_cfg;
    SuppressionGainTuning post_sg_tun;
    float* post_sg_storage;
    float *post_near_power, *error_power, *render_i16;
    const float* aligned_ref;
    const Complex* fused_near;
    const float *fused_r2, *fused_comfort;
} RefPost;

#define REF_PSD_SCALE (32768.0f * 32768.0f)

static void ref_mag2(float* out, const Complex* input, int n) {
    int i;
    for (i = 0; i < n; ++i) {
        float real = input[i].r;
        float imag = input[i].i;
        out[i] = real * real + imag * imag;
    }
}

static void ref_sg_init(RefPost* p) {
    int n = p->n_freqs, ma_n = p->post_sg_cfg.nearend_smoother_n;
    float* cursor = p->post_sg_storage;
    float* last_gain = cursor; cursor += n;
    float* last_near = cursor; cursor += n;
    float* last_echo = cursor; cursor += n;
    float* ma_buf = cursor; cursor += (size_t)ma_n * n;
    float* near_s = cursor; cursor += n;
    float* weighted = cursor; cursor += n;
    float* min_gain = cursor; cursor += n;
    float* max_gain = cursor; cursor += n;
    float* raw_gain = cursor; cursor += n;
    float* gain = cursor; cursor += n;
    float* sum = cursor;
    memset(p->post_sg_storage, 0, (size_t)(10 + ma_n) * (size_t)n * sizeof(float));
    suppression_gain_init(&p->post_sg, &p->post_sg_cfg, &p->post_sg_tun, last_gain, last_near,
                          last_echo, ma_buf, near_s, weighted, min_gain, max_gain, raw_gain,
                          gain, sum);
}

/* Power/reference preparation, then the suppressor. */
static const float* ref_post_res_gain(RefPost* p, int all_converged, float max_dt,
                                      float max_saturation, int clock_drift) {
    int n = p->n_freqs, hop = p->hop_size, k;
    ref_mag2(p->post_near_power, p->fused_near, n);
    for (k = 0; k < n; ++k) {
        float e2 = p->error_power[k];
        float n2 = p->post_near_power[k];
        p->post_near_power[k] = (all_converged ? fminf(e2, n2) : n2) * REF_PSD_SCALE;
    }
    for (k = 0; k < hop; ++k) p->render_i16[k] = p->aligned_ref[k] * 32768.0f;
    if (p->post_sg.initial_state && all_converged)
        suppression_gain_set_initial_state(&p->post_sg, 0);
    p->post_sg.dt_protect_active = max_dt > 0.2f;
    return suppression_gain_get_gain(&p->post_sg, p->post_near_power, p->fused_r2, p->fused_r2,
                                     p->fused_comfort, p->render_i16, clock_drift,
                                     max_saturation > 0.5f);
}


/* Inputs with a weak near-end and a strong residual-echo estimate: the gain
 * is driven to the floors, so a change of which floor applies shows. */
static void echo_dominated_inputs(int n, Complex* err, Complex* near, float* r2, float* cn) {
    int k;
    for (k = 0; k < n; ++k) {
        err[k].r = 1e-4f * srand_f(); err[k].i = 1e-4f * srand_f();
        near[k].r = 1e-3f * srand_f(); near[k].i = 1e-3f * srand_f();
        r2[k] = 1e12f;
        cn[k] = 1e2f;
    }
}

/* render_hop is scaled by exactly 32768 to the int16 power scale. The scaled
 * mean power of a constant block is compared with the split-floor latch
 * power; levels 3e-5 (relative power) either side of it are separated by the
 * scale factors 32767 and 32769 (relative power change 6.1e-5), and a block
 * of equal samples has an exact mean. */
static void test_render_scale(void) {
    int g, p, side;
    for (g = 0; g < N_GRIDS; ++g) {
        for (p = 0; p < N_LEVELS; ++p) {
            AecConfig cfg = make_cfg(g, p);
            size_t need = aec_res_get_mem_size(&cfg);
            void* mem = aligned_block(need);
            AecRes* r = aec_res_init(mem, need, &cfg);
            int n, hop, k;
            Complex *err, *near;
            float *r2, *cn, *rend, *gain_by_side[2];
            int latched[2] = {-1, -1};
            AecResInput in;
            char id[48];
            snprintf(id, sizeof id, "%d/%d/level%d", GRIDS[g][0], GRIDS[g][1], p);
            if (!r) { CHECK(0, "%s: init", id); free(mem); continue; }
            n = r->n_bins; hop = r->hop_size;
            err = (Complex*)calloc((size_t)n, sizeof(Complex));
            near = (Complex*)calloc((size_t)n, sizeof(Complex));
            r2 = (float*)calloc((size_t)n, sizeof(float));
            cn = (float*)calloc((size_t)n, sizeof(float));
            rend = (float*)calloc((size_t)hop, sizeof(float));
            gain_by_side[0] = (float*)calloc((size_t)n, sizeof(float));
            gain_by_side[1] = (float*)calloc((size_t)n, sizeof(float));
            echo_dominated_inputs(n, err, near, r2, cn);
            in.error_spec = err; in.near_spec = near; in.r2 = r2; in.comfort_noise = cn;
            in.render_hop = rend; in.all_converged = 0; in.max_dt = 0.0f;
            in.max_saturation = 0.0f; in.clock_drift = 0;
            CHECK(r->sg.cfg.split_floor_latch_power > 1.0f &&
                  r->sg.cfg.split_floor_latch_power < 32768.0f * 32768.0f,
                  "%s: the latch power is reachable by a render block", id);
            for (side = 0; side < 2; ++side) {
                /* scaled sample level s with s*s = latch * (1 -/+ 3e-5); v = s / 32768 is exact */
                float s = (float)sqrt((double)r->sg.cfg.split_floor_latch_power *
                                      (side ? 1.0 + 3e-5 : 1.0 - 3e-5));
                float v = s / 32768.0f;
                const float* out;
                for (k = 0; k < hop; ++k) rend[k] = v;
                aec_res_reset(r);
                out = aec_res_process(r, &in);
                latched[side] = r->sg.far_active_latched;
                if (out) memcpy(gain_by_side[side], out, (size_t)n * sizeof(float));
            }
            CHECK(latched[0] == 0, "%s: a block just below the latch power does not latch", id);
            CHECK(latched[1] == 1, "%s: a block just above the latch power latches", id);
            CHECK(memcmp(gain_by_side[0], gain_by_side[1], (size_t)n * sizeof(float)) != 0,
                  "%s: the latch changes the gain", id);
            free(err); free(near); free(r2); free(cn); free(rend);
            free(gain_by_side[0]); free(gain_by_side[1]); free(mem);
        }
    }
}

/* The suppressor starts in its initial state and leaves it when every filter
 * has converged. Both initial-state reads in the suppressor are OR-ed with a
 * flag the shared configuration builder sets to 1 (dne_use_during_initial_phase,
 * lf_smoothing_during_initial_phase), so through AecRes as shipped the state
 * has no effect on the gain. This test clears the two flags in the stored and
 * the reference configuration; the gain then depends on the state, and the
 * stage must track the reference through the transition. */
static void test_initial_phase(void) {
    int g;
    for (g = 0; g < N_GRIDS; ++g) {
        AecConfig cfg = make_cfg(g, 1);
        Aec *lane = NULL;
        size_t need = aec_res_get_mem_size(&cfg);
        void* mem = aligned_block(need);
        void* mem_on = aligned_block(need);
        AecRes *r = aec_res_init(mem, need, &cfg), *r_on = aec_res_init(mem_on, need, &cfg);
        RefPost ref;
        int n, hop, hh, k, same_ref = 1, flags_matter = 0;
        Complex *err, *near;
        float *r2, *cn, *rend;
        AecResInput in;
        char id[32];
        snprintf(id, sizeof id, "%d/%d", GRIDS[g][0], GRIDS[g][1]);
        if (!r || !r_on || (lane = aec_create(&cfg)) == NULL) { CHECK(0, "%s: setup", id); continue; }

        memset(&ref, 0, sizeof ref);
        n = ref.n_freqs = lane->n_freqs;
        hop = ref.hop_size = lane->hop_size;
        ref.post_sg_cfg = lane->a3_sg.cfg;
        ref.post_sg_tun = lane->a3_sg.tun;
        ref.post_sg_cfg.dne_use_during_initial_phase = 0;
        ref.post_sg_cfg.lf_smoothing_during_initial_phase = 0;
        ref.post_sg_storage = (float*)calloc((size_t)(10 + ref.post_sg_cfg.nearend_smoother_n) * n,
                                             sizeof(float));
        ref.post_near_power = (float*)calloc((size_t)n, sizeof(float));
        ref.error_power = (float*)calloc((size_t)n, sizeof(float));
        ref.render_i16 = (float*)calloc((size_t)hop, sizeof(float));
        ref_sg_init(&ref);
        r->sg.cfg.dne_use_during_initial_phase = 0;
        r->sg.cfg.lf_smoothing_during_initial_phase = 0;
        aec_res_reset(r);
        CHECK(r->sg.initial_state == 1 && r_on->sg.initial_state == 1,
              "%s: instances start in the initial state", id);

        err = (Complex*)calloc((size_t)n, sizeof(Complex));
        near = (Complex*)calloc((size_t)n, sizeof(Complex));
        r2 = (float*)calloc((size_t)n, sizeof(float));
        cn = (float*)calloc((size_t)n, sizeof(float));
        rend = (float*)calloc((size_t)hop, sizeof(float));
        for (hh = 0; hh < 400; ++hh) {
            /* near-end speech over a weak echo; every filter converged from hop 150 on */
            int all_conv = hh >= 150;
            const float *a, *b, *c;
            for (k = 0; k < n; ++k) {
                err[k].r = 0.06f * srand_f(); err[k].i = 0.06f * srand_f();
                near[k].r = 0.2f * srand_f(); near[k].i = 0.2f * srand_f();
                r2[k] = 0.12f * (near[k].r * near[k].r + near[k].i * near[k].i) * 1.0737418e9f * (0.8f + 0.4f * urand());
                cn[k] = 1e3f * urand();
            }
            for (k = 0; k < hop; ++k) rend[k] = 0.05f * srand_f();
            ref.aligned_ref = rend; ref.fused_near = near; ref.fused_r2 = r2; ref.fused_comfort = cn;
            ref_mag2(ref.error_power, err, n);
            a = ref_post_res_gain(&ref, all_conv, 0.0f, 0.0f, 0);
            in.error_spec = err; in.near_spec = near; in.r2 = r2; in.comfort_noise = cn;
            in.render_hop = rend; in.all_converged = all_conv; in.max_dt = 0.0f;
            in.max_saturation = 0.0f; in.clock_drift = 0;
            b = aec_res_process(r, &in);
            c = aec_res_process(r_on, &in);
            if (!a || !b || !c) { same_ref = 0; break; }
            if (memcmp(a, b, (size_t)n * sizeof(float)) != 0) {
                same_ref = 0;
                CHECK(0, "%s: gain differs from the reference at hop %d", id, hh);
                break;
            }
            if (!all_conv && memcmp(b, c, (size_t)n * sizeof(float)) != 0) flags_matter = 1;
        }
        CHECK(same_ref, "%s: gain equals the reference through the initial-state exit", id);
        CHECK(flags_matter, "%s: with the two flags cleared the initial state changes the gain", id);
        CHECK(r->sg.initial_state == 0 && ref.post_sg.initial_state == 0,
              "%s: the stage leaves the initial state when all filters converged", id);

        aec_destroy(lane);
        free(ref.post_sg_storage); free(ref.post_near_power); free(ref.error_power);
        free(ref.render_i16); free(err); free(near); free(r2); free(cn); free(rend);
        free(mem); free(mem_on);
    }
}

/* After a saturated-echo hop the minimum gain is 0 and the maximum gain is
 * last_gain * increase, raised to floor_first_increase so that a gain that has
 * fallen to ~0 can start to recover. The limit is in the suppressor's power
 * domain; the returned gain is its square root. */
static void test_saturated_recovery(void) {
    int g;
    for (g = 0; g < N_GRIDS; ++g) {
        AecConfig cfg = make_cfg(g, 1);
        size_t need = aec_res_get_mem_size(&cfg);
        void* mem = aligned_block(need);
        AecRes* r = aec_res_init(mem, need, &cfg);
        int n, hop, k, hh, flat = 1;
        Complex *err, *near;
        float *r2, *cn, *rend, ffi, ffi_amp;
        const float* out = NULL;
        AecResInput in;
        char id[32];
        snprintf(id, sizeof id, "%d/%d", GRIDS[g][0], GRIDS[g][1]);
        if (!r) { CHECK(0, "%s: init", id); free(mem); continue; }
        n = r->n_bins; hop = r->hop_size;
        ffi = r->sg.cfg.floor_first_increase;
        sk_fast_sqrt_f32(&ffi, &ffi_amp, 1);
        err = (Complex*)calloc((size_t)n, sizeof(Complex));
        near = (Complex*)calloc((size_t)n, sizeof(Complex));
        r2 = (float*)calloc((size_t)n, sizeof(float));
        cn = (float*)calloc((size_t)n, sizeof(float));
        rend = (float*)calloc((size_t)hop, sizeof(float));
        echo_dominated_inputs(n, err, near, r2, cn);
        in.error_spec = err; in.near_spec = near; in.r2 = r2; in.comfort_noise = cn;
        in.render_hop = rend; in.all_converged = 0; in.max_dt = 0.0f;
        in.max_saturation = 1.0f; in.clock_drift = 0;
        for (hh = 0; hh < 6; ++hh) out = aec_res_process(r, &in);   /* gain collapses */
        CHECK(out != NULL && out[n / 2] < ffi_amp, "%s: saturated echo collapses the gain below the floor", id);
        for (k = 0; k < n; ++k) r2[k] = 0.0f;                       /* echo gone, still saturated */
        out = aec_res_process(r, &in);
        if (out) for (k = 1; k < n - 1; ++k) if (out[k] != ffi_amp) flat = 0;
        CHECK(out != NULL && flat, "%s: the first recovery hop is limited to floor_first_increase", id);
        free(err); free(near); free(r2); free(cn); free(rend); free(mem);
    }
}

/* ---- F ---- */
/* Far-end signal for hop h of 400: silence, a quiet steady floor with an occasional click (low-render
 * detector on, then off through the click), a loud burst (crosses the far-active latch), a second
 * silence and a moderate stationary tail. */
static void far_hop(int h, float* x, int hop) {
    int k;
    float amp;
    if (h < 40) amp = 0.0f;
    else if (h < 120) amp = 2e-5f;
    else if (h < 160) amp = 3e-4f;
    else if (h < 200) amp = 0.6f;
    else if (h < 240) amp = 0.0f;
    else if (h < 330) amp = 1e-5f;
    else amp = 0.05f;
    for (k = 0; k < hop; ++k) x[k] = amp * srand_f();
    if (h >= 60 && h < 120 && h % 9 == 0) x[hop / 3] = 5e-4f;    /* click above 3x the average */
}

static void test_far_end_detectors(void) {
    int g, p;
    enum { HOPS = 400 };
    for (g = 0; g < N_GRIDS; ++g) {
        for (p = 0; p < N_LEVELS; ++p) {
            AecConfig cfg = make_cfg(g, p);
            size_t need = aec_res_get_mem_size(&cfg);
            void *mem = aligned_block(need);
            AecRes *r = aec_res_init(mem, need, &cfg);
            int n, hop, h, k, low_on = 0, low_off = 0, unlatched_early = 0, latched_late = 0, gains_ok = 1;
            Complex *err, *near;
            float *r2, *cn, *rend;
            char id[48];
            snprintf(id, sizeof id, "%d/%d/level%d", GRIDS[g][0], GRIDS[g][1], p);
            if (!r) { CHECK(0, "%s: init", id); free(mem); continue; }
            n = r->n_bins; hop = r->hop_size;
            err = (Complex*)calloc((size_t)n, sizeof(Complex));
            near = (Complex*)calloc((size_t)n, sizeof(Complex));
            r2 = (float*)calloc((size_t)n, sizeof(float));
            cn = (float*)calloc((size_t)n, sizeof(float));
            rend = (float*)calloc((size_t)hop, sizeof(float));
            for (h = 0; h < HOPS; ++h) {
                AecResInput in;
                const float *gain;
                float peak_sq = 0.0f;
                float echo_lvl = (h / 50) % 2 ? 1e8f : 1e4f;
                float near_amp = (h / 70) % 2 ? 3e-3f : 3e-4f;
                for (k = 0; k < n; ++k) {
                    err[k].r = near_amp * srand_f(); err[k].i = near_amp * srand_f();
                    near[k].r = near_amp * srand_f(); near[k].i = near_amp * srand_f();
                    r2[k] = echo_lvl * (0.1f + urand());
                    cn[k] = 1e2f * (0.5f + urand());
                }
                far_hop(h, rend, hop);
                in.error_spec = err; in.near_spec = near;
                in.r2 = r2; in.comfort_noise = cn;
                in.all_converged = (h > 100 && h % 3 != 0);
                in.max_dt = (h % 11 == 0) ? 0.5f : 0.0f;
                in.max_saturation = (h % 23 == 0) ? 1.0f : 0.0f;
                in.clock_drift = (h % 17 == 0);
                in.render_hop = rend;
                for (k = 0; k < hop; ++k) {
                    float sq = rend[k] * 32768.0f;
                    sq *= sq;
                    if (sq > peak_sq) peak_sq = sq;
                }

                /* low-render detector decision this hop, from the instance about to see it */
                {
                    float avg = r->sg.low_render_avg_power;
                    int low = avg < r->sg.cfg.low_render_threshold &&
                              peak_sq < 3.0f * (avg * (64.0f / (float)hop));
                    if (low) low_on++; else low_off++;
                }
                gain = aec_res_process(r, &in);
                if (!gain) gains_ok = 0;
                if (h == 5 && !r->sg.far_active_latched) unlatched_early = 1;
                if (h >= 160 && r->sg.far_active_latched) latched_late = 1;
            }
            CHECK(gains_ok, "%s: every hop returns a gain", id);
            CHECK(low_on > 0 && low_off > 0, "%s: both low-render branches exercised (%d on, %d off)", id, low_on, low_off);
            CHECK(unlatched_early && latched_late, "%s: far-active latch is off before the burst and on after", id);
            free(err); free(near); free(r2); free(cn); free(rend); free(mem);
        }
    }
}

int main(void) {
    test_config();
    test_contract();
    test_render_scale();
    test_initial_phase();
    test_saturated_recovery();
    test_far_end_detectors();
    printf("test_aec_res: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
