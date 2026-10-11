/* test_aec_res.c -- aec_res.h: standalone residual-echo suppression stage.
 *
 *  A  configuration: the suppressor an AecRes builds is bit-for-bit the one an
 *     Aec lane builds (heap and caller-pool lanes), on every supported grid
 *     and preset, and equals the checked-in digests of the lane
 *     configuration (so the shared builder cannot drift from what the
 *     library produced before it was factored out).
 *  B  gain: aec_res_process against a transcription of the multi-lane
 *     pipeline's post-beam RES stage (power preparation, thresholds,
 *     suppressor call, preset retarget, reset), bit-identical per hop over
 *     long randomised runs that cross every branch of the preparation.
 *  C  aec_cng_fill against transcriptions of the two pipelines' comfort-noise
 *     loops, bit-identical spectrum and generator state.
 *  D  contract: argument checks, sizing, alignment, no write outside the
 *     block, reset and preset semantics.
 *  E  details the comparison in B does not discriminate: the exact 32768
 *     scale of the render block (a block straddling the split-floor latch
 *     power), the initial-state exit (with the two flags that make it inert in
 *     the shipped configuration cleared), and the floor_first_increase limit on
 *     the first recovery hop after a saturated echo.
 *
 * usage: test_aec_res <sg_config golden> */
#include "src/aec_res.c"

#include "aec_simd_kernels.h"
#include "support/sha256.h"

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
#define N_PRESETS 3

static AecConfig make_cfg(int g, int preset) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, (AecPreset)preset, GRIDS[g][0]);
    cfg.fft_size = GRIDS[g][1];
    return cfg;
}

static void sg_digest(const SuppressionGain* sg, char* hex) {
    Sha h;
    const float* t[6];
    int i, n = sg->tun.table_len;
    t[0] = sg->tun.nearend_enr_tr; t[1] = sg->tun.nearend_enr_su; t[2] = sg->tun.nearend_emr_tr;
    t[3] = sg->tun.normal_enr_tr; t[4] = sg->tun.normal_enr_su; t[5] = sg->tun.normal_emr_tr;
    sha_init(&h);
    sha_update(&h, &sg->cfg, sizeof sg->cfg);
    sha_update(&h, &n, sizeof n);
    for (i = 0; i < 6; ++i) sha_update(&h, t[i], (size_t)n * sizeof(float));
    sha_final(&h, hex);
}

static int tun_equal(const SuppressionGainTuning* a, const SuppressionGainTuning* b) {
    return a->table_len == b->table_len &&
           a->nearend_enr_tr == b->nearend_enr_tr && a->nearend_enr_su == b->nearend_enr_su &&
           a->nearend_emr_tr == b->nearend_emr_tr && a->normal_enr_tr == b->normal_enr_tr &&
           a->normal_enr_su == b->normal_enr_su && a->normal_emr_tr == b->normal_emr_tr;
}

static void* aligned_block(size_t bytes) {
    void* p = NULL;
    if (posix_memalign(&p, 16, bytes) != 0) return NULL;
    return p;
}

/* ---- A ---- */
static void test_config(const char* golden_path) {
    FILE* f = fopen(golden_path, "r");
    char line[256];
    int g, p;
    CHECK(f != NULL, "open sg_config golden");
    for (g = 0; g < N_GRIDS; ++g) {
        for (p = 0; p < N_PRESETS; ++p) {
            AecConfig cfg = make_cfg(g, p);
            Aec heap_lane;
            Aec* pool_lane;
            void *pool, *res_mem;
            AecRes* r;
            size_t need = aec_res_get_mem_size(&cfg);
            char hex[65], want[80], id[48];
            int found = 0;
            snprintf(id, sizeof id, "%d/%d/preset%d", GRIDS[g][0], GRIDS[g][1], p);

            CHECK(need > 0, "%s: aec_res_get_mem_size", id);
            res_mem = aligned_block(need);
            r = aec_res_init(res_mem, need, &cfg);
            CHECK(r != NULL, "%s: aec_res_init", id);
            CHECK(aec_create(&heap_lane, &cfg) == 0, "%s: aec_create", id);
            pool = aligned_block(aec_get_mem_size(&cfg));
            pool_lane = aec_init(pool, aec_get_mem_size(&cfg), &cfg);
            CHECK(pool_lane != NULL, "%s: aec_init", id);
            if (!r || !pool_lane) continue;

            CHECK(memcmp(&heap_lane.a3_sg.cfg, &r->sg_cfg, sizeof r->sg_cfg) == 0,
                  "%s: config bytes equal the heap lane's", id);
            CHECK(memcmp(&pool_lane->a3_sg.cfg, &r->sg_cfg, sizeof r->sg_cfg) == 0,
                  "%s: config bytes equal the pool lane's", id);
            CHECK(tun_equal(&heap_lane.a3_sg.tun, &r->sg_tun), "%s: tuning tables are the lane's", id);
            CHECK(memcmp(&r->sg.cfg, &r->sg_cfg, sizeof r->sg_cfg) == 0, "%s: live config equals stored", id);
            CHECK(r->n_bins == heap_lane.n_freqs && r->hop_size == heap_lane.hop_size,
                  "%s: dimensions equal the lane's", id);

            sg_digest(&heap_lane.a3_sg, hex);
            if (f) {
                rewind(f);
                while (fgets(line, sizeof line, f)) {
                    char gid[48], dg[80];
                    if (line[0] == '#') continue;
                    if (sscanf(line, "%47s %79s", gid, dg) == 2 && !strcmp(gid, id)) {
                        strcpy(want, dg); found = 1; break;
                    }
                }
            }
            CHECK(found && !strcmp(hex, want), "%s: lane configuration digest equals the golden", id);
            sg_digest(&r->sg, hex);
            CHECK(found && !strcmp(hex, want), "%s: aec_res configuration digest equals the golden", id);

            aec_destroy(&heap_lane);
            aec_destroy(pool_lane);
            free(pool);
            free(res_mem);
        }
    }
    if (f) fclose(f);
}

/* ---- B: transcription of the multi-lane pipeline's post-beam RES stage ---- */
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

/* post_res_gain: power/reference preparation, then the suppressor. */
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

static int ref_set_preset(RefPost* p, AecPreset preset, float ramp_ms) {
    float db;
    if (aec_preset_floor_db(preset, &db) != 0) return -1;
    if (suppression_gain_set_split_floor_far_active_db(&p->post_sg, db, ramp_ms) != 0) return -1;
    p->post_sg_cfg.split_floor_far_active = p->post_sg.cfg.split_floor_far_active;
    return 0;
}

static void test_gain(void) {
    static const float dts[] = {0.0f, 0.1f, 0.2f, 0.20000002f, 0.5f, 1.0f};
    static const float sats[] = {0.0f, 0.5f, 0.50000006f, 1.0f};
    int g;
    for (g = 0; g < N_GRIDS; ++g) {
        AecConfig cfg = make_cfg(g, 1);
        Aec lane;
        size_t need = aec_res_get_mem_size(&cfg);
        void* mem = aligned_block(need);
        AecRes* r = aec_res_init(mem, need, &cfg);
        RefPost ref;
        int n, hop, hh, k, ok = 1, floors_ok = 1;
        Complex *err, *near;
        float *r2, *cn, *rend;
        AecResInput in;
        char id[32];
        snprintf(id, sizeof id, "%d/%d", GRIDS[g][0], GRIDS[g][1]);
        if (!r || aec_create(&lane, &cfg) != 0) { CHECK(0, "%s: setup", id); continue; }

        /* The transcribed stage takes its configuration from a lane, as the
         * pipeline does. */
        memset(&ref, 0, sizeof ref);
        n = ref.n_freqs = lane.n_freqs;
        hop = ref.hop_size = lane.hop_size;
        ref.post_sg_cfg = lane.a3_sg.cfg;
        ref.post_sg_tun = lane.a3_sg.tun;
        ref.post_sg_storage = (float*)calloc((size_t)(10 + ref.post_sg_cfg.nearend_smoother_n) * n,
                                             sizeof(float));
        ref.post_near_power = (float*)calloc((size_t)n, sizeof(float));
        ref.error_power = (float*)calloc((size_t)n, sizeof(float));
        ref.render_i16 = (float*)calloc((size_t)hop, sizeof(float));
        ref_sg_init(&ref);

        err = (Complex*)calloc((size_t)n, sizeof(Complex));
        near = (Complex*)calloc((size_t)n, sizeof(Complex));
        r2 = (float*)calloc((size_t)n, sizeof(float));
        cn = (float*)calloc((size_t)n, sizeof(float));
        rend = (float*)calloc((size_t)hop, sizeof(float));

        for (hh = 0; hh < 3200; ++hh) {
            int all_conv, clock_drift;
            float max_dt, max_sat, level, echo_level;
            const float *a, *b;
            /* phases: quiet, near-end speech, echo, double talk, saturated echo */
            int phase = (hh / 200) % 5;
            if (hh == 1000) {
                CHECK(ref_set_preset(&ref, AEC_PRESET_AGGRESSIVE, 120.0f) == 0 &&
                      aec_res_set_preset(r, AEC_PRESET_AGGRESSIVE, 120.0f) == 0,
                      "%s: ramped preset", id);
            }
            if (hh == 2000) {
                CHECK(ref_set_preset(&ref, AEC_PRESET_MILD, 0.0f) == 0 &&
                      aec_res_set_preset(r, AEC_PRESET_MILD, 0.0f) == 0, "%s: immediate preset", id);
            }
            if (hh == 2400) {
                ref_sg_init(&ref);
                aec_res_reset(r);
            }
            level = phase == 0 ? 1e-4f : phase == 1 ? 0.2f : phase == 2 ? 0.05f : phase == 3 ? 0.1f : 0.3f;
            echo_level = (phase == 2 || phase == 3 || phase == 4) ? 0.5f * level : 1e-5f;
            for (k = 0; k < n; ++k) {
                err[k].r = level * 0.3f * srand_f(); err[k].i = level * 0.3f * srand_f();
                near[k].r = level * srand_f(); near[k].i = level * srand_f();
                if ((rnd64() & 63) == 0) { near[k].r = 0.0f; near[k].i = 0.0f; }
                r2[k] = (rnd64() & 31) == 0 ? 0.0f : echo_level * echo_level * 1e9f * urand();
                cn[k] = (rnd64() & 31) == 0 ? -1.0f : 1e3f * urand();
            }
            for (k = 0; k < hop; ++k) rend[k] = (phase == 0 ? 0.0f : 0.3f) * srand_f();
            all_conv = (hh / 50) % 3 != 0;
            max_dt = dts[(hh / 7) % 6];
            max_sat = sats[(hh / 11) % 4];
            clock_drift = (hh / 300) % 2;

            ref.aligned_ref = rend;
            ref.fused_near = near;
            ref.fused_r2 = r2;
            ref.fused_comfort = cn;
            ref_mag2(ref.error_power, err, n);
            a = ref_post_res_gain(&ref, all_conv, max_dt, max_sat, clock_drift);

            in.error_spec = err; in.near_spec = near; in.r2 = r2; in.comfort_noise = cn;
            in.render_hop = rend; in.all_converged = all_conv; in.max_dt = max_dt;
            in.max_saturation = max_sat; in.clock_drift = clock_drift;
            b = aec_res_process(r, &in);
            if (!a || !b || memcmp(a, b, (size_t)n * sizeof(float)) != 0) {
                ok = 0;
                CHECK(0, "%s: gain differs at hop %d", id, hh);
                break;
            }
            {
                float live_a = ref.post_sg.split_floor_far_active_live, live_b, tgt_b;
                aec_res_split_floor(r, &live_b, &tgt_b);
                if (live_a != live_b || ref.post_sg_cfg.split_floor_far_active != tgt_b) floors_ok = 0;
            }
        }
        CHECK(ok, "%s: gains bit-identical to the pipeline stage over 3200 hops", id);
        CHECK(floors_ok, "%s: split floor (live, target) tracks the pipeline stage", id);

        aec_destroy(&lane);
        free(ref.post_sg_storage); free(ref.post_near_power); free(ref.error_power);
        free(ref.render_i16); free(err); free(near); free(r2); free(cn); free(rend);
        free(mem);
    }
}

/* ---- C: transcriptions of the two pipelines' comfort-noise loops ---- */
#define REF_CNG_NR_GAIN_FLOOR 0.31622777f

/* Multi-lane pipeline: scales by the fused RES gain `res_gain`, advances its own
 * generator one step per bin. */
static void ref_cng_4ch(uint32_t* rng_state, Complex* output_spec, const float* fused_comfort,
                        const float* res_gain, const float* nr_gain, int n) {
    int k;
    for (k = 1; k < n - 1; ++k) {
        float n_amp = fused_comfort[k] / REF_PSD_SCALE;
        float gain2 = 1.0f - res_gain[k] * res_gain[k];
        float amplitude;
        uint32_t ix;
        n_amp = n_amp > 0.0f ? sqrtf(n_amp) : 0.0f;
        gain2 = gain2 > 0.0f ? sqrtf(gain2) : 0.0f;
        amplitude = n_amp * gain2 * fmaxf(nr_gain[k], REF_CNG_NR_GAIN_FLOOR);
        *rng_state = (*rng_state * 69069u + 1u) & 0x7FFFFFFFu;
        ix = *rng_state >> 26;
        output_spec[k].r += amplitude * AEC3B_SQRT2_SIN_LUT[ix];
        output_spec[k].i += amplitude * AEC3B_SQRT2_SIN_LUT[(ix + 8u) & 31u];
    }
}

/* Mono pipeline: the loop reads the AEC's own res_gain and the NR gain. */
static void ref_cng_mono(uint32_t* rng_state, Complex* spec, const float* comfort_noise,
                         const float* res_gain, const float* nr_gain, int n_freqs) {
    int k;
    for (k = 1; k < n_freqs - 1; k++) {
        float n_amp = comfort_noise[k] / REF_PSD_SCALE;
        float ng2, noise_gain, a;
        uint32_t ix;
        n_amp = (n_amp > 0.0f) ? sqrtf(n_amp) : 0.0f;
        ng2 = 1.0f - res_gain[k] * res_gain[k];
        noise_gain = (ng2 > 0.0f) ? sqrtf(ng2) : 0.0f;
        a = noise_gain * n_amp * fmaxf(nr_gain[k], REF_CNG_NR_GAIN_FLOOR);
        *rng_state = (*rng_state * 69069u + 1u) & 0x7FFFFFFFu;
        ix = *rng_state >> 26;
        spec[k].r += a * AEC3B_SQRT2_SIN_LUT[ix];
        spec[k].i += a * AEC3B_SQRT2_SIN_LUT[(ix + 8u) & 31u];
    }
}

static void test_cng(void) {
    static const int sizes[] = {3, 4, 5, 9, 129, 257, 513};
    enum { MAXN = 513 };
    static Complex s0[MAXN], s1[MAXN], s2[MAXN];
    static float cn[MAXN], gr[MAXN], gn[MAXN], ones[MAXN];
    int si, it, k, ok4 = 1, okm = 1, oku = 1, oknull = 1;
    for (k = 0; k < MAXN; ++k) ones[k] = 1.0f;
    for (si = 0; si < (int)(sizeof sizes / sizeof sizes[0]); ++si) {
        int n = sizes[si];
        for (it = 0; it < 400; ++it) {
            uint32_t ra = (uint32_t)rnd64() & 0x7FFFFFFFu, rb = ra, rc = ra, rd = ra;
            for (k = 0; k < n; ++k) {
                int mode = (int)(rnd64() & 7);
                cn[k] = mode == 0 ? 0.0f : mode == 1 ? -3.0f : mode == 2 ? 1e12f
                      : mode == 3 ? 1e-30f : 1e5f * urand();
                mode = (int)(rnd64() & 7);
                gr[k] = mode == 0 ? 0.0f : mode == 1 ? 1.0f : mode == 2 ? 1.2f
                      : mode == 3 ? -0.5f : urand();
                mode = (int)(rnd64() & 7);
                gn[k] = mode == 0 ? 0.0f : mode == 1 ? 1.0f : mode == 2 ? 0.31622777f
                      : mode == 3 ? 1.7f : urand();
                s0[k].r = srand_f(); s0[k].i = srand_f();
                s1[k] = s0[k]; s2[k] = s0[k];
            }
            ref_cng_4ch(&ra, s0, cn, gr, gn, n);
            aec_cng_fill(&rb, s1, cn, gr, gn, REF_CNG_NR_GAIN_FLOOR, n);
            if (memcmp(s0, s1, (size_t)n * sizeof(Complex)) != 0 || ra != rb) ok4 = 0;

            for (k = 0; k < n; ++k) { s0[k] = s2[k]; s1[k] = s2[k]; }
            ref_cng_mono(&rc, s0, cn, gr, gn, n);
            aec_cng_fill(&rd, s1, cn, gr, gn, REF_CNG_NR_GAIN_FLOOR, n);
            if (memcmp(s0, s1, (size_t)n * sizeof(Complex)) != 0 || rc != rd) okm = 0;
            if (ra != rc) okm = 0;

            /* no second gain == unity vector */
            {
                uint32_t rx = rd, ry = rd;
                for (k = 0; k < n; ++k) { s0[k] = s2[k]; s1[k] = s2[k]; }
                ref_cng_4ch(&rx, s0, cn, gr, ones, n);
                aec_cng_fill(&ry, s1, cn, gr, NULL, REF_CNG_NR_GAIN_FLOOR, n);
                if (memcmp(s0, s1, (size_t)n * sizeof(Complex)) != 0 || rx != ry) oknull = 0;
            }
            /* a floor above 1 and a zero floor */
            {
                float floors[2] = {0.0f, 2.5f};
                int fi;
                for (fi = 0; fi < 2; ++fi) {
                    uint32_t rx = rd, ry = rd;
                    for (k = 0; k < n; ++k) { s0[k] = s2[k]; s1[k] = s2[k]; }
                    /* transcription with the floor as a parameter */
                    for (k = 1; k < n - 1; ++k) {
                        float n_amp = cn[k] / REF_PSD_SCALE, gain2 = 1.0f - gr[k] * gr[k], amp;
                        uint32_t ix;
                        n_amp = n_amp > 0.0f ? sqrtf(n_amp) : 0.0f;
                        gain2 = gain2 > 0.0f ? sqrtf(gain2) : 0.0f;
                        amp = n_amp * gain2 * fmaxf(gn[k], floors[fi]);
                        rx = (rx * 69069u + 1u) & 0x7FFFFFFFu;
                        ix = rx >> 26;
                        s0[k].r += amp * AEC3B_SQRT2_SIN_LUT[ix];
                        s0[k].i += amp * AEC3B_SQRT2_SIN_LUT[(ix + 8u) & 31u];
                    }
                    aec_cng_fill(&ry, s1, cn, gr, gn, floors[fi], n);
                    if (memcmp(s0, s1, (size_t)n * sizeof(Complex)) != 0 || rx != ry) oku = 0;
                }
            }
        }
    }
    CHECK(ok4, "aec_cng_fill equals the 4-channel pipeline loop (spectrum and generator)");
    CHECK(okm, "aec_cng_fill equals the mono pipeline loop (spectrum and generator)");
    CHECK(oknull, "aec_cng_fill with no second gain equals a unity vector");
    CHECK(oku, "aec_cng_fill equals the loop for floors 0 and 2.5");
    {
        /* bins 0 and n-1 and the argument guards leave everything untouched */
        Complex a[8], b[8];
        uint32_t rng = 12345u, rng0 = rng;
        float c8[8] = {1e5f, 1e5f, 1e5f, 1e5f, 1e5f, 1e5f, 1e5f, 1e5f}, g8[8] = {0};
        memset(a, 0, sizeof a);
        memcpy(b, a, sizeof a);
        aec_cng_fill(&rng, a, c8, g8, NULL, 0.3f, 8);
        CHECK(a[0].r == 0.0f && a[0].i == 0.0f && a[7].r == 0.0f && a[7].i == 0.0f,
              "aec_cng_fill leaves bin 0 and the last bin alone");
        CHECK(rng != rng0, "aec_cng_fill advances the generator");
        rng = rng0;
        aec_cng_fill(NULL, b, c8, g8, NULL, 0.3f, 8);
        aec_cng_fill(&rng, NULL, c8, g8, NULL, 0.3f, 8);
        aec_cng_fill(&rng, b, NULL, g8, NULL, 0.3f, 8);
        aec_cng_fill(&rng, b, c8, NULL, NULL, 0.3f, 8);
        aec_cng_fill(&rng, b, c8, g8, NULL, 0.3f, 2);
        CHECK(rng == rng0 && memcmp(b, "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 16) == 0,
              "aec_cng_fill ignores NULL arguments and n < 3");
    }
}

/* ---- D ---- */
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

    /* presets */
    CHECK(aec_res_split_floor(NULL, &live, &target) == -1, "split_floor rejects NULL");
    CHECK(aec_res_split_floor(r, NULL, NULL) == 0, "split_floor accepts NULL outputs");
    aec_res_split_floor(r, &live, &target);
    floor_mild = 0.0f; floor_aggr = 0.0f;
    CHECK(live == target && target > 0.0f, "fresh instance: live floor equals target");
    CHECK(aec_res_set_preset(NULL, AEC_PRESET_MILD, 0.0f) == -1, "set_preset rejects NULL");
    CHECK(aec_res_set_preset(r, (AecPreset)9, 0.0f) == -1, "set_preset rejects an unknown preset");
    CHECK(aec_res_set_preset(r, AEC_PRESET_MILD, -1.0f) == -1, "set_preset rejects a negative ramp");
    CHECK(aec_res_set_preset(r, AEC_PRESET_MILD, 1e9f) == -1, "set_preset rejects a ramp past the limit");
    { float l2, t2; aec_res_split_floor(r, &l2, &t2);
      CHECK(l2 == live && t2 == target, "a rejected preset change writes nothing"); }
    CHECK(aec_res_set_preset(r, AEC_PRESET_AGGRESSIVE, 0.0f) == 0, "set_preset aggressive");
    aec_res_split_floor(r, &live, &floor_aggr);
    CHECK(aec_res_process(r, &in) != NULL, "process after preset change");
    aec_res_split_floor(r, &live, &target);
    CHECK(live == floor_aggr && target == floor_aggr, "immediate retarget lands on the next hop");
    CHECK(aec_res_set_preset(r, AEC_PRESET_MILD, 0.0f) == 0, "set_preset mild");
    aec_res_split_floor(r, &live, &floor_mild);
    CHECK(floor_mild > floor_aggr, "mild floor is above aggressive floor");
    aec_res_reset(r);
    aec_res_split_floor(r, &live, &target);
    CHECK(target == floor_mild && live == floor_mild, "reset keeps the retargeted floor");
    aec_res_reset(NULL);

    /* reset: a reset instance reproduces a fresh one */
    {
        unsigned char* blk2 = (unsigned char*)aligned_block(need);
        AecRes* f;
        int same = 1;
        memset(blk2, 0x5A, need);
        f = aec_res_init(blk2, need, &cfg);
        aec_res_set_preset(f, AEC_PRESET_MILD, 0.0f);
        aec_res_reset(r);
        for (i = 0; i < 50; ++i) {
            g1 = aec_res_process(r, &in);
            g2 = aec_res_process(f, &in);
            if (!g1 || !g2 || memcmp(g1, g2, (size_t)n * sizeof(float)) != 0) same = 0;
        }
        CHECK(same, "reset instance equals a fresh instance with the same preset");
        free(blk2);
    }
    /* the result does not depend on what the block held before init */
    {
        unsigned char* blk3 = (unsigned char*)aligned_block(need);
        AecRes* f;
        int same = 1;
        memset(blk3, 0x00, need);
        f = aec_res_init(blk3, need, &cfg);
        aec_res_set_preset(f, AEC_PRESET_MILD, 0.0f);
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

/* ---- E: stage details the transcription in B does not discriminate ---- */

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
        for (p = 0; p < N_PRESETS; ++p) {
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
            snprintf(id, sizeof id, "%d/%d/preset%d", GRIDS[g][0], GRIDS[g][1], p);
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
            CHECK(r->sg_cfg.split_floor_latch_power > 1.0f &&
                  r->sg_cfg.split_floor_latch_power < 32768.0f * 32768.0f,
                  "%s: the latch power is reachable by a render block", id);
            for (side = 0; side < 2; ++side) {
                /* scaled sample level s with s*s = latch * (1 -/+ 3e-5); v = s / 32768 is exact */
                float s = (float)sqrt((double)r->sg_cfg.split_floor_latch_power *
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
 * the transcribed configuration; the gain then depends on the state, and the
 * stage must track the transcription through the transition. */
static void test_initial_phase(void) {
    int g;
    for (g = 0; g < N_GRIDS; ++g) {
        AecConfig cfg = make_cfg(g, 1);
        Aec lane;
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
        if (!r || !r_on || aec_create(&lane, &cfg) != 0) { CHECK(0, "%s: setup", id); continue; }

        memset(&ref, 0, sizeof ref);
        n = ref.n_freqs = lane.n_freqs;
        hop = ref.hop_size = lane.hop_size;
        ref.post_sg_cfg = lane.a3_sg.cfg;
        ref.post_sg_tun = lane.a3_sg.tun;
        ref.post_sg_cfg.dne_use_during_initial_phase = 0;
        ref.post_sg_cfg.lf_smoothing_during_initial_phase = 0;
        ref.post_sg_storage = (float*)calloc((size_t)(10 + ref.post_sg_cfg.nearend_smoother_n) * n,
                                             sizeof(float));
        ref.post_near_power = (float*)calloc((size_t)n, sizeof(float));
        ref.error_power = (float*)calloc((size_t)n, sizeof(float));
        ref.render_i16 = (float*)calloc((size_t)hop, sizeof(float));
        ref_sg_init(&ref);
        r->sg_cfg.dne_use_during_initial_phase = 0;
        r->sg_cfg.lf_smoothing_during_initial_phase = 0;
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
                CHECK(0, "%s: gain differs from the transcription at hop %d", id, hh);
                break;
            }
            if (!all_conv && memcmp(b, c, (size_t)n * sizeof(float)) != 0) flags_matter = 1;
        }
        CHECK(same_ref, "%s: gain equals the transcription through the initial-state exit", id);
        CHECK(flags_matter, "%s: with the two flags cleared the initial state changes the gain", id);
        CHECK(r->sg.initial_state == 0 && ref.post_sg.initial_state == 0,
              "%s: the stage leaves the initial state when all filters converged", id);

        aec_destroy(&lane);
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
        ffi = r->sg_cfg.floor_first_increase;
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

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s <sg_config golden>\n", argv[0]); return 2; }
    test_config(argv[1]);
    test_gain();
    test_cng();
    test_contract();
    test_render_scale();
    test_initial_phase();
    test_saturated_recovery();
    printf("test_aec_res: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
