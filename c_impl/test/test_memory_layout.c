/* Pool size, final carve position, and tail guard across optional allocations. */
#include "aec.h"
#include "aec_white_box.h"
#include "aec_test_tuning.h"
#include "aec_internal.h"
#include "hpf.h"
#include "res_gain_of.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t align16(size_t n) { return (n + 15u) & ~(size_t)15u; }

/* Absolute pins, LP64 only: sizeof(Aec) and the single-channel pool sizes at the balanced level, default delay mode, with and
 * without the shadow filter. They are literals, not recomputed from the build under test. */
#define AEC_SIZEOF_PIN 6048u
static int check_pinned_sizes(void) {
    static const struct { int sr, fft; size_t with_shadow, without_shadow; } pin[3] = {
        {16000, 256, 239760u, 211872u}, {16000, 512, 352128u, 312016u}, {48000, 1024, 776064u, 675504u} };
    int i, bad = 0;
    if (sizeof(void*) != 8 || sizeof(size_t) != 8) {
        printf("test_memory_layout: absolute size pins are LP64 values; skipped on this %zu-byte-pointer ABI\n",
               sizeof(void*));
        return 0;
    }
    if (sizeof(Aec) != AEC_SIZEOF_PIN) {
        fprintf(stderr, "sizeof(Aec) %zu, pinned %u\n", sizeof(Aec), AEC_SIZEOF_PIN);
        bad = 1;
    }
    for (i = 0; i < 3; ++i) {
        AecTestCfg cfg;
        size_t now_s, now_n;
        aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, pin[i].sr);
        cfg.c.fft_size = pin[i].fft;
        now_s = aec_get_mem_size(&cfg.c);
        cfg.t.enable_shadow = 0;
        now_n = aec_tcfg_mem_size(&cfg);
        if (now_s != pin[i].with_shadow || now_n != pin[i].without_shadow) {
            fprintf(stderr, "%d/%d pool %zu / %zu (shadow on / off), pinned %zu / %zu\n", pin[i].sr, pin[i].fft,
                    now_s, now_n, pin[i].with_shadow, pin[i].without_shadow);
            bad = 1;
        }
    }
    if (!bad)
        printf("test_memory_layout: sizeof(Aec) %zu; balanced pool 16k/256 %zu, 16k/512 %zu, 48k/1024 %zu B\n",
               sizeof(Aec), pin[0].with_shadow, pin[1].with_shadow, pin[2].with_shadow);
    return bad;
}

/* ABI facts the headers used to pin at compile time. They fail here at run time, with the fact named. */
#define LAYOUT_EXPECT(cond, what) do { if (!(cond)) { fprintf(stderr, "test_memory_layout: FAILED: %s\n", (what)); bad = 1; } } while (0)

/* A pool is 16-byte aligned and the Aec header is carved at its start, so struct Aec must not need more. The C99 spelling of
 * alignment query: the offset of a member of the type behind one char. */
struct AecAlignProbe { char c; struct Aec a; };
static int check_header_facts(void) {
    int bad = 0;
    LAYOUT_EXPECT(sizeof(AecMemReq) == 32, "sizeof(AecMemReq) must be exactly 32 bytes (public descriptor ABI)");
    LAYOUT_EXPECT(offsetof(AecMemReq, bytes) == 24, "AecMemReq.bytes must sit at offset 24 with no padding before it");
    LAYOUT_EXPECT(offsetof(struct AecAlignProbe, a) <= 16, "struct Aec must not need more alignment than the 16-byte pool alignment");
    LAYOUT_EXPECT(AEC_DELAY_NUM_FILTERS_MAX == DA_NUM_FILTERS, "AEC_DELAY_NUM_FILTERS_MAX (aec.h) must equal DA_NUM_FILTERS, the delay estimator's bank limit");
    LAYOUT_EXPECT(AEC_DELAY_FILTER_SIZE == DA_FILTER_SIZE, "AEC_DELAY_FILTER_SIZE (aec.h) must equal DA_FILTER_SIZE, the matched filter length");
    LAYOUT_EXPECT(AEC_DELAY_FILTER_INTRA_SHIFT == DA_FILTER_INTRA_SHIFT, "AEC_DELAY_FILTER_INTRA_SHIFT (aec.h) must equal DA_FILTER_INTRA_SHIFT, the filter spacing");
    LAYOUT_EXPECT(AEC_DELAY_DOWN_SAMPLING_FACTOR == DA_DOWN_SAMPLING_FACTOR, "AEC_DELAY_DOWN_SAMPLING_FACTOR (aec.h) must equal DA_DOWN_SAMPLING_FACTOR, the estimator decimation");
    if (!bad)
        printf("test_memory_layout: sizeof(AecMemReq) %zu; struct Aec alignment <= 16; AEC_DELAY_* equal DA_*\n", sizeof(AecMemReq));
    return bad;
}

/* The pool descriptor: layout_version is a literal (bump it, and the pin, when a carved layout changes size or order) and, on LP64
 * (where the ABI hash is the identity), so is the FNV-1a-32 of the carve text, which every edit of that text moves. */
#define AEC_LAYOUT_VERSION_PIN 4u
#define AEC_CARVE_HASH_PIN_LP64 3368782680u
static int check_descriptor_pin(void) {
    AecConfig cfg;
    AecMemReq req;
    int bad = 0;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    if (aec_get_mem_requirements(&cfg, &req) != 0) { fprintf(stderr, "descriptor query refused\n"); return 1; }
    if (AEC_LAYOUT_VERSION != AEC_LAYOUT_VERSION_PIN || req.layout_version != AEC_LAYOUT_VERSION_PIN) {
        fprintf(stderr, "layout_version %u (macro %u), pinned %u\n", (unsigned)req.layout_version, (unsigned)AEC_LAYOUT_VERSION,
                AEC_LAYOUT_VERSION_PIN);
        bad = 1;
    }
    if (sizeof(void*) == 8 && sizeof(size_t) == 8 && req.build_flags_hash != AEC_CARVE_HASH_PIN_LP64) {
        fprintf(stderr, "build_flags_hash %u, pinned %u\n", (unsigned)req.build_flags_hash, (unsigned)AEC_CARVE_HASH_PIN_LP64);
        bad = 1;
    }
    if (!bad) printf("test_memory_layout: layout_version %u, build_flags_hash %u\n", (unsigned)req.layout_version, (unsigned)req.build_flags_hash);
    return bad;
}

#ifndef AEC_NO_ARRAY
/* Absolute pins of the four-microphone array pool (n_mics 4, balanced level, default delay mode, shadow filter on), LP64:
 * the flow buffers (with the array's one synthesis window), the alignment ring and estimator, the gain stage (with its decoded
 * ENR/SU tables), one shared FFT handle, and four lanes. A lane (aec_lane_get_mem_size) is the EXTERNAL_ALIGNED
 * single-channel pool minus what that contract never runs: the delay-shift snapshots, the synthesis window and output
 * buffers, the gain buffers and decoded tables, and the FFT handle; a STATE lane also omits the residual-echo estimator and
 * comfort-noise arrays. */
static int check_array_pins(void) {
    static const struct { int sr, fft; size_t state, estimate, full; } pin[3] = {
        {16000, 256, 561456u, 656992u, 666992u}, {16000, 512, 849136u, 1038880u, 1057072u},
        {48000, 1024, 1894000u, 2272160u, 2309808u} };
    int i, bad = 0;
    if (sizeof(void*) != 8 || sizeof(size_t) != 8) {
        printf("test_memory_layout: array pool pins are LP64 values; skipped on this %zu-byte-pointer ABI\n", sizeof(void*));
        return 0;
    }
    for (i = 0; i < 3; ++i) {
        AecConfig cfg;
        size_t now_s, now_e, now_f;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, pin[i].sr);
        cfg.fft_size = pin[i].fft;
        cfg.n_mics = AEC_ARRAY_MICS;
        cfg.residual = AEC_RESIDUAL_STATE;    now_s = aec_get_mem_size(&cfg);
        cfg.residual = AEC_RESIDUAL_ESTIMATE; now_e = aec_get_mem_size(&cfg);
        cfg.residual = AEC_RESIDUAL_FULL;     now_f = aec_get_mem_size(&cfg);
        if (now_s != pin[i].state || now_e != pin[i].estimate || now_f != pin[i].full) {
            fprintf(stderr, "array %d/%d pool %zu / %zu / %zu (STATE / ESTIMATE / FULL), pinned %zu / %zu / %zu\n", pin[i].sr,
                    pin[i].fft, now_s, now_e, now_f, pin[i].state, pin[i].estimate, pin[i].full);
            bad = 1;
        }
    }
    if (!bad)
        printf("test_memory_layout: array pool 16k/256 %zu, 16k/512 %zu, 48k/1024 %zu B (FULL)\n", pin[0].full, pin[1].full,
               pin[2].full);
    return bad;
}

/* Array lanes across every carve toggle (grid, residual, delay mode, shadow, live observer, mic high-pass): init at need - 1 is
 * refused and writes nothing, init at need succeeds (a lane whose carve ended anywhere but at its announced size is refused), the
 * tail canary survives, and 60 hops plus a reset plus 10 hops run clean, so a buffer a lane omits faults if a lane touches it. */
static float array_noise(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return ((float)(*state >> 8) / 16777216.0f - 0.5f) * 0.4f;
}

static int run_array_hops(Aec* a, AecResidualMode residual, Complex* weights, int hop, uint32_t* rng, int hops) {
    float mics[512 * 4], ref[512];
    int h, i;
    for (h = 0; h < hops; ++h) {
        AecBeam beam;
        for (i = 0; i < hop; ++i) {
            float far = array_noise(rng);
            int ch;
            for (ch = 0; ch < 4; ++ch) mics[i * 4 + ch] = 0.4f * far + 0.1f * array_noise(rng);
            ref[i] = far;
        }
        if (aec_process_linear(a, mics, ref) != AEC_OK) return 1;
        if (residual != AEC_RESIDUAL_STATE) {
            AecResidualView rv;
            beam.weights = weights;
            beam.beamformed_error = NULL;
            if (aec_process_residual(a, &beam) != AEC_OK) return 1;
            memset(&rv, 0, sizeof rv);
            rv.struct_size = sizeof rv;
            if (aec_get_residual(a, &rv) != AEC_OK) return 1;
            if (residual == AEC_RESIDUAL_FULL && !res_gain_of(a)) return 1;
        }
    }
    return 0;
}

static int check_array_toggles(void) {
    static const int grid[3][2] = { {16000, 256}, {16000, 512}, {48000, 1024} };
    static const AecResidualMode modes[3] = { AEC_RESIDUAL_STATE, AEC_RESIDUAL_ESTIMATE, AEC_RESIDUAL_FULL };
    int g, m, d, shadow, observer, highpass, checked = 0;
    for (g = 0; g < 3; ++g)
    for (m = 0; m < 3; ++m)
    for (d = 0; d < 3; ++d)
    for (shadow = 0; shadow < 2; ++shadow)
    for (observer = 0; observer < 2; ++observer)
    for (highpass = 0; highpass < 2; ++highpass) {
        AecTestCfg cfg;
        size_t need, i;
        unsigned char* pool = NULL;
        Aec* a;
        Complex* weights;
        uint32_t rng = 12345u + (uint32_t)checked;
        int hop, n, k, bad = 0;
        aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, grid[g][0]);
        cfg.c.fft_size = grid[g][1];
        cfg.c.n_mics = AEC_ARRAY_MICS;
        cfg.c.residual = modes[m];
        cfg.t.enable_shadow = shadow;
        cfg.t.live_observer_enabled = observer;
        cfg.t.enable_highpass = highpass;
        if (d == 0) {
            cfg.c.delay_mode = AEC_DELAY_MATCHED;
        } else if (d == 1) {
            cfg.c.delay_mode = AEC_DELAY_FIXED;
            cfg.c.fixed_delay_samples = 100;
        } else {
            cfg.c.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
            cfg.c.fixed_delay_samples = -1;
        }
        need = aec_tcfg_mem_size(&cfg);
        if (!need) { fprintf(stderr, "array pool size refused: grid=%d residual=%d delay=%d shadow=%d observer=%d highpass=%d\n", g, m, d, shadow, observer, highpass); return 1; }
        if (posix_memalign((void**)&pool, 16, need + 64u) != 0) return 1;
        memset(pool, 0xA5, need + 64u);
        if (aec_tcfg_init(pool, need - 1u, &cfg) != NULL) bad = 1;
        for (i = 0; i < need + 64u; ++i) if (pool[i] != 0xA5) bad = 1;
        a = aec_tcfg_init(pool, need, &cfg);
        if (!a) { fprintf(stderr, "array init refused: grid=%d residual=%d delay=%d shadow=%d observer=%d highpass=%d\n", g, m, d, shadow, observer, highpass); free(pool); return 1; }
        hop = aec_hop_size(a);
        n = hop + 1;
        weights = (Complex*)malloc((size_t)4 * (size_t)n * sizeof(Complex));
        if (!weights) { free(pool); return 1; }
        for (k = 0; k < 4 * n; ++k) { weights[k].r = 0.25f; weights[k].i = 0.0f; }
        if (run_array_hops(a, modes[m], weights, hop, &rng, 60)) bad = 1;
        aec_reset(a);
        if (run_array_hops(a, modes[m], weights, hop, &rng, 10)) bad = 1;
        aec_destroy(a);
        for (i = need; i < need + 64u; ++i) if (pool[i] != 0xA5) bad = 1;
        free(weights);
        free(pool);
        if (bad) {
            fprintf(stderr, "array toggle walk failed: grid=%d residual=%d delay=%d shadow=%d observer=%d highpass=%d\n", g, m, d, shadow, observer, highpass);
            return 1;
        }
        ++checked;
    }
    if (checked != 3 * 3 * 3 * 2 * 2 * 2) { fprintf(stderr, "array toggle walk covered %d configurations, expected 216\n", checked); return 1; }
    printf("test_memory_layout: %d array configurations passed\n", checked);
    return 0;
}
#endif /* !AEC_NO_ARRAY */

/* The entry points refuse a destroyed single-channel handle (its pool is still alive): process calls answer AEC_ERR_ARG, and
 * aec_get_linear() leaves struct_size set and every other byte zero. Returns nonzero on a violation. */
static int check_destroyed_single(Aec* a) {
    static float mic[4096], ref[4096];
    AecLinearView lv;
    const unsigned char* b = (const unsigned char*)&lv;
    size_t n, declared;
    int bad = 0;
    memset(&lv, 0x5A, sizeof lv);
    lv.struct_size = sizeof lv;
    if (aec_process_linear(a, mic, ref) != AEC_ERR_ARG) bad = 1;
    if (aec_process_linear_shared_far_from(a, mic, ref, NULL) != AEC_ERR_ARG) bad = 1;
    if (aec_get_linear(a, &lv) != AEC_ERR_ARG) bad = 1;
    memcpy(&declared, &lv, sizeof declared);
    if (declared != sizeof lv) bad = 1;
    for (n = sizeof declared; n < sizeof lv; ++n) if (b[n]) bad = 1;
    return bad;
}

int main(void) {
    if (check_header_facts()) return 1;
    if (check_pinned_sizes()) return 1;
    if (check_descriptor_pin()) return 1;
#ifndef AEC_NO_ARRAY
    if (check_array_pins()) return 1;
    if (check_array_toggles()) return 1;
#else
    {
        AecConfig cfg;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
        cfg.n_mics = AEC_ARRAY_MICS;
        cfg.residual = AEC_RESIDUAL_FULL;
        if (aec_get_mem_size(&cfg) != 0) { fprintf(stderr, "AEC_NO_ARRAY: n_mics 4 must have no pool size\n"); return 1; }
    }
#endif
    const int rates[] = {16000, 16000, 48000};
    const int ffts[] = {256, 512, 1024};
    int checked = 0;
    for (int g = 0; g < 3; ++g)
    for (int shadow = 0; shadow < 2; ++shadow)
    for (int observer = 0; observer < 2; ++observer)
    for (int highpass = 0; highpass < 2; ++highpass)
    for (int delay = 0; delay < 3; ++delay) {
        AecTestCfg cfg;
        aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, rates[g]);
        cfg.c.fft_size = ffts[g];
        cfg.t.enable_shadow = shadow;
        cfg.t.live_observer_enabled = observer;
        cfg.t.enable_highpass = highpass;
        if (delay == 0) {
            cfg.c.delay_mode = AEC_DELAY_MATCHED;
        } else if (delay == 1) {
            cfg.c.delay_mode = AEC_DELAY_FIXED;
            cfg.c.fixed_delay_samples = 123;
        } else {
            cfg.c.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
        }
        size_t need = aec_tcfg_mem_size(&cfg);
        unsigned char* pool = NULL;
        if (!need || posix_memalign((void**)&pool, 16, need + 64u) != 0)
            return 1;
        memset(pool, 0xA5, need + 64u);
        if (aec_tcfg_init(pool, need - 1u, &cfg) != NULL) {
            fprintf(stderr, "undersized pool accepted\n");
            free(pool);
            return 1;
        }
        for (size_t i = 0; i < need + 64u; ++i) {
            if (pool[i] != 0xA5) {
                fprintf(stderr, "rejected init wrote to pool\n");
                free(pool);
                return 1;
            }
        }
        Aec* a = aec_tcfg_init(pool, need, &cfg);
        if (!a) { free(pool); return 1; }
        /* HPF is the optional final block; scr_e2ref_arr precedes it. */
        const unsigned char* end = a->hp_mic
            ? (const unsigned char*)a->hp_mic + align16(hpf_get_mem_size())
            : (const unsigned char*)a->scr_e2ref_arr
                + align16((size_t)a->n_freqs * sizeof(float));
        int failed = end != pool + need;
        for (size_t i = need; i < need + 64u; ++i)
            if (pool[i] != 0xA5) failed = 1;
        aec_destroy(a);
        if (check_destroyed_single(a)) failed = 1;
        for (size_t i = need; i < need + 64u; ++i)
            if (pool[i] != 0xA5) failed = 1;
        free(pool);
        if (failed) {
            fprintf(stderr, "pool layout failed: grid=%d shadow=%d observer=%d "
                    "highpass=%d delay=%d\n", g, shadow, observer, highpass, delay);
            return 1;
        }
        ++checked;
    }
    printf("test_memory_layout: %d configurations passed\n", checked);
    return 0;
}
