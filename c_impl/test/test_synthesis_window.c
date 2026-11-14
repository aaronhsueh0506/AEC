/* test_synthesis_window.c -- aec_synthesis_window() public accessor.
 *
 *   1. On the 3 contracted grids (16k/256, 16k/512, 48k/1024) and the 8k/256 grid, with both shadow settings, the returned
 *      table has the length the instance synthesizes with (== fft_size == block_size) and is element-for-element identical
 *      (memcmp) to the reference window of aec_ref_tables.h (the Python twin's table, generated outside the library: the
 *      library rebuilds its window at init from a half table, so its own tables are not an oracle); the pointer is stable
 *      across processing and aec_reset(), and an AcIstft configured with it (the documented way) initialises and
 *      borrows exactly that table. The window lives in the instance pool: every use below ends before aec_destroy().
 *   1b. The same on every residual mode of a single channel and on the 4-mic array (one window per array).
 *   1c. aec_synth_window_build() rebuilds every whitelisted length from the half table, and refuses nothing it should not.
 *   2. A NULL instance, NULL window or NULL length is refused with -1 and leaves the outputs untouched.
 *
 * Build and run: `make test-synthesis-window` (from c_impl/). */
#include "aec.h"
#include "aec_test_tuning.h"
#include "aec_balanced_config.h"
#include "aec_config_build.h"
#include "aec_ref_tables.h"
#include "audio_stft.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, ...) \
    do { \
        if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); g_fail++; } \
        else         { g_pass++; } \
    } while (0)

static const AecRefTables* find_ref(int sr, int fft) {
    int i;
    for (i = 0; i < AECREF_N_GRIDS; ++i)
        if (AECREF_GRIDS[i].sample_rate == sr && AECREF_GRIDS[i].fft_size == fft) return &AECREF_GRIDS[i];
    return NULL;
}

static void check_grid(int sr, int fft, int shadow) {
    AecConfig cfg;
    AecTuning tn;
    Aec* a = NULL;
    const AecBalancedRateDims* rd = aecb_rate_cfg(sr, fft);
    const AecRefTables* ref = find_ref(sr, fft);
    const float* w = NULL;
    const float* w2 = NULL;
    int n = 0, n2 = 0, i;
    float hop_in[1024], far_in[1024];
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, sr);
    cfg.fft_size = fft;
    aec_testing_tuning_defaults(&tn, &cfg);
    tn.enable_shadow = shadow;
    CHECK(rd != NULL && ref != NULL, "%d/%d: rate row and reference window exist", sr, fft);
    CHECK((a = aec_testing_create(&cfg, &tn)) != NULL, "%d/%d shadow=%d: create", sr, fft, shadow);
    if (!rd || !ref || !a) { if (a) aec_destroy(a); return; }

    CHECK(aec_synthesis_window(a, &w, &n) == 0, "%d/%d: accessor succeeds", sr, fft);
    CHECK(n == fft && n == rd->synth_window_len, "%d/%d: length %d == fft_size %d == row length %d", sr, fft, n, fft,
          rd->synth_window_len);
    CHECK(w != NULL, "%d/%d: window pointer set", sr, fft);
    CHECK(w && memcmp(w, ref->window, (size_t)fft * sizeof(float)) == 0,
          "%d/%d: values equal the reference window", sr, fft);

    {   /* the documented AcIstft recipe borrows exactly this table */
        AcStftConfig sc;
        size_t sz;
        void* mem = NULL;
        AcIstft* is;
        ac_stft_config_default(&sc, n);
        sc.window = w;
        sz = ac_istft_get_mem_size(&sc);
        CHECK(sz != 0 && posix_memalign(&mem, 16, sz) == 0, "%d/%d: AcIstft sized", sr, fft);
        is = mem ? ac_istft_init(mem, sz, &sc) : NULL;
        CHECK(is != NULL, "%d/%d: AcIstft initialised with the borrowed table", sr, fft);
        if (is) {
            AcStftConfig live;
            CHECK(ac_istft_get_config(is, &live) == 0 && live.window == w, "%d/%d: AcIstft borrows the table", sr, fft);
        }
        free(mem);
    }

    {   /* stable across processing and reset */
        int hop = fft / 2, h;
        for (i = 0; i < hop; ++i) { hop_in[i] = 0.01f * (float)((i * 7) % 13 - 6); far_in[i] = 0.01f * (float)((i * 5) % 11 - 5); }
        for (h = 0; h < 8; ++h) { float out[1024]; aec_process(a, hop_in, far_in, out); }
        CHECK(aec_synthesis_window(a, &w2, &n2) == 0 && w2 == w && n2 == n, "%d/%d: stable across processing", sr, fft);
        aec_reset(a);
        w2 = NULL; n2 = 0;
        CHECK(aec_synthesis_window(a, &w2, &n2) == 0 && w2 == w && n2 == n, "%d/%d: stable across aec_reset", sr, fft);
    }
    aec_destroy(a);
}

/* Every residual mode of a single channel keeps its window (the pool carves it whatever the mode), and the 4-mic array
 * owns one of its own. */
static void check_modes_and_array(void) {
    static const AecResidualMode modes[4] = { AEC_RESIDUAL_OFF, AEC_RESIDUAL_STATE, AEC_RESIDUAL_ESTIMATE, AEC_RESIDUAL_FULL };
    static const int grids[4][2] = { { 16000, 256 }, { 16000, 512 }, { 48000, 1024 }, { 8000, 256 } };
    int g, m;
    for (g = 0; g < 4; ++g) {
        const AecRefTables* ref = find_ref(grids[g][0], grids[g][1]);
        for (m = 0; m < 5; ++m) {
            AecConfig cfg;
            Aec* a;
            const float* w = NULL;
            int n = 0;
            aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, grids[g][0]);
            cfg.fft_size = grids[g][1];
            if (m < 4) cfg.residual = modes[m];
            else {   /* the array (product grids only; an AEC_NO_ARRAY library has none) */
#ifdef AEC_NO_ARRAY
                continue;
#endif
                if (grids[g][0] == 8000) continue;
                cfg.n_mics = AEC_ARRAY_MICS;
                cfg.residual = AEC_RESIDUAL_FULL;
                cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
                cfg.fixed_delay_samples = -1;
            }
            a = aec_create(&cfg);
            CHECK(a != NULL && ref != NULL, "%d/%d %s: create", grids[g][0], grids[g][1], m < 4 ? "mode" : "array");
            if (!a) continue;
            CHECK(aec_synthesis_window(a, &w, &n) == 0 && n == grids[g][1] && ref &&
                  memcmp(w, ref->window, (size_t)n * sizeof(float)) == 0,
                  "%d/%d %s %d: window equals the reference", grids[g][0], grids[g][1], m < 4 ? "mode" : "array", m);
            aec_destroy(a);   /* w is dead from here */
        }
    }
}

/* aec_synth_window_build(): every grid length from the half table, into a dirty buffer; the reference is the twin's table. */
static void check_builder(void) {
    int i, j;
    for (i = 0; i < AECREF_N_GRIDS; ++i) {
        const AecRefTables* r = &AECREF_GRIDS[i];
        float w[1024 + 4];
        for (j = 0; j < 1024 + 4; ++j) w[j] = -7.0f;
        aec_synth_window_build(w, r->fft_size);
        CHECK(memcmp(w, r->window, (size_t)r->fft_size * sizeof(float)) == 0, "%d/%d: builder equals the reference",
              r->sample_rate, r->fft_size);
        CHECK(w[r->fft_size] == -7.0f, "%d/%d: builder writes exactly fft_size floats", r->sample_rate, r->fft_size);
        CHECK(w[0] == 0.0f && w[r->fft_size / 2] == 1.0f, "%d/%d: window starts at 0 and peaks at 1", r->sample_rate, r->fft_size);
        for (j = 1; j < r->fft_size; ++j)
            if (w[j] != w[r->fft_size - j]) break;
        CHECK(j == r->fft_size, "%d/%d: window is symmetric", r->sample_rate, r->fft_size);
    }
}

static void check_refusal(void) {
    AecConfig cfg;
    Aec* a = NULL;
    static const float sentinel_w[1] = { 0.0f };
    const float* w = sentinel_w;
    int n = -12345;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    cfg.fft_size = 512;
    CHECK((a = aec_create(&cfg)) != NULL, "refusal: create");
    CHECK(aec_synthesis_window(NULL, &w, &n) == -1, "NULL instance refused");
    CHECK(w == sentinel_w && n == -12345, "NULL instance leaves outputs untouched");
    CHECK(aec_synthesis_window(a, NULL, &n) == -1, "NULL window out refused");
    CHECK(n == -12345, "NULL window out leaves length untouched");
    CHECK(aec_synthesis_window(a, &w, NULL) == -1, "NULL length out refused");
    CHECK(w == sentinel_w, "NULL length out leaves window untouched");
    CHECK(aec_synthesis_window(NULL, NULL, NULL) == -1, "all NULL refused");
    aec_destroy(a);
}

int main(void) {
    static const int grids[3][2] = { { 16000, 256 }, { 16000, 512 }, { 48000, 1024 } };
    int g, sh;
    for (g = 0; g < 3; ++g)
        for (sh = 0; sh <= 1; ++sh) check_grid(grids[g][0], grids[g][1], sh);
    for (sh = 0; sh <= 1; ++sh) check_grid(8000, 256, sh);
    check_modes_and_array();
    check_builder();
    check_refusal();
    printf("test_synthesis_window: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
