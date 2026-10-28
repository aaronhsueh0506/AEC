/* test_synthesis_window.c -- aec_synthesis_window() public accessor.
 *
 *   1. On the 3 contracted grids (16k/256, 16k/512, 48k/1024) and both shadow settings, the returned table has the
 *      length the instance synthesizes with (== fft_size == block_size) and is element-for-element identical (memcmp)
 *      to the library's internal per-rate table (AECB_RATE_TABLE, aec_balanced_config.h); the pointer is stable
 *      across processing and aec_reset(), and an AcIstft configured with it (the documented way) initialises and
 *      borrows exactly that table.
 *   2. A NULL instance, NULL window or NULL length is refused with -1 and leaves the outputs untouched.
 *
 * Build and run: `make test-synthesis-window` (from c_impl/). */
#include "aec.h"
#include "aec_balanced_config.h"
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

static void check_grid(int sr, int fft, int shadow) {
    AecConfig cfg;
    Aec* a = (Aec*)calloc(1, sizeof(Aec));
    const AecBalancedRateDims* rd = aecb_rate_cfg(sr, fft);
    const float* w = NULL;
    const float* w2 = NULL;
    int n = 0, n2 = 0, i;
    float hop_in[1024], far_in[1024];
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, sr);
    cfg.fft_size = fft;
    cfg.enable_shadow = shadow;
    CHECK(rd != NULL, "%d/%d: rate row exists", sr, fft);
    CHECK(a && aec_create(a, &cfg) == 0, "%d/%d shadow=%d: create", sr, fft, shadow);
    if (!rd || !a) return;

    CHECK(aec_synthesis_window(a, &w, &n) == 0, "%d/%d: accessor succeeds", sr, fft);
    CHECK(n == fft && n == rd->synth_window_len, "%d/%d: length %d == fft_size %d == table length %d", sr, fft, n, fft,
          rd->synth_window_len);
    CHECK(w != NULL, "%d/%d: window pointer set", sr, fft);   /* not compared by address: the header table is static per TU */
    CHECK(w && memcmp(w, rd->synth_window, (size_t)rd->synth_window_len * sizeof(float)) == 0,
          "%d/%d: values equal the internal table", sr, fft);

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
    free(a);
}

static void check_refusal(void) {
    AecConfig cfg;
    Aec* a = (Aec*)calloc(1, sizeof(Aec));
    static const float sentinel_w[1] = { 0.0f };
    const float* w = sentinel_w;
    int n = -12345;
    aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
    cfg.fft_size = 512;
    CHECK(a && aec_create(a, &cfg) == 0, "refusal: create");
    CHECK(aec_synthesis_window(NULL, &w, &n) == -1, "NULL instance refused");
    CHECK(w == sentinel_w && n == -12345, "NULL instance leaves outputs untouched");
    CHECK(aec_synthesis_window(a, NULL, &n) == -1, "NULL window out refused");
    CHECK(n == -12345, "NULL window out leaves length untouched");
    CHECK(aec_synthesis_window(a, &w, NULL) == -1, "NULL length out refused");
    CHECK(w == sentinel_w, "NULL length out leaves window untouched");
    CHECK(aec_synthesis_window(NULL, NULL, NULL) == -1, "all NULL refused");
    if (a) { aec_destroy(a); free(a); }
}

int main(void) {
    static const int grids[3][2] = { { 16000, 256 }, { 16000, 512 }, { 48000, 1024 } };
    int g, sh;
    for (g = 0; g < 3; ++g)
        for (sh = 0; sh <= 1; ++sh) check_grid(grids[g][0], grids[g][1], sh);
    check_refusal();
    printf("test_synthesis_window: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
