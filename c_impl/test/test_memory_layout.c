/* Pool size, final carve position, and tail guard across optional allocations. */
#include "aec.h"
#include "hpf.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t align16(size_t n) { return (n + 15u) & ~(size_t)15u; }

/* Absolute pins (LP64: 8-byte pointers and size_t, the host and the target platform), balanced level, default delay
 * mode. They are literals, not recomputed from the build under test. Derivation from the sizes recorded on the 115e10c
 * tree (sizeof(Aec) 6336; pool 364416 / 491600 / 1156544 with the shadow filter, 321536 / 430912 / 1002608 without):
 *   sizeof(Aec) 6352 = 6336 + 8 (AecPostRunScratch.output_base) + 8 (delay_shift_judgment_pre_H_error)
 *                      - 8 (AecPost.sqrt2_sin_lut) + 4 (AecPost.reset_serial) + 4 (AecPost.output_guard_active);
 *   pool = 115e10c pool + align16(K * sizeof(Complex))   rendering-base buffer (aec_get_output_base)
 *                       + align16(K * sizeof(float))     main-filter H_error snapshot of the bad-relock revert
 *                       + align16(6352) - 6336           growth of the 16-byte-aligned Aec struct
 *   with K = fft / 2 + 1: +1040 + 528 + 16 (16k/256), +2064 + 1040 + 16 (16k/512), +4112 + 2064 + 16 (48k/1024). */
#define AEC_SIZEOF_PIN 6352u
static int check_pinned_sizes(void) {
    static const struct { int sr, fft; size_t with_shadow, without_shadow; } pin[3] = {
        {16000, 256, 366000u, 323120u}, {16000, 512, 494720u, 434032u}, {48000, 1024, 1162736u, 1008800u} };
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
        AecConfig cfg;
        size_t now_s, now_n;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, pin[i].sr);
        cfg.fft_size = pin[i].fft;
        now_s = aec_get_mem_size(&cfg);
        cfg.enable_shadow = 0;
        now_n = aec_get_mem_size(&cfg);
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

int main(void) {
    if (check_pinned_sizes()) return 1;
    const int rates[] = {16000, 16000, 48000};
    const int ffts[] = {256, 512, 1024};
    int checked = 0;
    for (int g = 0; g < 3; ++g)
    for (int shadow = 0; shadow < 2; ++shadow)
    for (int observer = 0; observer < 2; ++observer)
    for (int highpass = 0; highpass < 2; ++highpass)
    for (int delay = 0; delay < 3; ++delay) {
        AecConfig cfg;
        aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, rates[g]);
        cfg.fft_size = ffts[g];
        cfg.enable_shadow = shadow;
        cfg.live_observer_enabled = observer;
        cfg.enable_highpass = highpass;
        if (delay == 0) {
            cfg.delay_mode = AEC_DELAY_MATCHED;
        } else if (delay == 1) {
            cfg.delay_mode = AEC_DELAY_FIXED;
            cfg.fixed_delay_samples = 123;
        } else {
            cfg.delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
            cfg.enable_delay_est = 0;
        }
        size_t need = aec_get_mem_size(&cfg);
        unsigned char* pool = NULL;
        if (!need || posix_memalign((void**)&pool, 16, need + 64u) != 0)
            return 1;
        memset(pool, 0xA5, need + 64u);
        if (aec_init(pool, need - 1u, &cfg) != NULL) {
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
        Aec* a = aec_init(pool, need, &cfg);
        if (!a) { free(pool); return 1; }
        /* HPF is the optional final block; scr_erl_arr precedes it. */
        const unsigned char* end = a->hp_mic
            ? (const unsigned char*)a->hp_mic + align16(hpf_get_mem_size())
            : (const unsigned char*)a->scr_erl_arr
                + align16((size_t)a->n_freqs * sizeof(float));
        int failed = end != pool + need;
        for (size_t i = need; i < need + 64u; ++i)
            if (pool[i] != 0xA5) failed = 1;
        aec_destroy(a);
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
