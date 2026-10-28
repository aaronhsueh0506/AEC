/* Pool size, final carve position, and tail guard across optional allocations. */
#include "aec.h"
#include "hpf.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t align16(size_t n) { return (n + 15u) & ~(size_t)15u; }

int main(void) {
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
