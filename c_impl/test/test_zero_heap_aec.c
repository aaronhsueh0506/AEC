/* Strict zero-heap acceptance test for the caller-owned pool path.
 *
 * Proves that AEC makes no malloc/calloc/realloc/free calls across:
 *   aec_init -> N hops aec_process -> aec_reset -> N hops -> aec_destroy
 * for 50 lifecycles on both FFT backends. The platform-specific allocator
 * hook lives in audio_common/test/zero_heap_hook.c. Run with
 * `make BACKEND=kiss|ne10 test-zero-heap`; -fno-builtin is required so the
 * hook sanity allocation cannot be optimized away.
 */
#include "aec.h"
#include "zero_heap_hook.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HOPS_PER_PHASE 200
#define LIFECYCLES     50

static unsigned lcg_state = 0x12345678u;
static float lcg_sample(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((float)(int)(lcg_state >> 9) / 4194304.0f - 1.0f) * 0.25f;
}

static int run_phase(Aec* aec, int hop, int hops) {
    static float far_hop[1024], mic_hop[1024], out_hop[1024];
    for (int h = 0; h < hops; ++h) {
        for (int i = 0; i < hop; ++i) {
            far_hop[i] = lcg_sample();
            mic_hop[i] = 0.3f * far_hop[i] + 0.01f * lcg_sample();
        }
        aec_process(aec, mic_hop, far_hop, out_hop);
        for (int i = 0; i < hop; ++i) {
            if (out_hop[i] != out_hop[i]) return -1;   /* NaN guard */
        }
    }
    return 0;
}

int main(void) {
    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
    cfg.enable_cng = 1;

    size_t need = aec_get_mem_size(&cfg);
    if (need == 0) { fprintf(stderr, "FAIL: get_mem_size == 0\n"); return 1; }

    void* pool = NULL;
    if (posix_memalign(&pool, 16, need) != 0 || !pool) {
        fprintf(stderr, "FAIL: pool alloc\n"); return 1;
    }
    memset(pool, 0xA5, need);   /* dirty pool: init must not rely on zeros */

    int hop = cfg.fft_size / 2;
    long m, c, r, f;
    int failures = 0;

    /* Hook sanity: a deliberate allocation must be observed. */
    zh_reset_counters();
    {
        void* p = malloc(32);
        if (p) free(p);
    }
    zh_get_counters(&m, &c, &r, &f);
    if (m != 1 || f != 1) {
        fprintf(stderr, "FAIL: hook not counting (m=%ld f=%ld) — build with "
                        "the hook dylib/--wrap and -fno-builtin\n", m, f);
        free(pool);
        return 1;
    }

    for (int cycle = 0; cycle < LIFECYCLES; ++cycle) {
        zh_reset_counters();

        Aec* aec = aec_init(pool, need, &cfg);
        if (!aec) { fprintf(stderr, "FAIL: aec_init cycle %d\n", cycle); failures++; break; }
        if (run_phase(aec, hop, HOPS_PER_PHASE) != 0) {
            fprintf(stderr, "FAIL: non-finite output cycle %d\n", cycle); failures++; break;
        }
        aec_reset(aec);
        if (run_phase(aec, hop, HOPS_PER_PHASE) != 0) {
            fprintf(stderr, "FAIL: non-finite output post-reset cycle %d\n", cycle); failures++; break;
        }
        aec_destroy(aec);   /* pool path: must be a true no-op */
        aec_destroy(aec);   /* idempotent: second call must also be a no-op */

        zh_get_counters(&m, &c, &r, &f);
        if (m | c | r | f) {
            fprintf(stderr, "FAIL: allocator calls in cycle %d: malloc=%ld "
                            "calloc=%ld realloc=%ld free=%ld\n", cycle, m, c, r, f);
            failures++;
            break;
        }
    }

    free(pool);
    if (failures) return 1;
    printf("PASS: %d init/process/reset/destroy cycles, allocator calls all "
           "zero (init->destroy strict zero-heap)\n", LIFECYCLES);
    return 0;
}
