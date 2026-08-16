/* delay_pool_test_util.h — one pool-first DelayAec3 construction for the
 * test/tool harnesses that need a bare estimator (no Aec around it).
 *
 * DelayAec3 owns no arrays of its own: every instance needs a caller block
 * sized for its exact (sample_rate, hop, num_filters) triple. That makes the
 * construction sequence four steps that must stay in this order --
 * get_mem_size -> aligned alloc -> pre-dirty -> init -- and five harnesses
 * were each spelling it out, which is five chances for one of them to drift
 * into a weaker construction than the code under test actually faces
 * (an unaligned block, a conveniently zeroed one, an unchecked size).
 *
 * The 0xA5 pre-dirty is load-bearing, not hygiene: a field the estimator
 * fails to initialise then reads as garbage rather than as a plausible 0,
 * so a missed init surfaces as a failure instead of passing by luck.
 *
 * Header-only (static inline) on purpose: several of these harnesses are
 * built standalone from a single .c file with no shared object, so anything
 * requiring its own translation unit would have to be listed in each of
 * their build lines and in the Makefile.
 *
 * Reporting is left to the caller -- the harnesses disagree on it (CHECK
 * macros vs printf+counter vs fprintf+exit) -- so failure returns NULL and
 * hands back a reason string through *why instead of choosing one.
 */
#ifndef DELAY_POOL_TEST_UTIL_H
#define DELAY_POOL_TEST_UTIL_H

#include "delay_aec3.h"
#include <stdlib.h>
#include <string.h>

/* Build one DelayAec3 on a freshly allocated, 16-byte-aligned, pre-dirtied
 * pool. Returns the pool -- which the caller owns and must free() once the
 * estimator is done -- or NULL, with *why set to a static reason string.
 * `why` may be NULL if the caller only needs the success/failure split. */
static inline void* delay_pool_init(DelayAec3* d, int sample_rate, int hop,
                                    int num_filters, const char** why) {
    size_t need = delay_aec3_get_mem_size(sample_rate, hop, num_filters);
    void* pool = NULL;
    if (why) *why = NULL;
    if (need == 0) {
        if (why) *why = "delay_aec3_get_mem_size returned 0";
        return NULL;
    }
    if (posix_memalign(&pool, 16, need) != 0 || !pool) {
        if (why) *why = "delay pool alloc failed";
        return NULL;
    }
    memset(pool, 0xA5, need);
    if (delay_aec3_init(d, pool, need, sample_rate, hop, num_filters) != 0) {
        if (why) *why = "delay_aec3_init failed";
        free(pool);
        return NULL;
    }
    return pool;
}

#endif /* DELAY_POOL_TEST_UTIL_H */
