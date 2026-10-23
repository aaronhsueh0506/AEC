/* delay_pool_test_util.h -- one pool-first DelayAec construction for the test/tool harnesses that need a bare estimator (no
 * Aec around it).
 *
 * DelayAec owns no arrays of its own: every instance needs a caller block sized for its exact (sample_rate, hop, num_filters)
 * triple, in four steps that must stay in this order: get_mem_size -> aligned alloc -> pre-dirty -> init. Sharing the sequence
 * keeps each harness from drifting into a weaker construction than the code under test faces (an unaligned block, a zeroed
 * one, an unchecked size).
 *
 * The 0xA5 pre-dirty is load-bearing: a field the estimator fails to initialise reads as garbage rather than a plausible 0, so
 * a missed init fails instead of passing by luck.
 *
 * Header-only (static inline) because several harnesses are built standalone from a single .c file with no shared object.
 *
 * Reporting is left to the caller (the harnesses use CHECK macros, printf + counter, or fprintf + exit): failure returns NULL
 * and hands back a reason string through *why.
 */
#ifndef DELAY_POOL_TEST_UTIL_H
#define DELAY_POOL_TEST_UTIL_H

#include "delay_aec.h"
#include <stdlib.h>
#include <string.h>

/* Build one DelayAec on a freshly allocated, 16-byte-aligned, pre-dirtied
 * pool. Returns the pool -- which the caller owns and must free() once the
 * estimator is done -- or NULL, with *why set to a static reason string.
 * `why` may be NULL if the caller only needs the success/failure split. */
static inline void* delay_pool_init(DelayAec* d, int sample_rate, int hop,
                                    int num_filters, const char** why) {
    size_t need = delay_aec_get_mem_size(sample_rate, hop, num_filters);
    void* pool = NULL;
    if (why) *why = NULL;
    if (need == 0) {
        if (why) *why = "delay_aec_get_mem_size returned 0";
        return NULL;
    }
    if (posix_memalign(&pool, 16, need) != 0 || !pool) {
        if (why) *why = "delay pool alloc failed";
        return NULL;
    }
    memset(pool, 0xA5, need);
    if (delay_aec_init(d, pool, need, sample_rate, hop, num_filters) != 0) {
        if (why) *why = "delay_aec_init failed";
        free(pool);
        return NULL;
    }
    return pool;
}

#endif /* DELAY_POOL_TEST_UTIL_H */
