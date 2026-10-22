#ifndef AEC_POOL_TEST_UTIL_H
#define AEC_POOL_TEST_UTIL_H

#include <stddef.h>
#include <stdlib.h>

/* Caller-pool APIs require 16 bytes even when malloc guarantees only 8. */
static void *aec_test_pool_alloc(size_t bytes) {
    void *pool = NULL;
    if (!bytes || posix_memalign(&pool, 16u, bytes) != 0) return NULL;
    return pool;
}

#endif
