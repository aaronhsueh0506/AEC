#ifndef AEC_POOL_TEST_UTIL_H
#define AEC_POOL_TEST_UTIL_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"

/* Caller-pool APIs require 16 bytes even when malloc guarantees only 8. */
static inline void *aec_test_pool_alloc(size_t bytes) {
    void *pool = NULL;
    if (!bytes || posix_memalign(&pool, 16u, bytes) != 0) return NULL;
    return pool;
}

/* 64-byte-aligned pool filled with `fill`: a dirty pool shows that init clears what it reads. */
static inline void *aec_test_pool_alloc_fill(size_t bytes, int fill) {
    void *p = NULL;
    if (!bytes || posix_memalign(&p, 64u, bytes) != 0) return NULL;
    memset(p, fill, bytes);
    return p;
}

/* aec_init() on a fresh filled pool; *mem_out receives the pool (free it) even when the init failed. */
static inline Aec *aec_test_new_aec(const AecConfig *cfg, void **mem_out, int fill) {
    size_t sz = aec_get_mem_size(cfg);
    void *mem = aec_test_pool_alloc_fill(sz, fill);
    Aec *a = (sz && mem) ? aec_init(mem, sz, cfg) : NULL;
    *mem_out = mem;
    return a;
}

/* One hop through the split entry points: the linear stage, then the residual stage. First non-AEC_OK status, or AEC_OK. */
static inline int aec_test_split_hop(Aec *a, const float *mic, const float *ref) {
    int st = aec_process_linear(a, mic, ref);
    return st == AEC_OK ? aec_process_residual(a, NULL) : st;
}

/* 1 when all n floats are zero (n <= 0 counts as all zero). */
static inline int aec_test_all_zero_f32(const float *x, int n) {
    int i;
    for (i = 0; i < n; ++i) if (x[i] != 0.0f) return 0;
    return 1;
}

/* xorshift32 noise in [-amp, amp]; every test seeds its own state so its stimulus stays fixed. */
static inline float aec_test_rng_uniform(unsigned *state, float amp) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    {
        float u = (float)(*state & 0xFFFFFFu) / (float)0xFFFFFFu;
        return amp * (2.0f * u - 1.0f);
    }
}

#endif
