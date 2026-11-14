/* aec_test_tuning.h -- a public AecConfig together with the internal AecTuning it resolves to, for tests that override tuning
 * (the shadow filter, the high-pass, a retiming constant, the delay ring, ...). Needs the AEC_TESTING hooks (aec_test_hooks.h)
 * and include/internal/ on the include path. A test that never overrides tuning uses aec_config_from_level() and the public calls as a product
 * does. Changing c.sample_rate or c.level after the fact leaves t stale: call aec_tcfg_refresh() (it drops every override). */
#ifndef AEC_TEST_TUNING_H
#define AEC_TEST_TUNING_H

#include "aec_pool_test_util.h"
#include "aec_test_hooks.h"
#include "aec_tuning.h"

typedef struct AecTestCfg {
    AecConfig c;
    AecTuning t;
} AecTestCfg;

static inline void aec_tcfg_refresh(AecTestCfg* x) {
    aec_testing_tuning_defaults(&x->t, &x->c);
}

static inline void aec_tcfg_from_level(AecTestCfg* x, AecLevel level, int sample_rate) {
    aec_config_from_level(&x->c, level, sample_rate);
    aec_tcfg_refresh(x);
}

static inline size_t aec_tcfg_mem_size(const AecTestCfg* x) {
    return aec_testing_get_mem_size(&x->c, &x->t);
}

static inline Aec* aec_tcfg_init(void* mem, size_t mem_size, const AecTestCfg* x) {
    return aec_testing_init(mem, mem_size, &x->c, &x->t);
}

static inline int aec_tcfg_mem_requirements(const AecTestCfg* x, AecMemReq* out) {
    return aec_testing_get_mem_requirements(&x->c, &x->t, out);
}

static inline Aec* aec_tcfg_create(const AecTestCfg* x) {
    return aec_testing_create(&x->c, &x->t);
}

/* aec_test_new_aec() for an AecTestCfg: a filled pool, *mem_out receives it (free it) even when the init failed. */
static inline Aec* aec_tcfg_new_aec(const AecTestCfg* x, void** mem_out, int fill) {
    size_t sz = aec_tcfg_mem_size(x);
    void* mem = aec_test_pool_alloc_fill(sz, fill);
    Aec* a = (sz && mem) ? aec_tcfg_init(mem, sz, x) : NULL;
    *mem_out = mem;
    return a;
}

#endif
