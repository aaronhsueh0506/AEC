/* Test-only entry points into otherwise-static AEC internals.
 *
 * NOT installed and NOT part of the public API. Everything declared here is
 * compiled only when the library is built with -DAEC_TESTING (`make
 * AEC_TESTING=1`), which keys its own obj/bin tree off CFG_SIG, so a
 * production build and a testing build never share objects.
 *
 * `make test-no-testing-symbols` asserts both directions: the production
 * archive exports none of these symbols, and the AEC_TESTING archive exports
 * all of them. A one-directional check would pass against a library that
 * had stopped defining them at all.
 *
 * Add to this header only when a property genuinely cannot be observed through
 * the public API. Each entry below records why.
 */
#ifndef AEC_TEST_HOOKS_H
#define AEC_TEST_HOOKS_H

#ifndef AEC_TESTING
#error "aec_test_hooks.h requires -DAEC_TESTING (build with make AEC_TESTING=1)"
#endif

#include "aec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Drives _update_simple_mu_ratio for one hop.
 *
 * Why the public API will not do: test_rate_structural check (d6) recovers the
 * retention each branch APPLIED by running the same stimulus from two different
 * `simple_mu_ratio` starting points, so the incoming ratio cancels. Through
 * aec_process() that cancellation does not hold -- simple_mu_ratio is read
 * earlier in the same hop to scale mu, so it feeds back into the error signal
 * the recovery needs held constant, and the two probes would differ in more
 * than one variable. */
void aec_testing_update_simple_mu_ratio(Aec* a, const float* output,
                                        const float* far_end, int n);

#ifdef __cplusplus
}
#endif

#endif /* AEC_TEST_HOOKS_H */
