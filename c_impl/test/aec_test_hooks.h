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

/* Drives the latch-reset portion shared by internal delay recovery and
 * external realignment.
 *
 * Why the public API will not do: every public caller also resets or shifts
 * filters and delay state, so it cannot isolate whether the shared seam itself
 * clears evidence derived from the discarded taps. */
void aec_testing_reset_filter_latches(Aec* a);

/* Exposes the legacy-10 ms additive-rate retiming helper.
 *
 * Why the public API will not do: the claim under test is that the helper
 * returns its authored argument BIT-FOR-BIT at the grid the constant was
 * calibrated on (hop=160, sample_rate=16000). That grid is a 320-sample frame,
 * which AEC_GRID_TABLE does not admit, so no aec_create() can build an
 * instance carrying that value and no public reader can observe it. Check (d2)
 * still asserts the four SHIPPED grids through the constructed instance's
 * field; this hook covers only the unconstructible reference point. */
float aec_testing_legacy10ms_rate(float legacy_per_hop, int hop,
                                  int sample_rate);

/* Array instance (src/aec_array.c) -- the shared estimator's duty-cycle machine and the lane far-spectrum borrow chain.
 *
 * Why the public API will not do: the duty cycle decimates the matched-filter ANALYSIS (one hop in K); the census
 * (AecDebugStatus duty_hops_total / _run) shows how many hops ran but not the leaky ERLE peak the collapse watchdog compares
 * against, nor the per-grid leak it converted at init, and a test of the C-only decimation needs the same stream with the
 * decimation pinned off (pin_duty_full_rate: removes ONLY the decimation, idempotent, inert outside MATCHED, not undone by
 * aec_reset()). far_spec_provenance is the per-hop evidence that all four lanes consumed one far spectrum (0 after any
 * bail-out and on a mixed warm/soft realign hop): the fusion trusts it instead of comparing every bin, so the guard that
 * stands behind it can only be exercised by watching it. 0 / 0.0f for a non-array or destroyed handle. */
void  aec_testing_array_pin_duty_full_rate(Aec* a);
/* The delay held for confirmation by the two-step admission, -1 when none is held (and for a non-array or destroyed handle).
 * Why the public API will not do: the admission's candidate has no public reader; the array baselines hash it. */
int   aec_testing_array_pending_delay_candidate(const Aec* a);
float aec_testing_array_duty_erle_leak_db(const Aec* a);
float aec_testing_array_duty_erle_peak(const Aec* a);
int   aec_testing_array_far_spec_provenance(const Aec* a);

/* The far-only analysis hoist: set_far_hoist(0) makes every lane compute its own far-only stages (the all-local reference;
 * like pin_duty_full_rate it is not undone by aec_reset()). far_sync_lanes is the bit set of lanes (bit ch) whose far history
 * has equalled lane 0's since the last reset; far_stale_lanes those that skipped their own analysis update on the last hop and
 * lean on lane 0's state. */
void     aec_testing_array_set_far_hoist(Aec* a, int enabled);
unsigned aec_testing_array_far_sync_lanes(const Aec* a);
unsigned aec_testing_array_far_stale_lanes(const Aec* a);
/* Lane ch (read-only; its analysis state is compared against an all-local twin). NULL for a non-array or a bad index. */
const Aec* aec_testing_array_lane(const Aec* a, int ch);

/* The array's gain stage (src/aec_res.c) as a standalone instance, fed from an AecResidualView exactly as the array feeds its
 * own stage (error_spec, near_spec, r2, comfort_noise, far_hop, all_converged, max_dt, max_saturation, clock_drift).
 *
 * Why the public API will not do: the stage is internal to an array instance, so a replay of a recorded frame sequence
 * (Audio_ALG pipelines/tests/pyc_parity, mode replay) cannot recompute G_res without it; recorded gains would make the
 * build-to-build comparison of the stage vacuous. The memory size and the instance follow aec_res_get_mem_size() / aec_res_init()
 * for a config at the frame's grid and the level under test; the returned gain is valid until the next call on r. */
typedef struct AecTestingRes AecTestingRes;
size_t         aec_testing_res_mem_size(const AecConfig* cfg);
AecTestingRes* aec_testing_res_init(void* mem, size_t bytes, const AecConfig* cfg);
const float*   aec_testing_res_process(AecTestingRes* r, const AecResidualView* view);

#ifdef __cplusplus
}
#endif

#endif /* AEC_TEST_HOOKS_H */
