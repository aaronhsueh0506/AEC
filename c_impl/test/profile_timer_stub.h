/* Substitute stamp for the audit-profile regression target.
 *
 * Stands in for the microsecond timer a target whose libc has no POSIX
 * CLOCK_MONOTONIC supplies through -DAEC_NOW_US=<fn>. Returning a constant is
 * the documented way to keep AEC_STAGE_TIMING on and read zeros, and it is
 * what lets the audit assert that the substituted build links no clock at
 * all. Not shipped in the library; test fixture only. */
#ifndef AEC_TEST_PROFILE_TIMER_STUB_H
#define AEC_TEST_PROFILE_TIMER_STUB_H

#include <stdint.h>

static inline uint32_t aec_test_stub_now_us(void) { return 0u; }

#endif /* AEC_TEST_PROFILE_TIMER_STUB_H */
