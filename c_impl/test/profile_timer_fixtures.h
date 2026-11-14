/* Timer stamps for the audit-profile regression target. Each is substituted through -DAEC_NOW_US=<fn> exactly like a target's own
 * timer, and brought in with -include; a translation unit that does not name one of them does not emit it. Not shipped in the
 * library; test fixtures only. */
#ifndef AEC_TEST_PROFILE_TIMER_FIXTURES_H
#define AEC_TEST_PROFILE_TIMER_FIXTURES_H

#include <stdint.h>

/* Substitute stamp: stands in for the microsecond timer a target whose libc has no POSIX CLOCK_MONOTONIC supplies. Returning a
 * constant is the documented way to keep AEC_STAGE_TIMING on and read zeros, and it is what lets the audit assert that the
 * substituted build links no clock at all. */
static inline uint32_t aec_test_stub_now_us(void) { return 0u; }

/* Counting stamp for the stamp-topology assertion: a strictly increasing microsecond count that advances by one per READ. Each
 * stage window then measures the number of stamps it contains rather than a wall-clock duration, which turns the stamp layout
 * into an exact integer the driver can pin: a stamp that is added, removed or moved between stages changes the tuple, while a
 * machine that is merely fast or slow does not. The library's aec.c is the only translation unit that calls the stamp, so the
 * file-static counter below is a single counter, not one per includer. */
static inline uint32_t aec_test_count_now_us(void) {
    static uint32_t c = 0u;
    return c++;
}

#endif /* AEC_TEST_PROFILE_TIMER_FIXTURES_H */
