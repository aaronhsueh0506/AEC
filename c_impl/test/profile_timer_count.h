/* Counting stamp for the audit-profile stamp-topology assertion.
 *
 * Substituted through -DAEC_NOW_US=<fn> exactly like a target's own timer,
 * but returning a strictly increasing microsecond count that advances by one
 * per READ. Each stage window then measures the number of stamps it contains
 * rather than a wall-clock duration, which turns the stamp layout into an
 * exact integer the driver can pin: a stamp that is added, removed or moved
 * between stages changes the tuple, while a machine that is merely fast or
 * slow does not. The library's aec.c is the only translation unit that calls
 * the stamp, so the file-static counter below is a single counter, not one
 * per includer. Not shipped in the library; test fixture only. */
#ifndef AEC_TEST_PROFILE_TIMER_COUNT_H
#define AEC_TEST_PROFILE_TIMER_COUNT_H

#include <stdint.h>

static inline uint32_t aec_test_count_now_us(void) {
    static uint32_t c = 0u;
    return c++;
}

#endif /* AEC_TEST_PROFILE_TIMER_COUNT_H */
