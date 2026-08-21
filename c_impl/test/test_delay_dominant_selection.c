/* test_delay_dominant_selection.c — the product delay is the DOMINANT
 * matched-filter peak, not the pre-echo one.
 *
 * AEC3 runs its delay estimator inside a RenderDelayController and wants the
 * earliest echo-path onset, so upstream substitutes the pre-echo candidate for
 * the reported delay while still qualifying it with the dominant peak's
 * histogram. This port's product path is different: the reported sample delay
 * aligns a short PBFDKF directly, so an onset further ahead of the dominant
 * echo than that filter spans puts the echo it must cancel outside the filter
 * -- while confidence still reads 1.0, because confidence describes the
 * dominant histogram and not the substituted delay.
 *
 * The pre-echo histogram stays: it is the upstream reference and a diagnostic.
 * Only its role in selecting the production delay is gone.
 *
 * The signal is dual-path on purpose: a weak early reflection plus a stronger
 * dominant one, separated by more than the PBFDKF span. A run in which the two
 * aggregators agree proves nothing about which one was reported, so every test
 * below asserts that premise before its result -- see check_two_candidates().
 *
 * Build (standalone, from c_impl/ -- same recipe as test_delay_reset.c):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude \
 *       -I../../audio_common/include \
 *       src/delay_aec3.c src/aec3_scale.c test/test_delay_dominant_selection.c \
 *       -lm -o bin/test_delay_dominant_selection
 * Run:
 *   ./bin/test_delay_dominant_selection
 */
#include "delay_aec3.h"
#include "delay_pool_test_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; }               \
        else         { printf("PASS: %s\n", (msg)); }                         \
    } while (0)

static unsigned int g_rng = 1u;
static void rng_seed(unsigned int s) { g_rng = s ? s : 1u; }
static float rng_next(void) {
    g_rng = g_rng * 1103515245u + 12345u;
    return (float)((int)((g_rng >> 9) & 0x7FFFu) - 16384) / 16384.0f;
}

/* Separation is 1600 samples at 16 kHz = 100 ms, far wider than the 832-sample
 * (52 ms) PBFDKF the reported delay aligns. Taken from the a6941aef far-end
 * single-talk fixture, whose dominant echo sits ~6210 samples out and whose
 * earlier onset lands at 4608. */
#define DOMINANT_16K 6208
#define EARLY_16K    4608
#define EARLY_GAIN   0.30f

/* near = EARLY_GAIN * far[n - early] + far[n - dominant]. */
static int feed_dual_path(DelayAec3 *d, unsigned int seed, int n_hops, int hop,
                          int early, int dominant) {
    int total = n_hops * hop + dominant;
    float *far = (float *)malloc((size_t)total * sizeof(float));
    float *near = (float *)calloc((size_t)total, sizeof(float));
    int i, last = -1;
    if (!far || !near) { free(far); free(near); return -1; }
    rng_seed(seed);
    for (i = 0; i < total; ++i) far[i] = rng_next();
    for (i = 0; i < total; ++i) {
        float v = 0.0f;
        if (i >= early)    v += EARLY_GAIN * far[i - early];
        if (i >= dominant) v += far[i - dominant];
        near[i] = v;
    }
    for (i = 0; i < n_hops; ++i) {
        int cur;
        delay_aec3_accumulate(d, near + (size_t)i * hop, far + (size_t)i * hop, hop);
        cur = delay_aec3_estimated_delay(d);
        if (cur >= 0) last = cur;
    }
    free(far);
    free(near);
    return last;
}

/* The premise every result below rests on. */
static int check_two_candidates(const DelayAec3 *d, const char *label) {
    int dominant = d->est.aggregator.highest_peak.candidate;
    int pre_echo = d->est.aggregator.pre_echo.pre_echo_candidate;
    char msg[160];
    snprintf(msg, sizeof msg,
             "%s: dominant(%d) and pre-echo(%d) candidates are distinct",
             label, dominant, pre_echo);
    CHECK(dominant > 0 && pre_echo > 0 && dominant != pre_echo, msg);
    return dominant > 0 && pre_echo > 0 && dominant != pre_echo;
}

static void test_reported_delay_is_the_dominant_peak(int sample_rate, int hop) {
    DelayAec3 d;
    const char *why = NULL;
    void *pool = delay_pool_init(&d, sample_rate, hop, DA_NUM_FILTERS, &why);
    int scale = sample_rate / 16000;
    int early = EARLY_16K * scale, dominant = DOMINANT_16K * scale;
    int reported, selected_candidate, from_dominant, from_early;
    char msg[200];
    if (!pool) { printf("FAIL: pool (%s)\n", why ? why : "?"); g_fail++; return; }

    reported = feed_dual_path(&d, 11u, 900, hop, early, dominant);
    snprintf(msg, sizeof msg, "%d Hz/hop %d: an estimate was produced",
             sample_rate, hop);
    CHECK(reported >= 0, msg);
    if (reported < 0) { free(pool); return; }

    snprintf(msg, sizeof msg, "%d Hz/hop %d", sample_rate, hop);
    if (!check_two_candidates(&d, msg)) { free(pool); return; }

    selected_candidate =
        d.est.aggregator.highest_peak.candidate *
        DA_DOWN_SAMPLING_FACTOR * d.rate_factor;
    snprintf(msg, sizeof msg,
             "%d Hz/hop %d: public delay %d is exactly the dominant "
             "candidate %d in caller-rate samples",
             sample_rate, hop, reported, selected_candidate);
    CHECK(reported == selected_candidate, msg);

    from_dominant = abs(reported - dominant);
    from_early = abs(reported - early);
    snprintf(msg, sizeof msg,
             "%d Hz/hop %d: reported %d tracks the dominant echo %d "
             "(off by %d) rather than the earlier onset %d (off by %d)",
             sample_rate, hop, reported, dominant, from_dominant, early,
             from_early);
    CHECK(from_dominant < from_early, msg);

    snprintf(msg, sizeof msg,
             "%d Hz/hop %d: confidence describes the delay it is attached to",
             sample_rate, hop);
    CHECK(delay_aec3_confidence(&d) >= 0.5f, msg);
    free(pool);
}

/* A patch that deleted the pre-echo aggregator would satisfy every assertion
 * above; this one says it must still be computed. */
static void test_pre_echo_is_still_computed(void) {
    DelayAec3 d;
    const char *why = NULL;
    void *pool = delay_pool_init(&d, 16000, 256, DA_NUM_FILTERS, &why);
    if (!pool) { printf("FAIL: pool (%s)\n", why ? why : "?"); g_fail++; return; }
    feed_dual_path(&d, 11u, 900, 256, EARLY_16K, DOMINANT_16K);
    CHECK(d.est.aggregator.pre_echo.pre_echo_candidate > 0,
          "the pre-echo histogram is still maintained for reference");
    free(pool);
}

static void test_reset_reacquires_the_dominant_candidate(void) {
    DelayAec3 d;
    const char *why = NULL;
    void *pool = delay_pool_init(&d, 16000, 256, DA_NUM_FILTERS, &why);
    int reported;
    if (!pool) { printf("FAIL: pool (%s)\n", why ? why : "?"); g_fail++; return; }
    feed_dual_path(&d, 11u, 900, 256, EARLY_16K, DOMINANT_16K);
    delay_aec3_reset(&d);
    CHECK(d.est.aggregator.pre_echo.pre_echo_candidate == 0,
          "reset: no stale pre-echo candidate survives");
    reported = feed_dual_path(&d, 13u, 900, 256, EARLY_16K, DOMINANT_16K);
    CHECK(reported >= 0, "reset: the estimator re-acquires");
    if (reported >= 0 && check_two_candidates(&d, "after reset"))
        CHECK(abs(reported - DOMINANT_16K) < abs(reported - EARLY_16K),
              "reset: re-acquisition again reports the dominant candidate");
    free(pool);
}

int main(void) {
    test_reported_delay_is_the_dominant_peak(16000, 128);
    test_reported_delay_is_the_dominant_peak(16000, 256);
    test_reported_delay_is_the_dominant_peak(48000, 512);
    test_pre_echo_is_still_computed();
    test_reset_reacquires_the_dominant_candidate();
    if (g_fail) { printf("\n%d check(s) FAILED\n", g_fail); return 1; }
    printf("\ntest_delay_dominant_selection: ALL PASS\n");
    return 0;
}
