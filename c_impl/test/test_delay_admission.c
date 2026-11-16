/* test_delay_admission.c -- the single-channel Path B admission of a shifted delay (aec.c aec_track_delay_shift()), driven with
 * chosen offers on an instance whose estimator has locked: a movement inside AEC_DELAY_CHANGE_MIN_SAMPLES is absorbed, a first
 * sighting is held, a repeat within AEC_DELAY_CHANGE_CONFIRM_SAMPLES of the held value is admitted (the alignment moves, the
 * bad-relock judgment arms, nothing stays held), a repeat 16 or more away replaces the candidate, and a candidate that spends
 * AEC_DELAY_CHANGE_CANDIDATE_TTL hops unrefreshed expires. Needs the testing library (make test-delay-admission).
 *
 * Run from c_impl/:  make test-delay-admission */
#include "aec.h"
#include "aec_white_box.h"
#include "aec_test_hooks.h"
#include "aec_test_tuning.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SR   16000
#define FFT  256
#define HOP  128
#define D    400
#define HOPS 700

static int g_fail = 0;
static long g_checks = 0;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_fail++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static unsigned long long g_rng = 0x9E3779B97F4A7C15ull;
static float noise_next(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return ((float)((g_rng >> 11) & 0xFFFFFF) / 8388608.0f - 1.0f) * 0.25f;
}

/* A MATCHED instance whose estimator has locked a 400-sample echo path. */
static Aec* locked_instance(void) {
    static float far_hist[D + HOPS * HOP];
    AecTestCfg cfg;
    Aec* a;
    int h, i;
    for (i = 0; i < (int)(sizeof far_hist / sizeof far_hist[0]); ++i) far_hist[i] = noise_next();
    aec_tcfg_from_level(&cfg, AEC_LEVEL_BALANCED, SR);
    cfg.c.fft_size = FFT;
    cfg.c.residual = AEC_RESIDUAL_OFF;
    cfg.c.delay_mode = AEC_DELAY_MATCHED;
    a = aec_tcfg_create(&cfg);
    if (!a) return NULL;
    for (h = 0; h < HOPS; ++h) {
        float mic[HOP], out[HOP];
        for (i = 0; i < HOP; ++i) mic[i] = 0.5f * far_hist[D + h * HOP + i - D];
        if (aec_process(a, mic, far_hist + D + h * HOP, out) != AEC_OK) { aec_destroy(a); return NULL; }
    }
    return a;
}

static void check_admission(void) {
    Aec* a = locked_instance();
    int base, moved, i;
    CHECK(a != NULL, "the instance runs");
    if (!a) return;
    base = a->current_delay;
    CHECK(base >= 0 && delay_aec_confidence(&a->delay) >= 0.5f, "the estimator has locked (delay %d, confidence %.2f)",
          base, (double)delay_aec_confidence(&a->delay));
    if (base < 0) { aec_destroy(a); return; }
    moved = base + 160;

    aec_delay_admission_clear(&a->delay_admission);
    aec_testing_track_delay_shift(a, base + AEC_DELAY_CHANGE_MIN_SAMPLES, 1);
    CHECK(a->current_delay == base && a->delay_admission.ttl == 0, "a movement of exactly the band is absorbed, not held");
    aec_testing_track_delay_shift(a, base + AEC_DELAY_CHANGE_MIN_SAMPLES + 1, 1);
    CHECK(a->current_delay == base && a->delay_admission.ttl == AEC_DELAY_CHANGE_CANDIDATE_TTL &&
              a->delay_admission.candidate == base + AEC_DELAY_CHANGE_MIN_SAMPLES + 1,
          "one sample beyond the band is held with a full life");

    /* The confirmation window is strict: a repeat CONFIRM_SAMPLES away replaces the candidate, one sample nearer admits it. */
    aec_testing_track_delay_shift(a, moved, 1);
    CHECK(a->current_delay == base && a->delay_admission.candidate == moved, "a sighting elsewhere replaces the candidate");
    aec_testing_track_delay_shift(a, moved + AEC_DELAY_CHANGE_CONFIRM_SAMPLES, 1);
    CHECK(a->current_delay == base && a->delay_admission.candidate == moved + AEC_DELAY_CHANGE_CONFIRM_SAMPLES &&
              a->delay_admission.ttl == AEC_DELAY_CHANGE_CANDIDATE_TTL,
          "a repeat exactly the confirmation window away is a new first sighting");
    aec_testing_track_delay_shift(a, moved + 2 * AEC_DELAY_CHANGE_CONFIRM_SAMPLES - 1, 1);
    CHECK(a->current_delay == moved + 2 * AEC_DELAY_CHANGE_CONFIRM_SAMPLES - 1 && a->delay_admission.ttl == 0 &&
              a->delay_admission.candidate == -1 && a->delay_shift_judgment_active &&
              a->delay_shift_judgment_pre_delay == base,
          "a repeat inside the window is admitted: the alignment moves, the judgment arms, nothing stays held");

    /* A candidate that is not refreshed expires after its life, whatever the estimator did in between. */
    base = a->current_delay;
    moved = base + 200;
    aec_testing_track_delay_shift(a, moved, 1);
    for (i = 0; i < AEC_DELAY_CHANGE_CANDIDATE_TTL - 1; ++i) aec_testing_track_delay_shift(a, base, 0);
    CHECK(a->delay_admission.ttl == 1 && a->delay_admission.candidate == moved, "the candidate survives its life minus one hop");
    aec_testing_track_delay_shift(a, base, 0);
    CHECK(a->delay_admission.ttl == 0 && a->delay_admission.candidate == -1, "it expires one hop later");
    aec_testing_track_delay_shift(a, moved, 1);
    CHECK(a->current_delay == base && a->delay_admission.ttl == AEC_DELAY_CHANGE_CANDIDATE_TTL,
          "a single reappearance after expiry is a new first sighting, not an admission");

    /* An ineligible hop never offers: the repeat is not admitted, and the life is spent. */
    aec_testing_track_delay_shift(a, moved, 0);
    CHECK(a->current_delay == base && a->delay_admission.ttl == AEC_DELAY_CHANGE_CANDIDATE_TTL - 1,
          "an ineligible hop ages the candidate and admits nothing");
    aec_reset(a);
    CHECK(a->delay_admission.ttl == 0 && a->delay_admission.candidate == -1, "aec_reset() drops a held candidate");
    aec_destroy(a);
}

int main(void) {
    check_admission();
    printf("%ld checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
