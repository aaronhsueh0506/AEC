/* test_duty_census.c — permanent regression test for the matched-filter
 * duty-cycle engagement census (Aec::duty_hops_total / duty_hops_run,
 * surfaced through AecDebugStatus).
 *
 * Background: the duty-cycle machine's doc comment in aec.h claims it cuts
 * "~90% of the matched-filter cost", and the cost model downstream of it has
 * been carrying that as a DESIGN estimate. Neither figure was ever measured
 * on a real run: the machine re-arms every time the estimate moves, so on an
 * unstable echo path the analysis never decimates at all and the realised
 * saving is nothing like the design figure. The census makes it measurable:
 * duty_hops_run/duty_hops_total is the realised engagement, so the saving
 * stops being a design intention and becomes a number a bench (or
 * example/aec_wav.c's one-line report) can print every run (productization
 * plan §7 measurement method).
 *
 * What this test proves (nothing else in the suite does):
 *   1. duty_hops_total counts EVERY hop through the delay block under
 *      AEC_DELAY_MATCHED -- it equals the number of aec_process() calls.
 *   2. duty_hops_run is strictly LESS than the total on a stable echo path.
 *      This is the assertion that can actually fail: a census wired to a
 *      plain hop counter instead of to run_filter, or a duty machine that
 *      never arms, both make run == total and turn this red. The signal is
 *      a fixed single-tap echo held for 3 s, which is exactly the condition
 *      the duty machine is designed to decimate (estimate solid + unchanged
 *      for delay_est_init_s), so "it never armed" is a real failure here,
 *      not a property of the stimulus.
 *   3. The realised engagement lands in the decimated regime rather than at
 *      full rate: with the default 0.5 s period at a 128-sample hop, K = 10,
 *      so a machine that arms early and stays armed tends toward ~10%
 *      engagement plus the un-armed warm-up. Asserted as a band (>0%, <60%)
 *      rather than a point value -- the exact figure depends on how fast the
 *      estimate goes solid, which is stimulus-dependent and not what this
 *      test pins.
 *   4. Only AEC_DELAY_MATCHED constructs a matched filter to decimate (see
 *      aec.h's `has_delay` doc): both AEC_DELAY_FIXED and
 *      AEC_DELAY_EXTERNAL_ALIGNED leave BOTH census counters at 0, and for
 *      the SAME reason in both cases -- has_delay is 0, so the duty block
 *      (nested inside `if (a->has_delay)`) never runs, not merely "the
 *      estimator produced nothing". Guards the reporting path from dividing
 *      by zero and from claiming a saving on a machine that was never built.
 *   5. aec_reset() zeroes the census, matching the "cumulative since
 *      aec_create/aec_init/aec_reset" contract on AecDebugStatus -- carrying
 *      pre-reset hops across would blend two duty regimes into one ratio --
 *      and counting resumes (not stuck-at-zero) afterwards.
 *   6. AecDebugStatus::delay_confidence is three-state and never reads a
 *      DelayAec3 that was never constructed: MATCHED tracks the estimator
 *      (a 5-hop run reports strictly less than a converged 3 s one), while
 *      FIXED and EXTERNAL_ALIGNED both report a constant 1.0 because their
 *      alignment is a caller contract, not a measurement. The same
 *      run_case() sweep already builds all three modes, so this is where the
 *      claim is cheapest to pin -- and it is the only test that would catch
 *      a reporting path reverting to the never-initialised struct (which
 *      answers 0.0, i.e. "estimator present but unconverged").
 *   7. The ERLE watchdog leaks the RETIMED per-hop amount
 *      (Aec::duty_erle_leak_db), reconstructed bit-exactly from the peak
 *      trajectory. test_rate_structural (d2)/(d2b) pin the field's value at
 *      every grid; only this proves the watchdog branch reads it instead of
 *      a frozen literal.
 *
 * The census is diagnostic: it must not perturb the signal path. That is not
 * asserted here (a single-process test cannot compare against a build
 * without the counters); it is verified out-of-band by rendering real
 * recordings through the pre-census and post-census builds and comparing
 * output bytes (see the productization-plan step-4 CHANGELOG entry).
 *
 * Build (standalone, from c_impl/ -- mirrors test_delay_num_filters.c):
 *   make -C ../../audio_common BACKEND=ne10 lib
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       test/test_duty_census.c $(find src -name '*.c') \
 *       $(make -s -C ../../audio_common BACKEND=ne10 print-lib-path) -lm \
 *       -o bin/test_duty_census
 * Run:  ./bin/test_duty_census
 * Also wired into the Makefile: `make test-duty-census`.
 */
#include "aec.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { printf("FAIL: %s\n", (msg)); g_fail++; } \
        else         { printf("pass: %s\n", (msg)); g_pass++; } \
    } while (0)

#define SR         16000
#define HOP        128
#define SECONDS    3
#define N_SAMPLES  (SR * SECONDS)
#define DELAY      2400              /* 150 ms @ 16 kHz -- well inside reach */

static float g_far[N_SAMPLES + DELAY];
static float g_near[N_SAMPLES + DELAY];
static float g_out[HOP];

static unsigned g_rng_state = 0x5EED1234u;
static float rng_uniform(float amp) {
    g_rng_state ^= g_rng_state << 13;
    g_rng_state ^= g_rng_state >> 17;
    g_rng_state ^= g_rng_state << 5;
    float u = (float)(g_rng_state & 0xFFFFFFu) / (float)0xFFFFFFu;
    return amp * (2.0f * u - 1.0f);
}

/* far = white noise; near = far delayed by DELAY and attenuated. A FIXED
 * delay for the whole run: the duty machine only decimates once the estimate
 * has been solid AND unchanged, so a moving echo path would legitimately hold
 * it at full rate and check 2 would be measuring the stimulus, not the code. */
static void build_signals(void) {
    int i;
    g_rng_state = 0x5EED1234u;
    for (i = 0; i < N_SAMPLES + DELAY; ++i) g_far[i] = rng_uniform(0.35f);
    for (i = 0; i < N_SAMPLES + DELAY; ++i)
        g_near[i] = (i >= DELAY) ? 0.5f * g_far[i - DELAY] : 0.0f;
}

/* Second stimulus, for check 7 only: the same converging echo path for the
 * first WD_CLEAN_S seconds, then an independent near-end noise floor on top
 * of it. build_signals()' signal cannot serve here -- its windowed ERLE only
 * ever climbs, so the watchdog's decay branch is never taken and a leak check
 * driven by it would pass over an empty loop. The added near-end raises the
 * error energy without touching the echo path, so erle_windowed falls back
 * below its running peak while the delay estimate stays exactly where it was
 * (same far, same single tap) and the duty machine stays armed -- which is
 * the state the decay branch lives in. Kept separate rather than folded into
 * build_signals() so checks 1-6 keep measuring the signal they were written
 * against. Sized by measurement, not by margin: 3 s of clean echo is what
 * checks 1-3 already show is enough for this topology to converge and arm,
 * and a 2 s tail yields 248 decay hops -- the same count an 8 s run produced,
 * so the extra seconds bought nothing. */
#define WD_SECONDS   5
#define WD_CLEAN_S   3
#define WD_SAMPLES   (SR * WD_SECONDS)
#define WD_HOPS      (WD_SAMPLES / HOP)
static float g_wd_far[WD_SAMPLES + DELAY];
static float g_wd_near[WD_SAMPLES + DELAY];

static void build_watchdog_signals(void) {
    int i;
    g_rng_state = 0x0DDBA11u;
    for (i = 0; i < WD_SAMPLES + DELAY; ++i) g_wd_far[i] = rng_uniform(0.35f);
    for (i = 0; i < WD_SAMPLES + DELAY; ++i)
        g_wd_near[i] = (i >= DELAY) ? 0.5f * g_wd_far[i - DELAY] : 0.0f;
    for (i = SR * WD_CLEAN_S; i < WD_SAMPLES + DELAY; ++i)
        g_wd_near[i] += rng_uniform(0.05f);
}

/* Run `hops` hops under the given delay_mode and return the census through
 * *total / *run, plus the polled delay_confidence through *conf (check 6).
 * fixed_delay_samples is only meaningful (and only applied) for
 * AEC_DELAY_FIXED; every other mode keeps the preset default (-1). */
static int run_case(AecDelayMode mode, int fixed_delay_samples, int hops,
                    unsigned long long* total, unsigned long long* run,
                    float* conf) {
    AecConfig cfg;
    Aec aec;
    AecDebugStatus st;
    int i;

    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
    cfg.delay_mode = mode;
    if (mode == AEC_DELAY_FIXED) cfg.fixed_delay_samples = fixed_delay_samples;
    if (aec_create(&aec, &cfg) != 0) return -1;

    for (i = 0; i < hops; ++i)
        aec_process(&aec, &g_near[i * HOP], &g_far[i * HOP], g_out);

    aec_debug_status(&aec, &st);
    *total = st.duty_hops_total;
    *run   = st.duty_hops_run;
    *conf  = st.delay_confidence;
    aec_destroy(&aec);
    return 0;
}

int main(void) {
    unsigned long long total = 0, run = 0;
    float conf = -1.0f, conf_matched = -1.0f;
    const int hops = N_SAMPLES / HOP;
    char msg[192];

    build_signals();
    build_watchdog_signals();

    /* ---- 1/2/3: AEC_DELAY_MATCHED ----------------------------------------- */
    if (run_case(AEC_DELAY_MATCHED, -1, hops, &total, &run, &conf) != 0) {
        printf("FAIL: aec_create (MATCHED)\n");
        return 1;
    }
    snprintf(msg, sizeof msg,
             "duty_hops_total == hops processed (%llu == %d)",
             total, hops);
    CHECK(total == (unsigned long long)hops, msg);

    snprintf(msg, sizeof msg,
             "duty_hops_run < duty_hops_total on a stable echo path "
             "(%llu < %llu)", run, total);
    CHECK(run < total, msg);

    CHECK(run > 0, "duty_hops_run > 0 (the matched filter did run)");

    {
        double eng = total ? 100.0 * (double)run / (double)total : -1.0;
        snprintf(msg, sizeof msg,
                 "engagement in the decimated band 0%%<e<60%% (%.2f%%)", eng);
        CHECK(eng > 0.0 && eng < 60.0, msg);
    }

    /* ---- 6a: MATCHED reports the ESTIMATOR's confidence, and reports it
     *          DYNAMICALLY. After 3 s of a fixed single-tap echo the value is
     *          on the converged end of delay_aec3_confidence's 0.0/0.5/1.0
     *          ladder; after 5 hops the estimator cannot have got there yet,
     *          so the same query must return strictly less. Pinning the long
     *          run alone would pass just as happily against a helper that had
     *          stopped consulting the estimator and returned a legal
     *          constant -- which is exactly the regression this guards, since
     *          the two other modes DO return a constant. */
    conf_matched = conf;
    snprintf(msg, sizeof msg,
             "MATCHED delay_confidence is on the estimator's ladder after 3 s "
             "(0.5/1.0, got %.2f)", (double)conf_matched);
    CHECK(conf_matched == 0.5f || conf_matched == 1.0f, msg);
    {
        unsigned long long cold_total = 0, cold_run = 0;
        float conf_cold = -1.0f;
        if (run_case(AEC_DELAY_MATCHED, -1, 5, &cold_total, &cold_run,
                     &conf_cold) != 0) {
            printf("FAIL: aec_create (MATCHED, cold)\n");
            return 1;
        }
        snprintf(msg, sizeof msg,
                 "MATCHED delay_confidence tracks the estimator (5 hops %.2f "
                 "< 3 s %.2f), not a constant",
                 (double)conf_cold, (double)conf_matched);
        CHECK(conf_cold < conf_matched, msg);
    }

    /* ---- 4: AEC_DELAY_FIXED and AEC_DELAY_EXTERNAL_ALIGNED — no estimator
     *        is ever constructed (has_delay == 0 in both), so the duty block
     *        never runs and both counters stay at 0 for the same structural
     *        reason, not two different ones. ------------------------------- */
    if (run_case(AEC_DELAY_FIXED, DELAY + HOP, hops, &total, &run, &conf) != 0) {
        printf("FAIL: aec_create (FIXED)\n");
        return 1;
    }
    CHECK(total == 0 && run == 0,
          "AEC_DELAY_FIXED leaves both census counters at 0 (no estimator)");
    /* ---- 6b ---- */
    snprintf(msg, sizeof msg,
             "AEC_DELAY_FIXED delay_confidence == 1.0 (measured out of band, "
             "got %.2f)", (double)conf);
    CHECK(conf == 1.0f, msg);

    if (run_case(AEC_DELAY_EXTERNAL_ALIGNED, -1, hops, &total, &run,
                 &conf) != 0) {
        printf("FAIL: aec_create (EXTERNAL_ALIGNED)\n");
        return 1;
    }
    CHECK(total == 0 && run == 0,
          "AEC_DELAY_EXTERNAL_ALIGNED leaves both census counters at 0 "
          "(no estimator)");
    /* ---- 6c: the caller CONTRACTED that ref arrives aligned, so there is no
     *          uncertainty to report -- and specifically not the 0.0 a read of
     *          the never-constructed DelayAec3 would produce. Same value
     *          aec_get_linear_context() has always reported for this mode. */
    snprintf(msg, sizeof msg,
             "AEC_DELAY_EXTERNAL_ALIGNED delay_confidence == 1.0 (caller "
             "contract, not a never-built estimator's 0.0; got %.2f)",
             (double)conf);
    CHECK(conf == 1.0f, msg);

    /* ---- 5: reset zeroes the census, MATCHED mode -------------------------- */
    {
        AecConfig cfg;
        Aec aec;
        AecDebugStatus st;
        int i;

        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
        if (aec_create(&aec, &cfg) != 0) {
            printf("FAIL: aec_create (reset case)\n");
            return 1;
        }
        for (i = 0; i < hops; ++i)
            aec_process(&aec, &g_near[i * HOP], &g_far[i * HOP], g_out);
        aec_debug_status(&aec, &st);
        CHECK(st.duty_hops_total > 0, "census non-zero before reset (control)");

        aec_reset(&aec);
        aec_debug_status(&aec, &st);
        CHECK(st.duty_hops_total == 0 && st.duty_hops_run == 0,
              "aec_reset zeroes the census");

        /* And it counts again afterwards -- a reset that left the counters
         * permanently stuck at zero would also pass the check above. */
        for (i = 0; i < 10; ++i)
            aec_process(&aec, &g_near[i * HOP], &g_far[i * HOP], g_out);
        aec_debug_status(&aec, &st);
        CHECK(st.duty_hops_total == 10,
              "census resumes counting after reset");
        aec_destroy(&aec);
    }

    /* ---- 7: the ERLE watchdog leaks the RETIMED amount, on the audio path.
     *
     * The leak is the machine's only wall-clock rate, and it was a bare
     * 0.001f per hop -- 0.1 dB/s only on the 10 ms grid it was calibrated on,
     * and 0.125 dB/s at this test's 8 ms hop. Aec::duty_erle_leak_db now
     * carries it, derived once at carve time; test_rate_structural (d2)
     * asserts that field's VALUE at all four shipped grids and (d2b) its
     * bit-exactness at the authoring grid.
     *
     * What neither of those can show is that the watchdog READS it. A field
     * can be retimed, asserted on, and then ignored by the branch it was
     * derived for -- the same failure mode test_rate_structural (d4) exists
     * for. So this walks the leaky peak hop by hop and reconstructs the
     * subtraction: on every hop where the peak decayed, the new peak must be
     * exactly the previous one minus duty_erle_leak_db. Bit-exact, not
     * within a tolerance -- it is the identical float operation, so anything
     * else means the branch subtracted a different number.
     *
     * Coverage is asserted, not assumed: a stimulus that never arms the duty
     * machine would satisfy an empty loop. ------------------------------- */
    {
        AecConfig cfg;
        Aec aec;
        int i, leak_hops = 0, mismatches = 0;

        aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, SR);
        if (aec_create(&aec, &cfg) != 0) {
            printf("FAIL: aec_create (watchdog leak case)\n");
            return 1;
        }

        /* 0.1 dB/s at this grid's 8 ms hop. Stated as the arithmetic rather
         * than as a magic literal so the intent survives a grid change; the
         * bit-exactness claim belongs to the authoring grid, not this one. */
        snprintf(msg, sizeof msg,
                 "duty_erle_leak_db = 0.1 dB/s x %d/%d s hop (%.9g/hop)",
                 HOP, SR, (double)aec.duty_erle_leak_db);
        CHECK(fabs((double)aec.duty_erle_leak_db * SR / HOP - 0.1) <= 1e-6,
              msg);
        snprintf(msg, sizeof msg,
                 "the 8 ms hop does NOT carry the 10 ms literal (%.9g != "
                 "0.001)", (double)aec.duty_erle_leak_db);
        CHECK(aec.duty_erle_leak_db != 0.001f, msg);

        for (i = 0; i < WD_HOPS; ++i) {
            float prev_peak = aec.duty_erle_peak;
            int   prev_active = aec.duty_active;
            aec_process(&aec, &g_wd_near[i * HOP], &g_wd_far[i * HOP], g_out);
            /* Only the steady decay branch is reconstructible here: arming
             * jumps the peak to the current ERLE, a rise re-seeds it from
             * the same, and a watchdog trip zeroes it and disarms. Each of
             * those is excluded by its own state change, not by a threshold. */
            if (prev_active && aec.duty_active &&
                aec.duty_erle_peak < prev_peak) {
                leak_hops++;
                if (aec.duty_erle_peak != prev_peak - aec.duty_erle_leak_db)
                    mismatches++;
            }
        }

        snprintf(msg, sizeof msg,
                 "the watchdog subtracts duty_erle_leak_db itself (%d leak "
                 "hops, %d mismatched)", leak_hops, mismatches);
        CHECK(mismatches == 0, msg);
        snprintf(msg, sizeof msg,
                 "the decay branch was actually exercised (%d leak hops > 0)",
                 leak_hops);
        CHECK(leak_hops > 0, msg);
        aec_destroy(&aec);
    }

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
