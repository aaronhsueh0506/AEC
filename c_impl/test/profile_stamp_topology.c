/* profile_stamp_topology.c — audit-profile's value-level half.
 *
 * The rest of audit-profile proves the PROFILE knob reaches the compile and
 * that a release build links no clock. That says nothing about WHERE the
 * stamps sit, and a stage window whose opening stamp is missing reports 0 --
 * indistinguishable from a stage that genuinely cost under a microsecond.
 * steering_us is the field most exposed to that: it exists precisely because
 * the window between the main filter and the post block had no stamps at all
 * and read as free.
 *
 * Built against a library whose stamp advances by one per read (see
 * profile_timer_count.h), so each window's value is the number of stamps it
 * spans. One hop through aec_process() then produces an exact tuple:
 *
 *   read 0  entry, opens frontend
 *   read 1  opens the delay bracket        delay_us    = 2 - 1     = 1
 *   read 2  closes it
 *   read 3  closes frontend, opens linear  frontend_us = (3-0) - 1 = 2
 *   read 4  closes linear, opens steering  linear_us   = 4 - 3     = 1
 *   read 5  closes steering, opens res     steering_us = 5 - 4     = 1
 *   read 6  closes res                     res_us      = 6 - 5     = 1
 *
 * Seven reads, five windows, no window empty -- which is the contract
 * AecStageTiming's doc comment states.
 */
#include "aec.h"
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;

static void expect(const char *label, unsigned long got, unsigned long want) {
    if (got == want) {
        printf("pass: %s == %lu\n", label, want);
    } else {
        printf("FAIL: %s == %lu (got %lu)\n", label, want, got);
        fails++;
    }
}

int main(void) {
    AecConfig cfg;
    AecStageTiming t;
    Aec *a;
    float *mic, *far, *out;
    int hop;

    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, 16000);
    a = (Aec *)malloc(sizeof(Aec));
    if (!a) { printf("FAIL: out of memory\n"); return 2; }
    if (aec_create(a, &cfg) != 0) { printf("FAIL: aec_create\n"); free(a); return 2; }

    hop = a->hop_size;
    mic = (float *)calloc((size_t)hop, sizeof(float));
    far = (float *)calloc((size_t)hop, sizeof(float));
    out = (float *)malloc((size_t)hop * sizeof(float));
    if (!mic || !far || !out) { printf("FAIL: out of memory\n"); return 2; }

    aec_process(a, mic, far, out);
    aec_get_last_timing(a, &t);

    printf("stamp topology: delay=%u frontend=%u linear=%u steering=%u res=%u\n",
           t.delay_us, t.frontend_us, t.linear_us, t.steering_us, t.res_us);

    expect("delay_us stamps",    t.delay_us,    1);
    expect("frontend_us stamps", t.frontend_us, 2);
    expect("linear_us stamps",   t.linear_us,   1);
    expect("steering_us stamps", t.steering_us, 1);
    expect("res_us stamps",      t.res_us,      1);

    free(mic); free(far); free(out);
    aec_destroy(a); free(a);

    printf(fails ? "profile_stamp_topology: %d FAILED\n" : "profile_stamp_topology: all pass\n", fails);
    return fails != 0;
}
