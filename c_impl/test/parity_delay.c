/* parity_delay.c — replay a binary golden through the C AEC3 matched-filter
 * delay estimator (DelayAec3 / delay_aec3_*) and assert the public outputs
 * are BIT-EXACT every hop:
 *   estimated_delay (int, EXACT)
 *   n_updates       (int, EXACT)
 *   is_solid        (int, EXACT)
 *   confidence      (double 0.0/0.5/1.0, EXACT)
 *
 * ⚠ The delay chain's matched-filter arithmetic is now hardcoded to the
 * float32/sliding-x2/NEON path (see delay_aec3.c's "matched-filter
 * arithmetic" note) — an intentional, sampled-cost-free divergence from the
 * Python float64 reference. python/diag/gen_delay_golden.py's golden (built
 * from Python's LegacyDelayShim output) is no longer guaranteed to match and
 * is not the primary target any more; the recommended golden is now the
 * C-REGRESSION one from test/gen_delay_c_golden.c (records this C code's own
 * output as "expected", so this checker instead catches accidental future
 * changes to delay_aec3.c). Both goldens share the same binary layout, so
 * this checker works unmodified against either.
 *
 * Build + run (from c_impl/, standalone -- does NOT link aec.c):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude \
 *       -I../../audio_common/include \
 *       src/delay_aec3.c src/aec3_scale.c test/parity_delay.c -lm -o /tmp/p_delay
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 -Iinclude -Iexample \
 *       -I../../audio_common/include \
 *       src/delay_aec3.c src/aec3_scale.c test/gen_delay_c_golden.c -lm \
 *       -o /tmp/gen_delay_golden
 *   /tmp/gen_delay_golden /tmp/delay_golden.bin
 *   /tmp/p_delay /tmp/delay_golden.bin
 */
#include "delay_aec3.h"
#include "delay_pool_test_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/delay_golden.bin";
    FILE *f = fopen(path, "rb");
    int hop, n_hops, i;
    float *near = NULL, *far = NULL;
    DelayAec3 d;
    int mism_delay = 0, mism_nupd = 0, mism_solid = 0, mism_conf = 0;
    int first_bad = -1;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &hop, 4) || !rd(f, &n_hops, 4)) { fprintf(stderr, "short header\n"); return 2; }

    near = malloc((size_t)hop * sizeof(float));
    far  = malloc((size_t)hop * sizeof(float));
    if (!near || !far) { fprintf(stderr, "oom\n"); return 2; }

    /* Pool-first construction (plan step 2): DelayAec3 carries no arrays of
     * its own, so the golden replay owns the block too, for the whole replay.
     * The golden was recorded at the default full bank, so this replays at
     * DA_NUM_FILTERS. */
    {
        const char *why = NULL;
        if (!delay_pool_init(&d, 16000, hop, DA_NUM_FILTERS, &why)) {
            fprintf(stderr, "%s\n", why); return 2;
        }
    }

    for (i = 0; i < n_hops; ++i) {
        int    exp_delay, exp_nupd, exp_solid;
        double exp_conf;
        int    got_delay, got_nupd, got_solid, bad = 0;
        double got_conf;
        int    hdr[3];

        if (!rd(f, near, (size_t)hop * sizeof(float)) ||
            !rd(f, far,  (size_t)hop * sizeof(float)) ||
            !rd(f, hdr, sizeof(hdr)) ||
            !rd(f, &exp_conf, sizeof(double))) {
            fprintf(stderr, "short read at hop %d\n", i);
            return 2;
        }
        exp_delay = hdr[0]; exp_nupd = hdr[1]; exp_solid = hdr[2];

        delay_aec3_accumulate(&d, near, far, hop);
        got_delay = delay_aec3_estimated_delay(&d);
        got_nupd  = delay_aec3_n_updates(&d);
        got_solid = delay_aec3_is_solid(&d);
        got_conf  = delay_aec3_confidence(&d);

        if (got_delay != exp_delay) { mism_delay++; bad = 1; }
        if (got_nupd  != exp_nupd)  { mism_nupd++;  bad = 1; }
        if (got_solid != exp_solid) { mism_solid++; bad = 1; }
        if (got_conf  != exp_conf)  { mism_conf++;  bad = 1; }
        if (bad && first_bad < 0) {
            first_bad = i;
            fprintf(stderr,
                    "FIRST MISMATCH hop %d:\n"
                    "  delay  got=%d exp=%d\n"
                    "  nupd   got=%d exp=%d\n"
                    "  solid  got=%d exp=%d\n"
                    "  conf   got=%.17g exp=%.17g\n",
                    i, got_delay, exp_delay, got_nupd, exp_nupd,
                    got_solid, exp_solid, got_conf, exp_conf);
        }
    }
    fclose(f);
    free(near); free(far);

    {
        int total = mism_delay + mism_nupd + mism_solid + mism_conf;
        printf("delay parity: %d hops\n", n_hops);
        printf("  estimated_delay mismatches: %d\n", mism_delay);
        printf("  n_updates       mismatches: %d\n", mism_nupd);
        printf("  is_solid        mismatches: %d\n", mism_solid);
        printf("  confidence      mismatches: %d\n", mism_conf);
        if (total) { printf(">>> FAIL (first bad hop %d)\n", first_bad); return 1; }
        printf(">>> PASS (bit-exact: estimated_delay/n_updates/is_solid/confidence all match every hop)\n");
    }
    return 0;
}
