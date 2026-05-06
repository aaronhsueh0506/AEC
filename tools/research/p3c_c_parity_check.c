/* Quick standalone check: simulate the same state Python observes on
 * 7GT (n_updates=2, last_par=119, estimated_delay=12606) and verify
 * delay_est_confidence returns 1.0 when prev==current (fast-path),
 * and 0.0 when prev != current. */
#include "delay_est.h"
#include <stdio.h>

int main(void) {
    DelayEst d;
    delay_est_init(&d, 16000, 1024.0, 0.5, 2.0);

    /* fast-path off: legacy n_updates>=3 gate */
    d.fast_path_enabled = 0;
    d.estimated_delay = 12606;
    d.prev_estimated_delay = 12606;
    d.last_par = 119.0;
    d.n_updates = 2;
    printf("legacy n=2 par=119: confidence=%.3f (expect 0.000)\n",
           delay_est_confidence(&d));

    /* fast-path on, prev matches: should return 1.0 */
    d.fast_path_enabled = 1;
    d.fast_par_threshold = 40.0;
    printf("fp n=2 par=119 prev==cur: confidence=%.3f (expect 1.000)\n",
           delay_est_confidence(&d));

    /* fast-path on, prev mismatch: should NOT promote */
    d.prev_estimated_delay = 12500;
    printf("fp n=2 par=119 prev!=cur: confidence=%.3f (expect 0.000)\n",
           delay_est_confidence(&d));

    /* fast-path on, par below threshold: should NOT promote */
    d.prev_estimated_delay = 12606;
    d.last_par = 30.0;
    printf("fp n=2 par=30 prev==cur: confidence=%.3f (expect 0.000)\n",
           delay_est_confidence(&d));

    /* fast-path on, prev=-1 (first estimate ever): should NOT promote */
    d.prev_estimated_delay = -1;
    d.last_par = 119.0;
    printf("fp n=2 par=119 prev=-1: confidence=%.3f (expect 0.000)\n",
           delay_est_confidence(&d));

    /* legacy gate still works at n>=3 */
    d.fast_path_enabled = 1;
    d.n_updates = 3;
    d.prev_estimated_delay = -1;
    d.last_par = 10.0;  /* between 5 and 8? actually 10 > 8 → 1.0 */
    printf("n=3 par=10: confidence=%.3f (expect 1.000)\n",
           delay_est_confidence(&d));

    delay_est_free(&d);
    return 0;
}
