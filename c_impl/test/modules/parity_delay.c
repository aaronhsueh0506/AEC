/* parity_delay.c — Wave 1.3 gate.
 *
 * GCC-PHAT delay estimator parity. Bit-exact NOT achievable (kiss_fft vs
 * pocketfft). Gate:
 *   - estimated_delay: exact integer match across all frames
 *   - n_updates:       exact integer match
 *   - last_par:        rtol < 1e-3 when both > 0
 */
#include "delay_est.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int read_exact(FILE* f, void* b, size_t n) {
    return fread(b, 1, n, f) == n ? 0 : -1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { fprintf(stderr, "usage: %s <delay.bin>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }

    int32_t n_frames, hop, sample_rate;
    double  max_delay_ms, init_s, period_s;
    if (read_exact(f, &n_frames, 4) || read_exact(f, &hop, 4) ||
        read_exact(f, &sample_rate, 4) ||
        read_exact(f, &max_delay_ms, 8) || read_exact(f, &init_s, 8) ||
        read_exact(f, &period_s, 8)) {
        fprintf(stderr, "header read failed\n"); return 2;
    }
    fprintf(stderr, "[delay] n_frames=%d hop=%d sr=%d max_ms=%.1f init_s=%.2f period_s=%.2f\n",
            n_frames, hop, sample_rate, max_delay_ms, init_s, period_s);

    DelayEst d;
    delay_est_init(&d, sample_rate, max_delay_ms, init_s, period_s);

    float* mic = (float*)malloc((size_t)hop * sizeof(float));
    float* ref = (float*)malloc((size_t)hop * sizeof(float));

    int    int_diffs   = 0;
    int    nup_diffs   = 0;
    double max_par_rel = 0.0;
    int    fail        = 0;

    for (int fi = 0; fi < n_frames; ++fi) {
        if (read_exact(f, mic, (size_t)hop * sizeof(float)) ||
            read_exact(f, ref, (size_t)hop * sizeof(float))) {
            fprintf(stderr, "frame %d sample read failed\n", fi); fail = 1; break;
        }
        int   exp_delay; double exp_par; int exp_nup;
        if (read_exact(f, &exp_delay, 4) || read_exact(f, &exp_par, 8) ||
            read_exact(f, &exp_nup, 4)) {
            fprintf(stderr, "frame %d expected read failed\n", fi); fail = 1; break;
        }

        delay_est_accumulate(&d, mic, ref, (size_t)hop);

        if (d.estimated_delay != exp_delay) {
            if (int_diffs < 5) {
                fprintf(stderr, "[delay] frame %d delay mismatch: C=%d PY=%d\n",
                        fi, d.estimated_delay, exp_delay);
            }
            int_diffs++;
        }
        if (d.n_updates != exp_nup) {
            if (nup_diffs < 5) {
                fprintf(stderr, "[delay] frame %d n_updates mismatch: C=%d PY=%d\n",
                        fi, d.n_updates, exp_nup);
            }
            nup_diffs++;
        }
        if (exp_par > 0.0 && d.last_par > 0.0) {
            double rel = fabs(d.last_par - exp_par) / exp_par;
            if (rel > max_par_rel) max_par_rel = rel;
        }
    }
    fclose(f);
    free(mic); free(ref);

    fprintf(stderr,
            "[delay] delay_diffs=%d  n_updates_diffs=%d  max_par_rel=%.3e\n",
            int_diffs, nup_diffs, max_par_rel);

    delay_est_free(&d);

    if (fail || int_diffs > 0 || nup_diffs > 0 || max_par_rel > 1e-3) {
        fprintf(stderr, "[delay] FAIL\n");
        return 1;
    }
    fprintf(stderr, "[delay] PASS (delay+n_updates exact, PAR rtol<1e-3)\n");
    return 0;
}
