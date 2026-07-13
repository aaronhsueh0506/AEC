/* parity_saturation_detector.c — replay the binary golden from
 * python/diag/gen_saturation_detector_golden.py through the C
 * SaturationDetector and assert exact (bit-for-bit bool) match.
 * WS5 Phase 5.2 (AecState layer).
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=gnu99 \
 *       -Ic_impl/include \
 *       c_impl/src/saturation_detector.c \
 *       c_impl/test/parity_saturation_detector.c \
 *       -lm -o /tmp/p_satdet
 *   python3 .../python/diag/gen_saturation_detector_golden.py /tmp/satdet_golden.bin
 *   /tmp/p_satdet /tmp/satdet_golden.bin
 */
#include "saturation_detector.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int rd(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/satdet_golden.bin";
    FILE *f = fopen(path, "rb");
    int n_frames, i, mism = 0, n_true = 0;
    float *rb = NULL;
    int rb_cap = 0;
    SaturationDetector d;

    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (!rd(f, &n_frames, 4)) return 2;

    saturation_detector_init(&d);

    for (i = 0; i < n_frames; ++i) {
        int rb_len;
        int32_t bools[2];
        double scal[3];  /* refined, coarse, echo_path_gain */
        unsigned char exp_flag;

        if (!rd(f, &rb_len, 4)) { fprintf(stderr, "short read len %d\n", i); return 2; }
        if (rb_len > rb_cap) {
            free(rb);
            rb = malloc((size_t)rb_len * sizeof(float));
            rb_cap = rb_len;
        }
        if (rb_len > 0 && !rd(f, rb, (size_t)rb_len * sizeof(float))) {
            fprintf(stderr, "short read rb %d\n", i); return 2;
        }
        if (!rd(f, bools, sizeof(bools)) ||
            !rd(f, scal, sizeof(scal)) ||
            !rd(f, &exp_flag, 1)) {
            fprintf(stderr, "short read scal %d\n", i); return 2;
        }

        saturation_detector_update(&d, rb, rb_len,
                                   (int)bools[0], (int)bools[1],
                                   scal[0], scal[1], scal[2]);
        {
            int got = saturation_detector_saturated_echo(&d);
            if (got) n_true++;
            if (got != (int)exp_flag) {
                if (mism < 10) {
                    fprintf(stderr,
                        "frame %d MISMATCH: C=%d golden=%d "
                        "(len=%d sat=%d usable=%d refined=%.6g coarse=%.6g epg=%.9g)\n",
                        i, got, (int)exp_flag, rb_len, (int)bools[0], (int)bools[1],
                        scal[0], scal[1], scal[2]);
                }
                mism++;
            }
        }
    }
    fclose(f);
    free(rb);

    printf("saturation_detector parity: %d frames, %d saturated_echo=True, "
           "mismatches=%d\n", n_frames, n_true, mism);
    if (mism) { printf(">>> FAIL\n"); return 1; }
    printf(">>> PASS (bit-exact)\n");
    return 0;
}
