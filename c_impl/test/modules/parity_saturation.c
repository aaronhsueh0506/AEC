/* parity_saturation.c -- parity gate for the Saturation detector.
 *
 * Usage: parity_saturation <sat.bin>. The file holds int32 n_frames, int32 hop, float64 threshold, then per frame float64
 * level_in, float64 level_out and float32 input[hop] (the Python detector's state before and after each frame). The C port
 * must reproduce level_out from level_in and the input exactly.
 *
 * Gate: the smoothed level matches Python bit-exactly (both use fp64 internally with the same op order).
 */
#include "saturation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int read_exact(FILE* f, void* buf, size_t bytes) {
    return fread(buf, 1, bytes, f) == bytes ? 0 : -1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) { fprintf(stderr, "usage: %s <sat.bin>\n", argv[0]); return 2; }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }

    int32_t n_frames, hop;
    double threshold;
    if (read_exact(f, &n_frames, 4) || read_exact(f, &hop, 4) ||
        read_exact(f, &threshold, 8)) {
        fprintf(stderr, "header read failed\n"); return 2;
    }
    fprintf(stderr, "[parity_saturation] n_frames=%d hop=%d thr=%.3f\n",
            n_frames, hop, threshold);

    Saturation s;
    /* The golden's header carries hop but not sample_rate; 16000 reproduces the authored constants
     * at hop=256. */
    saturation_init(&s, threshold, hop, 16000);

    float* in_buf = (float*)malloc((size_t)hop * sizeof(float));
    int    fail   = 0;
    double max_state_err = 0.0;
    int    first_diff_frame = -1;

    for (int fi = 0; fi < n_frames; ++fi) {
        double lvl_in_py, lvl_out_py;
        if (read_exact(f, &lvl_in_py, 8) || read_exact(f, &lvl_out_py, 8) ||
            read_exact(f, in_buf, (size_t)hop * sizeof(float))) {
            fprintf(stderr, "frame %d read failed\n", fi); fail = 1; break;
        }
        if (s.saturation_level != lvl_in_py) {
            fprintf(stderr, "[saturation] frame %d STATE-IN drift: "
                    "C=%.17g PY=%.17g\n",
                    fi, s.saturation_level, lvl_in_py);
            fail = 1; break;
        }
        double lvl_c = saturation_detect(&s, in_buf, (size_t)hop);
        double err   = fabs(lvl_c - lvl_out_py);
        if (err > max_state_err) max_state_err = err;
        if (lvl_c != lvl_out_py && first_diff_frame < 0) first_diff_frame = fi;
    }
    fclose(f);
    free(in_buf);

    fprintf(stderr, "[parity_saturation] max_state_err=%.3e", max_state_err);
    if (first_diff_frame >= 0)
        fprintf(stderr, " first_diff frame=%d", first_diff_frame);
    fprintf(stderr, "\n");

    if (fail || max_state_err != 0.0) {
        fprintf(stderr, "[parity_saturation] FAIL\n");
        return 1;
    }
    fprintf(stderr, "[parity_saturation] PASS (bit-exact)\n");
    return 0;
}
