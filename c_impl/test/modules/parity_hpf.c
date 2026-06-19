/* parity_hpf.c — replay the binary golden from
 * python/diag/gen_hpf_golden.py through the C Hpf (mic-path high-pass biquad)
 * and assert bit-exact float32 match + per-frame fp64 state-drift checks.
 *
 * Build (standalone, from anywhere):
 *   gcc -Wall -Wextra -O2 -ffp-contract=off -std=c99 -DUSE_STANDARD_MATH \
 *       -I<path-to-repo>/c_impl/include \
 *       <path-to-repo>/c_impl/src/hpf.c \
 *       <path-to-repo>/c_impl/test/modules/parity_hpf.c \
 *       -lm -o /tmp/p_hpf
 *   python3 .../python/diag/gen_hpf_golden.py /tmp/hpf_golden.bin
 *   /tmp/p_hpf /tmp/hpf_golden.bin
 *
 * (hpf.c uses only <math.h> tan/sqrt — no fast_math.h — so -DUSE_STANDARD_MATH
 * is a no-op here, but is passed for consistency with the parity-test suite.)
 *
 * Golden format (LE):
 *   header:  int32 n_frames, int32 hop, int32 sample_rate, float64 cutoff_hz
 *   per frame: float64 z1_in, z2_in, z1_out, z2_out,
 *              float32 input[hop], float32 expected_output[hop]
 *
 * Gate: max-abs error across all samples must be 0 (bit-exact, since Python
 * and C use identical fp64 internal arithmetic with f32 output cast).
 */
#include "hpf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int read_exact(FILE* f, void* buf, size_t bytes) {
    return fread(buf, 1, bytes, f) == bytes ? 0 : -1;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <hpf.bin>\n", argv[0]);
        return 2;
    }
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }

    int32_t n_frames, hop, sample_rate;
    double cutoff_hz;
    if (read_exact(f, &n_frames, 4) || read_exact(f, &hop, 4) ||
        read_exact(f, &sample_rate, 4) || read_exact(f, &cutoff_hz, 8)) {
        fprintf(stderr, "header read failed\n"); return 2;
    }
    fprintf(stderr, "[parity_hpf] n_frames=%d hop=%d sr=%d cutoff=%.2fHz\n",
            n_frames, hop, sample_rate, cutoff_hz);

    Hpf h;
    hpf_init(&h, cutoff_hz, sample_rate);

    float* in_buf  = (float*)malloc((size_t)hop * sizeof(float));
    float* out_buf = (float*)malloc((size_t)hop * sizeof(float));
    float* exp_buf = (float*)malloc((size_t)hop * sizeof(float));

    int    fail            = 0;
    double max_abs_err     = 0.0;
    int    first_diff_frame = -1, first_diff_idx = -1;

    for (int fi = 0; fi < n_frames; ++fi) {
        double z1_in_py, z2_in_py, z1_out_py, z2_out_py;
        if (read_exact(f, &z1_in_py, 8) || read_exact(f, &z2_in_py, 8) ||
            read_exact(f, &z1_out_py, 8) || read_exact(f, &z2_out_py, 8) ||
            read_exact(f, in_buf,  (size_t)hop * sizeof(float)) ||
            read_exact(f, exp_buf, (size_t)hop * sizeof(float))) {
            fprintf(stderr, "frame %d read failed\n", fi); fail = 1; break;
        }

        /* Verify pre-call state matches Python (cumulative drift check) */
        if (h.z1 != z1_in_py || h.z2 != z2_in_py) {
            fprintf(stderr, "[parity_hpf] frame %d STATE-IN mismatch: "
                    "C(z1=%.17g z2=%.17g) PY(z1=%.17g z2=%.17g)\n",
                    fi, h.z1, h.z2, z1_in_py, z2_in_py);
            fail = 1; break;
        }

        hpf_process(&h, in_buf, out_buf, (size_t)hop);

        for (int k = 0; k < hop; ++k) {
            float diff = fabsf(out_buf[k] - exp_buf[k]);
            if (diff > max_abs_err) max_abs_err = diff;
            if (diff != 0.0f && first_diff_frame < 0) {
                first_diff_frame = fi;
                first_diff_idx   = k;
            }
        }

        /* Verify post-call state */
        if (h.z1 != z1_out_py || h.z2 != z2_out_py) {
            fprintf(stderr, "[parity_hpf] frame %d STATE-OUT mismatch: "
                    "C(z1=%.17g z2=%.17g) PY(z1=%.17g z2=%.17g)\n",
                    fi, h.z1, h.z2, z1_out_py, z2_out_py);
            fail = 1; break;
        }
    }
    fclose(f);

    fprintf(stderr, "[parity_hpf] max_abs_err=%.3e", max_abs_err);
    if (first_diff_frame >= 0) {
        fprintf(stderr, " first_diff frame=%d idx=%d",
                first_diff_frame, first_diff_idx);
    }
    fprintf(stderr, "\n");

    free(in_buf); free(out_buf); free(exp_buf);

    /* Bit-exact gate: HPF runs in fp64 internally + float-cast on store →
     * Python and C should produce byte-identical float32 outputs. */
    if (fail || max_abs_err != 0.0) {
        fprintf(stderr, "[parity_hpf] FAIL\n");
        return 1;
    }
    fprintf(stderr, "[parity_hpf] PASS (bit-exact)\n");
    return 0;
}
