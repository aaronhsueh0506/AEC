/* parity_hpf.c — Wave 1.1 gate.
 *
 * Reads python-dumped hpf_{mic,ref}.npz (input/output/state per frame) via
 * a tiny side-car loader (we don't link libnpy in C — instead the Python
 * test driver writes a flat .bin alongside the .npz with a known layout).
 *
 * To keep this self-contained without bringing in a npz parser, the harness
 * ships its own small float dumper in python/parity_export_hpf.py that
 * writes a .bin we read here. Format:
 *
 *   header:  int32 n_frames, int32 hop, int32 sample_rate, float64 cutoff_hz
 *   per frame: float64 z1_in, z2_in, z1_out, z2_out,
 *              float32 input[hop], float32 expected_output[hop]
 *
 * Gate: max-abs error across all samples must be 0 (bit-exact, since Python
 * and C use identical fp64 internal arithmetic).
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
