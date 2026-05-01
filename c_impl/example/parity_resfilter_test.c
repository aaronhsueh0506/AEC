/**
 * parity_resfilter_test.c - Phase 3 ResFilter parity verifier.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "res_filter.h"

#define BLOCK    512
#define FRAME    320
#define HOP      160
#define N_FREQS  257
#define N_FRAMES 200
#define SR       16000


int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    /* Allocate input arrays */
    size_t hop_n = (size_t)N_FRAMES * HOP;
    size_t spec_n = (size_t)N_FRAMES * N_FREQS;
    float* error_hop = (float*)malloc(hop_n * sizeof(float));
    float* echo_re   = (float*)malloc(spec_n * sizeof(float));
    float* echo_im   = (float*)malloc(spec_n * sizeof(float));
    float* far_re    = (float*)malloc(spec_n * sizeof(float));
    float* far_im    = (float*)malloc(spec_n * sizeof(float));
    float* near_re   = (float*)malloc(spec_n * sizeof(float));
    float* near_im   = (float*)malloc(spec_n * sizeof(float));
    float* far_pwr   = (float*)malloc(N_FRAMES * sizeof(float));
    float* erle_factor = (float*)malloc(N_FRAMES * sizeof(float));
    float* dt_indicator = (float*)malloc(N_FRAMES * sizeof(float));
    float* over_sub  = (float*)malloc(N_FRAMES * sizeof(float));
    float* erl_estimate = (float*)malloc(N_FRAMES * sizeof(float));
    int32_t* fconv   = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* is_stat = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* epc_act = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    float* sat_level = (float*)malloc(N_FRAMES * sizeof(float));

    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { perror(argv[1]); return 2; }
    fread(error_hop,    sizeof(float),   hop_n,    fp);
    fread(echo_re,      sizeof(float),   spec_n,   fp);
    fread(echo_im,      sizeof(float),   spec_n,   fp);
    fread(far_re,       sizeof(float),   spec_n,   fp);
    fread(far_im,       sizeof(float),   spec_n,   fp);
    fread(near_re,      sizeof(float),   spec_n,   fp);
    fread(near_im,      sizeof(float),   spec_n,   fp);
    fread(far_pwr,      sizeof(float),   N_FRAMES, fp);
    fread(erle_factor,  sizeof(float),   N_FRAMES, fp);
    fread(dt_indicator, sizeof(float),   N_FRAMES, fp);
    fread(over_sub,     sizeof(float),   N_FRAMES, fp);
    fread(erl_estimate, sizeof(float),   N_FRAMES, fp);
    fread(fconv,        sizeof(int32_t), N_FRAMES, fp);
    fread(is_stat,      sizeof(int32_t), N_FRAMES, fp);
    fread(epc_act,      sizeof(int32_t), N_FRAMES, fp);
    fread(sat_level,    sizeof(float),   N_FRAMES, fp);
    fclose(fp);

    /* Build ResConfig matching Python parity_resfilter_gen.py */
    ResConfig rc;
    memset(&rc, 0, sizeof(rc));
    rc.block_size = BLOCK;
    rc.frame_size = FRAME;
    rc.hop_size = HOP;
    rc.n_freqs = N_FREQS;
    rc.g_min_db = -55.0f;
    rc.max_drop_db_per_frame = 6.0f;
    rc.max_rise_db_per_frame = 6.0f;
    rc.spectral_floor_db = -38.0f;
    rc.ne_protect_db = -16.0f;
    rc.enable_cng = 0;
    rc.alpha_echo_psd = 0.4f;
    rc.alpha_error_psd = 0.5f;
    rc.echo_method = RES_ECHO_DIRECT;
    rc.gain_type = RES_GAIN_ENR;
    rc.enable_reverb = 1;
    rc.reverb_decay = 0.65f;
    rc.reverb_gain = 1.4f;
    rc.enr_scale = 0.85f;

    ResFilter* res = res_create(&rc);
    if (!res) { fprintf(stderr, "res_create failed\n"); return 3; }

    Complex* echo_spec = (Complex*)malloc(N_FREQS * sizeof(Complex));
    Complex* far_spec  = (Complex*)malloc(N_FREQS * sizeof(Complex));
    Complex* near_spec = (Complex*)malloc(N_FREQS * sizeof(Complex));
    float*   out_hop_buf = (float*)malloc(HOP * sizeof(float));

    FILE* out_fp = fopen(argv[2], "wb");
    if (!out_fp) { perror(argv[2]); return 4; }
    int32_t hdr[6] = { BLOCK, FRAME, HOP, N_FREQS, N_FRAMES, SR };
    fwrite(hdr, sizeof(int32_t), 6, out_fp);

    for (int f = 0; f < N_FRAMES; f++) {
        for (int k = 0; k < N_FREQS; k++) {
            echo_spec[k].r = echo_re[f * N_FREQS + k];
            echo_spec[k].i = echo_im[f * N_FREQS + k];
            far_spec[k].r  = far_re [f * N_FREQS + k];
            far_spec[k].i  = far_im [f * N_FREQS + k];
            near_spec[k].r = near_re[f * N_FREQS + k];
            near_spec[k].i = near_im[f * N_FREQS + k];
        }
        res_process(res, error_hop + f * HOP,
                    echo_spec, far_spec, near_spec,
                    far_pwr[f],
                    fconv[f], erle_factor[f], dt_indicator[f],
                    over_sub[f], 0.0f /* divergence */,
                    is_stat[f], 0.0f /* shadow_dt */,
                    erl_estimate[f], epc_act[f], sat_level[f],
                    out_hop_buf);

        fwrite(out_hop_buf, sizeof(float), HOP, out_fp);
        fwrite(res_get_gain_smooth(res), sizeof(float), N_FREQS, out_fp);
        fwrite(res_get_echo_psd(res),    sizeof(float), N_FREQS, out_fp);
        fwrite(res_get_error_psd(res),   sizeof(float), N_FREQS, out_fp);
        fwrite(res_get_noise_psd(res),   sizeof(float), N_FREQS, out_fp);
    }

    fclose(out_fp);
    res_destroy(res);
    free(error_hop); free(echo_re); free(echo_im);
    free(far_re); free(far_im); free(near_re); free(near_im);
    free(far_pwr); free(erle_factor); free(dt_indicator);
    free(over_sub); free(erl_estimate); free(fconv); free(is_stat);
    free(epc_act); free(sat_level);
    free(echo_spec); free(far_spec); free(near_spec); free(out_hop_buf);
    printf("Wrote %s\n", argv[2]);
    return 0;
}
