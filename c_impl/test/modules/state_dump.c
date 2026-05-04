/* state_dump.c — dumps per-frame AEC v2 state for parity diff vs Python.
 * Format matches Python python/dump_state.py:
 *   header: u32 hop, u32 n_frames
 *   per frame: u32 idx + 16 floats + 4 ints
 */
#include "aec.h"
#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char* argv[]) {
    if (argc < 4) { fprintf(stderr, "usage: %s mic.wav ref.wav out.bin\n", argv[0]); return 1; }
    WavReader* mr = wav_open_read(argv[1]);
    WavReader* rr = wav_open_read(argv[2]);
    if (!mr || !rr) return 2;
    int sr = mr->info.sample_rate;
    int n  = mr->info.num_samples < rr->info.num_samples
           ? mr->info.num_samples : rr->info.num_samples;

    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
    Aec a;
    aec_create(&a, &cfg);
    int hop = aec_hop_size(&a);
    int n_frames = (n - hop) / hop + 1;

    FILE* fp = fopen(argv[3], "wb");
    uint32_t hdr[2] = { (uint32_t)hop, (uint32_t)n_frames };
    fwrite(hdr, 4, 2, fp);

    float* mic = (float*)malloc(hop*sizeof(float));
    float* ref = (float*)malloc(hop*sizeof(float));
    float* out = (float*)malloc(hop*sizeof(float));
    int frame_idx = 0;
    for (int i = 0; i + hop <= n; i += hop) {
        wav_read_float(mr, mic, hop);
        wav_read_float(rr, ref, hop);
        /* compute pre-process scalars (HPF would have applied — but we want
         * the same fields Python records inside .process) */
        aec_process(&a, mic, ref, out);
        /* Compute per-frame far_pwr and mic_pwr like Python does */
        float far_pwr = 0, mic_pwr = 0;
        for (int k = 0; k < hop; ++k) { far_pwr += ref[k]*ref[k]; mic_pwr += mic[k]*mic[k]; }
        far_pwr /= hop; mic_pwr /= hop;
        float out_e = 0;
        for (int k = 0; k < hop; ++k) out_e += out[k]*out[k];
        out_e /= hop;
        uint32_t idx = (uint32_t)frame_idx++;
        /* mu_scale: scalar approximation from simple_mu_ratio (matches diag) */
        float mu_scale = (float)a.simple_mu_ratio;
        /* Compute erle_factor like Python (same way at end of process) */
        double erle_inst = (a.near_power < 1e-10 && a.raw_error_power < 1e-10) ? 0.0 :
            10.0 * log10((a.near_power + 1e-10)/(a.raw_error_power + 1e-10));
        double erle_w = 10.0 * log10((a.erle_window_near + 1e-10)/(a.erle_window_err + 1e-10));
        double erle_for_factor = erle_inst > erle_w ? erle_inst : erle_w;
        double erle_factor = erle_for_factor / 10.0;
        if (erle_factor < 0) erle_factor = 0; if (erle_factor > 1) erle_factor = 1;

        float floats[15] = {
            mu_scale,
            (float)a.erle_factor_prev,   /* erle_factor we passed to RES this frame */
            (float)a.last_dt_indicator,
            0.0f,                              /* raw_dt */
            (float)a.res.far_activity,         /* matches Python res._diag_far_activity */
            far_pwr, mic_pwr,
            (float)a.raw_error_power,
            (float)a.erl_estimate,
            (float)a.main_err_smooth,
            (float)a.shadow_err_smooth,
            (float)a.simple_mu_ratio,
            (float)a.dt_analyzer.dt_from_energy,
            (float)a.dt_analyzer.dt_from_shadow,
            (float)a.res.diag_gain_mean,
        };
        float out_eng = out_e;
        uint32_t ints[4] = {
            (uint32_t)a.convergence.converged,
            (uint32_t)a.convergence.once_converged,
            (uint32_t)a.res.residual_est.using_render_based,
            (uint32_t)a.epc.active,
        };
        fwrite(&idx, 4, 1, fp);
        fwrite(floats, 4, 15, fp);
        fwrite(&out_eng, 4, 1, fp);
        fwrite(ints, 4, 4, fp);
    }
    fclose(fp);
    fprintf(stderr, "Dumped %d frames\n", frame_idx);
    aec_destroy(&a);
    free(mic); free(ref); free(out);
    return 0;
}
