/**
 * aec_dump.c - Run AEC + dump per-frame state for parity diff vs Python.
 *
 * Format must match python/dump_state.py output:
 *   header: u32 hop, u32 n_frames
 *   per frame: u32 frame_idx + 15 floats + 4 ints (matches Python struct)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aec.h"
#include "wav_io.h"
#include "hpf.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <mic.wav> <ref.wav> <out.bin>\n", argv[0]);
        return 1;
    }
    WavReader* mr = wav_open_read(argv[1]);
    WavReader* rr = wav_open_read(argv[2]);
    if (!mr || !rr) return 2;

    int sr = mr->info.sample_rate;
    int n = mr->info.num_samples < rr->info.num_samples
            ? mr->info.num_samples : rr->info.num_samples;

    AecConfig cfg = aec_config_from_preset(AEC_PRESET_BALANCED, sr);
    cfg.enable_cng = 1;
    cfg.enable_delay_est = 1;  /* match Python default for parity */
    Aec* aec = aec_create(&cfg);
    int hop = aec_get_hop_size(aec);

    /* HPF 80 Hz Butterworth applied externally — matches Python's internal
     * enable_highpass=True default. */
    Hpf* hp_mic = hpf_create(80.0f, sr);
    Hpf* hp_ref = hpf_create(80.0f, sr);

    float* mic = (float*)malloc(hop * sizeof(float));
    float* ref = (float*)malloc(hop * sizeof(float));
    float* out = (float*)malloc(hop * sizeof(float));

    FILE* fp = fopen(argv[3], "wb");
    int n_frames = (n - hop) / hop + 1;
    uint32_t hdr[2] = { (uint32_t)hop, (uint32_t)n_frames };
    fwrite(hdr, sizeof(uint32_t), 2, fp);

    int frame_idx = 0;
    for (int i = 0; i + hop <= n; i += hop) {
        wav_read_float(mr, mic, hop);
        wav_read_float(rr, ref, hop);
        hpf_process(hp_mic, mic, hop);
        hpf_process(hp_ref, ref, hop);

        float far_pwr = 0.0f, mic_pwr = 0.0f;
        for (int k = 0; k < hop; k++) {
            far_pwr += ref[k] * ref[k];
            mic_pwr += mic[k] * mic[k];
        }
        far_pwr /= hop; mic_pwr /= hop;

        aec_process(aec, mic, ref, out);
        float out_e = 0.0f;
        for (int k = 0; k < hop; k++) out_e += out[k] * out[k];
        out_e /= hop;

        uint32_t i_frame = (uint32_t)frame_idx++;
        float floats[15] = {
            aec_get_mu_scale_last(aec),
            aec_get_erle_factor_last(aec),
            aec_get_dt_indicator(aec),
            0.0f, /* raw_dt — not exported */
            aec_get_far_activity(aec),
            far_pwr, mic_pwr,
            aec_get_raw_error_power(aec),
            aec_get_erl_estimate(aec),
            aec_get_main_err_smooth(aec),
            aec_get_shadow_err_smooth(aec),
            aec_get_simple_mu_ratio(aec),
            aec_get_dt_from_energy(aec),
            aec_get_dt_from_shadow(aec),
            aec_get_res_gain_mean(aec),
        };
        float out_eng = out_e;
        uint32_t ints[4] = {
            (uint32_t)aec_is_converged(aec),
            (uint32_t)aec_is_filter_once_converged(aec),
            (uint32_t)aec_is_using_render_based(aec),
            (uint32_t)aec_is_epc_active(aec),
        };
        fwrite(&i_frame, sizeof(uint32_t), 1, fp);
        fwrite(floats, sizeof(float), 15, fp);
        fwrite(&out_eng, sizeof(float), 1, fp);
        fwrite(ints, sizeof(uint32_t), 4, fp);
    }
    fclose(fp);
    aec_destroy(aec);
    hpf_destroy(hp_mic); hpf_destroy(hp_ref);
    wav_close_read(mr); wav_close_read(rr);
    free(mic); free(ref); free(out);
    fprintf(stderr, "Dumped %d frames → %s\n", n_frames, argv[3]);
    return 0;
}
