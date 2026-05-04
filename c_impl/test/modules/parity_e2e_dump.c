/* Run aec on a wav and dump per-frame raw_output + final_output to .bin
 * for diff vs Python dump_module_state.py.
 */
#include "aec.h"
#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    if (argc < 4) { fprintf(stderr, "usage: %s mic.wav ref.wav out.bin [n_frames]\n", argv[0]); return 1; }
    WavReader* mr = wav_open_read(argv[1]);
    WavReader* rr = wav_open_read(argv[2]);
    if (!mr || !rr) return 2;
    int sr = mr->info.sample_rate;
    int n  = mr->info.num_samples < rr->info.num_samples
           ? mr->info.num_samples : rr->info.num_samples;

    AecConfig cfg;
    aec_config_from_preset(&cfg, AEC_PRESET_BALANCED, sr);
    /* Match Python CLI default: enable_cng=False */
    Aec aec;
    aec_create(&aec, &cfg);
    int hop = aec_hop_size(&aec);
    int max_frames = (argc > 4) ? atoi(argv[4]) : 100;
    int n_frames = (n - hop) / hop + 1;
    if (n_frames > max_frames) n_frames = max_frames;

    FILE* fp = fopen(argv[3], "wb");
    fwrite(&hop, 4, 1, fp); fwrite(&n_frames, 4, 1, fp);

    float *mic = (float*)malloc(hop*sizeof(float));
    float *ref = (float*)malloc(hop*sizeof(float));
    float *out = (float*)malloc(hop*sizeof(float));
    for (int i = 0; i < n_frames; ++i) {
        wav_read_float(mr, mic, hop);
        wav_read_float(rr, ref, hop);
        aec_process(&aec, mic, ref, out);
        /* Dump raw_output (pre-RES), res_output (post-RES, pre-limiter),
         * final out (post-limiter), and a few scalars. */
        fwrite(aec.raw_output, sizeof(float), hop, fp);
        fwrite(aec.res_output, sizeof(float), hop, fp);
        fwrite(out, sizeof(float), hop, fp);
        double sat = aec.saturation_level;
        double erl = aec.erl_estimate;
        double mes = aec.main_err_smooth;
        double mu  = aec.simple_mu_ratio;
        double lim = aec.limiter_gain;
        fwrite(&sat,8,1,fp); fwrite(&erl,8,1,fp); fwrite(&mes,8,1,fp);
        fwrite(&mu, 8,1,fp); fwrite(&lim,8,1,fp);
    }
    fclose(fp);
    fprintf(stderr, "Dumped %d frames\n", n_frames);
    aec_destroy(&aec);
    free(mic); free(ref); free(out);
    return 0;
}
