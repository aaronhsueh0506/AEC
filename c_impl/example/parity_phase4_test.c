/**
 * parity_phase4_test.c - EPC + ShadowCopy parity verifier.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "epc_detector.h"
#include "shadow_copy.h"

#define N_FRAMES 200
#define EPC_HANGOVER 20
#define EPC_TOTAL_RISE 1.5f
#define EPC_DELTA_THR 0.3f
#define SHADOW_COPY_THRESHOLD 0.7f
#define SHADOW_COPY_HYSTERESIS 5


int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    float* far_pwr_global = (float*)malloc(N_FRAMES * sizeof(float));
    float* main_err       = (float*)malloc(N_FRAMES * sizeof(float));
    float* shadow_err     = (float*)malloc(N_FRAMES * sizeof(float));
    int32_t* is_stat      = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* fconv        = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* main_paused  = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    float* sat_level      = (float*)malloc(N_FRAMES * sizeof(float));
    float* dt_energy      = (float*)malloc(N_FRAMES * sizeof(float));
    int32_t* epc_act_arr  = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));

    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { perror(argv[1]); return 2; }
    fread(far_pwr_global, sizeof(float), N_FRAMES, fp);
    fread(main_err,      sizeof(float),   N_FRAMES, fp);
    fread(shadow_err,    sizeof(float),   N_FRAMES, fp);
    fread(is_stat,       sizeof(int32_t), N_FRAMES, fp);
    fread(fconv,         sizeof(int32_t), N_FRAMES, fp);
    fread(main_paused,   sizeof(int32_t), N_FRAMES, fp);
    fread(sat_level,     sizeof(float),   N_FRAMES, fp);
    fread(dt_energy,     sizeof(float),   N_FRAMES, fp);
    fread(epc_act_arr,   sizeof(int32_t), N_FRAMES, fp);
    fclose(fp);

    EpcDetector* epc = epc_create(EPC_HANGOVER, EPC_TOTAL_RISE, EPC_DELTA_THR);
    ShadowCopyController* sc = sc_create(SHADOW_COPY_THRESHOLD, SHADOW_COPY_HYSTERESIS, EPC_HANGOVER);

    int32_t* epc_active        = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* epc_hangover_arr  = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    float*   epc_gain_fast     = (float*)malloc(N_FRAMES * sizeof(float));
    float*   epc_gain_slow     = (float*)malloc(N_FRAMES * sizeof(float));
    int32_t* epc_event_source  = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* sc_pause_arr      = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* sc_boost_q_arr    = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* sc_reverse_arr    = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    float*   sc_baseline_arr   = (float*)malloc(N_FRAMES * sizeof(float));

    for (int f = 0; f < N_FRAMES; f++) {
        EpcEvent epv = epc_update_epv(epc, far_pwr_global[f], fconv[f], main_paused[f]);
        EpcEvent sr  = epc_update_shadow_rise(epc, main_err[f], shadow_err[f], is_stat[f]);
        int src;
        if (epv.fired) src = 2;
        else if (sr.fired) src = 3;
        else { src = 0; epc_tick_hangover(epc); }
        epc_event_source[f] = src;
        epc_active[f]       = epc_is_active(epc);
        epc_hangover_arr[f] = epc_get_hangover(epc);
        epc_gain_fast[f]    = epc_get_gain_fast(epc);
        epc_gain_slow[f]    = epc_get_gain_slow(epc);

        ShadowCopyDecision d = sc_update(sc, f, far_pwr_global[f],
                                          main_err[f], shadow_err[f],
                                          epc_act_arr[f], sat_level[f],
                                          dt_energy[f]);
        sc_pause_arr[f]    = d.pause_main;
        sc_boost_q_arr[f]  = d.boost_q;
        sc_reverse_arr[f]  = d.reverse_copy;
        sc_baseline_arr[f] = sc_copy_err_baseline(sc);
    }

    FILE* out = fopen(argv[2], "wb");
    if (!out) { perror(argv[2]); return 3; }
    int32_t hdr[1] = { N_FRAMES };
    fwrite(hdr, sizeof(int32_t), 1, out);
    fwrite(epc_active,       sizeof(int32_t), N_FRAMES, out);
    fwrite(epc_hangover_arr, sizeof(int32_t), N_FRAMES, out);
    fwrite(epc_gain_fast,    sizeof(float),   N_FRAMES, out);
    fwrite(epc_gain_slow,    sizeof(float),   N_FRAMES, out);
    fwrite(epc_event_source, sizeof(int32_t), N_FRAMES, out);
    fwrite(sc_pause_arr,     sizeof(int32_t), N_FRAMES, out);
    fwrite(sc_boost_q_arr,   sizeof(int32_t), N_FRAMES, out);
    fwrite(sc_reverse_arr,   sizeof(int32_t), N_FRAMES, out);
    fwrite(sc_baseline_arr,  sizeof(float),   N_FRAMES, out);
    fclose(out);

    epc_destroy(epc);
    sc_destroy(sc);
    free(far_pwr_global); free(main_err); free(shadow_err);
    free(is_stat); free(fconv); free(main_paused);
    free(sat_level); free(dt_energy); free(epc_act_arr);
    free(epc_active); free(epc_hangover_arr); free(epc_gain_fast);
    free(epc_gain_slow); free(epc_event_source);
    free(sc_pause_arr); free(sc_boost_q_arr); free(sc_reverse_arr);
    free(sc_baseline_arr);

    printf("Wrote %s\n", argv[2]);
    return 0;
}
