/**
 * parity_aux_test.c - Phase 2bcd parity verifier
 * (RenderActivityDetector + DoubleTalkAnalyzer + FilterConvergenceAnalyzer).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "render_activity.h"
#include "dt_analyzer.h"
#include "convergence.h"

#define HOP        160
#define N_FRAMES   200
#define SHADOW_DTD_OFFSET            1.5f
#define SHADOW_DTD_ADVANTAGE_SCALE   3.0f


int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    /* Allocate input buffers */
    size_t fe_n = (size_t)N_FRAMES * HOP;
    float* far_end     = (float*)malloc(fe_n * sizeof(float));
    float* far_pwr_arr = (float*)malloc(N_FRAMES * sizeof(float));
    float* mic_pwr_arr = (float*)malloc(N_FRAMES * sizeof(float));
    float* erl_arr     = (float*)malloc(N_FRAMES * sizeof(float));
    int32_t* far_act   = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    float* main_err    = (float*)malloc(N_FRAMES * sizeof(float));
    float* shadow_err  = (float*)malloc(N_FRAMES * sizeof(float));
    int32_t* shadow_far = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    float* near_pwr    = (float*)malloc(N_FRAMES * sizeof(float));
    float* raw_err     = (float*)malloc(N_FRAMES * sizeof(float));
    int32_t* warmup    = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));

    FILE* fp = fopen(argv[1], "rb");
    if (!fp) { perror(argv[1]); return 2; }
    fread(far_end, sizeof(float), fe_n, fp);
    fread(far_pwr_arr, sizeof(float), N_FRAMES, fp);
    fread(mic_pwr_arr, sizeof(float), N_FRAMES, fp);
    fread(erl_arr, sizeof(float), N_FRAMES, fp);
    fread(far_act, sizeof(int32_t), N_FRAMES, fp);
    fread(main_err, sizeof(float), N_FRAMES, fp);
    fread(shadow_err, sizeof(float), N_FRAMES, fp);
    fread(shadow_far, sizeof(int32_t), N_FRAMES, fp);
    fread(near_pwr, sizeof(float), N_FRAMES, fp);
    fread(raw_err, sizeof(float), N_FRAMES, fp);
    fread(warmup, sizeof(int32_t), N_FRAMES, fp);
    fclose(fp);

    RenderActivityDetector* ra = ra_create();
    DoubleTalkAnalyzer* dta = dta_create(SHADOW_DTD_OFFSET, SHADOW_DTD_ADVANTAGE_SCALE);
    FilterConvergenceAnalyzer* fc = fc_create();

    FILE* out = fopen(argv[2], "wb");
    if (!out) { perror(argv[2]); return 3; }
    int32_t hdr[2] = { HOP, N_FRAMES };
    fwrite(hdr, sizeof(int32_t), 2, out);

    /* Per-frame: ra (3 fields) + dta (3 fields) + fc (3 fields) = 9 floats per frame
     * but with int fields. Use arrays per category. Output layout:
     *   ra_active[N_FRAMES] int32
     *   ra_stationary[N_FRAMES] int32
     *   ra_far_pwr[N_FRAMES] float
     *   dt_energy[N_FRAMES] float
     *   dt_shadow[N_FRAMES] float
     *   dt_advantage[N_FRAMES] float
     *   fc_converged[N_FRAMES] int32
     *   fc_once[N_FRAMES] int32
     *   fc_div[N_FRAMES] float
     */
    int32_t* ra_active     = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* ra_stationary = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    float* ra_far_pwr      = (float*)malloc(N_FRAMES * sizeof(float));
    float* dt_energy       = (float*)malloc(N_FRAMES * sizeof(float));
    float* dt_shadow_arr   = (float*)malloc(N_FRAMES * sizeof(float));
    float* dt_advantage    = (float*)malloc(N_FRAMES * sizeof(float));
    int32_t* fc_converged  = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    int32_t* fc_once       = (int32_t*)malloc(N_FRAMES * sizeof(int32_t));
    float* fc_div          = (float*)malloc(N_FRAMES * sizeof(float));

    for (int f = 0; f < N_FRAMES; f++) {
        RenderActivityState st = ra_update(ra, far_end + f * HOP, HOP);
        ra_active[f]     = st.is_active;
        ra_stationary[f] = st.is_stationary;
        ra_far_pwr[f]    = st.far_pwr;

        dta_update_shadow_dt(dta, f, shadow_far[f], main_err[f], shadow_err[f]);
        dta_update_energy_dt(dta, far_act[f], far_pwr_arr[f], mic_pwr_arr[f], erl_arr[f]);
        dt_energy[f]    = dta_get_dt_from_energy(dta);
        dt_shadow_arr[f]= dta_get_dt_from_shadow(dta);
        dt_advantage[f] = dta_get_shadow_advantage(dta);

        fc_update_divergence(fc, near_pwr[f], raw_err[f]);
        fc_update_convergence(fc, near_pwr[f], raw_err[f],
                              far_act[f], warmup[f]);
        fc_converged[f] = fc_is_converged(fc);
        fc_once[f]      = fc_is_once_converged(fc);
        fc_div[f]       = fc_get_divergence(fc);
    }

    fwrite(ra_active,     sizeof(int32_t), N_FRAMES, out);
    fwrite(ra_stationary, sizeof(int32_t), N_FRAMES, out);
    fwrite(ra_far_pwr,    sizeof(float),   N_FRAMES, out);
    fwrite(dt_energy,     sizeof(float),   N_FRAMES, out);
    fwrite(dt_shadow_arr, sizeof(float),   N_FRAMES, out);
    fwrite(dt_advantage,  sizeof(float),   N_FRAMES, out);
    fwrite(fc_converged,  sizeof(int32_t), N_FRAMES, out);
    fwrite(fc_once,       sizeof(int32_t), N_FRAMES, out);
    fwrite(fc_div,        sizeof(float),   N_FRAMES, out);
    fclose(out);

    ra_destroy(ra);
    dta_destroy(dta);
    fc_destroy(fc);
    free(far_end); free(far_pwr_arr); free(mic_pwr_arr); free(erl_arr);
    free(far_act); free(main_err); free(shadow_err); free(shadow_far);
    free(near_pwr); free(raw_err); free(warmup);
    free(ra_active); free(ra_stationary); free(ra_far_pwr);
    free(dt_energy); free(dt_shadow_arr); free(dt_advantage);
    free(fc_converged); free(fc_once); free(fc_div);

    printf("Wrote %s\n", argv[2]);
    return 0;
}
