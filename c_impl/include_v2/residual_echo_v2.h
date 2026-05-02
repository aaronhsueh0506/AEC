/* residual_echo_v2.h — port of aec.py ResidualEchoEstimator (2507-2636).
 *
 * Only 'legacy' mode is implemented (used by BALANCED preset). 'split' mode
 * (Phase B2 ablation) deferred — Plan §取捨.
 */
#ifndef AEC_V2_RESIDUAL_ECHO_H
#define AEC_V2_RESIDUAL_ECHO_H

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ResidualEchoV2 {
    int    n_freqs;
    int    using_render_based;
    int    render_based_hold;
} ResidualEchoV2;

void residual_echo_v2_init(ResidualEchoV2* r, int n_freqs);
void residual_echo_v2_reset(ResidualEchoV2* r);

/* attribute_legacy(...) — outputs residual_echo_psd[n_freqs].
 * Mutates r->using_render_based / r->render_based_hold.
 *
 * Inputs (1:1 Python kwargs):
 *   echo_psd[n_freqs]    fp32 — |echo_spec|² already computed by caller
 *   error_psd[n_freqs]   fp32
 *   coh2[n_freqs]        fp32
 *   far_spec[n_freqs]    Complex (NULL ⇒ treat as zero PSD)
 *   far_power            fp64 (matches Python float)
 *   erle_factor          fp64
 *   dt_for_fs            fp64
 *   far_activity         fp64
 *   epc_active           int
 *   saturation_level     fp64
 *   filter_converged     int
 *   erl_estimate         fp64
 *   filter_erle[n_freqs] fp32 (mirrors aec.py filter_erle.erle)
 *   fb_erle              fp64 (mirrors aec.py fb_erle.fb_erle)
 *   residual_echo_psd_out[n_freqs] fp32 (output)
 */
void residual_echo_v2_attribute_legacy(
    ResidualEchoV2* r,
    const float*  echo_psd,
    const float*  error_psd,
    const float*  coh2,
    const Complex* far_spec,         /* NULL allowed */
    double        far_power,
    double        erle_factor,
    double        dt_for_fs,
    double        far_activity,
    int           epc_active,
    double        saturation_level,
    int           filter_converged,
    double        erl_estimate,
    const float*  filter_erle,
    double        fb_erle,
    float*        residual_echo_psd_out);

#ifdef __cplusplus
}
#endif

#endif
