/* residual_echo.c — see header. 1:1 port of attribute_legacy
 * (aec.py 2556-2619).
 */
#include "residual_echo.h"
#include "erle.h"
#include <math.h>
#include <string.h>

void residual_echo_init(ResidualEcho* r, int n_freqs) {
    r->n_freqs = n_freqs;
    r->using_render_based = 0;
    r->render_based_hold  = 0;
    r->long_window_far_psd    = NULL;
    r->long_window_n_updates  = 0;
    r->long_window_alpha      = 0.993;
}
void residual_echo_set_lw_buf(ResidualEcho* r, float* lw_buf) {
    r->long_window_far_psd = lw_buf;
    if (lw_buf) memset(lw_buf, 0, (size_t)r->n_freqs * sizeof(float));
    r->long_window_n_updates = 0;
}
void residual_echo_reset_ex(ResidualEcho* r, int preserve_long_window_ema) {
    r->using_render_based = 0;
    r->render_based_hold  = 0;
    if (!preserve_long_window_ema) {
        if (r->long_window_far_psd)
            memset(r->long_window_far_psd, 0, (size_t)r->n_freqs * sizeof(float));
        r->long_window_n_updates = 0;
    }
}
void residual_echo_reset(ResidualEcho* r) {
    residual_echo_reset_ex(r, 0);
}

void residual_echo_attribute_legacy(
    ResidualEcho* r,
    const float*  echo_psd,
    const float*  error_psd,
    const float*  coh2,
    const Complex* far_spec,
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
    float*        residual_echo_psd_out) {
    const int K = r->n_freqs;

    /* confidence = compute_erle_confidence(filter_erle, fb_erle) */
    double confidence = erle_compute_confidence(filter_erle, K, fb_erle);

    /* erle_corrected = max(confidence * filter_erle + (1-conf), 0.5)
     * direct_est     = echo_psd
     * erle_est       = echo_psd / erle_corrected
     * If far_power > 1e-4: nonlinear_floor = error_psd * coh2 * far_activity * dt_weight
     *   apply max() to direct_est and erle_est. */
    double dt_weight = 1.0 - dt_for_fs;
    int active_far   = (far_power > 1e-4);

    /* Allocate scratch on stack: assume K <= 8192 */
    float erle_corrected[8192];
    float direct_est[8192];
    float erle_est[8192];
    for (int k = 0; k < K; ++k) {
        double v = confidence * (double)filter_erle[k] + (1.0 - confidence);
        if (v < 0.5) v = 0.5;
        erle_corrected[k] = (float)v;
        direct_est[k] = echo_psd[k];
        erle_est[k]   = echo_psd[k] / erle_corrected[k];
        if (active_far) {
            float nl_floor = (float)((double)error_psd[k]
                                    * (double)coh2[k]
                                    * far_activity * dt_weight);
            if (nl_floor > direct_est[k]) direct_est[k] = nl_floor;
            if (nl_floor > erle_est[k])   erle_est[k]   = nl_floor;
        }
        residual_echo_psd_out[k] = (float)((1.0 - erle_factor) * (double)direct_est[k]
                                          + erle_factor * (double)erle_est[k]);
    }

    /* v3.10.0 — maintain long-window far-PSD EMA every frame regardless of
     * render mode. Skip on silence to avoid poisoning the EMA. */
    if (far_spec != NULL && far_power > 1e-6 && r->long_window_far_psd != NULL) {
        if (r->long_window_n_updates == 0) {
            for (int k = 0; k < K; ++k) {
                double xr = far_spec[k].r, xi = far_spec[k].i;
                r->long_window_far_psd[k] = (float)(xr * xr + xi * xi);
            }
        } else {
            double a = r->long_window_alpha;
            double b = 1.0 - a;
            for (int k = 0; k < K; ++k) {
                double xr = far_spec[k].r, xi = far_spec[k].i;
                double inst = xr * xr + xi * xi;
                r->long_window_far_psd[k] =
                    (float)(a * (double)r->long_window_far_psd[k] + b * inst);
            }
        }
        r->long_window_n_updates++;
    }

    if (active_far) {
        double err_mean = 0.0;
        for (int k = 0; k < K; ++k) err_mean += (double)error_psd[k];
        err_mean /= (double)K;
        err_mean += 1e-10;
        double enr = far_power / err_mean;
        double clipped = enr / (enr + 1.0);
        if (clipped < 0.3) clipped = 0.3;
        if (clipped > 0.7) clipped = 0.7;
        double switching_threshold = 0.5 * clipped;
        const double hysteresis = 0.05;
        double effective_threshold = r->using_render_based
            ? (switching_threshold + hysteresis)
            : switching_threshold;
        int force_render = epc_active
                        || (saturation_level > 0.5)
                        || !filter_converged;
        int want_render = (erle_factor < effective_threshold) || force_render;
        if (want_render && !r->using_render_based) r->render_based_hold = 5;
        if (r->using_render_based) {
            r->render_based_hold = (r->render_based_hold > 1)
                                 ? (r->render_based_hold - 1) : 0;
        }
        int can_exit = !want_render && (r->render_based_hold == 0);
        r->using_render_based = want_render
                             || (r->using_render_based && !can_exit);

        if (r->using_render_based) {
            double blend = 1.0 - erle_factor / effective_threshold;
            if (blend < 0.0) blend = 0.0;
            if (blend > 1.0) blend = 1.0;
            /* v3.10.1: long-window EMA blended with instantaneous when
             * available. warmup_weight = min(n_updates/100, 1); lw_weight = 0.7*warmup_weight. */
            double warmup_weight = (double)r->long_window_n_updates / 100.0;
            if (warmup_weight > 1.0) warmup_weight = 1.0;
            if (warmup_weight < 0.0) warmup_weight = 0.0;
            double lw_weight = 0.7 * warmup_weight;
            int has_lw = (r->long_window_far_psd != NULL) && (lw_weight > 0.0);
            for (int k = 0; k < K; ++k) {
                double inst = 0.0;
                if (far_spec) {
                    double xr = far_spec[k].r, xi = far_spec[k].i;
                    inst = xr * xr + xi * xi;
                }
                double far_psd;
                if (has_lw) {
                    far_psd = (1.0 - lw_weight) * inst
                            + lw_weight * (double)r->long_window_far_psd[k];
                } else {
                    far_psd = inst;
                }
                double render_based_echo = far_psd * erl_estimate;
                residual_echo_psd_out[k] = (float)((1.0 - blend) * (double)residual_echo_psd_out[k]
                                                  + blend * render_based_echo);
            }
        }
    }
}
