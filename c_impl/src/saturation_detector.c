/* saturation_detector.c — C port of
 * python/modules/state/saturation_detector.py.
 *
 * The else-branch reduction (max over |render_block|) runs in float32 to match
 * `np.abs(render_block).max()` on a float32 array; the resulting f32 scalar is
 * then promoted to double (Python `float(...)`) and the margin/gain multiplies +
 * the final compare are all double. The usable-branch comparisons are f64 vs
 * f64. Built with -ffp-contract=off so nothing fuses.
 */
#include "saturation_detector.h"

void saturation_detector_init(SaturationDetector *d) {
    d->saturated_echo = 0;
}

void saturation_detector_update(SaturationDetector *d,
                                const float *render_block, int render_block_len,
                                int saturated_capture,
                                int usable_linear_estimate,
                                double subtractor_s_refined_max_abs,
                                double subtractor_s_coarse_max_abs,
                                double echo_path_gain) {
    d->saturated_echo = 0;
    if (!saturated_capture) {
        return;
    }
    if (usable_linear_estimate) {
        /* refined_max_abs > THR or coarse_max_abs > THR  (all f64). */
        d->saturated_echo =
            (subtractor_s_refined_max_abs > SATDET_SATURATION_THRESHOLD ||
             subtractor_s_coarse_max_abs  > SATDET_SATURATION_THRESHOLD) ? 1 : 0;
    } else {
        /* max_sample = float(np.abs(render_block).max()) if size>0 else 0.0
         *   → float32 max-abs reduction, then promote to double. */
        double max_sample;
        if (render_block_len > 0) {
            float max_f = -1.0f;            /* |x| >= 0, so the first sets it */
            int i;
            for (i = 0; i < render_block_len; ++i) {
                float a = render_block[i];
                float av = a < 0.0f ? -a : a;   /* np.abs (float32) */
                if (av > max_f) max_f = av;     /* np.max (float32) */
            }
            max_sample = (double)max_f;     /* float(...) promote f32→f64 */
        } else {
            max_sample = 0.0;
        }
        /* peak = max_sample * echo_path_gain * _RENDER_MARGIN  (all f64). */
        {
            double peak = max_sample * echo_path_gain * SATDET_RENDER_MARGIN;
            d->saturated_echo = (peak > SATDET_INT16_SATURATION) ? 1 : 0;
        }
    }
}

int saturation_detector_saturated_echo(const SaturationDetector *d) {
    return d->saturated_echo;
}
