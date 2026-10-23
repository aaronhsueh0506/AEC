/* saturation_detector.c -- see saturation_detector.h. All comparisons are float32. */
#include "saturation_detector.h"
#include "utility/simd_aec.h"

void saturation_detector_init(SaturationDetector *d) {
    d->saturated_echo = 0;
}

void saturation_detector_update(SaturationDetector *d,
                                const float *render_block, int render_block_len,
                                int saturated_capture,
                                int usable_linear_estimate,
                                float subtractor_s_refined_max_abs,
                                float subtractor_s_coarse_max_abs,
                                float echo_path_gain) {
    d->saturated_echo = 0;
    if (!saturated_capture) {
        return;
    }
    if (usable_linear_estimate) {
        d->saturated_echo =
            (subtractor_s_refined_max_abs > SATDET_SATURATION_THRESHOLD ||
             subtractor_s_coarse_max_abs  > SATDET_SATURATION_THRESHOLD) ? 1 : 0;
    } else {
        float max_sample = sk_max_abs_f32(render_block, render_block_len);
        {
            float peak = max_sample * echo_path_gain * SATDET_RENDER_MARGIN;
            d->saturated_echo = (peak > SATDET_INT16_SATURATION) ? 1 : 0;
        }
    }
}

int saturation_detector_saturated_echo(const SaturationDetector *d) {
    return d->saturated_echo;
}
