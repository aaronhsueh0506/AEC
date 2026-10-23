/* Echo-saturation flag, set only while the capture itself is saturated.
 * Usable linear estimate: subtractor output max-abs vs 20000 (int16 units).
 * Otherwise: render peak * echo_path_gain * 10 vs 32000.
 * Python reference: modules/state/saturation_detector.py. */
#ifndef SATURATION_DETECTOR_H
#define SATURATION_DETECTOR_H

#define SATDET_SATURATION_THRESHOLD 20000.0f
#define SATDET_RENDER_MARGIN        10.0f
#define SATDET_INT16_SATURATION     32000.0f

typedef struct {
    int saturated_echo;   /* 0/1 */
} SaturationDetector;

/* Clears the flag. */
void saturation_detector_init(SaturationDetector *d);

/* render_block has render_block_len (>= 0) samples; flags are 0/1; the other inputs are float32.
 * Re-evaluates the flag. */
void saturation_detector_update(SaturationDetector *d,
                                const float *render_block, int render_block_len,
                                int saturated_capture,
                                int usable_linear_estimate,
                                float subtractor_s_refined_max_abs,
                                float subtractor_s_coarse_max_abs,
                                float echo_path_gain);

/* Returns the saturated-echo flag (0/1). */
int saturation_detector_saturated_echo(const SaturationDetector *d);

#endif /* SATURATION_DETECTOR_H */
