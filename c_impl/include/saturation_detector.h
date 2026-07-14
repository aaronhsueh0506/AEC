/* saturation_detector.h — C port of
 * python/modules/state/saturation_detector.py (AecState::SaturationDetector,
 * mirrors docs/aec3_extracts/src/aec3/aec_state.cc:446-478).
 *
 * Flags echo saturation. Two paths:
 *   - usable linear estimate → look at the subtractor (refined/coarse) output
 *     max-abs vs kSaturationThreshold (20000, raw int16 amplitude units).
 *   - otherwise → look at the render block peak, scaled by echo_path_gain and a
 *     render margin (10), vs the int16 saturation point (32000).
 * Only ever fires while the capture itself is saturated.
 *
 * FLOAT32-BY-DESIGN (numpy 1.26 → C):
 *   Captured from the real pipeline (aec_state.py:333 call site):
 *     - render_block  : float32, 1-D (hop=160)
 *     - saturated_capture / usable_linear_estimate : Python bool
 *     - subtractor_s_refined_max_abs / subtractor_s_coarse_max_abs : float32
 *       (formerly kept as fp64; converted for uniformity, drift accepted)
 *     - echo_path_gain : float32 (formerly fp64)
 *     - output _saturated_echo : Python bool
 *
 *   The only numeric work is the else-branch:
 *     max_sample = np.abs(render_block).max()   # f32 reduction
 *     peak = max_sample * echo_path_gain * _RENDER_MARGIN   # all f32
 *   so the max-abs reduction runs in float32 (a `max`, not a sum — no numpy
 *   pairwise concern) and the two multiplies + the final compare are now all
 *   float32 throughout. The usable-branch comparisons are f32 vs f32. Built
 *   with -ffp-contract=off.
 *
 *   _SATURATION_THRESHOLD = 20000.0, _RENDER_MARGIN = 10.0,
 *   _INT16_SATURATION = 32000.0 (float32-by-design).
 */
#ifndef SATURATION_DETECTOR_H
#define SATURATION_DETECTOR_H

#define SATDET_SATURATION_THRESHOLD 20000.0f
#define SATDET_RENDER_MARGIN        10.0f
#define SATDET_INT16_SATURATION     32000.0f

typedef struct {
    int saturated_echo;   /* bool: _saturated_echo */
} SaturationDetector;

/* Constructor: _saturated_echo = False. */
void saturation_detector_init(SaturationDetector *d);

/* update() — strict port of saturation_detector.py:22-43.
 *   render_block : float32 array, length render_block_len (>=0).
 *   saturated_capture / usable_linear_estimate : 0/1.
 *   subtractor_s_refined_max_abs / subtractor_s_coarse_max_abs / echo_path_gain
 *     : float32 (formerly double / Python float).
 * Re-evaluates _saturated_echo. */
void saturation_detector_update(SaturationDetector *d,
                                const float *render_block, int render_block_len,
                                int saturated_capture,
                                int usable_linear_estimate,
                                float subtractor_s_refined_max_abs,
                                float subtractor_s_coarse_max_abs,
                                float echo_path_gain);

/* saturated_echo() accessor. */
int saturation_detector_saturated_echo(const SaturationDetector *d);

#endif /* SATURATION_DETECTOR_H */
