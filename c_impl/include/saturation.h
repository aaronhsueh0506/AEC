/* Clipping detector: smoothed saturation level in [0, 1] (asymmetric EMA) plus a soft_clip helper.
 * State and accumulation are float32. Python reference: modules/preprocessing.py (SaturationDetector). */
#ifndef AEC_SATURATION_H
#define AEC_SATURATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Saturation {
    float threshold;
    float saturation_level;
    float alpha_attack;
    float alpha_release;
} Saturation;

/* hop_size/sample_rate retime the attack/release EMAs from the 256/16000 authoring grid.
 * Non-positive values leave the detector un-retimed (init cannot report errors). */
void  saturation_init(Saturation* s, float threshold,   /* default 0.95 */
                      int hop_size, int sample_rate);
void  saturation_reset(Saturation* s);
/* Returns smoothed saturation_level (also stored in s->saturation_level). */
float saturation_detect(Saturation* s, const float* signal, size_t n);

/* Soft-clips `signal` above `knee` (typically 0.8) into `out`; `out` may alias `signal`. */
void  saturation_soft_clip(const float* signal, float* out, size_t n,
                              float knee);

#ifdef __cplusplus
}
#endif

#endif /* AEC_SATURATION_H */
