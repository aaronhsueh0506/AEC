/* saturation.h — 1:1 port of python/aec.py SaturationDetector
 * (lines 1034-1094).
 *
 * Detects clipping in a signal and returns a smoothed saturation level
 * in [0, 1] via an asymmetric EMA. Also offers a static soft_clip helper.
 *
 * Float32-by-design: state and accumulation run in `float`, matching sample
 * I/O (converted for uniformity as part of the f32 campaign; the reference
 * Python ran fp64, drift accepted).
 */
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

/* hop_size/sample_rate retime the attack/release EMAs off their authored
 * hop=256/16000 (16 ms) grid — NOT the 10 ms reference the filter EMAs use.
 * Returns void, so a non-positive hop_size/sample_rate cannot be reported:
 * aec3_growth_rehop() absorbs it by returning the authoring value, leaving the
 * detector un-retimed rather than driving it with a NaN retention. */
void  saturation_init(Saturation* s, float threshold,   /* default 0.95 */
                      int hop_size, int sample_rate);
void  saturation_reset(Saturation* s);
/* Returns smoothed saturation_level (also stored in s->saturation_level). */
float saturation_detect(Saturation* s, const float* signal, size_t n);

/* Pure helper, mirrors Python staticmethod soft_clip(signal, knee=0.8).
 * Writes result into `out`. `out` may alias `signal`. */
void  saturation_soft_clip(const float* signal, float* out, size_t n,
                              float knee);

#ifdef __cplusplus
}
#endif

#endif /* AEC_SATURATION_H */
