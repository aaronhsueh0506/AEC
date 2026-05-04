/* saturation.h — 1:1 port of python/aec.py SaturationDetector
 * (lines 1034-1094).
 *
 * Detects clipping in a signal and returns a smoothed saturation level
 * in [0, 1] via an asymmetric EMA. Also offers a static soft_clip helper.
 *
 * Python runs in fp64; C mirrors with `double` for state and accumulation,
 * `float` for sample I/O.
 */
#ifndef AEC_SATURATION_H
#define AEC_SATURATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Saturation {
    double threshold;
    double saturation_level;
    double alpha_attack;
    double alpha_release;
} Saturation;

void   saturation_init(Saturation* s, double threshold);   /* default 0.95 */
void   saturation_reset(Saturation* s);
/* Returns smoothed saturation_level (also stored in s->saturation_level). */
double saturation_detect(Saturation* s, const float* signal, size_t n);

/* Pure helper, mirrors Python staticmethod soft_clip(signal, knee=0.8).
 * Writes result into `out`. `out` may alias `signal`. */
void   saturation_soft_clip(const float* signal, float* out, size_t n,
                               double knee);

#ifdef __cplusplus
}
#endif

#endif /* AEC_SATURATION_H */
