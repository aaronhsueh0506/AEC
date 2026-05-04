/* hpf.h — 1:1 port of python/aec.py HighPassFilter (lines 989-1032).
 *
 * 2nd-order Butterworth IIR HPF (bilinear transform), Direct Form II, with
 * two delay states z1/z2. Python computes coefficients in fp64 and runs the
 * sample loop in Python `float` (fp64). For bit-exact parity, this C port
 * uses `double` internally — even though Plan §設計紀律 says float32-only,
 * the rule there is "match Python dtype". Here Python is fp64 so C follows.
 */
#ifndef AEC_HPF_H
#define AEC_HPF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Hpf {
    /* Mirrors Python __init__ field order/names exactly */
    double b0, b1, b2;
    double a1, a2;
    double z1, z2;
} Hpf;

void hpf_init(Hpf* h, double cutoff_hz, int sample_rate);
void hpf_reset(Hpf* h);
/* Mirrors Python `process(x)`: in-place over a block of length n. */
void hpf_process(Hpf* h, const float* x_in, float* x_out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* AEC_HPF_H */
