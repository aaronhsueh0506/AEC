/* reverb_model.h — C port of python/modules/residual/reverb_model.py
 * (which mirrors AEC3 reverb_model.cc). Tiny exponential IIR accumulator over a
 * per-bin power spectrum:
 *
 *     reverb[k] = (reverb[k] + power_spectrum[k] * scaling) * decay
 *
 * Parity: the real pipeline feeds float32 power_spectrum + float32/scalar
 * scaling, so the whole update runs in float32 — three separate float32
 * roundings (mul, add, mul). Built with -ffp-contract=off so the compiler does
 * not fuse them. `reverb` is therefore `float`, not `aec3_real`.
 */
#ifndef REVERB_MODEL_H
#define REVERB_MODEL_H

typedef struct {
    int    n_bins;
    float *reverb;   /* owned; length n_bins */
} ReverbModel;

void  reverb_model_init(ReverbModel *m, float *reverb_storage, int n_bins);
void  reverb_model_reset(ReverbModel *m);

/* UpdateReverbNoFreqShaping (cc:28-39): scalar scaling. decay<=0 → no-op. */
void  reverb_model_update_no_freq_shaping(ReverbModel *m,
                                          const float *power_spectrum,
                                          float scaling, float decay);

/* UpdateReverb (cc:41-52): per-bin scaling array. decay<=0 → no-op. */
void  reverb_model_update(ReverbModel *m, const float *power_spectrum,
                          const float *scaling, float decay);

#endif /* REVERB_MODEL_H */
