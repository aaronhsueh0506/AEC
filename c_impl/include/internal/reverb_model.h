/* Per-bin exponential IIR accumulator over a power spectrum, float32:
 * reverb[k] = (reverb[k] + power_spectrum[k] * scaling) * decay, rounded separately at each step
 * (no fused multiply-add). Python reference: modules/residual/reverb_model.py. */
#ifndef REVERB_MODEL_H
#define REVERB_MODEL_H

typedef struct {
    int    n_bins;
    float *reverb;   /* owned; length n_bins */
} ReverbModel;

void  reverb_model_init(ReverbModel *m, float *reverb_storage, int n_bins);
void  reverb_model_reset(ReverbModel *m);

/* Scalar scaling; decay <= 0 is a no-op. */
void  reverb_model_update_no_freq_shaping(ReverbModel *m,
                                          const float *power_spectrum,
                                          float scaling, float decay);

/* Per-bin scaling array; decay <= 0 is a no-op. */
void  reverb_model_update(ReverbModel *m, const float *power_spectrum,
                          const float *scaling, float decay);

#endif
