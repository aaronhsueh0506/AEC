/* delay_est_v2.h — 1:1 port of python/aec.py DelayEstimator (lines 394-521).
 *
 * GCC-PHAT delay estimator with overlapping segments (50 % hop) and EMA
 * cross-spectrum smoothing. Cross-spectrum is fp64 (Python uses
 * np.complex128); FFT itself is kiss_fft float (numpy uses pocketfft) — so
 * bit-exact parity is not achievable, but the integer delay output and
 * PAR scalar should match within rtol < 1e-3.
 *
 * Field names mirror Python __init__ exactly (with `_` private prefix
 * collapsed for C readability where it's already a static-like field).
 */
#ifndef AEC_V2_DELAY_EST_H
#define AEC_V2_DELAY_EST_H

#include <stddef.h>
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DelayEstV2 {
    int    sample_rate;
    int    max_delay_samples;
    double init_seconds;
    double period_seconds;

    int    seg_size;
    int    seg_hop;
    int    n_freqs;

    /* fp64 cross-spectrum (Python np.complex128) */
    double* cross_re;        /* [n_freqs] */
    double* cross_im;        /* [n_freqs] */
    double  alpha;           /* 0.6 */
    int     n_updates;

    /* fp32 sliding buffers (Python np.float32) */
    float* mic_buf;          /* [seg_size] */
    float* ref_buf;          /* [seg_size] */
    int    buf_pos;

    /* state */
    int    estimated_delay;     /* -1 until first estimate */
    int    samples_accumulated;
    int    samples_since_est;
    int    init_done;            /* 0/1 */
    int    init_samples;
    int    period_samples;
    int    n_estimates;
    double last_par;             /* PAR (peak/avg-excl-peak) */

    /* FFT scratch (uses fft_wrapper.h on top of kiss_fft) */
    FftHandle*  fft;
    Complex*    mic_spec;        /* [n_freqs] */
    Complex*    ref_spec;        /* [n_freqs] */
    Complex*    phat;            /* [n_freqs] */
    float*      gcc;             /* [seg_size] (irfft output) */
} DelayEstV2;

void delay_est_v2_init(DelayEstV2* d, int sample_rate,
                       double max_delay_ms,        /* 250.0 default */
                       double init_seconds,        /* 0.5 */
                       double period_seconds);     /* 2.0 */
void delay_est_v2_free(DelayEstV2* d);
void delay_est_v2_reset(DelayEstV2* d);

/* Mirrors Python accumulate(mic, ref) -> bool. Returns 1 if a new
 * estimate was produced this call, else 0. */
int  delay_est_v2_accumulate(DelayEstV2* d,
                             const float* mic, const float* ref, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* AEC_V2_DELAY_EST_H */
