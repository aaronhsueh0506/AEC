/* delay_est.h — 1:1 port of python/aec.py DelayEstimator (lines 394-521).
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
#ifndef AEC_DELAY_EST_H
#define AEC_DELAY_EST_H

#include <stddef.h>
#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct DelayEst {
    int    sample_rate;
    int    max_delay_samples;
    double init_seconds;
    double period_seconds;

    /* v3.10.0 — PAR-based confidence. Two thresholds replace the single
     * 5.0 gate that v3.8.x used. Linear blend in [low, solid]. */
    double par_low_threshold;     /* 5.0 */
    double par_solid_threshold;   /* 8.0 */

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
    int    prev_estimated_delay; /* P3c Phase 1a: lag of previous estimate
                                  * (saved in estimate() before overwrite),
                                  * used by the high-PAR fast-path's
                                  * same-lag-twice guard. -1 until first
                                  * estimate. */
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

    int is_static;               /* 1 = state placed in caller buffer */

    /* P3c Phase 1a — high-PAR fast-path knobs. When fast_path_enabled,
     * confidence promotes to 1.0 at n_updates >= 2 if last_par >=
     * fast_par_threshold AND prev_estimated_delay == estimated_delay.
     * Defaults match Python AecConfig (enabled=True, threshold=40.0). */
    int    fast_path_enabled;
    double fast_par_threshold;
} DelayEst;

void delay_est_init(DelayEst* d, int sample_rate,
                       double max_delay_ms,        /* v3.10.4: 1024.0 default */
                       double init_seconds,        /* 0.5 */
                       double period_seconds);     /* 2.0 */
size_t delay_est_get_mem_size(int sample_rate, double max_delay_ms);
void   delay_est_init_static(DelayEst* d, void* mem, size_t mem_size,
                              int sample_rate, double max_delay_ms,
                              double init_seconds, double period_seconds);
void delay_est_free(DelayEst* d);
void delay_est_reset(DelayEst* d);

/* Mirrors Python accumulate(mic, ref) -> bool. Returns 1 if a new
 * estimate was produced this call, else 0. */
int  delay_est_accumulate(DelayEst* d,
                             const float* mic, const float* ref, size_t n);

/* v3.10.0 — confidence(): linear 0..1 blend between par_low_threshold and
 * par_solid_threshold. Returns 0 when _n_updates < 3 or estimated_delay < 0.
 * P3c Phase 1a: when fast_path_enabled, returns 1.0 at n_updates >= 2 if
 * last_par >= fast_par_threshold AND prev_estimated_delay == estimated_delay
 * (overwhelming PAR + same-lag confirmation). */
double delay_est_confidence(const DelayEst* d);

/* v3.10.0 — is_solid(): confidence >= 1.0 (PAR > solid AND _n_updates >= 3). */
int    delay_est_is_solid(const DelayEst* d);

#ifdef __cplusplus
}
#endif

#endif /* AEC_DELAY_EST_H */
